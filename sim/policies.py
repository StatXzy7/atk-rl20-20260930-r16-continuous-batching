"""调度策略：FCFS 静态分批 与 continuous batching（chunked prefill）。

每个策略实现 ``plan(ctx, now) -> list[Action]``：在给定时刻观察
*当前运行态*（已 prefill / 已生成长度、持有页、队列年龄），产出
本 step 要执行的动作。策略不允许读取 RequestSpec.output_len。

动作只描述"做什么"；真正推进时间、记账、终止判定由 engine 完成。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from .memory import PagePool
from .model import PolicyKind, Request, SimConfig, Status


# ---------------------------------------------------------------------------
# 动作
# ---------------------------------------------------------------------------


@dataclass
class PrefillAction:
    """处理 req 的一段 prefill（可能是首轮，也可能是抢占后的重算）。"""

    req: Request
    tokens: int  # 本段处理的 token 数（chunk 边界由 planner 决定）

    @property
    def kind(self) -> str:
        return "prefill"


@dataclass
class DecodeAction:
    req: Request

    @property
    def kind(self) -> str:
        return "decode"


Action = object  # 实际为 PrefillAction | DecodeAction


# ---------------------------------------------------------------------------
# 规划上下文：封装本 step 内的预算、页池与抢占操作
# ---------------------------------------------------------------------------


class PreemptionDenied(RuntimeError):
    """无法找到合法的抢占受害者。"""


class SchedulingContext:
    """planner 与引擎资源之间的唯一接口。

    页申请时机严格遵守：
      * prefill chunk：执行前按"写完本段后的 token 位置"在页边界申请；
      * decode：执行前申请，跨边界时先拿新页。
    """

    def __init__(self, cfg: SimConfig, pool: PagePool, active: Dict[str, Request],
                 log) -> None:
        self.cfg = cfg
        self.pool = pool
        self.active = active
        self.log = log
        self.budget_left = cfg.token_budget
        self.prefill_tokens = 0
        self.decode_tokens = 0
        # 本 step 已获得动作的请求，避免同一请求每步重复推进
        self._scheduled: set = set()
        # 本 step 已为每个请求规划的 prefill token（同一请求可由
        # "保底 chunk + 余量补齐"组成两个动作），页数计算必须计入
        self._planned_prefill: Dict[str, int] = {}

    # ---- 观察 ----
    def waiting_requests(self) -> List[Request]:
        return [r for r in self.active.values() if r.status == Status.WAITING]

    def prefill_requests(self) -> List[Request]:
        return [r for r in self.active.values() if r.status == Status.PREFILL]

    def decoders(self) -> List[Request]:
        return [r for r in self.active.values() if r.status == Status.DECODING]

    def is_scheduled(self, req: Request) -> bool:
        return req.rid in self._scheduled

    # ---- 页分配 ----
    def pages_after_prefill(self, req: Request, tokens: int) -> int:
        positions = (req.tokens_in_kv
                     + self._planned_prefill.get(req.rid, 0) + tokens)
        return -(-max(positions, 1) // self.cfg.page_size)

    def pages_after_decode(self, req: Request) -> int:
        positions = req.tokens_in_kv + 1
        return -(-max(positions, 1) // self.cfg.page_size)

    def _alloc_for(self, req: Request, need_pages: int) -> bool:
        have = len(req.pages)
        if need_pages <= have:
            return True
        want = need_pages - have
        if self.pool.free_pages < want:
            return False
        req.pages.extend(self.pool.allocate(want))
        return True

    # ---- 抢占 ----
    def preempt(self, victim: Request, now: float, reason: str) -> None:
        """抢占一个请求：释放全部私有页，计入重算工作量。

        * DECODING 受害者：丢弃全部 KV，之后需重算 prompt+已生成位置；
        * PREFILL 受害者（部分 prefill 的新请求）：丢弃部分进度，重头再来。
        必须先检查抢占上限，再做任何不可逆修改。
        """
        if victim.rid in self._scheduled:
            raise PreemptionDenied("不能抢占本 step 已调度的请求")
        if victim.num_preemptions >= self.cfg.max_preemptions:
            raise PreemptionDenied(
                f"{victim.rid} 抢占次数达到上限 "
                f"{self.cfg.max_preemptions}，拒绝再次抢占")
        wasted = victim.tokens_in_kv
        n = len(victim.pages)
        victim.num_preemptions += 1
        victim.recompute_tokens += wasted
        victim.recompute_pending = wasted
        victim.prefilled = 0
        self.pool.free_many(victim.pages)
        victim.pages = []
        if victim.status == Status.DECODING:
            # 首次抢占 decoder：需要重算 prompt + 已生成的位置
            victim.recompute_extra = victim.generated
        else:
            # 抢占一个正在重算的 partial prefill：仍需重放到原来的
            # prompt+generated 位置，recompute_extra 必须保留。
            pass
        victim.status = Status.WAITING
        victim.decode_gap = 0
        victim.prefill_enqueue_time = now
        self.log({
            "event": "preempt", "time": now, "rid": victim.rid,
            "wasted_tokens": wasted, "freed_pages": n,
            "preempt_count": victim.num_preemptions, "reason": reason,
        })

    # ---- 授权动作 ----
    def grant_prefill(self, req: Request, now: float,
                      allow_preempt: bool = False,
                      want: Optional[int] = None,
                      protected: bool = False) -> Optional[PrefillAction]:
        if req.status == Status.WAITING:
            # 安全准入：只能从空闲页一次性预留整个 prompt 所需页，
            # 绝不抢占任何 decoder（这是防活锁的关键规则）。
            need_whole = -(-req.prefill_target // self.cfg.page_size)
            if self.pool.free_pages < need_whole:
                return None
            if not self._alloc_for(req, need_whole):
                return None
            req.status = Status.PREFILL
            req.prefilled = 0
            if req.admit_time is None:
                req.admit_time = now
            self.log({"event": "admit", "time": now, "rid": req.rid,
                      "policy": PolicyKind.CONT.value,
                      "pages": need_whole})
        if req.status != Status.PREFILL:
            return None
        if req.rid in self._scheduled:
            return None
        already = self._planned_prefill.get(req.rid, 0)
        remaining = req.prefill_target - req.prefilled - already
        hard_cap = self.cfg.chunk_max - already
        cap = hard_cap if want is None else min(want, hard_cap)
        tokens = min(remaining, cap, self.budget_left)
        if tokens <= 0:
            return None
        need_pages = self.pages_after_prefill(req, tokens)
        if not self._alloc_for(req, need_pages):
            # prefill 绝不驱逐 decoder；受保护请求也只能放弃更年轻的
            # partial prefill（见 _victims_for），做不到就等待。
            if not allow_preempt:
                return None
            try:
                self._evict_for(req, need_pages, now, reason="prefill",
                                protected=protected)
            except PreemptionDenied:
                return None
            if not self._alloc_for(req, need_pages):
                return None
        act = PrefillAction(req, tokens)
        self.budget_left -= tokens
        self.prefill_tokens += tokens
        self._planned_prefill[req.rid] = already + tokens
        self._scheduled.add(req.rid)
        return act

    def grant_decode(self, req: Request, now: float,
                     allow_preempt: bool = False,
                     protected: bool = False) -> Optional[DecodeAction]:
        if req.status != Status.DECODING or req.rid in self._scheduled:
            return None
        if self.budget_left < 1:
            return None
        need_pages = self.pages_after_decode(req)
        if not self._alloc_for(req, need_pages):
            if not allow_preempt:
                return None
            try:
                self._evict_for(req, need_pages, now, reason="decode",
                                protected=protected)
            except PreemptionDenied:
                return None
            if not self._alloc_for(req, need_pages):
                return None
        act = DecodeAction(req)
        self.budget_left -= 1
        self.decode_tokens += 1
        self._scheduled.add(req.rid)
        return act

    # ---- 抢占受害者选择 ----
    def _evict_for(self, req: Request, need_pages: int, now: float,
                   reason: str, protected: bool = False) -> None:
        """为让 req 获得 need_pages 页，按最小损失顺序驱逐其他请求。

        受保护（更老）的 prefill 请求之间不互相驱逐；正在本 step
        调度的请求不可驱逐。
        """
        victims = self._victims_for(req, need_pages, now, reason, protected)
        freed = 0
        have = len(req.pages)
        target = need_pages - have
        for v in victims:
            if freed >= target:
                break
            freed += len(v.pages)
            self.preempt(v, now, reason=reason)
        if self.pool.free_pages < target:
            raise PreemptionDenied("驱逐后仍无足够页")

    def _victims_for(self, req: Request, need_pages: int, now: float,
                     reason: str, protected: bool) -> List[Request]:
        others = [r for r in self.active.values()
                  if r.rid != req.rid and not self.is_scheduled(r)
                  and r.status in (Status.DECODING, Status.PREFILL)]
        others = [r for r in others if r.pages]

        def age(r: Request) -> float:
            return now - r.prefill_enqueue_time

        if reason == "decode":
            # decode 饥饿：先牺牲最新的 partial prefill，再牺牲已投入
            # token 最少（重算成本最低）的 decoder。受保护 prefill
            # 绝不被 decode 抢占。
            partials = [r for r in others if r.status == Status.PREFILL
                        and age(r) < self.cfg.starvation_threshold]
            partials.sort(key=lambda r: (age(r),))
            decoders = [r for r in others if r.status == Status.DECODING]
            decoders.sort(key=lambda r: (r.tokens_in_kv,
                                         -r.num_preemptions, -age(r)))
            return partials + decoders

        # prefill：任何 prefill 请求都*不能*驱逐 decoder（防活锁）；
        # 只能放弃比自己更年轻（且本身未受保护）的 partial prefill。
        chosen = [r for r in others
                  if r.status == Status.PREFILL
                  and age(r) < age(req)
                  and not (protected and
                           age(r) >= self.cfg.starvation_threshold)]
        chosen.sort(key=lambda r: (r.tokens_in_kv, -age(r)))
        return chosen


# ---------------------------------------------------------------------------
# FCFS：静态分批，无抢占，严格 head-of-line 准入
# ---------------------------------------------------------------------------


class FCFSPlanner:
    """先到先服务基线。

    规则（可解释、可核验）：
      1. 空闲时按到达顺序准入一批请求：为每个请求预留其全部 prompt 页，
         第一个无法预留的请求阻塞其后所有请求（head-of-line blocking）；
      2. 整批先做 chunked prefill（不与 decode 混合），prefill 阶段
         每步在预算内按顺序推进批内请求；
      3. 全部 prefill 完成后整批 decode；每个请求每步至多一个 decode
         token，按轮转指针选请求；
      4. 不做抢占；decode 跨页无空闲页时该请求等待，其他请求继续。
    """

    IDLE = "idle"
    PREFILL = "prefill"
    DECODE = "decode"

    def __init__(self) -> None:
        self.phase = self.IDLE
        self.batch: List[str] = []
        self.pointer = 0

    def plan(self, ctx: SchedulingContext, now: float) -> List[Action]:
        # 清理已离开系统的批次成员（完成/取消）
        self.batch = [rid for rid in self.batch
                      if rid in ctx.active
                      and ctx.active[rid].status
                      in (Status.PREFILL, Status.DECODING)]

        # 阶段切换（准入 / prefill->decode / 批次耗尽）不消耗时间，
        # 在本调用内部循环直到产出动作或确认无法推进。
        for _ in range(4):
            if self.phase == self.IDLE:
                self._admit_batch(ctx, now)
                if not self.batch:
                    return []
                self.phase = self.PREFILL
                self.pointer = 0
                continue

            if self.phase == self.PREFILL:
                actions = self._plan_prefill(ctx, now)
                if actions:
                    return actions
                remaining = [ctx.active[r] for r in self.batch
                             if ctx.active[r].status == Status.PREFILL]
                if remaining:
                    return []
                if any(ctx.active[r].status == Status.DECODING
                       for r in self.batch):
                    self.phase = self.DECODE
                    continue
                self._reset()
                continue

            if self.phase == self.DECODE:
                actions = self._plan_decode(ctx, now)
                still = [rid for rid in self.batch
                         if ctx.active[rid].status == Status.DECODING]
                if actions:
                    return actions
                if still:
                    return []
                self._reset()
                continue
        return []

    def _reset(self) -> None:
        self.phase = self.IDLE
        self.batch = []
        self.pointer = 0

    def _admit_batch(self, ctx: SchedulingContext, now: float) -> None:
        # 准入即预留整个 prompt 页；严格按到达顺序，队首放不下则
        # head-of-line 阻塞（其后请求本轮一律不准入，不抢占）。
        waiting = sorted(ctx.waiting_requests(),
                         key=lambda r: (r.arrival, r.rid))
        for req in waiting:
            need = -(-req.prompt_len // ctx.cfg.page_size)
            if ctx.pool.free_pages < need:
                break
            if not ctx._alloc_for(req, need):
                break
            if req.admit_time is None:
                req.admit_time = now
            req.status = Status.PREFILL
            req.prefill_enqueue_time = now
            self.batch.append(req.rid)
            ctx.log({"event": "admit", "time": now, "rid": req.rid,
                     "policy": PolicyKind.FCFS.value, "pages": need})
        self.pointer = 0

    def _plan_prefill(self, ctx: SchedulingContext, now: float) -> List[Action]:
        actions: List[Action] = []
        for rid in self.batch:
            req = ctx.active.get(rid)
            if req is None or req.status != Status.PREFILL:
                continue
            act = ctx.grant_prefill(req, now, allow_preempt=False)
            if act is not None:
                actions.append(act)
        return actions

    def _plan_decode(self, ctx: SchedulingContext, now: float) -> List[Action]:
        members = [ctx.active[r] for r in self.batch
                   if ctx.active[r].status == Status.DECODING]
        if not members:
            return []
        if self.pointer >= len(members):
            self.pointer = 0
        actions: List[Action] = []
        n = len(members)
        for offset in range(n):
            req = members[(self.pointer + offset) % n]
            act = ctx.grant_decode(req, now, allow_preempt=False)
            if act is not None:
                actions.append(act)
        self.pointer = (self.pointer + 1) % n
        return actions


# ---------------------------------------------------------------------------
# Continuous batching：chunked prefill 即时混入 + 防饥饿 + 抢占重算
# ---------------------------------------------------------------------------


class ContinuousPlanner:
    """continuous batching 策略。

    防饥饿规则（显式、可解释；详见 README 的适用条件与上界推导）：
      * 年龄规则：在 prefill 队列中等待超过 ``starvation_threshold``
        的请求进入*受保护集合*；每个 step 保证至少获得
        ``min_protected_chunk`` 个 prefill token（预算允许时），
        且不会被其他 prefill 抢占；
      * decode 轮转：所有 decoder 按连续未被选中的 step 数
        （decode_gap）降序选择，每个有预算/页的 step 都尝试全部
        decoder 一轮，从而把 decode 间隔压在 D_gap 个 step 内
        （资源不足的情况除外）；
      * 抢占重算：仅当受保护 prefill 或饥饿 decoder 无法获得页时才
        驱逐；受害者按已投入 token 最少（重算成本最低）选择，
        重算 token 全额计费；每个请求抢占次数有上限，超限停止驱逐。
    """

    def plan(self, ctx: SchedulingContext, now: float) -> List[Action]:
        actions: List[Action] = []

        candidates = [r for r in ctx.active.values()
                      if r.status in (Status.WAITING, Status.PREFILL)]
        ages = {r.rid: now - r.prefill_enqueue_time for r in candidates}
        protected = sorted(
            (r for r in candidates
             if ages[r.rid] >= ctx.cfg.starvation_threshold),
            key=lambda r: (-ages[r.rid], r.arrival, r.rid),
        )

        # Pass A：受保护 prefill，最老的先来，允许抢占
        for req in protected:
            act = ctx.grant_prefill(
                req, now, allow_preempt=True,
                want=min(ctx.cfg.min_protected_chunk, ctx.cfg.chunk_max),
                protected=True)
            if act is not None:
                actions.append(act)

        # Pass B：decoder 按 decode_gap 降序（最久没跑的先来）
        decoders = sorted(ctx.decoders(),
                          key=lambda r: (-r.decode_gap, r.arrival, r.rid))
        starved = [r for r in decoders
                   if r.decode_gap >= ctx.cfg.decode_starvation_gap - 1]
        for req in starved:
            act = ctx.grant_decode(req, now, allow_preempt=True,
                                   protected=True)
            if act is not None:
                actions.append(act)
        for req in decoders:
            if ctx.is_scheduled(req):
                continue
            act = ctx.grant_decode(req, now, allow_preempt=True)
            if act is not None:
                actions.append(act)

        # Pass C：尽力而为的 prefill，按入队年龄排序，不抢占。
        # 本 step 已拿到保底 chunk 的受保护请求不再重复调度；其"保底"
        # 语义是"至少 min_protected_chunk"，剩余预算留给其他请求。
        rest = [r for r in candidates if not ctx.is_scheduled(r)]
        rest.sort(key=lambda r: (-ages.get(r.rid, 0.0), r.arrival, r.rid))
        for req in rest:
            act = ctx.grant_prefill(req, now, allow_preempt=False)
            if act is not None:
                actions.append(act)
        return actions


def make_planner(cfg: SimConfig):
    if cfg.policy == PolicyKind.FCFS:
        return FCFSPlanner()
    if cfg.policy == PolicyKind.CONT:
        return ContinuousPlanner()
    raise ValueError(f"未知策略: {cfg.policy}")
