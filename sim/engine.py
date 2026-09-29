"""离散事件主循环。

时间语义：
  * 事件（到达/取消）只在精确时间点发生；
  * 每次调度产出一个 batch（step），step 是原子的：在 step 起点
    [now] 处理完该时刻全部事件后调度，step 持续 cost model 给出的
    时长，动作在 [now, now+dur] 内执行，token 在 step 结束时刻
    [now+dur] 落盘并触发环境终止判定；
  * 到达发生在 step 之间，绝不打断正在执行的 step。

终止完全由环境负责：环境持有 RequestSpec.output_len（调度器不可见），
在 decode 使 generated 达到 output_len 时宣布完成。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .memory import PagePool
from .model import (
    Cancellation,
    PolicyKind,
    Request,
    RequestSpec,
    SimConfig,
    Status,
)
from .policies import (
    DecodeAction,
    PrefillAction,
    SchedulingContext,
    make_planner,
)


class DeadlockError(RuntimeError):
    """所有在系统内的请求都无法推进。"""


@dataclass
class RequestMetrics:
    rid: str
    status: str
    arrival: float
    prompt_len: int
    output_len: int
    admit_time: Optional[float] = None
    first_token_time: Optional[float] = None
    completion_time: Optional[float] = None
    completion_latency: Optional[float] = None   # completion_time - arrival
    ttft: Optional[float] = None                 # first_token_time - arrival
    itls: List[float] = field(default_factory=list)  # 相邻 decode token 间隔
    mean_itl: Optional[float] = None
    max_itl: Optional[float] = None
    queue_delay: Optional[float] = None          # admit_time - arrival
    preemptions: int = 0
    recompute_tokens: int = 0
    decode_tokens: int = 0

    @property
    def itl_note(self) -> str:
        if self.status != Status.COMPLETED.value:
            return "未完成，ITL 不适用"
        if self.output_len == 0:
            return "output_len=0：无生成 token，TTFT/ITL 不适用；" \
                   "完成延迟=首个可服务 step 的结束时刻"
        if self.output_len == 1:
            return "output_len=1：无相邻 token 对，ITL 序列为空；" \
                   "TTFT 即首 token（也是完成）时刻"
        return "正常"


@dataclass
class StepRecord:
    index: int
    start: float
    duration: float
    end: float
    actions: List[Dict[str, Any]]
    prefill_tokens: int
    decode_tokens: int
    fresh_prefill_tokens: int
    replay_tokens: int
    used_pages_before: int
    used_pages_after: int            # 步末、完成/取消释放之前
    used_pages_released: int = 0     # 该步因完成/取消释放的页数
    used_pages_final: int = 0        # 释放之后（=下一步 before）
    peak_pages: int = 0


@dataclass
class SimulationResult:
    policy: str
    cfg: Dict[str, Any]
    events: List[Dict[str, Any]]
    steps: List[StepRecord]
    requests: List[RequestMetrics]
    makespan: float
    # 汇总
    total_emitted_tokens: int = 0
    completed_emitted_tokens: int = 0
    goodput: float = 0.0                 # 已完成请求的 decode token / makespan
    token_throughput: float = 0.0        # 全部落盘 prefill+decode / makespan
    peak_pages: int = 0
    final_used_pages: int = 0
    final_internal_fragmentation: int = 0
    peak_internal_fragmentation: int = 0
    total_prefill_work: int = 0
    total_replay_work: int = 0
    total_decode_work: int = 0
    total_preemptions: int = 0
    num_completed: int = 0
    num_rejected: int = 0
    num_cancelled: int = 0
    deadlock: Optional[str] = None


class _Simulator:
    def __init__(self, cfg: SimConfig, specs: List[RequestSpec],
                 cancellations: Optional[List[Cancellation]] = None) -> None:
        self.cfg = cfg
        self.pool = PagePool(cfg.total_pages, cfg.page_size)
        self.active: Dict[str, Request] = {}
        self.all_requests: Dict[str, Request] = {}
        self.events: List[Dict[str, Any]] = []
        self.steps: List[StepRecord] = []

        self.specs = sorted(specs, key=lambda s: (s.arrival, s.rid))
        for s in self.specs:
            if s.rid in {x.rid for x in specs if x is not s}:
                raise ValueError(f"重复 rid: {s.rid}")
        self.arrivals = list(self.specs)
        self.cancellations = sorted(cancellations or [],
                                    key=lambda c: (c.at, c.rid))
        self.planner = make_planner(cfg)
        self.now = 0.0
        self.step_index = 0
        self.deadlock_reason: Optional[str] = None

        # 工作量总计
        self.total_fresh_prefill = 0
        self.total_replay = 0
        self.total_decode = 0

    # ---- 日志 ----
    def log(self, entry: Dict[str, Any]) -> None:
        self.events.append(entry)

    def _count_preempts(self) -> int:
        return sum(r.num_preemptions for r in self.active.values())

    def make_context(self):
        plan_events: List[Dict[str, Any]] = []
        ctx = SchedulingContext(self.cfg, self.pool, self.active,
                                plan_events.append)
        return ctx, plan_events

    # ---- 主循环 ----
    def run(self) -> SimulationResult:
        self.peak_fragmentation = 0
        # 活锁检测：连续 stall_window 个 step 都在发生抢占而
        # 没有任何 decode token 产出 / 请求完成，则判定活锁。
        self._stall_window = 64
        self._stall = 0
        self._total_preempt_ev = 0
        max_steps = 4000
        while True:
            self._process_arrivals_and_cancels_at(self.now)
            if self._system_empty():
                break

            ctx, plan_events = self.make_context()
            pages_before = self.pool.used_pages
            actions = self.planner.plan(ctx, self.now)
            # 规划阶段事件（准入/抢占）时间戳为 step 起点，必须排在
            # 执行期（end 时刻）事件之前，保证日志全局时间有序。
            for ev in plan_events:
                self.events.append(ev)

            if not actions:
                # FCFS 批次在 step 结束时清空并回到 idle：同刻到达的
                # 新请求必须先准入再决定是否推进时钟。
                if self.arrivals and self.arrivals[0].arrival <= self.now + 1e-12:
                    continue
                if self.cancellations and self.cancellations[0].at \
                        <= self.now + 1e-12:
                    continue
                if not self._advance_to_next_event():
                    self._raise_deadlock()
                    break
                continue

            pre_pre = self._count_preempts()
            self._execute_step(ctx, actions, pages_before)
            post_pre = self._count_preempts()
            if post_pre > pre_pre and ctx.decode_tokens == 0:
                self._stall += 1
            else:
                self._stall = 0
            self.step_index += 1
            if self.step_index > max_steps or self._stall > self._stall_window:
                self.deadlock_reason = (
                    "抢占活锁：连续 %d 个 step 仅发生抢占无 decode 进展"
                    % self._stall_window)
                break

        return self._build_result()

    def _system_empty(self) -> bool:
        return not self.active and not self.arrivals

    # ---- 到达 / 取消 ----
    def _process_arrivals_and_cancels_at(self, t: float) -> None:
        while self.arrivals and self.arrivals[0].arrival <= t + 1e-12:
            spec = self.arrivals.pop(0)
            self._arrive(spec, t)
        while self.cancellations and self.cancellations[0].at <= t + 1e-12:
            c = self.cancellations.pop(0)
            self._cancel(c, t)

    def _arrive(self, spec: RequestSpec, t: float) -> None:
        req = Request(spec=spec)
        req.prefill_enqueue_time = t
        self.all_requests[spec.rid] = req
        prompt_pages = -(-spec.prompt_len // self.cfg.page_size)
        if prompt_pages > self.cfg.total_pages:
            req.status = Status.REJECTED
            self.log({"event": "reject", "time": t, "rid": spec.rid,
                      "reason": f"prompt 需要 {prompt_pages} 页，"
                                f"超过总页数 {self.cfg.total_pages}"})
            return
        req.status = Status.WAITING
        self.active[spec.rid] = req
        self.log({"event": "arrival", "time": t, "rid": spec.rid,
                  "prompt_len": spec.prompt_len})

    def _cancel(self, c: Cancellation, t: float) -> None:
        req = self.all_requests.get(c.rid)
        if req is None:
            self.log({"event": "cancel_ignored", "time": t, "rid": c.rid,
                      "reason": "未知请求"})
            return
        if req.status in (Status.COMPLETED, Status.REJECTED,
                          Status.CANCELLED):
            self.log({"event": "cancel_ignored", "time": t, "rid": c.rid,
                      "reason": f"请求已处于 {req.status.value}"})
            return
        freed = self._release_all(req)
        req.status = Status.CANCELLED
        req.completion_time = t
        self.active.pop(req.rid, None)
        self.log({"event": "cancel", "time": t, "rid": req.rid,
                  "freed_pages": freed})

    # ---- 释放 ----
    def _release_all(self, req: Request) -> int:
        n = len(req.pages)
        if n:
            self.pool.free_many(req.pages)
            req.pages = []
        return n

    # ---- 执行 ----
    def _execute_step(self, ctx: SchedulingContext,
                      actions: List[Any], pages_before: int) -> None:
        dur = self.cfg.cost.step_duration(ctx.prefill_tokens,
                                          ctx.decode_tokens)
        end = self.now + dur
        action_logs: List[Dict[str, Any]] = []

        for act in actions:
            req = act.req
            if isinstance(act, PrefillAction):
                self._do_prefill(req, act.tokens, end, action_logs)
            elif isinstance(act, DecodeAction):
                self._do_decode(req, end, action_logs)
            else:
                raise RuntimeError(f"未知动作: {act}")

        # decode_gap 记账：未被选中的 decoder +1，被选中的清零
        chosen = {a.req.rid for a in actions if isinstance(a, DecodeAction)}
        for req in self.active.values():
            if req.status == Status.DECODING:
                if req.rid in chosen:
                    req.decode_gap = 0
                else:
                    req.decode_gap += 1

        frag = self._fragmentation()
        self.peak_fragmentation = max(self.peak_fragmentation, frag)
        self.log({"event": "frag_snapshot", "time": end,
                  "used_pages": self.pool.used_pages,
                  "used_tokens": sum(r.tokens_in_kv
                                     for r in self.active.values()),
                  "internal_fragmentation": frag})

        self.steps.append(StepRecord(
            index=self.step_index, start=self.now, duration=dur, end=end,
            actions=action_logs,
            prefill_tokens=ctx.prefill_tokens,
            decode_tokens=ctx.decode_tokens,
            fresh_prefill_tokens=sum(a.get("fresh", 0)
                                     for a in action_logs
                                     if a["kind"] == "prefill"),
            replay_tokens=sum(a.get("replay", 0)
                              for a in action_logs
                              if a["kind"] == "prefill"),
            used_pages_before=pages_before,
            used_pages_after=self.pool.used_pages,
            peak_pages=self.pool.peak_used,
        ))
        replay_in_step = sum(a.get("replay", 0) for a in action_logs
                             if a["kind"] == "prefill")
        fresh_in_step = ctx.prefill_tokens - replay_in_step
        self.total_fresh_prefill += fresh_in_step
        self.total_replay += replay_in_step
        self.total_decode += ctx.decode_tokens
        self.now = end

        # 终止后在 step 结束时处理（释放私有页）
        pages_at_end = self.pool.used_pages
        self._finalize_completed(end)
        released = pages_at_end - self.pool.used_pages
        self.steps[-1].used_pages_released = released
        self.steps[-1].used_pages_final = self.pool.used_pages

    def _do_prefill(self, req: Request, tokens: int, end: float,
                    action_logs: List[Dict[str, Any]]) -> None:
        start_pos = req.prefilled
        req.prefilled += tokens
        # 同一 step 内可能有多个 prefill 动作（保底 chunk + 余量补齐），
        # replay 拆分必须基于动作前的 pending 余量
        replay = min(tokens, req.recompute_pending)
        fresh = tokens - replay
        req.recompute_pending -= replay
        assert req.recompute_pending >= 0
        became_decoder = req.prefilled >= req.prefill_target
        if became_decoder:
            req.prefilled = req.prompt_len
            req.status = Status.DECODING
            req.decode_gap = 0
        self.log({"event": "prefill", "time": end, "rid": req.rid,
                  "tokens": tokens, "fresh": fresh, "replay": replay,
                  "pos": req.prefilled, "target": req.prefill_target,
                  "pages": len(req.pages),
                  "became_decoder": became_decoder})
        action_logs.append({"kind": "prefill", "rid": req.rid,
                            "tokens": tokens, "fresh": fresh,
                            "replay": replay, "pages": len(req.pages),
                            "became_decoder": became_decoder,
                            "kv_pos_after": req.tokens_in_kv,
                            "true_output_len": req._true_output_len()})

    def _do_decode(self, req: Request, end: float,
                   action_logs: List[Dict[str, Any]]) -> None:
        # 环境判定：调度器不知道终点，这里仅推进一个 token
        req.generated += 1
        req.decode_tokens_done += 1
        req.token_times.append(end)
        if req.first_token_time is None:
            req.first_token_time = end
        finished = req.generated >= req._true_output_len()
        self.log({"event": "decode", "time": end, "rid": req.rid,
                  "generated": req.generated, "pages": len(req.pages),
                  "env_finished": finished})
        action_logs.append({"kind": "decode", "rid": req.rid,
                            "generated": req.generated,
                            "pages": len(req.pages),
                            "env_finished": finished})

    def _finalize_completed(self, end: float) -> None:
        for rid in list(self.active.keys()):
            req = self.active[rid]
            if (req.status == Status.DECODING
                    and req.generated >= req._true_output_len()):
                req.status = Status.COMPLETED
                req.completion_time = end
                freed = self._release_all(req)
                self.active.pop(rid, None)
                self.log({"event": "complete", "time": end, "rid": rid,
                          "generated": req.generated, "freed_pages": freed})

    # ---- 空闲推进 ----
    def _advance_to_next_event(self) -> bool:
        """没有可执行动作时：若未来还有到达/取消，跳到该时刻。

        若系统内仍有请求但未来没有事件，则为资源死锁。
        """
        future = [s.arrival for s in self.arrivals]
        future += [c.at for c in self.cancellations
                   if c.at > self.now + 1e-12]
        future = [t for t in future if t > self.now + 1e-12]
        stuck = [r for r in self.active.values()
                 if r.status in (Status.WAITING, Status.PREFILL,
                                 Status.DECODING)]
        if not future:
            return False
        if stuck:
            return False
        nxt = min(future)
        self.log({"event": "idle_advance", "time": self.now,
                  "next_time": nxt})
        self.now = nxt
        return True

    def _raise_deadlock(self) -> None:
        stuck = [(r.rid, r.status.value, len(r.pages),
                  r.tokens_in_kv, r.prefill_target, r.generated)
                 for r in self.active.values()]
        self.deadlock_reason = f"无未来事件且存在无法推进的请求: {stuck}"
        self.log({"event": "deadlock", "time": self.now,
                  "requests": stuck})

    # ---- 碎片：每个 step 后快照 ----
    def _fragmentation(self) -> int:
        used_tokens = sum(r.tokens_in_kv for r in self.active.values())
        return self.pool.internal_fragmentation(used_tokens)

    # ---- 汇总 ----
    def _build_result(self) -> SimulationResult:
        req_metrics: List[RequestMetrics] = []
        emitted = 0
        completed_emitted = 0
        for req in self.all_requests.values():
            m = self._metrics_for(req)
            req_metrics.append(m)
            emitted += m.decode_tokens
            if m.status == Status.COMPLETED.value:
                completed_emitted += m.decode_tokens

        makespan = self.now
        cfg_dict = {
            "policy": self.cfg.policy.value,
            "token_budget": self.cfg.token_budget,
            "chunk_max": self.cfg.chunk_max,
            "page_size": self.cfg.page_size,
            "total_pages": self.cfg.total_pages,
            "starvation_threshold": self.cfg.starvation_threshold,
            "min_protected_chunk": self.cfg.min_protected_chunk,
            "decode_starvation_gap": self.cfg.decode_starvation_gap,
            "max_preemptions": self.cfg.max_preemptions,
            "cost_model": {
                "prefill_cost": self.cfg.cost.prefill_cost,
                "decode_cost": self.cfg.cost.decode_cost,
                "throughput": self.cfg.cost.throughput,
                "fixed_fee": self.cfg.cost.fixed_fee,
            },
        }
        res = SimulationResult(
            policy=self.cfg.policy.value,
            cfg=cfg_dict,
            events=self.events,
            steps=self.steps,
            requests=req_metrics,
            makespan=makespan,
            total_emitted_tokens=emitted,
            completed_emitted_tokens=completed_emitted,
            goodput=(completed_emitted / makespan) if makespan > 0 else 0.0,
            token_throughput=(
                (self.total_fresh_prefill + self.total_replay
                 + self.total_decode) / makespan) if makespan > 0 else 0.0,
            peak_pages=self.pool.peak_used,
            final_used_pages=self.pool.used_pages,
            final_internal_fragmentation=self._fragmentation(),
            peak_internal_fragmentation=self.peak_fragmentation,
            total_prefill_work=self.total_fresh_prefill,
            total_replay_work=self.total_replay,
            total_decode_work=self.total_decode,
            total_preemptions=sum(
                e.get("preempt_count", 0) for e in self.events
                if e.get("event") == "preempt"),
            num_completed=sum(
                1 for m in req_metrics
                if m.status == Status.COMPLETED.value),
            num_rejected=sum(
                1 for m in req_metrics
                if m.status == Status.REJECTED.value),
            num_cancelled=sum(
                1 for m in req_metrics
                if m.status == Status.CANCELLED.value),
            deadlock=self.deadlock_reason,
        )
        return res

    def _metrics_for(self, req: Request) -> RequestMetrics:
        itls: List[float] = []
        times = list(req.token_times)
        for a, b in zip(times, times[1:]):
            itls.append(round(b - a, 12))
        ttft = None
        if req.first_token_time is not None:
            ttft = req.first_token_time - req.arrival
        comp_lat = None
        if req.completion_time is not None:
            comp_lat = req.completion_time - req.arrival
        return RequestMetrics(
            rid=req.rid,
            status=req.status.value,
            arrival=req.arrival,
            prompt_len=req.prompt_len,
            output_len=req._true_output_len(),
            admit_time=req.admit_time,
            first_token_time=req.first_token_time,
            completion_time=req.completion_time,
            completion_latency=comp_lat,
            ttft=ttft,
            itls=itls,
            mean_itl=(sum(itls) / len(itls)) if itls else None,
            max_itl=max(itls) if itls else None,
            queue_delay=(req.admit_time - req.arrival)
            if req.admit_time is not None else None,
            preemptions=req.num_preemptions,
            recompute_tokens=req.recompute_tokens,
            decode_tokens=req.generated if req.status == Status.COMPLETED
            else req.generated,
        )

    def _peak_fragmentation(self) -> int:
        best = 0
        for e in self.events:
            if e.get("event") == "frag_snapshot":
                best = max(best, e["internal_fragmentation"])
        return best


def simulate(cfg: SimConfig, specs: List[RequestSpec],
             cancellations: Optional[List[Cancellation]] = None
             ) -> SimulationResult:
    """运行一次模拟并返回完整结果（含逐事件日志与逐 step 记录）。"""
    return _Simulator(cfg, specs, cancellations).run()
