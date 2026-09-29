"""Schedulers: FCFS/static batching and continuous batching (chunked prefill).

Schedulers only observe runtime Request state (arrival, prompt_len,
prompt_done, gen_done, pages, wait_time). They never receive RequestSpec or
gen_len; tests enforce this structurally.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .allocator import Allocator
from .model import Request, SimConfig, Status, pages_for_tokens


@dataclass
class StepItem:
    req: Request
    prefill: int = 0
    decode: int = 0


@dataclass
class Plan:
    items: list[StepItem] = field(default_factory=list)
    preempted: list[str] = field(default_factory=list)
    reject: Optional[str] = None
    reject_kind: Optional[str] = None  # prompt_too_large | runtime_too_large
    planned_head: Optional[str] = None
    reason: str = ""

    @property
    def prefill_tokens(self) -> int:
        return sum(i.prefill for i in self.items)

    @property
    def decode_tokens(self) -> int:
        return sum(i.decode for i in self.items)

    @property
    def recompute_tokens(self) -> int:
        return sum(i.prefill for i in self.items if i.req.is_recompute)


@dataclass
class Context:
    cfg: SimConfig
    alloc: Allocator
    now: float
    waiting: list[Request]
    active: list[Request]


class _BaseScheduler:
    name = "base"

    def reset(self) -> None:
        pass

    def plan(self, ctx: Context) -> Plan:  # pragma: no cover - interface
        raise NotImplementedError


def _incremental_page_need(cfg: SimConfig, items: list[StepItem]) -> int:
    """Additional pages needed to acquire all items in order, honoring
    per-request running page counts."""
    sim: dict[str, int] = {}
    need = 0
    for it in items:
        req = it.req
        pages = sim.get(req.rid, req.pages)
        if it.prefill:
            target = pages_for_tokens(req.prompt_done + it.prefill, cfg.page_size)
        elif it.decode:
            target = pages_for_tokens(req.kv_tokens + 1, cfg.page_size)
        else:
            target = pages
        if target > pages:
            need += target - pages
            sim[req.rid] = target
        else:
            sim[req.rid] = pages
    return need


# --------------------------------------------------------------------------
# Continuous batching with chunked prefill + age-based anti-starvation.
# --------------------------------------------------------------------------
class ContinuousBatching(_BaseScheduler):
    """Sliding-window continuous batching.

    Declared anti-starvation rule
    -----------------------------
    Requests keep their original arrival time forever (also across
    preemptions). The oldest waiting request is the *head*.

    * ``grace == 0`` (default) and ``max_chunk = q <= B``:
      - the head gets up to ``q`` *guaranteed* prefill tokens each step before
        decoders; while its prompt remains, decoders are capped at ``B - q`` so
        the reservation is real, not best-effort;
      - leftover budget is shared round-robin, oldest first, at most ``q``
        tokens per request per step.
    * ``preemption="recompute"``: when the entire feasible plan does not fit
      the pool, the *youngest* admitted requests are evicted (decoders before
      older prefills; the beneficiary is never evicted). All private pages are
      released and the full prompt is recomputed later (billed at the prefill
      rate via CostModel.recompute_factor). Evicted requests keep arrival time.
    * ``preemption="wait"``: nothing is evicted. A fresh request is admitted
      only if its complete prompt *plus one decode-growth page* fit at once
      (declared conservative admission). Requests that still cannot be served
      when no external event remains are refused explicitly
      (``resource_exhausted_wait_policy``) rather than looping forever.
    """

    name = "continuous_batching"

    def __init__(self, preemption: str = "recompute", grace: float = 0.0) -> None:
        if preemption not in ("recompute", "wait"):
            raise ValueError("preemption must be 'recompute' or 'wait'")
        self.preemption = preemption
        self.grace = grace

    # -- page pressure ------------------------------------------------------
    def _evict(self, ctx: Context, v: Request, plan: Plan, excluded: set[str]) -> None:
        ctx.alloc.release(v)
        v.status = Status.WAITING
        v.prompt_done = 0
        v.gen_done = 0
        v.is_recompute = True
        v.preemptions += 1
        ctx.active.remove(v)
        ctx.waiting.append(v)
        plan.preempted.append(v.rid)
        excluded.add(v.rid)
        # Drop any tentative work for the victim from this step; it restarts
        # prefill in a later step (excluded for the remainder of this one).
        plan.items = [it for it in plan.items if it.req.rid != v.rid]

    def _victims(
        self,
        ctx: Context,
        beneficiary: str,
        excluded: set[str],
        same_time_peers: bool = False,
    ):
        # Eviction eligibility is a stable total order keyed by
        # (arrival, rid). A beneficiary may evict only strictly-later keys;
        # ``same_time_peers`` controls whether equal-arrival requests are
        # eligible (used by the head's prefill guarantee to serialize a burst
        # cohort by rid) or excluded (decoders never evict a same-time peer,
        # which prevents cohort livelock).
        ben = self._by_id(ctx, beneficiary)
        if ben is None:
            return []
        ben_key = (ben.arrival, ben.rid)
        vics = []
        for r in list(ctx.active):
            if r.status not in (Status.DECODE, Status.PREFILL):
                continue
            if r.rid in excluded:
                continue
            key = (r.arrival, r.rid)
            later_arrival = r.arrival > ben.arrival
            same_time_later_rid = (
                same_time_peers and r.arrival == ben.arrival and r.rid > ben.rid
            )
            if later_arrival or same_time_later_rid:
                vics.append(r)
        vics.sort(
            key=lambda r: (
                r.admitted_at or r.arrival,
                0 if r.status == Status.DECODE else 1,
                r.rid,
            ),
            reverse=True,
        )
        return vics

    @staticmethod
    def _by_id(ctx: Context, rid: str):
        for r in ctx.active + ctx.waiting:
            if r.rid == rid:
                return r
        return None

    def _fits(self, ctx: Context, items: list[StepItem]) -> bool:
        return _incremental_page_need(ctx.cfg, items) <= ctx.alloc.free_pages

    def _try_add(
        self,
        ctx: Context,
        plan: Plan,
        excluded: set[str],
        candidate: StepItem,
        beneficiary: str,
    ) -> bool:
        """Append candidate; under recompute policy evict youngest requests
        until the whole plan fits. On failure restore plan state. Pages freed
        by eviction stay freed (correct -- eviction commits)."""
        plan.items.append(candidate)
        if self.preemption == "recompute":
            survivors = [it for it in plan.items if it.req.rid not in excluded]
            for v in self._victims(ctx, beneficiary, excluded):
                if _incremental_page_need(ctx.cfg, survivors) <= ctx.alloc.free_pages:
                    break
                self._evict(ctx, v, plan, excluded)
                survivors = [
                    it for it in plan.items if it.req.rid not in excluded
                ]
            ok = _incremental_page_need(ctx.cfg, survivors) <= ctx.alloc.free_pages
        else:
            ok = self._fits(ctx, plan.items)
        if not ok:
            plan.items.pop()
        return ok

    @staticmethod
    def _wait_admit_feasible(ctx: Context, req: Request,
                             planned_fresh: int = 0) -> bool:
        # Strict no-preemption admission:
        #  (1) prompt + one decode-growth page fits the whole pool;
        #  (2) no other request is active and none was admitted earlier in
        #      this same planning step (ctx.active only updates at execution).
        # Processing one request at a time makes progress provable without
        # eviction; concurrency is deliberately traded for liveness.
        if ctx.active or planned_fresh > 0:
            return False
        need = pages_for_tokens(req.prompt_len + 1, ctx.cfg.page_size)
        return need <= ctx.alloc.free_pages

    # -- main ---------------------------------------------------------------
    def plan(self, ctx: Context) -> Plan:
        plan = Plan()
        excluded: set[str] = set()
        B = ctx.cfg.token_budget
        q = min(ctx.cfg.max_chunk, B)

        def pref(req: Request) -> int:
            return sum(
                it.prefill for it in plan.items if it.req.rid == req.rid
            )

        fresh_count = [0]

        def survivors() -> list[StepItem]:
            return [it for it in plan.items if it.req.rid not in excluded]

        def fits(items: list[StepItem], spare: int = 0) -> bool:
            return (
                _incremental_page_need(ctx.cfg, items) + spare
                <= ctx.alloc.free_pages
            )

        def add(req: Request, *, prefill: int = 0, decode: int = 0,
                beneficiary: Optional[str] = None,
                allow_eviction: bool = True,
                same_time_peers: bool = False) -> bool:
            cand = StepItem(req=req, prefill=prefill, decode=decode)
            if self.preemption == "wait":
                fresh = req.status == Status.WAITING and prefill
                if fresh and not self._wait_admit_feasible(
                    ctx, req, fresh_count[0]
                ):
                    return False
                if decode and pages_for_tokens(
                    req.kv_tokens + 1, ctx.cfg.page_size
                ) > ctx.cfg.num_pages:
                    # Even owning the whole pool it cannot grow: physically
                    # impossible; the engine rejects it explicitly.
                    plan.reject = req.rid
                    plan.reject_kind = "runtime_too_large"
                    return False
                # Fresh admissions preserve the global spare (liveness token).
                # Continued prefills and decoders may consume available pages
                # including that spare: they are already committed and must be
                # able to drain, which releases pages again.
                spare = 1 if fresh else 0
                if not fits(survivors() + [cand], spare=spare):
                    return False
                plan.items.append(cand)
                if fresh:
                    fresh_count[0] += 1
                return True
            # Under recompute, allow_eviction=False means the work must fit in
            # currently free pages (used for fresh non-head admissions): no
            # younger request may be packed in by evicting older work.
            if not allow_eviction and not fits(survivors() + [cand]):
                return False
            plan.items.append(cand)
            ben = beneficiary or req.rid
            for v in self._victims(
                ctx, ben, excluded, same_time_peers=same_time_peers
            ):
                if fits(survivors()):
                    break
                self._evict(ctx, v, plan, excluded)
            if fits(survivors()):
                return True
            plan.items.pop()  # evictions remain committed (intended)
            return False

        def used_tokens() -> int:
            return sum(
                it.prefill + it.decode for it in survivors()
            )

        waiting = sorted(ctx.waiting, key=lambda r: (r.arrival, r.rid))
        if waiting and waiting[0].prompt_pages(ctx.cfg.page_size) > ctx.cfg.num_pages:
            plan.reject = waiting[0].rid
            plan.reject_kind = "prompt_too_large"
            return plan
        # A waiting head whose prompt fits but that could never even hold its
        # prompt plus one generated token in the whole pool is physically
        # unschedulable *now* under the conservative wait rule. Under recompute
        # it could still be admitted and run, but if it cannot be granted its
        # guaranteed slice without infinite eviction (prompt needs the whole
        # pool and decoders occupy it), the engine handles progress; here only
        # mark the trivially impossible prompt-pages case above.


        head = waiting[0] if waiting else None
        head_served = 0
        head_blocked = False

        # Pass 1: guaranteed head slice. Under the wait policy prompt
        # processing is serialized (no other request may be active); under
        # recompute the head always gets first claim and may evict strictly
        # younger requests for pages, while other prefills may proceed in
        # parallel with leftover budget (classic continuous batching).
        in_flight_prefill = any(
            r.status == Status.PREFILL for r in ctx.active
        )
        if head is not None and head.wait_time(ctx.now) >= self.grace:
            blocked_by_serial = (
                self.preemption == "wait"
                and head.status == Status.WAITING
                and in_flight_prefill
            )
            chunk = min(q, head.prompt_len, B - used_tokens())
            if chunk > 0 and not blocked_by_serial:
                if add(
                    head,
                    prefill=chunk,
                    beneficiary=head.rid,
                    same_time_peers=True,
                ):
                    head_served = chunk
                else:
                    head_blocked = True
            else:
                head_blocked = blocked_by_serial

        # Pass 2: decoders oldest first, reserving q for a progressing head.
        head_remaining = (head.prompt_len - head_served) if head else 0
        reserve = q if head_remaining > 0 and not head_blocked else 0
        for d in sorted(
            (r for r in ctx.active if r.status == Status.DECODE),
            key=lambda r: (r.admitted_at or r.arrival, r.rid),
        ):
            if used_tokens() >= B or B - used_tokens() - reserve <= 0:
                break
            # Every active decoder may advance one token. Under recompute the
            # add() helper evicts strictly-younger requests if a boundary page
            # is needed; under wait it simply skips if no page is available.
            add(d, decode=1, beneficiary=d.rid)

        # Pass 3: remaining budget -> prefills, continued first then fresh,
        # round-robin oldest first, cap q per request. Re-read live queues each
        # sweep so just-evicted requests are excluded.
        while used_tokens() < B:
            progressed = False
            continued = sorted(
                (
                    r
                    for r in ctx.active
                    if r.status == Status.PREFILL and r.rid not in excluded
                ),
                key=lambda r: (r.arrival, r.rid),
            )
            fresh = sorted(
                (r for r in ctx.waiting if r.rid not in excluded),
                key=lambda r: (r.arrival, r.rid),
            )
            for req in continued + fresh:
                if used_tokens() >= B:
                    break
                already = pref(req)
                if already >= q:
                    continue
                take = min(q - already, B - used_tokens(),
                           req.prompt_len - req.prompt_done - already)
                if take <= 0:
                    continue
                fresh_admit = req.status == Status.WAITING
                # A continued prefill that is older (by the stable
                # (arrival, rid) order) than an active peer may finish by
                # evicting strictly-younger peers, including same-arrival ones
                # ordered after it; this serializes a saturated burst cohort
                # deterministically. Fresh admissions never force eviction.
                older_than_a_peer = (
                    not fresh_admit
                    and any(
                        (o.arrival, o.rid) > (req.arrival, req.rid)
                        for o in ctx.active
                    )
                )
                if add(
                    req,
                    prefill=take,
                    beneficiary=req.rid,
                    allow_eviction=not fresh_admit,
                    same_time_peers=older_than_a_peer,
                ):
                    progressed = True
            if not progressed:
                break

        plan.items = survivors()
        plan.planned_head = head.rid if head is not None else None
        plan.reason = (
            f"head={plan.planned_head} mode={self.preemption}"
        )
        return plan


# --------------------------------------------------------------------------
# Strict FCFS with static batches.
# --------------------------------------------------------------------------
class FCFSScheduler(_BaseScheduler):
    """Strict first-come-first-served static batching.

    * No active batch: form from the arrival-order prefix.
      - head prompt larger than the whole pool -> reject;
      - head prompt longer than B -> head owns the batch, chunked prefill;
      - otherwise longest prefix with one-step total prefill (``<= B``) whose
        fully-resident prompt pages fit the pool at once.
    * The batch drains as one generation: no new admission until every member
      completes/cancels (head-of-line blocking). No preemption; reserving all
      prompt pages at formation guarantees later decode-boundary pages exist.
    """

    name = "fcfs"

    def __init__(self) -> None:
        self.batch: list[Request] = []

    def reset(self) -> None:
        self.batch = []

    def _live_batch(self) -> list[Request]:
        self.batch = [
            r for r in self.batch if r.status in (Status.PREFILL, Status.DECODE)
        ]
        return self.batch

    def plan(self, ctx: Context) -> Plan:
        plan = Plan()
        batch = self._live_batch()
        B, q = ctx.cfg.token_budget, ctx.cfg.max_chunk
        waiting = sorted(ctx.waiting, key=lambda r: (r.arrival, r.rid))

        if not batch:
            if not waiting:
                return Plan(reason="idle")
            head = waiting[0]
            if head.prompt_pages(ctx.cfg.page_size) > ctx.cfg.num_pages:
                plan.reject = head.rid
                return plan
            chosen: list[Request] = []
            if head.prompt_len > B:
                chosen = [head]
            else:
                tok = 0
                reserved = 0
                for req in waiting:
                    pp = req.prompt_pages(ctx.cfg.page_size)
                    if pp > ctx.cfg.num_pages:
                        break
                    if tok + req.prompt_len > B or reserved + pp > ctx.cfg.num_pages:
                        break
                    chosen.append(req)
                    tok += req.prompt_len
                    reserved += pp
                if not chosen:
                    chosen = [head]
            self.batch = chosen
            batch = chosen
            plan.reason = "form:" + ",".join(r.rid for r in chosen)

        items: dict[str, StepItem] = {}

        def slot(req: Request) -> StepItem:
            it = items.get(req.rid)
            if it is None:
                it = StepItem(req=req)
                items[req.rid] = it
            return it

        prefillers = [
            r for r in batch if r.status in (Status.WAITING, Status.PREFILL)
        ]
        used = 0
        if prefillers:
            if len(prefillers) == 1 and prefillers[0].prompt_len > B:
                sole = prefillers[0]
                chunk = min(q, B, sole.prompt_len - sole.prompt_done)
                slot(sole).prefill = chunk
                used += chunk
            else:
                for req in sorted(prefillers, key=lambda r: (r.arrival, r.rid)):
                    take = min(req.prompt_len - req.prompt_done, B - used)
                    if take <= 0:
                        break
                    slot(req).prefill = take
                    used += take
        else:
            decs = sorted(
                (r for r in batch if r.status == Status.DECODE),
                key=lambda r: (r.admitted_at or r.arrival, r.rid),
            )
            for d in decs[:B]:
                slot(d).decode += 1
                used += 1

        plan.items = list(items.values())
        if waiting:
            plan.planned_head = waiting[0].rid
        return plan
