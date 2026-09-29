"""Discrete-event simulation engine.

The engine owns the *environment truth* (actual gen_len). Schedulers receive a
Context containing only Request runtime objects (which carry no gen_len).
"""
from __future__ import annotations

from typing import Optional

from .allocator import Allocator, EventLogger
from .model import Request, RequestSpec, SimConfig, Status, pages_for_tokens
from .scheduler import Context, Plan, _BaseScheduler


class SimulationResult:
    def __init__(self, policy: str, config: SimConfig, specs: list[RequestSpec]):
        self.policy = policy
        self.config = config
        self.specs = {s.rid: s for s in specs}  # environment truth (report only)
        self.requests: dict[str, Request] = {}
        self.makespan: float = 0.0
        self.events: list[dict] = []
        self.steps: list[dict] = []
        self.logger = EventLogger()
        self.alloc: Optional[Allocator] = None


class Engine:
    def __init__(
        self,
        config: SimConfig,
        scheduler: _BaseScheduler,
        workload: list[RequestSpec],
        log: Optional[EventLogger] = None,
    ) -> None:
        self.cfg = config
        self.scheduler = scheduler
        self.workload = sorted(workload, key=lambda s: (s.arrival, s.rid))
        # Environment-private map: never handed to the scheduler.
        self._truth: dict[str, RequestSpec] = {s.rid: s for s in workload}
        if len(self._truth) != len(workload):
            raise ValueError("duplicate request ids")
        self.alloc = Allocator(config.num_pages, config.page_size)
        self.logger = log or EventLogger()
        self.now = 0.0
        self.waiting: list[Request] = []
        self.active: list[Request] = []
        self.requests: dict[str, Request] = {}
        self.steps_log: list[dict] = []
        self._unadmitted: list[RequestSpec] = list(self.workload)
        self.step_count = 0
        self._time_weighted_frag = 0.0  # token-slots * time

    # -- helpers ------------------------------------------------------------
    def _log(self, event: str, **fields) -> None:
        self.logger.emit(event, self.now, **fields)

    def _arrivals_due(self) -> list[RequestSpec]:
        due = [s for s in self._unadmitted if s.arrival <= self.now + 1e-12]
        due.sort(key=lambda s: (s.arrival, s.rid))
        for s in due:
            self._unadmitted.remove(s)
        return due

    def _next_event_time(self) -> Optional[float]:
        nxt: Optional[float] = None
        if self._unadmitted:
            nxt = min(s.arrival for s in self._unadmitted)
        for s in self._truth.values():
            if s.cancel_at is not None:
                req = self.requests.get(s.rid)
                if req is not None and not req.status.terminal:
                    nxt = s.cancel_at if nxt is None else min(nxt, s.cancel_at)
        return nxt

    def _admit_arrivals(self) -> None:
        for spec in self._arrivals_due():
            req = Request(
                rid=spec.rid,
                arrival=spec.arrival,
                prompt_len=spec.prompt_len,
                cancel_at=spec.cancel_at,
            )
            self.requests[spec.rid] = req
            self.waiting.append(req)
            self._log(
                "arrival",
                rid=spec.rid,
                prompt_len=spec.prompt_len,
                # gen_len intentionally not logged as scheduler-visible; it is
                # recorded by the environment only at completion.
            )

    def _apply_cancellations(self) -> None:
        for spec in list(self._truth.values()):
            if spec.cancel_at is None:
                continue
            req = self.requests.get(spec.rid)
            if req is None or req.status.terminal:
                continue
            if self.now + 1e-12 >= spec.cancel_at:
                self._terminate(req, Status.CANCELLED)

    def _terminate(self, req: Request, status: Status) -> None:
        if req.status.terminal:
            return
        released = self.alloc.release(req)
        req.status = status
        req.completed_at = self.now
        if req in self.active:
            self.active.remove(req)
        if req in self.waiting:
            self.waiting.remove(req)
        self._log(
            status.value,
            rid=req.rid,
            released_pages=released,
            prompt_done=req.prompt_done,
            gen_done=req.gen_done,
        )

    # -- main loop ----------------------------------------------------------
    def run(self) -> SimulationResult:
        self.scheduler.reset()
        while True:
            self._admit_arrivals()
            self._apply_cancellations()

            if not self.waiting and not self.active:
                if self._unadmitted:
                    jump = min(s.arrival for s in self._unadmitted)
                    if jump > self.now + 1e-12:
                        self.now = float(jump)
                        self._log("idle_jump", to=self.now)
                        continue
                    continue
                break  # all requests terminal, nothing future

            ctx = Context(self.cfg, self.alloc, self.now, self.waiting, self.active)
            plan = self.scheduler.plan(ctx)

            # Runtime-impossible decoder (sequence exceeds pool even owning it
            # fully): terminate it explicitly before building admissions.
            if plan.reject and getattr(plan, "reason", "") ==                     "sequence_exceeds_pool_at_runtime":
                req = self.requests[plan.reject]
                req.status = Status.REJECTED
                req.completed_at = self.now
                req.reject_reason = "sequence_exceeds_pool_at_runtime"
                if req in self.active:
                    self.alloc.release(req)
                    self.active.remove(req)
                self._log("rejected", rid=req.rid, reason=req.reject_reason,
                          kv_tokens=req.kv_tokens)
                continue

            # Rejected up front (prompt cannot physically fit the pool).
            if plan.reject:
                req = self.requests[plan.reject]
                req.status = Status.REJECTED
                req.completed_at = self.now
                req.reject_reason = "prompt_exceeds_page_pool"
                if req in self.waiting:
                    self.waiting.remove(req)
                self._log(
                    "rejected",
                    rid=req.rid,
                    reason=req.reject_reason,
                    prompt_len=req.prompt_len,
                    pool_pages=self.cfg.num_pages,
                )
                continue

            if not plan.items:
                policy = getattr(self.scheduler, "preemption", "none")

                # 1) Refuse physically impossible resident sequences first.
                impossible = [
                    r
                    for r in list(self.active)
                    if r.status in (Status.DECODE, Status.PREFILL)
                    and pages_for_tokens(r.kv_tokens + 1, self.cfg.page_size)
                    > self.cfg.num_pages
                ]
                for req in impossible:
                    self.alloc.release(req)
                    self.active.remove(req)
                    req.status = Status.REJECTED
                    req.completed_at = self.now
                    req.reject_reason = "sequence_exceeds_pool_at_runtime"
                    self._log("rejected", rid=req.rid, reason=req.reject_reason,
                              kv_tokens=req.kv_tokens)
                if impossible:
                    self._log("capacity_replan",
                              waiting=[r.rid for r in self.waiting])
                    continue

                nxt = self._next_event_time()

                # 2) Active resident work exists.
                if self.active:
                    if policy == "recompute":
                        # Normally eviction guarantees forward progress. An
                        # empty plan that repeats at the same instant means the
                        # pool is physically saturated by a same-time cohort
                        # with no strictly-younger victim to evict: the only
                        # resolution is to shed the youngest admitted request.
                        self._spin = getattr(self, "_spin", None)
                        key = (self.now, len(self.active), len(self.waiting))
                        self._spin = (
                            (key, 1)
                            if self._spin is None or self._spin[0] != key
                            else (key, self._spin[1] + 1)
                        )
                        if self._spin[1] >= 2:
                            victim = max(
                                self.active, key=lambda r: r.arrival
                            )
                            self.alloc.release(victim)
                            self.active.remove(victim)
                            victim.status = Status.REJECTED
                            victim.completed_at = self.now
                            victim.reject_reason = (
                                "capacity_exhausted_cohort_shed"
                            )
                            self._log(
                                "rejected",
                                rid=victim.rid,
                                reason=victim.reject_reason,
                                kv_tokens=victim.kv_tokens,
                            )
                            self._spin = None
                        continue
                    # wait mode: wait for next arrival/cancellation, which is
                    # the only thing that can change feasibility.
                    if nxt is not None and nxt > self.now + 1e-12:
                        self.now = float(nxt)
                        self._log("stall", to=self.now,
                                  waiting=[r.rid for r in self.waiting])
                        continue
                    raise RuntimeError(
                        f"wait-policy liveness failure at t={self.now}: active "
                        "requests cannot grow and no external event remains"
                    )

                # 3) Only waiting requests remain and none fits right now.
                if nxt is not None and nxt > self.now + 1e-12:
                    self.now = float(nxt)
                    self._log("stall", to=self.now,
                              waiting=[r.rid for r in self.waiting])
                    continue
                reason = (
                    "resource_exhausted_wait_policy"
                    if policy == "wait"
                    else "resource_exhausted"
                )
                for req in list(self.waiting):
                    req.status = Status.REJECTED
                    req.completed_at = self.now
                    req.reject_reason = reason
                    self.waiting.remove(req)
                    self._log("rejected", rid=req.rid, reason=reason,
                              used_pages=self.alloc.used_pages)
                break

            self._execute(plan)

        result = SimulationResult(self.scheduler.name, self.cfg, list(self._truth.values()))
        result.requests = self.requests
        result.makespan = self.now
        result.events = self.logger.events
        result.steps = self.steps_log
        result.alloc = self.alloc
        result.logger = self.logger
        return result

    def _oldest_waiting_id(self):
        if not self.waiting:
            return None
        return min(self.waiting, key=lambda r: (r.arrival, r.rid)).rid

    # -- step execution -----------------------------------------------------
    def _execute(self, plan: Plan) -> None:
        self.step_count += 1
        pre_preempt = sorted(plan.preempted)
        self._log(
            "step_begin",
            step=self.step_count,
            free_pages=self.alloc.free_pages,
            plan_prefill=plan.prefill_tokens,
            plan_decode=plan.decode_tokens,
            preempted=pre_preempt,
            planned_head=getattr(plan, "planned_head", None),
            waiting=[r.rid for r in self.waiting],
            note=plan.reason,
        )

        # Budget ledger check (scheduler contract).
        if plan.prefill_tokens + plan.decode_tokens > self.cfg.token_budget:
            raise AssertionError(
                f"step {self.step_count}: token budget exceeded "
                f"{plan.prefill_tokens}+{plan.decode_tokens}>{self.cfg.token_budget}"
            )

        # Admit waiting requests present in plan, then reserve pages up front.
        admitted: list[str] = []
        for item in plan.items:
            req = item.req
            if req.status == Status.WAITING:
                if req not in self.waiting:
                    raise AssertionError(f"{req.rid}: missing from waiting queue")
                self.waiting.remove(req)
                req.status = Status.PREFILL
                req.admitted_at = self.now
                self.active.append(req)
                admitted.append(req.rid)

        for item in plan.items:
            req = item.req
            if item.prefill:
                self.alloc.acquire_prefill(req, item.prefill)
            if item.decode:
                if item.prefill:
                    raise AssertionError(
                        f"step {self.step_count}: {req.rid} mixes prefill+decode"
                    )
                if req.status != Status.DECODE:
                    raise AssertionError(
                        f"step {self.step_count}: decode requested for "
                        f"{req.rid} in {req.status}"
                    )
                self.alloc.acquire_decode(req)

        if admitted:
            self._log("admit", rids=admitted)

        if self.alloc.used_pages > self.alloc.num_pages:
            raise AssertionError("page pool oversubscribed")

        dt = self.cfg.cost.step_time(
            plan.prefill_tokens, plan.decode_tokens, plan.recompute_tokens
        )
        t_start = self.now
        self.now += dt

        # Produce tokens / advance state, then let the environment signal end.
        finished_prefill: list[str] = []
        decoded: list[dict] = []
        for item in plan.items:
            req = item.req
            if item.prefill:
                before = req.prompt_done
                req.prompt_done += item.prefill
                if req.is_recompute:
                    req.recompute_tokens += item.prefill
                self._log(
                    "prefill_chunk",
                    rid=req.rid,
                    tokens=item.prefill,
                    prompt_done=req.prompt_done,
                    pages=req.pages,
                    recompute=req.is_recompute,
                )
                if req.prompt_done > req.prompt_len:
                    raise AssertionError(f"{req.rid}: prefill overran prompt")
                if req.prompt_done == req.prompt_len:
                    finished_prefill.append(req.rid)
            if item.decode:
                req.gen_done += 1
                req.decode_times.append(self.now)
                if req.first_token_at is None:
                    req.first_token_at = self.now
                decoded.append({"rid": req.rid, "gen_done": req.gen_done})

        # Environment checks termination against the *actual* gen_len.
        for rid in [r.rid for r in list(self.active)]:
            req = self.requests[rid]
            spec = self._truth[rid]
            if req.status == Status.PREFILL and req.prompt_done == req.prompt_len:
                if spec.gen_len == 0:
                    self._terminate(req, Status.COMPLETED)
                else:
                    req.status = Status.DECODE
                    self._log("prefill_done", rid=req.rid, pages=req.pages)
            elif req.status == Status.DECODE:
                if req.gen_done > spec.gen_len:
                    raise AssertionError(
                        f"{req.rid}: decoded {req.gen_done} > actual gen_len "
                        f"{spec.gen_len} (scheduler must not know this)"
                    )
                if req.gen_done == spec.gen_len:
                    self._terminate(req, Status.COMPLETED)

        # A cancellation deadline may land within this step.
        self._apply_cancellations()

        frag = self.alloc.internal_frag_tokens(self.active)
        self._time_weighted_frag += frag * dt
        self.alloc.sample_frag(self.active)

        self.steps_log.append(
            {
                "step": self.step_count,
                "t_start": round(t_start, 6),
                "t_end": round(self.now, 6),
                "duration": round(dt, 6),
                "prefill_tokens": plan.prefill_tokens,
                "decode_tokens": plan.decode_tokens,
                "recompute_tokens": plan.recompute_tokens,
                "used_pages": self.alloc.used_pages,
                "free_pages": self.alloc.free_pages,
                "peak_pages": self.alloc.peak_pages,
                "internal_frag_tokens": frag,
                "admitted": admitted,
                "preempted": pre_preempt,
                "waiting_head": self._oldest_waiting_id(),
                "planned_head": getattr(plan, "planned_head", None),
                "decoded": decoded,
                "finished_prefill": finished_prefill,
            }
        )
        self._log(
            "step_end",
            step=self.step_count,
            duration=round(dt, 6),
            used_pages=self.alloc.used_pages,
            internal_frag_tokens=frag,
            decoded=decoded,
        )
