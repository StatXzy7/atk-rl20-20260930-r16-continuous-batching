"""Stress sequences: long prefill + short bursts, page scarcity,
cancellations, preemption/recompute, and global invariants."""
import random

from sim.model import CostModel, RequestSpec, SimConfig, Status
from sim.run import run_once
from sim.metrics import request_metrics
from tests.helpers import assert_invariants


def stress_config(**kw):
    base = dict(
        page_size=4,
        num_pages=16,
        token_budget=8,
        max_chunk=4,
        cost=CostModel(prefill_time_per_token=1, decode_time_per_token=3),
    )
    base.update(kw)
    return SimConfig(**base)


def long_mix_workload():
    wl = [RequestSpec("Long", 0, 40, 6)]
    # short requests arrive in a burst while Long is prefilling
    for i in range(12):
        wl.append(RequestSpec(f"S{i}", 1 + 0.1 * i, 2 + (i % 3), 1 + (i % 4)))
    return wl


def test_long_prefill_mixed_shorts_completes():
    cfg = stress_config()
    wl = long_mix_workload()
    for pol, kw in (("fcfs", {}), ("cb", dict(preemption="recompute")),
                    ("cb", dict(preemption="wait"))):
        r = run_once(wl, cfg, pol, **kw)
        rows = request_metrics(r)
        assert all(x.status == Status.COMPLETED.value for x in rows), pol
        assert_invariants(r, cfg)


def test_burst_arrivals_monotone_and_budget_ok():
    cfg = stress_config(num_pages=8)
    wl = [RequestSpec(f"R{i}", 0.0, 3 + i % 5, 1 + i % 6) for i in range(20)]
    for pol, kw in (("fcfs", {}), ("cb", dict(preemption="recompute"))):
        r = run_once(wl, cfg, pol, **kw)
        assert r.makespan > 0
        for s in r.steps:
            assert s["prefill_tokens"] + s["decode_tokens"] <= cfg.token_budget
            assert s["used_pages"] <= cfg.num_pages
        assert_invariants(r, cfg)


def test_page_scarcity_triggers_preemption_and_bills_recompute():
    cfg = stress_config(num_pages=4, page_size=4, max_chunk=4, token_budget=4,
                        cost=CostModel(1, 2, 0, 1.0))
    wl = [
        RequestSpec("A", 0, 13, 1),
        RequestSpec("B", 0, 13, 1),
    ]
    r = run_once(wl, cfg, "cb", preemption="recompute")
    rows = {x.rid: x for x in request_metrics(r)}
    assert rows["A"].status == Status.COMPLETED.value
    assert rows["B"].status == Status.COMPLETED.value
    # B (younger) is preempted and its full prompt recomputed & billed.
    assert r.requests["B"].preemptions >= 1
    assert rows["B"].recompute_tokens == 13
    # recompute charged at prefill rate: B TTFT includes the recompute work.
    assert r.makespan >= 13 + 2  # recompute + decodes lower bound
    assert_invariants(r, cfg)


def test_wait_policy_serializes_under_scarcity_no_eviction():
    cfg = stress_config(num_pages=4, page_size=4, max_chunk=4, token_budget=4,
                        cost=CostModel(1, 2))
    wl = [RequestSpec("A", 0, 13, 1), RequestSpec("B", 0, 13, 1)]
    r = run_once(wl, cfg, "cb", preemption="wait")
    rows = {x.rid: x for x in request_metrics(r)}
    assert rows["A"].status == Status.COMPLETED.value
    assert rows["B"].status == Status.COMPLETED.value
    assert sum(x.preemptions for x in rows.values()) == 0
    assert sum(x.recompute_tokens for x in rows.values()) == 0
    # at most one long prompt resident at peak (conservative admission)
    assert r.alloc.peak_pages <= 4
    assert_invariants(r, cfg)


def test_cancellations_under_load_release_all_pages():
    cfg = stress_config(num_pages=10)
    wl = long_mix_workload()
    # cancel a few at different phases
    for rid, t in (("Long", 20), ("S0", 5), ("S5", 100)):
        spec = next(s for s in wl if s.rid == rid)
        object.__setattr__(spec, "cancel_at", t)
    for pol, kw in (("fcfs", {}), ("cb", dict(preemption="recompute"))):
        r = run_once(wl, cfg, pol, **kw)
        for req in r.requests.values():
            assert req.pages == 0
            assert req.status.terminal
        assert r.alloc.used_pages == 0
        assert_invariants(r, cfg)


def test_randomized_stress_deterministic_and_leak_free():
    rng = random.Random(1234)
    cfg = stress_config(num_pages=12)
    wl = []
    for i in range(40):
        gen = rng.choice([0, 1, rng.randint(1, 20)])
        wl.append(
            RequestSpec(
                f"R{i}",
                round(rng.uniform(0, 30), 2),
                rng.randint(1, 30),
                gen,
                cancel_at=round(rng.uniform(5, 60), 2) if rng.random() < 0.1 else None,
            )
        )
    for pol, kw in (("fcfs", {}), ("cb", dict(preemption="recompute")),
                    ("cb", dict(preemption="wait"))):
        r1 = run_once(list(wl), cfg, pol, **kw)
        r2 = run_once(list(wl), cfg, pol, **kw)
        assert r1.makespan == r2.makespan
        assert [s["t_end"] for s in r1.steps] == [s["t_end"] for s in r2.steps]
        for req in r1.requests.values():
            assert req.pages == 0
        assert_invariants(r1, cfg)


def test_fcfs_head_of_line_long_request_delays_shorts():
    # A short request arrives WHILE the long head is prefilling. Static FCFS
    # cannot admit it into the running batch (head-of-line blocking);
    # continuous batching interleaves it via chunked prefill.
    cfg = stress_config(num_pages=16, token_budget=8, max_chunk=4)
    wl = [RequestSpec("Long", 0, 40, 1),
          RequestSpec("Short", 2, 2, 1)]
    fcfs = run_once(wl, cfg, "fcfs")
    cb = run_once(wl, cfg, "cb", preemption="recompute")
    f_short = next(x for x in request_metrics(fcfs) if x.rid == "Short")
    c_short = next(x for x in request_metrics(cb) if x.rid == "Short")
    assert f_short.ttft > c_short.ttft
