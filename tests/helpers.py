from sim.model import CostModel, RequestSpec, SimConfig
from sim.run import run_once


def micro_config(**over):
    kw = dict(
        page_size=4,
        num_pages=8,
        token_budget=4,
        max_chunk=2,
        cost=CostModel(prefill_time_per_token=1, decode_time_per_token=1),
    )
    kw.update(over)
    return SimConfig(**kw)


def micro_workload():
    return [RequestSpec("A", 0, 3, 2), RequestSpec("B", 0, 2, 3)]


def run(policy, workload=None, config=None, **kw):
    return run_once(
        workload or micro_workload(), config or micro_config(), policy, **kw
    )


def compact_steps(result):
    return [
        (
            s["step"],
            s["t_start"],
            s["t_end"],
            s["prefill_tokens"],
            s["decode_tokens"],
            s["recompute_tokens"],
            s["used_pages"],
            tuple(s["admitted"]),
            tuple(s["preempted"]),
            tuple(d["rid"] for d in s["decoded"]),
        )
        for s in result.steps
    ]


def assert_invariants(result, cfg):
    """Cross-cutting invariants every valid simulation must satisfy."""
    # Time monotonicity and positive durations.
    prev_end = 0.0
    for s in result.steps:
        assert s["t_start"] >= prev_end - 1e-9
        assert s["t_end"] >= s["t_start"] - 1e-9
        assert s["duration"] >= 0
        prev_end = s["t_end"]

    # Event log strictly ordered in time (emitted within one step share t).
    last_t = -1.0
    for ev in result.events:
        assert ev["time"] >= last_t - 1e-9
        last_t = ev["time"]

    # Token budget per step never exceeded.
    for s in result.steps:
        assert s["prefill_tokens"] + s["decode_tokens"] <= cfg.token_budget

    # Page pool never oversubscribed; peak within pool.
    for s in result.steps:
        assert 0 <= s["used_pages"] <= cfg.num_pages
        assert s["used_pages"] + s["free_pages"] == cfg.num_pages
    assert result.alloc.peak_pages <= cfg.num_pages

    # No leaked pages: every terminal request holds zero pages and allocator
    # is fully drained at the end (all workloads here are finite).
    assert result.alloc.used_pages == 0
    assert result.alloc.free_pages == cfg.num_pages
    for req in result.requests.values():
        assert req.pages == 0

    # Internal fragmentation non-negative and consistent.
    for s in result.steps:
        assert s["internal_frag_tokens"] >= 0
        assert s["internal_frag_tokens"] < cfg.num_pages * cfg.page_size
