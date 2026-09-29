"""Special cases: zero/one generated tokens, oversized prompt rejection,
page-boundary allocation, cancellations, and label discipline."""
from sim.model import RequestSpec, Status
from sim.metrics import aggregate, request_metrics
from tests.helpers import assert_invariants, micro_config, run


def test_zero_generation_completes_at_prefill():
    cfg = micro_config()
    wl = [RequestSpec("Z", 0, 3, 0)]
    r = run("cb", wl, cfg)
    m = request_metrics(r)[0]
    assert m.status == Status.COMPLETED.value
    assert m.actual_gen_len == 0
    assert m.gen_done == 0
    assert m.ttft is None
    assert m.ttft_note == "no_generation"
    assert m.itls == []
    assert m.itl_mean is None
    assert m.prompt_latency == 3
    assert m.completion_latency == 3
    assert_invariants(r, cfg)


def test_single_generation_ttft_equals_completion():
    cfg = micro_config()
    wl = [RequestSpec("O", 0, 4, 1)]
    r = run("cb", wl, cfg)
    m = request_metrics(r)[0]
    assert m.gen_done == 1
    assert m.ttft == m.completion_latency
    assert m.itls == []
    assert m.ttft_note is None
    assert m.prompt_latency == 4
    assert_invariants(r, cfg)


def test_prompt_larger_than_pool_rejected_both_policies():
    cfg = micro_config(num_pages=2, page_size=4)  # pool holds 8 tokens
    wl = [RequestSpec("Big", 0, 9, 5)]
    for pol in ("fcfs", "cb"):
        r = run(pol, wl, cfg)
        req = r.requests["Big"]
        assert req.status == Status.REJECTED
        assert req.reject_reason == "prompt_exceeds_page_pool"
        assert req.pages == 0
        m = request_metrics(r)[0]
        assert m.ttft is None
        assert m.ttft_note == "rejected"
        assert_invariants(r, cfg)


def test_pages_acquired_exactly_at_boundaries():
    # page_size=4: 1-4 tokens -> 1 page, 5-8 -> 2, 9-12 -> 3.
    cfg = micro_config(page_size=4, num_pages=8, token_budget=2, max_chunk=2)
    wl = [RequestSpec("P", 0, 9, 0)]
    r = run("cb", wl, cfg)
    # prefill chunk end tokens: 2,4,6,8,9
    expected_pages = [1, 1, 2, 2, 3]
    prefill_steps = [s for s in r.steps if s["prefill_tokens"] > 0]
    seen = []
    for s in prefill_steps:
        # find P's page count after the chunk via prefill_chunk events
        pass
    pages_after = [
        ev["pages"] for ev in r.events if ev["event"] == "prefill_chunk"
    ]
    assert pages_after == expected_pages
    assert_invariants(r, cfg)


def test_cancellation_waiting_then_running():
    cfg = micro_config(num_pages=4)
    wl = [
        RequestSpec("A", 0, 4, 10, cancel_at=6),
        RequestSpec("B", 1, 2, 1, cancel_at=2),  # cancelled while waiting
    ]
    r = run("cb", wl, cfg, preemption="wait")
    a, b = r.requests["A"], r.requests["B"]
    assert b.status == Status.CANCELLED
    assert b.pages == 0
    assert a.status == Status.CANCELLED
    assert a.pages == 0
    # all private pages released
    assert r.alloc.used_pages == 0
    assert_invariants(r, cfg)


def test_results_labeled_prediction_not_measurement():
    r = run("cb")
    agg = aggregate(r)
    assert agg["label"] == "simulated_prediction"
    assert agg["not_hardware_measurement"] is True
    for ev in r.events:
        assert "measured" not in ev.get("event", "")
