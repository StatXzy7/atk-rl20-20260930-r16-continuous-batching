"""Anti-starvation guarantees and applicability conditions, plus the
structural 'no future knowledge' guard (scheduler cannot see actual gen_len).

Bounded-wait guarantee (preemption='recompute', grace=0, max_chunk=q <= B):
  When a request is the oldest *waiting* request, every step in which it is not
  itself preempted advances its prompt by up to q guaranteed tokens (pass 1
  runs before decoders and reserves budget; strictly-younger victims can be
  evicted for pages). A waiting head makes zero prompt progress only on a step
  where it is preempted by an even older beneficiary. Hence its number of
  stalled steps is bounded by its preemption count, and its
  waiting->prefill-finish time is bounded (ceil(prompt_len/q) work-steps plus
  billed recompute work) independently of other requests' unknown gen lengths.
The wait policy serializes prompt processing instead (no evictions); its bound
is workload-dependent (sum of older prompt work), documented in ANALYSIS.md.
"""
import pathlib

from sim.model import CostModel, RequestSpec, SimConfig, Status, pages_for_tokens
from sim.run import run_once
from sim.metrics import request_metrics


def cfg():
    return SimConfig(
        page_size=4,
        num_pages=12,
        token_budget=4,
        max_chunk=2,
        cost=CostModel(prefill_time_per_token=1, decode_time_per_token=2),
    )


def _prefill_events(result, rid):
    return [
        ev for ev in result.events
        if ev["event"] == "prefill_chunk" and ev.get("rid") == rid
    ]


def test_head_never_stalled_without_an_older_active_or_own_preemption():
    c = cfg()
    wl = [
        RequestSpec("A", 0, 6, 30),
        RequestSpec("B", 1, 6, 30),
        RequestSpec("C", 2, 6, 30),
    ]
    r = run_once(wl, c, "cb", preemption="recompute")
    arrivals = {rid: q.arrival for rid, q in r.requests.items()}
    for s in r.steps:
        head = s.get("planned_head")
        if head is None:
            continue
        served = any(ev["time"] == s["t_end"] for ev in _prefill_events(r, head))
        preempted_this_step = head in s["preempted"]
        if served or preempted_this_step:
            continue
        # The only legitimate reason the oldest waiting head does not advance
        # this step: strictly older requests are resident and hold the pages.
        older_active = any(
            rid != head and arrivals[rid] < arrivals[head]
            for rid in self_rids_at(r, s)
        )
        assert older_active, (s["step"], head)
    assert all(x.status == Status.COMPLETED.value for x in request_metrics(r))


def self_rids_at(result, step):
    ids = set(step["admitted"]) | {d["rid"] for d in step["decoded"]}
    for ev in result.events:
        if abs(ev["time"] - step["t_end"]) < 1e-9 and "rid" in ev:
            ids.add(ev["rid"])
    return ids


def test_head_prompt_finishes_in_prompt_over_q_chunks():
    # Long is the strictly-oldest head; younger decoders arrive later and may
    # be evicted for its guaranteed slices, so Long is never itself preempted
    # even under heavy (in-pool) decode pressure.
    c = SimConfig(page_size=4, num_pages=16, token_budget=8, max_chunk=4,
                  cost=CostModel(1, 3))
    wl = [RequestSpec("Long", 0, 40, 5)]
    for i in range(3):
        wl.append(RequestSpec(f"D{i}", 5 + i, 4, 8))
    r = run_once(wl, c, "cb", preemption="recompute")
    chunks = [ev for ev in _prefill_events(r, "Long") if not ev.get("recompute")]
    assert sum(ev["tokens"] for ev in chunks) == 40
    assert all(ev["tokens"] <= 4 for ev in chunks)
    assert r.requests["Long"].preemptions == 0
    assert r.requests["Long"].first_token_at is not None



def test_recompute_victims_strictly_younger_than_beneficiary():
    c = SimConfig(page_size=4, num_pages=4, token_budget=4, max_chunk=4,
                  cost=CostModel(1, 2))
    wl = [
        RequestSpec("A", 0, 9, 4),
        RequestSpec("B", 3, 5, 1),
        RequestSpec("C", 6, 5, 1),
    ]
    r = run_once(wl, c, "cb", preemption="recompute")
    arrs = {rid: q.arrival for rid, q in r.requests.items()}
    for s in r.steps:
        for victim in s["preempted"]:
            # The request progressing this step that the eviction benefits is
            # a prefiller admitted in the same step. In this scenario every
            # eviction is triggered to finish an older request's prompt.
            progressing = s["admitted"] + [
                d["rid"] for d in s["decoded"]
            ] + [
                ev.get("rid")
                for ev in r.events
                if ev["event"] == "prefill_chunk" and ev["time"] == s["t_end"]
            ]
            older_beneficiary = [b for b in set(progressing) if arrs[b] < arrs[victim]]
            assert older_beneficiary, (s["step"], victim, progressing)


def test_wait_policy_serializes_prefills():
    c = SimConfig(page_size=4, num_pages=6, token_budget=4, max_chunk=4,
                  cost=CostModel(1, 2))
    wl = [RequestSpec("A", 0, 9, 4), RequestSpec("B", 0, 9, 4)]
    r = run_once(wl, c, "cb", preemption="wait")
    assert all(x.status == Status.COMPLETED.value for x in request_metrics(r))
    for s in r.steps:
        assert s["used_pages"] <= 3


def test_physically_infeasible_runtime_sequence_explicitly_rejected():
    c = SimConfig(page_size=4, num_pages=4, token_budget=4, max_chunk=4,
                  cost=CostModel(1, 2))
    wl = [RequestSpec("A", 0, 9, 100)]
    r = run_once(wl, c, "cb", preemption="recompute")
    req = r.requests["A"]
    assert req.status == Status.REJECTED
    assert req.reject_reason == "sequence_exceeds_pool_at_runtime"
    assert pages_for_tokens(req.kv_tokens, 4) <= 4
    assert r.alloc.used_pages == 0


def test_scheduler_cannot_read_actual_gen_len():
    import ast

    root = pathlib.Path(__file__).resolve().parents[1]
    tree = ast.parse((root / "sim" / "scheduler.py").read_text())

    imported_names = set()
    used_names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for a in node.names:
                imported_names.add(a.asname or a.name)
        elif isinstance(node, ast.Attribute):
            if isinstance(node.value, ast.Name):
                used_names.add((node.value.id, node.attr))
        elif isinstance(node, ast.Name):
            used_names.add(("", node.id))

    assert "RequestSpec" not in imported_names
    # gen_len (environment truth) must never be referenced; gen_done (observed
    # progress) is allowed.
    assert all(attr != "gen_len" for _, attr in used_names)
    from sim.model import Request
    assert not hasattr(Request(rid="x", arrival=0, prompt_len=1), "gen_len")
