"""Hand-computed micro event tables.

Workload: A(arrival=0,prompt=3,gen=2), B(0,prompt=2,gen=3)
Config: page_size=4, pool=8 pages, B=4, q=2, prefill cost 1, decode cost 1.

FCFS (static, one request in flight at a time here because prompt prefix
3+2=5 > B=4):
  s1 [0,3)  prefill A 3              used pages ceil(3/4)=1 frag 1
  s2 [3,4)  decode A1   pages stay 1 frag 0
  s3 [4,5)  decode A2 (done) pages 0
  s4 [5,7)  prefill B 2  pages 1 frag 2
  s5 [7,8)  decode B1
  s6 [8,9)  decode B2
  s7 [9,10) decode B3 (done) -> makespan 10

Continuous batching (chunked, q=2):
  s1 [0,4)  A 2 + B 2 prefill, pages 2 (A1,B1) frag 4
  s2 [4,6)  A1 prefill + B1 decode; A prompt done; B gen1
  s3 [6,8)  A1 decode + B1 decode (A gen1, B gen2)
  s4 [8,10) A1 decode(done) + B1 decode(done) -> makespan 10

These numbers were computed by hand independently of the implementation; the
test asserts the exact step ledger, not just aggregates.
"""
from tests.helpers import compact_steps, micro_config, micro_workload, run
from sim.metrics import request_metrics


FCFS_TABLE = [
    (1, 0, 3, 3, 0, 0, 1, ("A",), (), ()),
    (2, 3, 4, 0, 1, 0, 1, (), (), ("A",)),
    (3, 4, 5, 0, 1, 0, 0, (), (), ("A",)),
    (4, 5, 7, 2, 0, 0, 1, ("B",), (), ()),
    (5, 7, 8, 0, 1, 0, 1, (), (), ("B",)),
    (6, 8, 9, 0, 1, 0, 1, (), (), ("B",)),
    (7, 9, 10, 0, 1, 0, 0, (), (), ("B",)),
]

CB_TABLE = [
    (1, 0, 4, 4, 0, 0, 2, ("A", "B"), (), ()),
    (2, 4, 6, 1, 1, 0, 2, (), (), ("B",)),
    (3, 6, 8, 0, 2, 0, 2, (), (), ("A", "B")),
    (4, 8, 10, 0, 2, 0, 0, (), (), ("A", "B")),
]


def test_fcfs_hand_table():
    r = run("fcfs")
    assert compact_steps(r) == FCFS_TABLE
    assert r.makespan == 10
    m = {x.rid: x for x in request_metrics(r)}
    assert m["A"].prompt_latency == 3
    assert m["A"].ttft == 4
    assert m["A"].completion_latency == 5
    assert m["A"].itls == [1]
    assert m["B"].prompt_latency == 7
    assert m["B"].ttft == 8
    assert m["B"].completion_latency == 10
    assert m["B"].itls == [1, 1]


def test_cb_hand_table():
    r = run("cb")
    assert compact_steps(r) == CB_TABLE
    assert r.makespan == 10
    m = {x.rid: x for x in request_metrics(r)}
    assert m["A"].prompt_latency == 6
    assert m["A"].ttft == 8
    assert m["A"].completion_latency == 10
    assert m["A"].itls == [2]
    assert m["B"].prompt_latency == 4
    assert m["B"].ttft == 6
    assert m["B"].completion_latency == 10
    assert m["B"].itls == [2, 2]


def test_both_use_identical_cost_and_budget():
    # One fixed cost model / budget shared by both policy runs.
    cfg = micro_config()
    a = run("fcfs", config=cfg)
    b = run("cb", config=cfg)
    assert a.config.cost.prefill_time_per_token == b.config.cost.prefill_time_per_token
    assert a.config.cost.decode_time_per_token == b.config.cost.decode_time_per_token
    assert a.config.cost.launch_overhead == b.config.cost.launch_overhead
    assert a.config.token_budget == b.config.token_budget
    assert a.config.page_size == b.config.page_size
    assert a.config.num_pages == b.config.num_pages
    assert a.config.max_chunk == b.config.max_chunk
