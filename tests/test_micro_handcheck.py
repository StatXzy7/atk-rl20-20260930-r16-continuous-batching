"""用手算事件表逐 step 核验微型用例。

配置（与 README 的手工表一致，两种策略用同一成本模型）：
    page_size=4, total_pages=6, token_budget=8, chunk_max=8
    CostModel(prefill=1, decode=1, throughput=1, fixed_fee=0)
    R1: arrival=0 prompt=5 output=3  （output_len 仅环境可见）
    R2: arrival=1 prompt=3 output=2

成本模型在比较前固定，两策略完全一致。
"""

import unittest

from sim import (
    CostModel,
    PolicyKind,
    RequestSpec,
    SimConfig,
    simulate,
)


def cfg(policy):
    return SimConfig(
        policy=policy, token_budget=8, chunk_max=8,
        page_size=4, total_pages=6,
        starvation_threshold=20.0, min_protected_chunk=2,
        decode_starvation_gap=4,
        cost=CostModel(1.0, 1.0, 1.0, 0.0),
    )


SPECS = [
    RequestSpec("R1", 0, 5, 3),
    RequestSpec("R2", 1, 3, 2),
]


class TestFCFSHandCheck(unittest.TestCase):
    def setUp(self):
        self.res = simulate(cfg(PolicyKind.FCFS), SPECS)

    def test_step_table(self):
        # (start, end, prefill_tokens, decode_tokens, pages_after)
        # 元组：(start,end,prefill,decode,after释放前,final释放后)
        expected = [
            (0.0, 5.0, 5, 0, 2, 2),    # batch1: R1 prefill 全量
            (5.0, 6.0, 0, 1, 2, 2),    # R1 decode #1 -> TTFT
            (6.0, 7.0, 0, 1, 2, 2),    # R1 decode #2
            (7.0, 8.0, 0, 1, 2, 0),    # R1 decode #3，步末释放 2 页
            (8.0, 11.0, 3, 0, 1, 1),   # batch2: R2 prefill
            (11.0, 12.0, 0, 1, 1, 1),  # R2 decode #1 -> TTFT
            (12.0, 13.0, 0, 1, 2, 0),  # R2 decode #2 跨页，步末释放
        ]
        got = [(s.start, s.end, s.prefill_tokens, s.decode_tokens,
                s.used_pages_after, s.used_pages_final)
               for s in self.res.steps]
        self.assertEqual(got, expected)

    def test_request_metrics(self):
        m = {x.rid: x for x in self.res.requests}
        self.assertEqual(m["R1"].queue_delay, 0.0)
        self.assertEqual(m["R1"].ttft, 6.0)
        self.assertEqual(m["R1"].completion_latency, 8.0)
        self.assertEqual(m["R1"].itls, [1.0, 1.0])
        self.assertEqual(m["R2"].queue_delay, 7.0)  # 8 - 1
        self.assertEqual(m["R2"].ttft, 11.0)
        self.assertEqual(m["R2"].completion_latency, 12.0)
        self.assertEqual(m["R2"].itls, [1.0])

    def test_peak_and_makespan(self):
        self.assertEqual(self.res.makespan, 13.0)
        self.assertEqual(self.res.peak_pages, 2)
        self.assertIsNone(self.res.deadlock)


class TestContinuousHandCheck(unittest.TestCase):
    def setUp(self):
        self.res = simulate(cfg(PolicyKind.CONT), SPECS)

    def test_step_table(self):
        # (start, end, P, D, pages_after, 动作集合)
        # (start,end,prefill,decode,after释放前,final释放后,动作集合)
        expected = [
            (0.0, 5.0, 5, 0, 2, 2, {("R1", "prefill", 5)}),
            (5.0, 9.0, 3, 1, 3, 3,
             {("R2", "prefill", 3), ("R1", "decode", 1)}),
            (9.0, 11.0, 0, 2, 3, 3,
             {("R2", "decode", 1), ("R1", "decode", 1)}),
            (11.0, 13.0, 0, 2, 4, 0,
             {("R1", "decode", 1), ("R2", "decode", 1)}),
        ]
        got = []
        for s in self.res.steps:
            acts = set()
            for a in s.actions:
                if a["kind"] == "prefill":
                    acts.add((a["rid"], "prefill", a["tokens"]))
                else:
                    acts.add((a["rid"], "decode", 1))
            got.append((s.start, s.end, s.prefill_tokens,
                        s.decode_tokens, s.used_pages_after,
                        s.used_pages_final, acts))
        self.assertEqual(got, expected)

    def test_request_metrics(self):
        m = {x.rid: x for x in self.res.requests}
        self.assertEqual(m["R1"].ttft, 9.0)
        self.assertEqual(m["R1"].completion_latency, 13.0)
        self.assertEqual(m["R1"].itls, [2.0, 2.0])
        self.assertEqual(m["R2"].ttft, 10.0)
        self.assertEqual(m["R2"].completion_latency, 12.0)
        self.assertEqual(m["R2"].itls, [2.0])

    def test_no_preemption_here(self):
        self.assertEqual(self.res.total_preemptions, 0)
        self.assertEqual(self.res.total_replay_work, 0)
        self.assertEqual(self.res.makespan, 13.0)


if __name__ == "__main__":
    unittest.main()
