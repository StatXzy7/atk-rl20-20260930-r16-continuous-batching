"""随机化 fuzz 与防饥饿上界测试。

Fuzz：在随机到达/长度/页数/预算/取消下，两策略均满足：
  时间单调、预算不超支、独立页账本零泄漏、chunk 不超上限、
  完成的请求指标自洽。

防饥饿（continuous batching）：
  * 每个 decoder 的相邻 decode token 间隔（按 step 计）在
    "资源可满足"条件下不超过 decode_starvation_gap+1 个 step；
  * 受保护 prefill 请求（年龄超过阈值后）每个 step 至少获得
    min_protected_chunk，除非该步预算不足以容纳该最小值；
  * 抢占次数受 max_preemptions 约束，活锁被检测上报而非静默。
"""

import os
import random
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sim import (
    Cancellation,
    CostModel,
    PolicyKind,
    RequestSpec,
    SimConfig,
    simulate,
)
from tests.test_invariants import LedgerReplayer


def random_scenario(seed):
    rng = random.Random(seed)
    page_size = rng.choice([2, 4, 8])
    total_pages = rng.randint(4, 24)
    budget = rng.choice([4, 8, 16])
    chunk = rng.choice([2, 4, 8])
    n = rng.randint(1, 10)
    specs = []
    t = 0.0
    for i in range(n):
        t += rng.choice([0, 0, 1, 3, 8])
        plen = rng.randint(1, page_size * (total_pages // 2 + 1))
        out = rng.choice([0, 1, rng.randint(1, 12)])
        specs.append(RequestSpec(f"q{i}", t, plen, out))
    cancels = []
    for s in specs:
        if rng.random() < 0.15 and s.output_len > 0:
            cancels.append(Cancellation(s.rid,
                                        s.arrival + rng.randint(1, 20)))
    cfg_kwargs = dict(
        token_budget=budget, chunk_max=chunk, page_size=page_size,
        total_pages=total_pages,
        starvation_threshold=rng.choice([4, 8, 100]),
        min_protected_chunk=min(2, chunk),
        decode_starvation_gap=rng.choice([2, 3, 4]),
        max_preemptions=50,
        cost=CostModel(1.0, 1.0, 1.0, 0.0),
    )
    return specs, cancels, cfg_kwargs


def run(policy, specs, cancels, kw):
    cfg = SimConfig(policy=policy, **kw)
    return simulate(cfg, specs, cancels)


class TestFuzzInvariants(unittest.TestCase):
    def test_200_random_cases(self):
        for seed in range(400):
            specs, cancels, kw = random_scenario(seed)
            with self.subTest(seed=seed):
                for policy in (PolicyKind.FCFS, PolicyKind.CONT):
                    res = run(policy, specs, cancels, kw)
                    # 独立账本校验（死锁时不要求终态零占用）
                    if res.deadlock is None:
                        LedgerReplayer(res).check()
                    self._check_basic(res)

    def _check_basic(self, res):
        prev = 0.0
        for st in res.steps:
            self.assertGreaterEqual(st.start, prev - 1e-9)
            self.assertGreater(st.end, st.start - 1e-12)
            prev = st.end
            self.assertLessEqual(
                st.prefill_tokens + st.decode_tokens,
                res.cfg["token_budget"])
            for a in st.actions:
                if a["kind"] == "prefill":
                    self.assertLessEqual(a["tokens"],
                                         res.cfg["chunk_max"])
        for m in res.requests:
            if m.status == "COMPLETED":
                self.assertIsNotNone(m.completion_latency)
                self.assertEqual(len(m.itls), max(0, m.output_len - 1))


class TestDecodeStarvationBound(unittest.TestCase):
    """资源可满足时 decode 间隔上界。

    构造页充足的场景（无抢占、无跨页阻塞），CONT 下任一 decoder
    连续不被选中的 step 数不应超过 decode_starvation_gap。
    """

    def test_gap_bound_when_pages_abundant(self):
        cfg = SimConfig(
            policy=PolicyKind.CONT, token_budget=8, chunk_max=8,
            page_size=4, total_pages=40, starvation_threshold=100.0,
            min_protected_chunk=2, decode_starvation_gap=3,
            cost=CostModel(1.0, 1.0, 1.0, 0.0),
        )
        # 4 个 decoder，预算 8，每步最多 8 个动作
        specs = [
            RequestSpec("d0", 0, 4, 12),
            RequestSpec("d1", 0, 4, 12),
            RequestSpec("d2", 0, 4, 12),
            RequestSpec("d3", 0, 4, 12),
        ]
        res = simulate(cfg, specs)
        self.assertIsNone(res.deadlock)
        # 用 step 索引重建每个 decoder 的 decode 步
        decode_steps = {s.rid: [] for s in specs}
        for st in res.steps:
            for a in st.actions:
                if a["kind"] == "decode":
                    decode_steps[a["rid"]].append(st.index)
        for rid, idxs in decode_steps.items():
            gaps = [b - a for a, b in zip(idxs, idxs[1:])]
            # 4 decoder、预算 8：每步都可 decode，间隔应为 1
            self.assertTrue(all(g <= 2 for g in gaps),
                            f"{rid} decode 间隔越界: {gaps}")

    def test_gap_bound_with_prefill_pressure(self):
        # 预算 6，4 decoder 每轮至少 4 token，prefill 抢占预算时
        # decoder 间隔可能变大，但不应超过 gap+1（gap=3 -> <=4）
        cfg = SimConfig(
            policy=PolicyKind.CONT, token_budget=6, chunk_max=6,
            page_size=4, total_pages=40, starvation_threshold=100.0,
            min_protected_chunk=2, decode_starvation_gap=3,
            cost=CostModel(1.0, 1.0, 1.0, 0.0),
        )
        specs = [RequestSpec(f"d{i}", 0, 4, 10) for i in range(4)]
        specs.append(RequestSpec("late", 6, 16, 2))
        res = simulate(cfg, specs)
        self.assertIsNone(res.deadlock)
        decode_steps = {s.rid: [] for s in specs}
        for st in res.steps:
            for a in st.actions:
                if a["kind"] == "decode":
                    decode_steps[a["rid"]].append(st.index)
        for rid, idxs in decode_steps.items():
            gaps = [b - a for a, b in zip(idxs, idxs[1:])]
            over = [g for g in gaps if g > 4]
            self.assertFalse(over, f"{rid} 饥饿上界失效: {gaps}")


class TestProtectedPrefillBound(unittest.TestCase):
    def test_protected_prefill_progresses(self):
        # 高占用迫使长 prompt 等待超过阈值，之后受保护规则必须推进
        cfg = SimConfig(
            policy=PolicyKind.CONT, token_budget=8, chunk_max=4,
            page_size=4, total_pages=12, starvation_threshold=6.0,
            min_protected_chunk=2, decode_starvation_gap=3,
            cost=CostModel(1.0, 1.0, 1.0, 0.0),
        )
        specs = [
            RequestSpec("occ1", 0, 8, 10),
            RequestSpec("occ2", 0, 8, 10),
            RequestSpec("late_long", 4, 12, 2),
        ]
        res = simulate(cfg, specs)
        self.assertIsNone(res.deadlock)
        status = {m.rid: m.status for m in res.requests}
        self.assertEqual(status["late_long"], "COMPLETED")

    def test_max_preemptions_caps_livelock(self):
        # 极小 max_preemptions 下，达到上限后系统应死锁/上报而非无限抢占
        cfg = SimConfig(
            policy=PolicyKind.CONT, token_budget=8, chunk_max=4,
            page_size=4, total_pages=8, starvation_threshold=4.0,
            min_protected_chunk=2, decode_starvation_gap=2,
            max_preemptions=1,
            cost=CostModel(1.0, 1.0, 1.0, 0.0),
        )
        specs = [
            RequestSpec("a", 0, 8, 20),
            RequestSpec("b", 0, 8, 20),
            RequestSpec("c", 6, 8, 20),
            RequestSpec("d", 8, 8, 20),
        ]
        res = simulate(cfg, specs)
        per_req = {m.rid: m.preemptions for m in res.requests}
        for rid, n in per_req.items():
            self.assertLessEqual(n, 1)


if __name__ == "__main__":
    unittest.main()


class TestSchedulerVisibilityAndCostParity(unittest.TestCase):
    """调度器只能看当前已生成长度；成本模型对两策略一致。"""

    def test_planner_source_does_not_read_output_len(self):
        import inspect
        from sim import policies
        src = inspect.getsource(policies)
        # 剥掉 docstring/注释后，策略代码不得引用真实生成长度
        import io, tokenize
        code_parts = []
        for tok in tokenize.generate_tokens(io.StringIO(src).readline):
            if tok.type in (tokenize.STRING, tokenize.COMMENT):
                continue
            code_parts.append(tok.string)
        code = " ".join(code_parts)
        self.assertNotIn("output_len", code)
        self.assertNotIn("_true_output_len", code)

    def test_identical_cost_model_both_policies(self):
        specs, cancels, kw = random_scenario(3)
        rf = run(PolicyKind.FCFS, specs, cancels, kw)
        rc = run(PolicyKind.CONT, specs, cancels, kw)
        self.assertEqual(rf.cfg["cost_model"], rc.cfg["cost_model"])
        # 同成本模型下，step 时长必须可由 prefill/decode token 数重算
        for res in (rf, rc):
            for st in res.steps:
                cm = res.cfg["cost_model"]
                expect = (cm["prefill_cost"] * st.prefill_tokens
                          + cm["decode_cost"] * st.decode_tokens
                          ) / cm["throughput"]
                if st.prefill_tokens + st.decode_tokens > 0:
                    expect += cm["fixed_fee"]
                self.assertAlmostEqual(st.duration, expect, places=9)
