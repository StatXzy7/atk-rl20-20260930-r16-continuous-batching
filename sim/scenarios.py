"""测试与演示场景。

成本模型在策略比较之前固定：所有策略对同一场景使用同一个 SimConfig
（含同一个 CostModel），保证公平。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from .model import Cancellation, CostModel, PolicyKind, RequestSpec, SimConfig


@dataclass
class Scenario:
    name: str
    description: str
    cfg: SimConfig
    specs: List[RequestSpec]
    cancellations: List[Cancellation]

    def with_policy(self, policy: PolicyKind) -> "Scenario":
        cfg = SimConfig(
            policy=policy,
            token_budget=self.cfg.token_budget,
            chunk_max=self.cfg.chunk_max,
            page_size=self.cfg.page_size,
            total_pages=self.cfg.total_pages,
            starvation_threshold=self.cfg.starvation_threshold,
            min_protected_chunk=self.cfg.min_protected_chunk,
            decode_starvation_gap=self.cfg.decode_starvation_gap,
            max_preemptions=self.cfg.max_preemptions,
            cost=self.cfg.cost,
        )
        return Scenario(self.name, self.description, cfg,
                        list(self.specs), list(self.cancellations))


# ---------------------------------------------------------------------------
# 微型手算用例（README 中有完整手工事件表）
#   页大小 4、总 6 页、预算 8、chunk 8、cp=cd=1
# ---------------------------------------------------------------------------

def micro_handcheck() -> Scenario:
    cfg = SimConfig(
        policy=PolicyKind.CONT,  # 占位，with_policy 时覆盖
        token_budget=8, chunk_max=8, page_size=4, total_pages=6,
        starvation_threshold=20.0, min_protected_chunk=2,
        decode_starvation_gap=4,
        cost=CostModel(prefill_cost=1.0, decode_cost=1.0,
                       throughput=1.0, fixed_fee=0.0),
    )
    specs = [
        RequestSpec("R1", arrival=0, prompt_len=5, output_len=3),
        RequestSpec("R2", arrival=1, prompt_len=3, output_len=2),
    ]
    return Scenario("micro_handcheck",
                    "微型用例：5+3 与 3+2 两请求，6 页，预算 8",
                    cfg, specs, [])


# ---------------------------------------------------------------------------
# 边界：output_len=0 / =1 特殊请求
# ---------------------------------------------------------------------------

def zero_and_single_token() -> Scenario:
    cfg = SimConfig(
        policy=PolicyKind.CONT, token_budget=8, chunk_max=8,
        page_size=4, total_pages=8, starvation_threshold=10.0,
        min_protected_chunk=2, decode_starvation_gap=4,
        cost=CostModel(1.0, 1.0, 1.0, 0.0),
    )
    specs = [
        RequestSpec("Z0", arrival=0, prompt_len=4, output_len=0),
        RequestSpec("O1", arrival=0, prompt_len=2, output_len=1),
        RequestSpec("N3", arrival=0, prompt_len=6, output_len=3),
    ]
    return Scenario("zero_and_single",
                    "output_len=0（仅 prefill）与 =1（单个 decode token）",
                    cfg, specs, [])


# ---------------------------------------------------------------------------
# 压力 1：长 prefill 中混入短请求（chunked prefill 的价值场景）
# ---------------------------------------------------------------------------

def long_prefill_mix() -> Scenario:
    cfg = SimConfig(
        policy=PolicyKind.CONT, token_budget=16, chunk_max=8,
        page_size=8, total_pages=12, starvation_threshold=12.0,
        min_protected_chunk=4, decode_starvation_gap=4,
        cost=CostModel(1.0, 1.0, 1.0, 0.0),
    )
    specs = [
        RequestSpec("LONG", 0,  48, 4),    # 长 prompt，需 6 页完成时
        RequestSpec("S1",   2,  4,  3),    # 短请求在长 prefill 期间到达
        RequestSpec("S2",   5,  8,  2),
        RequestSpec("S3",   9,  4,  2),
        RequestSpec("MID",  12, 24, 3),
    ]
    return Scenario("long_prefill_mix",
                    "长 prefill(48) 中穿插多个短请求",
                    cfg, specs, [])


# ---------------------------------------------------------------------------
# 压力 2：突发到达（同一时刻大量请求）
# ---------------------------------------------------------------------------

def burst_arrival() -> Scenario:
    cfg = SimConfig(
        policy=PolicyKind.CONT, token_budget=16, chunk_max=8,
        page_size=8, total_pages=16, starvation_threshold=10.0,
        min_protected_chunk=4, decode_starvation_gap=4,
        cost=CostModel(1.0, 1.0, 1.0, 0.0),
    )
    specs = [RequestSpec(f"B{i:02d}", 5, (i % 5 + 1) * 4,
                         (i % 3) + 1) for i in range(12)]
    return Scenario("burst_arrival",
                    "t=5 突发 12 个请求（共需页接近池上限）",
                    cfg, specs, [])


# ---------------------------------------------------------------------------
# 压力 3：页不足（迫使等待 / 抢占重算）
# ---------------------------------------------------------------------------

def page_starvation() -> Scenario:
    cfg = SimConfig(
        policy=PolicyKind.CONT, token_budget=16, chunk_max=8,
        page_size=4, total_pages=10, starvation_threshold=6.0,
        min_protected_chunk=4, decode_starvation_gap=3,
        max_preemptions=100,
        cost=CostModel(1.0, 1.0, 1.0, 0.0),
    )
    specs = [
        # 早期 decoder 会逐步吃满页；后到的大 prompt 需要抢占
        RequestSpec("A", 0,  12, 12),   # 完成时需要 (12+12)/4=6 页
        RequestSpec("B", 0,  12, 12),   # 与 A 合计完成时 12 页 > 10
        RequestSpec("C", 8,  16, 2),    # 自身需要 5 页，到达时页紧张
        RequestSpec("D", 14,  8, 6),    # 完成需 4 页
    ]
    return Scenario("page_starvation",
                    "页池 10 页，长期高占用 + 后到大 prompt，触发抢占重算",
                    cfg, specs, [])


# ---------------------------------------------------------------------------
# 压力 4：取消（等待中 / prefill 中 / decode 中）
# ---------------------------------------------------------------------------

def cancellation_mix() -> Scenario:
    cfg = SimConfig(
        policy=PolicyKind.CONT, token_budget=8, chunk_max=4,
        page_size=4, total_pages=8, starvation_threshold=8.0,
        min_protected_chunk=2, decode_starvation_gap=4,
        cost=CostModel(1.0, 1.0, 1.0, 0.0),
    )
    specs = [
        RequestSpec("K1", 0,  16, 20),   # 长 decode
        RequestSpec("K2", 1,  8,  20),
        RequestSpec("K3", 2,  4,  20),
        RequestSpec("K4", 6,  8,  5),
    ]
    cancels = [
        Cancellation("K3", 3),    # 等待中取消
        Cancellation("K1", 12),   # decode 中取消，释放页给后续
        Cancellation("KX", 4),    # 未知请求，忽略
    ]
    return Scenario("cancellation_mix",
                    "等待中与 decode 中取消，校验私有页全部释放",
                    cfg, specs, cancels)


# ---------------------------------------------------------------------------
# 压力 5：prompt 超过总页数 -> 明确拒绝
# ---------------------------------------------------------------------------

def oversized_prompt() -> Scenario:
    cfg = SimConfig(
        policy=PolicyKind.CONT, token_budget=8, chunk_max=8,
        page_size=4, total_pages=4, starvation_threshold=8.0,
        min_protected_chunk=2, decode_starvation_gap=4,
        cost=CostModel(1.0, 1.0, 1.0, 0.0),
    )
    specs = [
        RequestSpec("OK1", 0, 12, 2),    # 需要 3 页，可容纳
        RequestSpec("HUGE", 1, 20, 2),   # 需要 5 页 > 4，拒绝
        RequestSpec("OK2", 2, 8, 1),     # 拒绝不应影响后续正常请求
    ]
    return Scenario("oversized_prompt",
                    "prompt 自身超过总页数必须被明确拒绝",
                    cfg, specs, [])


def all_scenarios() -> List[Scenario]:
    return [
        micro_handcheck(),
        zero_and_single_token(),
        long_prefill_mix(),
        burst_arrival(),
        page_starvation(),
        cancellation_mix(),
        oversized_prompt(),
    ]
