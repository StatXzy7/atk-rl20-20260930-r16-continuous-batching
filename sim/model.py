"""核心数据模型：请求、配置、成本模型。

关键约束（见任务说明）：
  * RequestSpec 里的 ``output_len`` 是环境侧的"实际生成长度"，
    调度器在任何调度决策中都不允许读取它；终止完全由环境在生成到
    对应长度时发出（见 engine 中的环境回调）。
  * 调度器只能观察 Request 的已 prefill 长度 / 已生成长度等运行态。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


class PolicyKind(str, Enum):
    FCFS = "fcfs"          # 静态分批的先到先服务基线
    CONT = "cont"          # continuous batching + chunked prefill


class Status(str, Enum):
    # 尚未到达
    PENDING = "PENDING"
    # 已到达、已准入、尚未拿到任何 KV 页（排队）
    WAITING = "WAITING"
    # prefill 进行中（含 chunked prefill 的中间状态，以及抢占后的重算）
    PREFILL = "PREFILL"
    # prefill 完成、decode 进行中
    DECODING = "DECODING"
    # 正常完成
    COMPLETED = "COMPLETED"
    # 到达时 prompt 无法在页池中容纳，被明确拒绝
    REJECTED = "REJECTED"
    # 被取消请求（arrival 之后、完成之前）
    CANCELLED = "CANCELLED"


@dataclass(frozen=True)
class RequestSpec:
    """不可变的请求规格。``output_len`` 仅环境可见。"""

    rid: str
    arrival: float
    prompt_len: int          # prompt token 数（>=1）
    output_len: int          # 实际生成长度（>=0），调度器禁止读取

    def __post_init__(self) -> None:
        if self.prompt_len < 1:
            raise ValueError(f"{self.rid}: prompt_len 必须 >=1")
        if self.output_len < 0:
            raise ValueError(f"{self.rid}: output_len 必须 >=0")
        if self.arrival < 0:
            raise ValueError(f"{self.rid}: arrival 必须 >=0")

    def pages_needed(self, page_size: int) -> int:
        """整个请求（prompt + 全部 decode）需要的私有页数。"""
        return -(-(self.prompt_len + self.output_len) // page_size)


@dataclass(frozen=True)
class Cancellation:
    """取消事件：在 ``at`` 时刻取消请求 ``rid``（若尚未完成）。"""

    rid: str
    at: float


@dataclass
class Request:
    """请求的运行态。调度器只允许观察/操作这里的非环境字段。"""

    spec: RequestSpec
    status: Status = Status.PENDING

    # ---- 调度器可见的运行态 ----
    # 已计算并落盘到 KV 中的 prompt token 数（0..prompt_len）
    prefilled: int = 0
    # 已生成 token 数（0..真实 output_len；调度器不知道终点在哪）
    generated: int = 0
    # 当前持有的私有 KV 页列表（页 id，有序；最后一页可能未写满）
    pages: List[int] = field(default_factory=list)
    # 进入当前 prefill 轮次的时间：首次为 arrival；抢占后为重算开始时间
    prefill_enqueue_time: float = 0.0
    # 累计被抢占次数（每抢占一次，下一次需要从头重算 prompt）
    num_preemptions: int = 0
    # 抢占重算时，除 prompt 外还要重放的已生成 token 数；0 表示普通首轮 prefill
    recompute_extra: int = 0
    # 本轮尚需重算的 token 数（被抢占时写入，重算过程中递减）
    recompute_pending: int = 0
    # 成为 decoder 后，有多少个 step 没被选中 decode（防饥饿轮转用）
    decode_gap: int = 0

    # ---- 指标记录 ----
    admit_time: Optional[float] = None
    first_token_time: Optional[float] = None   # 第 1 个 decode token 产出时间
    completion_time: Optional[float] = None
    # 每个 decode token 的产出时间（含第 1 个），用于 ITL
    token_times: List[float] = field(default_factory=list)

    # ---- 工作量记账（含重算浪费）----
    useful_prefill_tokens: int = 0   # 当前轮次最终保留的 prefill token
    recompute_tokens: int = 0        # 因抢占被丢弃、必须重算的 prefill token
    decode_tokens_done: int = 0

    # ---- 环境专用：调度器禁止使用 ----
    def _true_output_len(self) -> int:
        return self.spec.output_len

    @property
    def rid(self) -> str:
        return self.spec.rid

    @property
    def prompt_len(self) -> int:
        return self.spec.prompt_len

    @property
    def arrival(self) -> float:
        return self.spec.arrival

    @property
    def prefill_target(self) -> int:
        """当前 prefill 轮次要计算到的位置：prompt；若被抢占则含重放。"""
        return self.prompt_len + self.recompute_extra

    @property
    def tokens_in_kv(self) -> int:
        """KV 中实际占用的 token 位置数。

        prefill 轮中 prefilled 从 0 计数（含重放段）；decode 阶段为
        prompt + 已生成。
        """
        if self.status == Status.PREFILL:
            return self.prefilled
        if self.status == Status.WAITING:
            return 0
        return self.prefilled + self.generated

    def pages_required_now(self, page_size: int) -> int:
        """容纳当前 KV token 数所需的页数（页边界向上取整）。"""
        return -(-max(self.tokens_in_kv, 1) // page_size)

    def will_need_page_for(self, new_tokens: int, page_size: int) -> bool:
        """若再写入 new_tokens 个 token，是否会越过当前页边界。"""
        before = self.pages_required_now(page_size)
        after = -(-(self.tokens_in_kv + new_tokens) // page_size)
        return after > before


@dataclass(frozen=True)
class CostModel:
    """每步处理成本（处理时间）模型，在策略比较之前固定，两种策略一致使用。

        duration(step) = (cp * prefill_tokens + cd * decode_tokens) / throughput
                         + fixed_fee

    这是一个线性近似的*预测模型*：它不代表任何真实硬件的延迟分布，
    不建模 kernel launch、SM 争用、KV 带宽、调度抖动等。
    """

    prefill_cost: float = 1.0       # 每个 prefill token 的相对成本
    decode_cost: float = 1.0        # 每个 decode token 的相对成本
    throughput: float = 1.0         # 成本单位 / 时间单位
    fixed_fee: float = 0.0          # 每个非空 step 的固定开销

    def step_duration(self, prefill_tokens: int, decode_tokens: int) -> float:
        work = self.prefill_cost * prefill_tokens + self.decode_cost * decode_tokens
        if prefill_tokens == 0 and decode_tokens == 0:
            return 0.0
        return work / self.throughput + self.fixed_fee


@dataclass(frozen=True)
class SimConfig:
    """模拟器配置。token 预算与 KV 页总数共同约束每个批次。"""

    policy: PolicyKind

    # 每步 token 工作预算（prefill 与 decode token 共享的预算池）
    token_budget: int = 8
    # 一次 prefill 分块最多处理多少 prompt token（chunk 大小上限）
    chunk_max: int = 8
    # KV 页大小（每个页保存 page_size 个 token 的 KV）
    page_size: int = 4
    # 固定大小 KV 页池中的总页数
    total_pages: int = 6

    # ---- continuous batching 防饥饿参数 ----
    # 等待/partial-prefill 请求在队列中等待超过该时间后进入受保护集合
    starvation_threshold: float = 20.0
    # 受保护 prefill 请求每步保证的最小 token 量（预算可容纳时）
    min_protected_chunk: int = 2
    # decoder 轮转：任一 decoder 最多连续/累计多少个 step 不被选中
    decode_starvation_gap: int = 4
    # 单个请求最多允许被抢占（重算）多少次
    max_preemptions: int = 100000

    cost: CostModel = field(default_factory=CostModel)

    def __post_init__(self) -> None:
        if self.token_budget < 1:
            raise ValueError("token_budget 必须 >=1")
        if self.chunk_max < 1:
            raise ValueError("chunk_max 必须 >=1")
        if self.page_size < 1:
            raise ValueError("page_size 必须 >=1")
        if self.total_pages < 1:
            raise ValueError("total_pages 必须 >=1")
        if self.min_protected_chunk < 1:
            raise ValueError("min_protected_chunk 必须 >=1")
        if self.decode_starvation_gap < 1:
            raise ValueError("decode_starvation_gap 必须 >=1")
