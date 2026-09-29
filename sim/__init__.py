"""离散事件自回归推理调度模拟器。

比较两种调度策略：
  * FCFS：静态分批（一个批次整体 prefill 后整体 decode）；
  * CONT：带 chunked prefill 的 continuous batching（新请求即时混入，
    配合可解释的防饥饿规则与抢占重算）。

本包给出的数字全部是成本模型驱动的*性能预测*，不是真实硬件测量。
"""

from .model import (
    Request,
    RequestSpec,
    SimConfig,
    CostModel,
    PolicyKind,
    Status,
    Cancellation,
)
from .memory import PagePool
from .engine import SimulationResult, simulate

__all__ = [
    "Request",
    "RequestSpec",
    "SimConfig",
    "CostModel",
    "PolicyKind",
    "Status",
    "Cancellation",
    "PagePool",
    "SimulationResult",
    "simulate",
]
