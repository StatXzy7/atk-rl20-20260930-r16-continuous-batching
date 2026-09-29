"""Core data model for the discrete-event autoregressive inference simulator.

Timing and memory units
-----------------------
* Wall-clock time is a *simulated* quantity (float), derived from the
  :class:`CostModel`. It is never a hardware measurement.
* Token work per step is split into ``prefill_tokens`` (compute-bound prompt
  work, chunkable) and ``decode_tokens`` (one per active decoder).
* KV cache memory is expressed in fixed-size pages of ``page_size`` tokens.

Memory accounting contract
--------------------------
Pages are *reserved before the tokens that occupy them are computed*:

* **Prefill chunk of size ``c``** (starting from ``d`` already-resident prompt
  tokens): the inclusive end is ``d + c``. The new high-water mark is
  ``ceil((d + c) / page_size)`` pages, so additional pages are acquired
  ``ceil(d / page_size) + 1 ... ceil((d+c)/page_size)`` -- a page is fetched at
  the *boundary*, i.e. exactly when the next token would spill into a new page.
* **Decode**: before generating token ``k`` (``k >= 1`` resident afterwards),
  the request must hold ``ceil(k / page_size)`` pages; a fresh page is acquired
  on boundary crossings.

Internal fragmentation for a request with ``r`` resident tokens and ``p``
private pages is ``p*page_size - r`` tokens of reserved-but-unused capacity.
All pages are private (no prefix sharing), so completing or cancelling a
request releases exactly its page count -- there is nothing to leak.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


def pages_for_tokens(tokens: int, page_size: int) -> int:
    """Pages required to hold ``tokens`` KV slots (0 tokens -> 0 pages)."""
    if tokens < 0:
        raise ValueError("token count must be non-negative")
    return (tokens + page_size - 1) // page_size


class Status(str, Enum):
    WAITING = "waiting"        # not admitted; no private pages
    PREFILL = "prefill"        # admitted, prompt partially processed
    DECODE = "decode"          # prompt done, generating
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    REJECTED = "rejected"

    @property
    def terminal(self) -> bool:
        return self in (Status.COMPLETED, Status.CANCELLED, Status.REJECTED)


@dataclass(frozen=True)
class RequestSpec:
    """Workload description for one request (the *environment's* truth).

    ``gen_len`` is the actual generation length. It is the environment's
    private information: schedulers never see it. Termination is issued by the
    environment once exactly ``gen_len`` decode tokens have been produced.
    ``gen_len=0`` is legal: the request completes at the end of prefill.
    """

    rid: str
    arrival: float
    prompt_len: int
    gen_len: int
    cancel_at: Optional[float] = None

    def __post_init__(self) -> None:
        if self.prompt_len <= 0:
            raise ValueError(f"{self.rid}: prompt_len must be >= 1")
        if self.gen_len < 0:
            raise ValueError(f"{self.rid}: gen_len must be >= 0")
        if self.arrival < 0:
            raise ValueError(f"{self.rid}: arrival must be >= 0")


@dataclass
class Request:
    """Mutable runtime state. Note: there is deliberately **no** gen_len
    attribute -- the scheduler cannot peek at the future."""

    rid: str
    arrival: float
    prompt_len: int
    cancel_at: Optional[float] = None
    status: Status = Status.WAITING
    # Prompt tokens already computed and resident.
    prompt_done: int = 0
    # Decode tokens already produced (0..unknown to scheduler).
    gen_done: int = 0
    pages: int = 0
    admitted_at: Optional[float] = None
    first_token_at: Optional[float] = None
    completed_at: Optional[float] = None
    # Raw decode completion timestamps (end time of each producing step).
    decode_times: list[float] = field(default_factory=list)
    preemptions: int = 0
    recompute_tokens: int = 0
    # True while re-processing a prompt after preemption.
    is_recompute: bool = False
    reject_reason: Optional[str] = None

    # -- scheduler-visible derived quantities -------------------------------
    @property
    def kv_tokens(self) -> int:
        """KV slots currently resident (prompt part + generated part)."""
        return self.prompt_done + self.gen_done

    def prompt_pages(self, page_size: int) -> int:
        """Pages required if the *entire prompt* were resident (admission)."""
        return pages_for_tokens(self.prompt_len, page_size)

    def resident_tokens(self) -> int:
        return self.kv_tokens

    def fragmentation_tokens(self, page_size: int) -> int:
        if self.status.terminal or self.pages == 0:
            return 0
        return self.pages * page_size - self.kv_tokens

    def wait_time(self, now: float) -> float:
        if self.status.terminal:
            end = self.completed_at if self.completed_at is not None else now
            return end - self.arrival
        return now - self.arrival


@dataclass
class CostModel:
    """Configurable input-processing cost, fixed *before* comparing policies
    and applied identically to both.

    step_time = launch_overhead
                + prefill_time_per_token * prefill_tokens
                + decode_time_per_token * decode_tokens
    Recomputed prefill tokens are charged at ``recompute_factor`` times the
    prefill rate (default 1.0: wasted-but-identical recompute work).
    """

    prefill_time_per_token: float = 1.0
    decode_time_per_token: float = 2.0
    launch_overhead: float = 0.0
    recompute_factor: float = 1.0

    def step_time(
        self,
        prefill_tokens: int,
        decode_tokens: int,
        recompute_tokens: int = 0,
    ) -> float:
        fresh = prefill_tokens - recompute_tokens
        if fresh < 0:
            raise ValueError("recompute tokens exceed prefill tokens")
        return (
            self.launch_overhead
            + self.prefill_time_per_token * fresh
            + self.prefill_time_per_token * self.recompute_factor * recompute_tokens
            + self.decode_time_per_token * decode_tokens
        )


@dataclass
class SimConfig:
    page_size: int = 4
    num_pages: int = 8          # total fixed-size KV pages in the pool
    token_budget: int = 4       # B: total token-work units per step
    max_chunk: int = 4          # q: max prefill tokens for one req per step
    cost: CostModel = field(default_factory=CostModel)

    def __post_init__(self) -> None:
        if self.page_size <= 0 or self.num_pages <= 0:
            raise ValueError("page_size and num_pages must be positive")
        if self.token_budget <= 0 or self.max_chunk <= 0:
            raise ValueError("token_budget and max_chunk must be positive")
        if self.max_chunk > self.token_budget:
            raise ValueError("max_chunk must be <= token_budget")

    @property
    def max_seq_pages(self) -> int:
        """Hardest admissible sequence is prompt-only (admission safety)."""
        return self.num_pages
