"""Page allocator, preemption support and structured event logging."""
from __future__ import annotations

from dataclasses import dataclass, field

from .model import Request, Status, pages_for_tokens


class OutOfPages(Exception):
    """Raised when an acquisition cannot be satisfied even after preemption."""


@dataclass
class Allocator:
    """Fixed pool of private KV pages. Tracks peak usage for reporting.

    All acquisitions reserve pages *before* the occupying tokens are produced,
    so the pool limit can never be exceeded.
    """

    num_pages: int
    page_size: int
    free_pages: int = field(init=False)
    used_pages: int = field(init=False, default=0)
    peak_pages: int = field(init=False, default=0)
    # time-weighted peak of *internal* fragmentation (tokens), sampled each step
    peak_frag_tokens: int = 0

    def __post_init__(self) -> None:
        self.free_pages = self.num_pages

    def _lend(self, req: Request, pages: int) -> None:
        if pages > self.free_pages:
            raise OutOfPages(
                f"need {pages} pages for {req.rid}, only {self.free_pages} free"
            )
        self.free_pages -= pages
        self.used_pages += pages
        req.pages += pages
        self.peak_pages = max(self.peak_pages, self.used_pages)

    def acquire_prefill(self, req: Request, chunk: int) -> int:
        """Reserve pages for a prefill chunk of ``chunk`` new prompt tokens.

        Works both for fresh admission (req.pages==0) and recompute after
        preemption (prompt_done reset to 0, pages==0). The running page count
        req.pages is the authoritative baseline; the target count is derived
        from prompt_done+chunk.
        """
        end = req.prompt_done + chunk
        target = pages_for_tokens(end, self.page_size)
        need = target - req.pages
        if need < 0:
            raise AssertionError(
                f"{req.rid}: prefill would shrink pages {req.pages}->{target}"
            )
        if need > 0:
            self._lend(req, need)
        return need

    def acquire_decode(self, req: Request) -> int:
        """Reserve the (at most one) page needed for one more decode token."""
        target_tokens = req.kv_tokens + 1
        need = pages_for_tokens(target_tokens, self.page_size) - req.pages
        if need > 0:
            self._lend(req, need)
        return need

    def release(self, req: Request) -> int:
        """Release all private pages (completion or cancellation)."""
        n = req.pages
        if n < 0:
            raise AssertionError(f"{req.rid}: negative page count")
        self.free_pages += n
        self.used_pages -= n
        if self.used_pages < 0:
            raise AssertionError("allocator used_pages negative")
        req.pages = 0
        return n

    def internal_frag_tokens(self, active: list[Request]) -> int:
        total = 0
        for req in active:
            if req.status in (Status.PREFILL, Status.DECODE):
                total += req.pages * self.page_size - req.kv_tokens
        return total

    def sample_frag(self, active: list[Request]) -> None:
        self.peak_frag_tokens = max(
            self.peak_frag_tokens, self.internal_frag_tokens(active)
        )


@dataclass
class EventLogger:
    """In-memory list of dict records; optionally mirrored to a JSONL file.

    Every record is a *simulator prediction* and is tagged as such via
    ``kind == "simulated"`` in serialized output (see :meth:`write_jsonl`).
    """

    events: list[dict] = field(default_factory=list)
    _seq: int = 0

    def emit(self, event: str, time: float, **fields) -> None:
        rec = {"seq": self._seq, "event": event, "time": round(float(time), 6)}
        rec.update(fields)
        self.events.append(rec)
        self._seq += 1

    def write_jsonl(self, path: str, policy: str) -> None:
        import json

        with open(path, "w") as fh:
            for rec in self.events:
                out = {"kind": "simulated", "policy": policy, **rec}
                fh.write(json.dumps(out, ensure_ascii=False) + "\n")
