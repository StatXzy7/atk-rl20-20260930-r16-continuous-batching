"""Per-request and aggregate metrics. All values are *simulator predictions*
derived from the configured CostModel -- they are never hardware measurements.

Metric definitions
------------------
* ``prompt_latency``: arrival -> end of the step that finishes the prompt.
* ``ttft`` (time to first token): arrival -> end of the first decode step.
    - gen_len == 0 -> ``ttft = null`` (note "no_generation"); completion
      happens at end of prefill.
    - gen_len == 1 -> ttft == completion latency; ``itls`` is empty (a single
      token has no interval).
* ``completion_latency``: arrival -> completion / cancellation / rejection.
* ``itls``: decode_times[i] - decode_times[i-1] for i >= 1 (gen_len-1 values).
* ``decode_throughput``: generated tokens / (last_decode_time - first_decode)
  for gen_len >= 2; null otherwise.
* Rejected requests: null ttft/itls, explicit flag and reason.
* Cancelled requests: metrics accumulated up to the cancellation instant.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional

from .engine import SimulationResult
from .model import Status


@dataclass
class RequestMetrics:
    rid: str
    status: str
    arrival: float
    prompt_len: int
    actual_gen_len: Optional[int]
    gen_done: int
    admitted_at: Optional[float]
    prompt_latency: Optional[float]
    ttft: Optional[float]
    completion_latency: Optional[float]
    itls: list[float]
    itl_mean: Optional[float]
    itl_max: Optional[float]
    decode_throughput: Optional[float]
    preemptions: int
    recompute_tokens: int
    pages_held_at_end: int
    ttft_note: Optional[str] = None
    reject_reason: Optional[str] = None


def request_metrics(result: SimulationResult) -> list[RequestMetrics]:
    rows: list[RequestMetrics] = []
    for rid in sorted(result.requests, key=lambda k: (result.requests[k].arrival, k)):
        req = result.requests[rid]
        spec = result.specs.get(rid)
        actual = spec.gen_len if spec else None
        admitted = req.admitted_at

        if req.status == Status.REJECTED:
            rows.append(
                RequestMetrics(
                    rid=rid,
                    status=req.status.value,
                    arrival=req.arrival,
                    prompt_len=req.prompt_len,
                    actual_gen_len=actual,
                    gen_done=0,
                    admitted_at=None,
                    prompt_latency=None,
                    ttft=None,
                    completion_latency=_sub(req.completed_at, req.arrival),
                    itls=[],
                    itl_mean=None,
                    itl_max=None,
                    decode_throughput=None,
                    preemptions=0,
                    recompute_tokens=0,
                    pages_held_at_end=req.pages,
                    ttft_note="rejected",
                    reject_reason=req.reject_reason,
                )
            )
            continue

        prompt_lat = None
        if req.prompt_done == req.prompt_len and req.prompt_len > 0:
            finish_t = _prompt_finish_time(result, req)
            if finish_t is None and req.status == Status.COMPLETED and req.gen_done == 0:
                finish_t = req.completed_at
            prompt_lat = _sub(finish_t, req.arrival)

        ttft = _sub(req.first_token_at, req.arrival)
        ttft_note = None
        if ttft is None:
            if actual == 0 and req.status == Status.COMPLETED:
                ttft_note = "no_generation"
            elif req.status == Status.CANCELLED and req.gen_done == 0:
                ttft_note = "cancelled_before_first_token"
            else:
                ttft_note = "no_token"

        comp = _sub(req.completed_at, req.arrival)
        itls = [
            round(req.decode_times[i] - req.decode_times[i - 1], 9)
            for i in range(1, len(req.decode_times))
        ]
        itl_mean = round(sum(itls) / len(itls), 6) if itls else None
        itl_max = round(max(itls), 6) if itls else None
        dec_tp = None
        if len(req.decode_times) >= 2:
            span = req.decode_times[-1] - req.decode_times[0]
            dec_tp = round((len(req.decode_times) - 1) / span, 6) if span > 0 else None

        rows.append(
            RequestMetrics(
                rid=rid,
                status=req.status.value,
                arrival=req.arrival,
                prompt_len=req.prompt_len,
                actual_gen_len=actual,
                gen_done=req.gen_done,
                admitted_at=admitted,
                prompt_latency=prompt_lat,
                ttft=ttft,
                completion_latency=comp,
                itls=itls,
                itl_mean=itl_mean,
                itl_max=itl_max,
                decode_throughput=dec_tp,
                preemptions=req.preemptions,
                recompute_tokens=req.recompute_tokens,
                pages_held_at_end=req.pages,
                ttft_note=ttft_note,
            )
        )
    return rows


def _sub(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None:
        return None
    return round(a - b, 6)


def _prompt_finish_time(result: SimulationResult, req) -> Optional[float]:
    for st in result.steps:
        if req.rid in st.get("finished_prefill", []):
            return st["t_end"]
    return None


def aggregate(result: SimulationResult) -> dict:
    rows = request_metrics(result)
    completed = [r for r in rows if r.status == Status.COMPLETED.value]
    cancelled = [r for r in rows if r.status == Status.CANCELLED.value]
    rejected = [r for r in rows if r.status == Status.REJECTED.value]

    total_decoded = sum(r.gen_done for r in completed)
    total_prefill = sum(r.prompt_len for r in completed) + sum(
        r.recompute_tokens for r in rows
    )
    makespan = result.makespan

    busy = sum(s["duration"] for s in result.steps)
    page_capacity_time = result.config.num_pages * makespan if makespan > 0 else 0.0
    used_page_time = sum(
        s["used_pages"] * s["duration"] for s in result.steps
    )

    def avg(vals):
        vals = [v for v in vals if v is not None]
        return round(sum(vals) / len(vals), 6) if vals else None

    # Time-weighted internal fragmentation: average wasted token slots while
    # pages are held. External fragmentation is 0 by paging design: capacity is
    # allocated in pages and any free page can serve any request, so the only
    # waste is intra-page slack (reported here).
    tw_frag_time = sum(
        s["internal_frag_tokens"] * s["duration"] for s in result.steps
    )
    mean_frag = round(tw_frag_time / busy, 6) if busy > 0 else 0.0
    peak_frag = result.alloc.peak_frag_tokens

    return {
        "label": "simulated_prediction",
        "not_hardware_measurement": True,
        "policy": result.policy,
        "requests": len(rows),
        "completed": len(completed),
        "cancelled": len(cancelled),
        "rejected": len(rejected),
        "makespan": round(makespan, 6),
        "busy_time": round(busy, 6),
        "idle_time": round(makespan - busy, 6),
        "steps": len(result.steps),
        "request_throughput": round(len(completed) / makespan, 6) if makespan else None,
        "decode_token_throughput": round(total_decoded / makespan, 6) if makespan else None,
        "total_decode_tokens": total_decoded,
        "total_prefill_tokens_charged": total_prefill,
        "total_recompute_tokens": sum(r.recompute_tokens for r in rows),
        "mean_ttft": avg([r.ttft for r in completed]),
        "mean_completion_latency": avg([r.completion_latency for r in completed]),
        "mean_itl": avg([r.itl_mean for r in completed]),
        "peak_pages": result.alloc.peak_pages,
        "page_pool_utilization": round(used_page_time / page_capacity_time, 6)
        if page_capacity_time
        else None,
        "mean_internal_frag_tokens": mean_frag,
        "peak_internal_frag_tokens": peak_frag,
        "external_fragmentation": 0,
        "external_frag_note": "0 by design: free pages are interchangeable; "
        "only intra-page slack wastes capacity.",
    }


def report(result: SimulationResult) -> dict:
    return {
        "summary": aggregate(result),
        "requests": [asdict(r) for r in request_metrics(result)],
        "steps": result.steps,
    }
