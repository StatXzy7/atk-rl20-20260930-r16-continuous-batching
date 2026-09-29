"""Convenience helpers to run a workload under one or both policies."""
from __future__ import annotations

import json

from .engine import Engine
from .metrics import report
from .model import RequestSpec, SimConfig
from .scheduler import ContinuousBatching, FCFSScheduler


def run_once(workload, cfg: SimConfig, policy: str, preemption="recompute",
             grace: float = 0.0) -> SimulationResult:
    if policy == "fcfs":
        sched = FCFSScheduler()
    elif policy in ("cb", "continuous_batching"):
        sched = ContinuousBatching(preemption=preemption, grace=grace)
    else:
        raise ValueError(f"unknown policy {policy}")
    return Engine(cfg, sched, list(workload)).run()


def compare(workload, cfg: SimConfig, preemption="recompute", grace: float = 0.0):
    return {
        "config": {
            "page_size": cfg.page_size,
            "num_pages": cfg.num_pages,
            "token_budget": cfg.token_budget,
            "max_chunk": cfg.max_chunk,
            "cost": {
                "prefill_time_per_token": cfg.cost.prefill_time_per_token,
                "decode_time_per_token": cfg.cost.decode_time_per_token,
                "launch_overhead": cfg.cost.launch_overhead,
                "recompute_factor": cfg.cost.recompute_factor,
            },
            "cost_fixed_before_comparison": True,
        },
        "fcfs": report(run_once(workload, cfg, "fcfs")),
        "continuous_batching": report(
            run_once(workload, cfg, "cb", preemption=preemption, grace=grace)
        ),
    }


def write_compare_json(path: str, workload, cfg: SimConfig, **kw) -> dict:
    data = compare(workload, cfg, **kw)
    with open(path, "w") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)
    return data
