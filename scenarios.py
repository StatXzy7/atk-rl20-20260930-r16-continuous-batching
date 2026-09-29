"""Run representative scenarios and emit reports + full JSONL event logs.

All outputs are SIMULATOR PREDICTIONS derived from a fixed CostModel; they are
not real-hardware measurements.

Usage:
    python3 scenarios.py
Writes:
    out/micro_compare.json          (hand-verifiable micro workload)
    out/stress_compare.json         (long prefill + burst + scarcity + cancels)
    out/<policy>_<scenario>.jsonl   (complete ordered event logs)
"""
from __future__ import annotations

import json
import os

from sim.metrics import report
from sim.model import CostModel, RequestSpec, SimConfig
from sim.run import run_once

OUT = os.path.join(os.path.dirname(__file__), "out")
os.makedirs(OUT, exist_ok=True)


def emit(scenario: str, workload, cfg: SimConfig, policies: dict):
    bundle = {"scenario": scenario, "config": _config_dict(cfg), "policies": {}}
    for name, kwargs in policies.items():
        result = run_once(workload, cfg, "cb", **kwargs) if name.startswith("cb") \
            else run_once(workload, cfg, name)
        pol_label = {
            "fcfs": "fcfs",
            "cb_recompute": "continuous_batching+recompute",
            "cb_wait": "continuous_batching+wait",
        }[name]
        rep = report(result)
        rep["summary"]["policy"] = pol_label
        bundle["policies"][name] = rep
        path = os.path.join(OUT, f"{scenario}_{name}.jsonl")
        result.logger.write_jsonl(path, pol_label)
        print(
            f"[{scenario}] {name:14s} makespan={result.makespan:8.2f} "
            f"peak_pages={result.alloc.peak_pages:3d} "
            f"recompute={sum(q.recompute_tokens for q in result.requests.values()):4d} "
            f"-> {os.path.relpath(path)}"
        )
    with open(os.path.join(OUT, f"{scenario}_compare.json"), "w") as fh:
        json.dump(bundle, fh, indent=2, ensure_ascii=False)


def _config_dict(cfg: SimConfig) -> dict:
    return {
        "page_size": cfg.page_size,
        "num_pages": cfg.num_pages,
        "token_budget": cfg.token_budget,
        "max_chunk": cfg.max_chunk,
        "cost_model": {
            "prefill_time_per_token": cfg.cost.prefill_time_per_token,
            "decode_time_per_token": cfg.cost.decode_time_per_token,
            "launch_overhead": cfg.cost.launch_overhead,
            "recompute_factor": cfg.cost.recompute_factor,
        },
        "fixed_before_comparison_and_identical_for_all_policies": True,
        "units": "simulated time units from CostModel (NOT hardware timings)",
    }


def micro():
    cfg = SimConfig(
        page_size=4, num_pages=8, token_budget=4, max_chunk=2,
        cost=CostModel(prefill_time_per_token=1, decode_time_per_token=1),
    )
    wl = [RequestSpec("A", 0, 3, 2), RequestSpec("B", 0, 2, 3)]
    emit("micro", wl, cfg, {"fcfs": {}, "cb_recompute": {"preemption": "recompute"}})


def stress():
    cfg = SimConfig(
        page_size=4, num_pages=12, token_budget=8, max_chunk=4,
        cost=CostModel(prefill_time_per_token=1, decode_time_per_token=3),
    )
    wl = [RequestSpec("Long", 0, 40, 6)]
    # burst of short requests during the long prefill
    for i in range(12):
        wl.append(RequestSpec(f"S{i}", round(1 + 0.1 * i, 2), 2 + i % 3, 1 + i % 4))
    # page-pressure pair (prompt needs most of the pool)
    wl.append(RequestSpec("Huge1", 8, 30, 2))
    # cancellations at different phases
    object.__setattr__(next(s for s in wl if s.rid == "S3"), "cancel_at", 12)
    object.__setattr__(next(s for s in wl if s.rid == "S9"), "cancel_at", 60)
    # an impossible-at-runtime long generator is intentionally NOT included;
    # covered by tests. Keep the stress workload within physical capacity.
    emit(
        "stress",
        wl,
        cfg,
        {
            "fcfs": {},
            "cb_recompute": {"preemption": "recompute"},
            "cb_wait": {"preemption": "wait"},
        },
    )


if __name__ == "__main__":
    micro()
    stress()
    print("\nAll numbers are simulator predictions, not hardware measurements.")
