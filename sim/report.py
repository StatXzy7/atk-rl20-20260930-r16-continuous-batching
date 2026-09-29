"""结果呈现：控制台对比、完整事件日志（JSON）、step 时间线。

注意：所有输出均标注为*模拟器预测*，不标注为硬件实测。
"""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Dict, List

from .engine import SimulationResult
from .model import PolicyKind
from .scenarios import Scenario
from .engine import simulate


def run_both(scn: Scenario) -> Dict[str, SimulationResult]:
    return {
        PolicyKind.FCFS.value: simulate(scn.with_policy(PolicyKind.FCFS).cfg,
                                        scn.specs, scn.cancellations),
        PolicyKind.CONT.value: simulate(scn.with_policy(PolicyKind.CONT).cfg,
                                        scn.specs, scn.cancellations),
    }


def result_to_dict(res: SimulationResult) -> Dict:
    return {
        "kind": "simulator-prediction",
        "disclaimer": (
            "本文件全部数字为线性成本模型驱动的模拟器性能预测，"
            "不代表任何真实硬件测量（非 measured / benchmark 结果）。"
        ),
        "policy": res.policy,
        "config": res.cfg,
        "makespan": res.makespan,
        "aggregate": {
            "completed": res.num_completed,
            "rejected": res.num_rejected,
            "cancelled": res.num_cancelled,
            "total_emitted_decode_tokens": res.total_emitted_tokens,
            "completed_decode_tokens": res.completed_emitted_tokens,
            "goodput_decode_tokens_per_time": res.goodput,
            "all_token_throughput_per_time": res.token_throughput,
            "peak_pages": res.peak_pages,
            "final_used_pages": res.final_used_pages,
            "final_internal_fragmentation_tokens":
                res.final_internal_fragmentation,
            "peak_internal_fragmentation_tokens":
                res.peak_internal_fragmentation,
            "total_fresh_prefill_work": res.total_prefill_work,
            "total_recompute_work": res.total_replay_work,
            "total_decode_work": res.total_decode_work,
            "total_preemptions": res.total_preemptions,
            "deadlock": res.deadlock,
        },
        "requests": [asdict(m) for m in res.requests],
        "steps": [asdict(st) for st in res.steps],
        "events": res.events,
    }


def write_full_log(res: SimulationResult, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result_to_dict(res), f, ensure_ascii=False, indent=2)


def compare_text(name: str, results: Dict[str, SimulationResult]) -> str:
    lines = [f"=== 场景: {name} ===",
             "（下列数字均为模拟器预测，非硬件实测）"]
    header = (f"{'指标':<34}{'FCFS':>16}{'CONT':>16}")
    lines.append(header)
    a = results[PolicyKind.FCFS.value]
    b = results[PolicyKind.CONT.value]

    def row(label, va, vb, fmt="{:.3f}"):
        fa = fmt.format(va) if isinstance(va, float) else str(va)
        fb = fmt.format(vb) if isinstance(vb, float) else str(vb)
        lines.append(f"{label:<34}{fa:>16}{fb:>16}")

    row("makespan 完成时刻", a.makespan, b.makespan)
    row("完成请求数", a.num_completed, b.num_completed, "{}")
    row("拒绝数", a.num_rejected, b.num_rejected, "{}")
    row("取消数", a.num_cancelled, b.num_cancelled, "{}")
    row("goodput(完成 decode/t)", a.goodput, b.goodput)
    row("总 token 吞吐/t", a.token_throughput, b.token_throughput)
    row("峰值页数", a.peak_pages, b.peak_pages, "{}")
    row("峰值内部碎片(槽位)", a.peak_internal_fragmentation,
        b.peak_internal_fragmentation, "{}")
    row("重算 token 工作量", a.total_replay_work, b.total_replay_work, "{}")
    row("抢占次数", a.total_preemptions, b.total_preemptions, "{}")
    row("死锁", a.deadlock or "-", b.deadlock or "-", "{}")

    # 每请求对比
    lines.append("")
    lines.append(f"{'请求':<8}{'策略':<6}{'状态':<10}{'TTFT':>9}"
                 f"{'完成延迟':>10}{'meanITL':>10}{'maxITL':>9}"
                 f"{'抢占':>5}{'重算tok':>9}")
    for policy, res in ((PolicyKind.FCFS.value, a),
                        (PolicyKind.CONT.value, b)):
        for m in res.requests:
            lines.append(
                f"{m.rid:<8}{policy:<6}{m.status:<10}"
                f"{(f'{m.ttft:.2f}' if m.ttft is not None else '-'):>9}"
                f"{(f'{m.completion_latency:.2f}' if m.completion_latency is not None else '-'):>10}"
                f"{(f'{m.mean_itl:.2f}' if m.mean_itl is not None else '-'):>10}"
                f"{(f'{m.max_itl:.2f}' if m.max_itl is not None else '-'):>9}"
                f"{m.preemptions:>5}{m.recompute_tokens:>9}")
    return "\n".join(lines)
