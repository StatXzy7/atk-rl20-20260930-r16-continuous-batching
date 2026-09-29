#!/usr/bin/env python3
"""运行全部场景，对比 FCFS 与 continuous batching，产出完整事件日志。

用法：
    python run_simulations.py                # 打印对比表 + 写 JSON 日志
    python run_simulations.py --outdir DIR

所有输出数字均为线性成本模型驱动的*模拟器预测*，非硬件实测。
"""

from __future__ import annotations

import argparse
import json
import os

from sim.report import compare_text, result_to_dict, run_both
from sim.timeline import render_timeline
from sim.scenarios import all_scenarios


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default="outputs")
    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    summary = {"kind": "simulator-prediction", "scenarios": []}
    for scn in all_scenarios():
        results = run_both(scn)
        print(compare_text(scn.name, results))
        print()
        for policy, res in results.items():
            path = os.path.join(args.outdir,
                                f"{scn.name}__{policy}.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(result_to_dict(res), f, ensure_ascii=False,
                          indent=2)
            tpath = os.path.join(args.outdir,
                                 f"{scn.name}__{policy}.timeline.txt")
            with open(tpath, "w", encoding="utf-8") as f:
                f.write(render_timeline(res))
        summary["scenarios"].append({
            "name": scn.name,
            "description": scn.description,
            "fcfs_makespan": results["fcfs"].makespan,
            "cont_makespan": results["cont"].makespan,
            "fcfs_completed": results["fcfs"].num_completed,
            "cont_completed": results["cont"].num_completed,
            "cont_preemptions": results["cont"].total_preemptions,
            "cont_recompute_work": results["cont"].total_replay_work,
            "fcfs_deadlock": results["fcfs"].deadlock,
            "cont_deadlock": results["cont"].deadlock,
        })
    with open(os.path.join(args.outdir, "_summary.json"), "w",
              encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"完整逐事件/逐 step 日志已写入 {args.outdir}/ 目录")
    print("提醒：以上全部为模拟器预测值，不是硬件测量结果。")


if __name__ == "__main__":
    main()
