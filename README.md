# 离散事件自回归推理调度模拟器

比较 **FCFS + 静态批处理** 与 **带 chunked prefill 的 continuous batching**（含
可配置的抢占重算 / 等待策略）。纯 Python 标准库实现，不调用任何现成 serving
框架；调度器是 `sim/scheduler.py` 中独立实现的离散事件策略。

> 所有时间/吞吐数字都是由固定 `CostModel` 推出的**模拟器预测**，事件日志用
> `"kind": "simulated"` 标注，**不是硬件实测值**。

## 运行

```bash
python3 -m pytest -q          # 全部测试（含手算事件表逐字段断言）
python3 scenarios.py          # 生成 out/*.json 报告与 out/*.jsonl 完整事件日志
```

## 代码结构

- `sim/model.py`：请求/请求规格、成本模型、配置、页需求与内存占用契约。
- `sim/allocator.py`：固定页池（边界申请、峰值/碎片统计）、结构化事件日志。
- `sim/scheduler.py`：`FCFSScheduler` 与 `ContinuousBatching`（recompute / wait）。
- `sim/engine.py`：离散事件循环；持有真实 gen_len（环境私有），在到达边界发终止。
- `sim/metrics.py`：每请求 TTFT / 完成延迟 / decode 间隔与聚合吞吐、峰值页、碎片。
- `sim/run.py`：运行/对比入口。
- `tests/`：手算事件表、特殊请求、拒绝、取消、抢占重算、压力序列、不变量、
  防饥饿上界条件、以及“调度器无法读取真实 gen_len”的结构性守卫。
- `scenarios.py` / `ANALYSIS.md`：场景产物与策略分析。

## 关键设计

- 调度器只见 `arrival/prompt_len/prompt_done/gen_done/pages`，**拿不到**真实
  `gen_len`；终止由环境在生成到实际长度时发出。
- 页在 token 计算**之前**于页边界申请；完成/取消释放全部私有页（零泄漏）。
- prompt 装不下池 → 准入即拒绝；运行中序列超池 → 环境在边界显式拒绝。
- 防饥饿：最老等待者每步有 `max_chunk` 保障切片；抢占仅限更年轻请求，重算全额
  计费。上界的成立条件与失效情形见 `ANALYSIS.md`。
