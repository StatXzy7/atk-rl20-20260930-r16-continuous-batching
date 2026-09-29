# 策略分析：FCFS/静态批处理 vs 带 chunked prefill 的 continuous batching

> **重要：本文档与 `out/` 中所有数字都是“模拟器预测”**，它们由一个固定的
> `CostModel`（每 token 的 prefill/decode 时间、可选启动开销、重算倍率）经离散
> 事件推演得到，**不是真实硬件测量值**，也不应被引用为实测性能。真实性能需要在
> 具体 GPU/框架上 benchmark。本模拟器不依赖任何现成 serving 框架，调度器为
> `sim/scheduler.py` 中的独立实现。

## 1. 模型与记账

- 时间是离散步进（step）推进的*模拟量*。每个 step：
  `duration = launch_overhead + t_prefill·fresh_prefill_tokens
            + t_prefill·recompute_factor·recompute_tokens
            + t_decode·decode_tokens`。
- 每 step 总 token 工作受 `token_budget = B` 约束；prefill 可分块，单请求单步
  prefill 上限 `max_chunk = q (q ≤ B)`；每个被选 decoder 单步至多推进 1 token。
- KV 内存是固定大小页池（`page_size` token/页，共 `num_pages` 页）。**页在产生
  占用它的 token 之前申请**：
  - prefill 块把已驻留 prompt 从 `d` 推到 `d+c`，页需求
    `ceil((d+c)/page_size)`，恰在跨页边界时申请新页；
  - decode 生成第 `k` 个 token 前，需持有 `ceil(k/page_size)` 页。
- 完成或取消时释放该请求**全部**私有页（无共享前缀），因此页泄漏可被测试直接
  捕获（终态 `used_pages == 0`、每个请求 `pages == 0`）。
- 内部碎片 = 已申请页内未用 token；分页设计下**外部碎片恒为 0**（任何空闲页可
  服务任何请求，浪费只来自页内 slack）。

成本模型在比较两种策略**之前固定**，并对两策略使用同一份参数（见
`scenarios.py` 与 `test_both_use_identical_cost_and_budget`）。

## 2. 两种策略

### FCFS + 静态批处理（`FCFSScheduler`）
- 严格按到达序成批：队首 prompt 超过 `B` 时独占并分块；否则取“一个 prefill
  step 能装下且完整 prompt 页同时可容纳”的最长到达前缀。
- 整批进入 decode 后必须一起排空，期间**不接纳新请求**（经典 head-of-line
  blocking）；无抢占，成批时一次性预留全部 prompt 页，因此后续 decode 跨页永远
  有页可用。

### Continuous batching + chunked prefill（`ContinuousBatching`）
- 每个 step 独立规划（滑动窗口），新请求可在任意 step 准入，完成即释放。
- 三段式规划：
  1. **队首保障**：最老等待请求每步先获得至多 `q` 个 prefill token 的预算；
  2. **decoder**：在跑请求按年龄各推进至多 1 token；
  3. **剩余预算**：round-robin（最老优先）给其它 prefill，单请求≤`q`。
- 两种内存策略：
  - `preemption="recompute"`：需要页时可驱逐**严格更年轻**（稳定全序
    `(arrival, rid)`）的请求；被驱逐者释放全部私有页、`prompt_done` 归零，之后
    **完整重算 prompt**，重算 token 以 prefill 费率（×`recompute_factor`）计费。
    decoder 不驱逐同到达时刻的同批 prefill，以避免同代互踢活锁；新准入（非队首）
    只使用当前空闲页、绝不触发抢占，抢占仅服务于队首保障与在跑 decoder 增长。
  - `preemption="wait"`：**绝不驱逐**。准入是保守的——只有“完整 prompt + 1 个
    decode 增长页”能放入整个池、且当前没有任何在跑请求时才准入（串行处理）。
    这用并发换取了无驱逐下的可证明活性；若实际生成长度（调度器不可预知）最终仍
    超过物理池，环境在到达边界时将该请求显式标记
    `sequence_exceeds_pool_at_runtime` 拒绝，而不是无限等待。

## 3. 防饥饿规则与（条件性）上界

关键前提：调度器只能看到 `prompt_done/gen_done/pages/arrival`，**看不到真实
gen_len**，因此任何界都不能依赖“别人还要生成多久”。

### recompute 策略
- 稳定年龄序 `(arrival, rid)` 决定谁能抢占谁：年长者永远不会被年轻者驱逐。
- 当请求 R 是最老等待者（head）时：
  - 每步它最多被保证 `q` 个 prefill token 的预算（pass 1 先于 decoder）；
  - 若它在某步**没有**推进，只可能因为：
    1. 该步它自身被一个**更老**受益者抢占（其被抢占次数有限，见下），或
    2. 严格更老的在跑请求占着内存（其完成时释放，R 随后获得保障）。
- 因此 R 从“成为最老等待者”到“prompt 完成”的忙碌步数与**任何其它请求的未知
  生成长度无关**，量级为 `ceil(prompt_len_R / q)` 的有界推进，外加被抢占导致的
  重算工作。R 被抢占的次数受限于“比它年轻、且在它等待窗口内被准入”的请求集合
  （每个这样的请求至多为给 R 让路贡献有限次驱逐）。
- **适用条件/失效情形**：
  - `q ≤ B`（保证切片确实能放入预算）且 `grace = 0`；
  - prompt 本身能放入池（否则在准入时即被拒绝）；
  - 同一时刻的突发同代请求若集合需求超池，按 rid 全序确定性串行（不是并行），
    属于“有界等待但不并行”；
  - 真实序列长度超过物理池时，环境在边界显式拒绝（非饥饿，是容量不足）。

### wait 策略
- 不驱逐、串行准入保证了活性，但**不存在与他人 decode 无关的通用 TTFT 上界**：
  R 的等待包含所有更老请求 prompt 工作之和，且在罕见的“所有在跑 decoder 同一步
  跨页边界且无空页”时只能等某个 decoder 完成释放。保守准入（完整 prompt + 增长
  页 + 单活跃）就是为把这种等待收敛为“至多等当前在跑者完成”而声明的规则。
- 这是明确的权衡：wait 无重算浪费（压测中 recompute token=0）、碎片更低，但
  并发度低；recompute 并发高、短请求 TTFT 更低，但有重算时间/工作量成本。

## 4. 指标定义（特殊请求）

- `prompt_latency`：到达 → prompt 全部处理完。
- `ttft`：到达 → 第一个 decode token 完成。
  - `gen_len = 0`：**不产生 token**，`ttft = null`（`ttft_note="no_generation"`），
    在 prefill 结束即完成；
  - `gen_len = 1`：`ttft == completion_latency`，`itls` 为空（单 token 无间隔）。
- `itls[i] = decode_time[i] - decode_time[i-1]`（共 gen_len−1 个）。
- `completion_latency`：到达 → 完成/取消/拒绝；被拒绝者 ttft/itl 为 null 且带
  `reject_reason`；被取消者报告截至取消时刻的累计指标。
- 聚合：请求/token 吞吐、makespan、忙碌/空闲时间、峰值页、时间加权平均与峰值内部
  碎片、页池利用率、总重算 token。

## 5. 手算微型事件表（已在测试中逐字段断言）

工作负载 `A(0,prompt=3,gen=2)`, `B(0,prompt=2,gen=3)`；`page_size=4, pool=8,
B=4, q=2, t_prefill=t_decode=1`。完整逐步表见
`tests/test_hand_table.py`（FCFS 7 步、CB 4 步，makespan 均为 10；区别在于逐请求
TTFT/ITL 与并发度，CB 下 B 的 TTFT 8→6）。

## 6. 压力序列观察（预测值，见 `out/stress_compare.json`）

- 长 prefill + 短请求突发：CB 让部分短请求 TTFT 大幅下降（例 S0 68→21 模拟单位），
  代价是少量重算（23 prefill token）与略高 makespan/碎片；
- 取消（等待中/运行中）后全部私有页释放，终态零泄漏；
- 页不足：recompute 通过驱逐+计费重算推进，wait 串行无重算但并发低。

## 7. 何时相信这些数字
- 可用于：策略逻辑正确性、相对趋势、上界结构、内存/预算不变量、事件因果。
- 不可用于：声称真实延迟/吞吐。线性成本模型忽略了 kernel 细节、SM 占用、KV 读写
  带宽、调度开销分布等；接入真实硬件前请把它们当作上界/趋势而非实测。
