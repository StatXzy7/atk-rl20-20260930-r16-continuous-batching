# 离散事件自回归推理调度模拟器

比较两种批次调度策略（**不使用任何现成 serving 框架，调度器为自研实现**）：

- **FCFS**：先到先服务的*静态分批*基线——一批请求整体 `chunked prefill`
  完成后才整体 decode，批次之间严格 head-of-line，不抢占；
- **CONT**：带 `chunked prefill` 的 *continuous batching*——新请求在步边界
  即时混入，decode 与 prefill 同批执行，配合可解释的防饥饿规则与
  *抢占重算*（recompute）应对 KV 页不足。

> **预测而非实测**：本仓库所有时间/吞吐数字都是一个**线性成本模型**驱动的
> 模拟器**性能预测**，日志与报告中均标注 `simulator-prediction`。
> 它不建模 kernel launch、SM 争用、HBM 带宽、调度抖动、尾延迟等真实硬件效应，
> **不能贴上 measured / benchmark 标签**。要得到真实数字需在目标 GPU 上用
> 生产引擎实测，本模拟器只用于策略逻辑与相对趋势的可解释推演。

## 目录结构

- `sim/model.py`：请求、配置、线性成本模型；`output_len` 仅环境可见
- `sim/memory.py`：固定大小 KV 页池、页边界申请、内部碎片统计
- `sim/policies.py`：`SchedulingContext` + FCFS / CONT 两个 planner
- `sim/engine.py`：离散事件主循环、环境终止、记账、指标、事件日志、死锁/活锁检测
- `sim/scenarios.py`：微型手算用例与 6 组压力/边界场景
- `sim/report.py`、`sim/timeline.py`：对比表、JSON 全量日志、人类可读时间线
- `run_simulations.py`：一键运行，产出 `outputs/` 下日志
- `tests/`：手算事件表核验、独立页账本重放不变量、400 组随机 fuzz、
  防饥饿上界、成本模型一致性与"调度器不可见 output_len"的静态检查

## 运行

```bash
python run_simulations.py                 # 打印对比表并写 outputs/*.json/.timeline.txt
python -m unittest discover -s tests -v   # 全部测试（含 400 组随机 fuzz）
```

无第三方依赖，仅用 Python 3 标准库。

## 模型语义

### 请求与信息可见性

每个请求规格 `RequestSpec(rid, arrival, prompt_len, output_len)`：

- `output_len` 是**实际生成长度，只有环境知道**。调度器在任何决策路径上都
  读不到它（`tests/.../test_planner_source_does_not_read_output_len` 会用
  tokenize 剥离注释/docstring 后静态断言策略代码不引用 `output_len`）。
- 调度器只能观察运行态：已 prefill 长度、已生成 `generated`、持有页、队列年龄。
- **终止由环境发出**：decode 使 `generated == output_len` 时环境在步末宣布完成。

### 批次约束（两种策略一致）

- 每个 step 的 prefill token + decode token 之和不超过 `token_budget`；
- 单个 prefill 分段不超过 `chunk_max`（chunked prefill）；
- 每个被选请求每个 step 至多推进 **1 个 decode token**；
- KV 缓存是 `total_pages` 个固定大小页，每页 `page_size` 个 token 槽位，
  页为请求私有（不建模 prefix sharing）。

### 内存占用时机与页边界

- **Prefill token**：在 chunk 执行*之前*，按"写完本段后的 token 位置"
  在页边界向上取整申请页；跨页的 chunk 一次性拿到它跨越的所有目标页，
  绝不写入未申请的页。
- **Decode token**：同样在执行前申请；只有当这次 decode 越过当前最后一页
  边界时才申请 1 个新页（先申请后写入）。
- **准入（CONT 安全规则）**：一个新请求只有在空闲页能一次性容纳其整个
  prompt 时才准入；prefill **永远不抢占 decoder**（这是防活锁的关键，
  见下文"为什么 prefill 不抢 decoder"）。FCFS 准入即预留 prompt 全部页。
- **释放**：请求完成、被取消、或被抢占时，释放其*全部*私有页。
  `tests/test_invariants.py` 的独立重放器逐 step 验证零泄漏。

### 拒绝、等待与抢占重算

- prompt 自身所需页数超过 `total_pages`：到达时**明确拒绝**（`reject` 事件），
  不影响其他请求。
- 运行中页不足：可以**等待**（FCFS 只能等待），或按 CONT 声明的策略
  **抢占重算**：
  - 抢占 decoder：丢弃其全部 KV，之后从 prompt 头开始重算，并*重放*
    （replay）此前已生成位置；
  - 抢占 partial prefill：丢弃已算段，从头再来（若它本身在重算，
    重放目标长度保持为 prompt+原 generated，不能因二次抢占而缩水——
    这是 fuzz 抓到并修复的一个真实 bug）；
  - **重算的处理时间和工作量全额计费**：事件中区分 fresh 与 replay token，
    汇总里单独报告 `total_recompute_work`，吞吐按全部工作计算。
  - 每请求抢占次数受 `max_preemptions` 上限约束；超过后不再驱逐。

### 成本模型（比较前固定，对两策略完全相同）

```
step_duration = (cp*prefill_tokens + cd*decode_tokens)/throughput + fixed_fee
```

线性、加性、无抢占式硬件效应。`SimConfig.cost` 在构造场景时确定，
`run_both` 用同一份配置派生两种策略，测试会断言二者 `cost_model` 完全相同。

### 时间语义（离散事件）

- step 是原子的：在步起点 `[t]` 处理完所有 `at<=t` 的到达/取消/抢占后调度，
  步内动作在 `[t, t+dur]` 执行，token 在步末 `[t+dur]` 落盘并由环境判完成；
- 到达/取消只在步边界生效，**绝不打断正在执行的 step**；
- 没有可执行动作但有未来事件：`idle_advance` 跳到下一事件时刻；
- 系统内仍有请求却无未来事件且无法推进：判定**死锁**并在结果中明确上报
  （不静默挂起）；连续 64 步只有抢占没有 decode 进展判定**活锁**。

## 手算事件表核验（微型用例）

配置：`page_size=4, total_pages=6, token_budget=8, chunk_max=8, cp=cd=1`；
R1 = (到达 0, prompt 5, 实际生成 3)，R2 = (到达 1, prompt 3, 实际生成 2)。
下列每一行都由 `tests/test_micro_handcheck.py` 逐字段断言。

### FCFS（静态分批）

R1 在 t=0 单独成批（R2 t=1 才到，且批次不中途加人）。

| step | 区间 | 动作 | P | D | 步末持有页(释放前→释放后) |
|---|---|---|---|---|---|
| 0 | 0→5 | R1 prefill 5 | 5 | 0 | 2→2 |
| 1 | 5→6 | R1 decode #1（TTFT=6） | 0 | 1 | 2→2 |
| 2 | 6→7 | R1 decode #2 | 0 | 1 | 2→2 |
| 3 | 7→8 | R1 decode #3，完成，释放 2 页 | 0 | 1 | 2→0 |
| 4 | 8→11 | R2 prefill 3 | 3 | 0 | 1→1 |
| 5 | 11→12 | R2 decode #1（TTFT=11） | 0 | 1 | 1→1 |
| 6 | 12→13 | R2 decode #2 跨到第 2 页，完成 | 0 | 1 | 2→0 |

请求指标：R1 TTFT 6、完成延迟 8、ITL [1,1]；R2 排队延迟 7（8−1）、
TTFT 11、完成延迟 12、ITL [1]。makespan=13。

### CONT（chunked prefill 即时混入）

R2 在 t=5（R1 prefill 结束、开始 decode 的那个步边界）准入。

| step | 区间 | 动作 | P | D | 持有页(释放前→后) |
|---|---|---|---|---|---|
| 0 | 0→5 | R1 prefill 5 | 5 | 0 | 2→2 |
| 1 | 5→9 | R1 decode#1；R2 prefill 3 | 3 | 1 | 3→3 |
| 2 | 9→11 | R2 decode#1；R1 decode#2 | 0 | 2 | 3→3 |
| 3 | 11→13 | R1 decode#3；R2 decode#2（跨页），同完成 | 0 | 2 | 4→0 |

请求指标：R1 TTFT 9、完成延迟 13、ITL [2,2]；R2 TTFT 10、完成延迟 12、
ITL [2]。makespan=13。

**手算结论（仅此成本模型）**：makespan 相同（13），总 token 工作量相同
（R1 8 + R2 5 = 13），差异在分布：FCFS 让 R1 尽快拿到 token（TTFT 6），
但 R2 因 head-of-line 等到 t=8 才准入（排队 7）；CONT 让 R2 提前准入
（TTFT 10 对 11），代价是与 R1 共享 decode 预算使两者 ITL 从 1 变为 2。
这说明 **continuous batching 不是无条件更快**——它的价值在于高并发、
decode 占主导、且请求到达错开时摊薄每步固定成本/提高吞吐，而非降低单请求延迟。

## CONT 的防饥饿规则（显式、可解释）

设年龄 `age = now - prefill_enqueue_time`（抢占后重置为重算入队时间）。

1. **Prefill 年龄保护**：`age >= starvation_threshold` 的 prefill/等待请求
   进入*受保护集合*。每个 step 的 Pass A 先服务最老的受保护请求，保证至少
   `min_protected_chunk` 个 prefill token（当该步剩余预算能容纳时），且它
   不会被其他 prefill 驱逐。
2. **Decode 轮转**：decoder 维护 `decode_gap`（连续未被选中的 step 数），
   Pass B 按 gap 降序选择；预算允许时每个 step 都尝试跑全部 decoder 一轮，
   从而把相邻 token 的 step 间隔压在 `decode_starvation_gap`（D_gap）内。
3. **最小代价抢占**：只有*饥饿 decoder 跨页*或受保护 prefill 拿不到页时才
   驱逐；受害者按"已投入 token 最少（重算成本最低）"选择；抢占次数有
   每请求上限。

### 防饥饿上界与**适用条件**

记单步 token 预算为 B，`c = min_protected_chunk`，`D_gap = decode_starvation_gap`。
在以下条件同时成立时：

- (C1) 每个受保护请求的最小 chunk `c <= B`；
- (C2) 每个 decoder 一步只需 1 token，且受保护 prefill 每步保底占用
  不超过 `B - k`（k 为需要保证的 decoder 数），即预算可同时容纳
  保底 prefill 与至少一个 decoder；
- (C3) 所需页可以从空闲页或可抢占受害者满足（总页池能容纳目标请求，
  且未触及每请求抢占上限）；

有：

- **Prefill 进展**：进入受保护集合后，每个 step 至少前进 c 个 token，
  剩余 prefill 长度 L 最多再用 `ceil(L/c)` 个 step 完成（条件 C1/C3）。
- **Decode 间隔**：任一 decoder 相邻 decode token 之间的 step 数
  `gap <= D_gap`（条件 C2/C3）；对应时间上界
  `ITL <= D_gap · Δmax + 重算停顿`，其中 Δmax 是单步最大时长。
- 当 C2 不成立（prefill 挤占全部预算）或 C3 不成立（物理上放不下、
  或抢占上限已到）时，上界**不再保证**；模拟器不会假装满足，而是
  如实表现为更大的间隔、等待，或在极端配置下报告死锁/活锁。
  `tests/test_fuzz_and_fairness.py` 在"页充足"和"prefill 压力"两个
  受控场景下数值核验该 gap 上界，并在 `page_starvation` 场景展示
  FCFS 因不满足 C3 而死锁、CONT 以抢占重算打破死锁的对照。

### 为什么 prefill 不允许抢占 decoder

早期版本允许受保护 prefill 驱逐 decoder，fuzz（突发 12 请求）暴露了
**活锁**：decoder 被驱逐→重算→还没跑完又被更老的 prefill 驱逐→……
单场景产生数百万次抢占、零 decode 完成。根因是 prefill 可无限制造对
decoder 页的争夺。最终规则改为：**prefill 只用空闲页或放弃更年轻的
partial prefill，绝不驱逐 decoder**；decoder 才允许在跨页饥饿时抢占。
这样每个准入的请求要么用空闲容量推进，要么显式等待/可计入地重算，
不会出现"只有驱逐没有进展"的循环（连续 64 步无 decode 进展会被判活锁）。

## 指标定义

每个请求报告：

- **TTFT**（首 token 延迟）= 第一个 decode token 产出时刻 − arrival；
- **完成延迟** = completion_time − arrival；
- **decode token 间隔（ITL）** = 相邻 token 产出时刻之差；被抢占重算造成
  的停顿自然体现在间隔里（重算期间无 token，下一个间隔变大），
  另以 `preemptions` 与 `recompute_tokens` 单列；
- 特殊请求：
  - `output_len = 0`（只有 prefill，无生成）：TTFT/ITL 不适用（null），
    完成延迟定义为 prefill 完成步的结束时刻；
  - `output_len = 1`：只有一个 token，ITL 序列为空，TTFT 即首 token
    （也是完成）时刻，完成延迟与之相等。

整体报告：makespan、完成/拒绝/取消数、**goodput**（已完成请求 decode
token / makespan）、总 token 吞吐（含重算）、**峰值页数**、
**内部碎片峰值/终值**（所有页私有，故只统计每请求最后一页未写满槽位；
无外部碎片）、fresh prefill / replay / decode 工作量、抢占次数、死锁。

## 压力场景结论（模拟预测）

来自 `python run_simulations.py`（`outputs/_summary.json`）：

- **long_prefill_mix**：长 prompt(48) 穿插短请求。FCFS 严格分批，短请求
  排到长请求 decode 完之后（S1 TTFT≈94）；CONT 让短请求早早开始
  （S1 TTFT≈38），但在该线性成本模型下长请求自身 TTFT 变大，且发生
  1 次代价 4 token 的重算。
- **burst_arrival**：12 请求同刻到达。两策略都能完成；CONT 因即时混入
  产生少量抢占重算（4 次、28 replay token），FCFS 无重算但严格排队。
- **page_starvation**：总页 10 而两个 decoder 完成时各需 6 页。FCFS
  在 t=40 两 decoder 同时跨页等待、无空闲页、不能抢占 → **明确报告死锁**
  （C/D 永远等待）；CONT 用 5 次抢占（共 51 replay token）打破死锁，
  4 个请求全部完成（makespan 131，重算代价已计入）。这正是"等待 vs
  抢占重算"的核心权衡：用可计费的重算工作量换取活性。
- **cancellation_mix**：等待中与 decode 中取消均正确释放全部私有页，
  未知请求的取消被忽略；CONT makespan 49 对 FCFS 53。
- **oversized_prompt**：prompt 超过总页数在到达时被明确拒绝，且不影响
  其他正常请求。

## 工程不变量（测试如何保证可信）

- 独立 `LedgerReplayer` 不信任引擎内部账本，仅依据事件/step 流重放页池：
  时间单调、预算不超支、chunk 不越限、写入前必有足够页、
  完成/取消/抢占后私有页全部释放、终态零泄漏；
- 400 组随机（到达/长度/页数/预算/取消/抢占上限）fuzz 同时跑两种策略；
- 抢占浪费 token 与重算工作量对账一致；
- 成本模型对两策略逐位相同，且每个 step 时长可由 token 数精确反推。
