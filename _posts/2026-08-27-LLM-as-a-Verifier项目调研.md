---
layout: post
toc: true
title: "LLM-as-a-Verifier 技术调研：用评分分布、重复评估与概率锦标赛扩展智能体验证"
categories: LLM
tags: [LLM, Agent, Verifier, RL, Robotics, Evaluation]
author:
  - vortezwohl
  - 吴子豪
excerpt: "LLM-as-a-Verifier 是 Stanford、UC Berkeley 与 NVIDIA 团队在 2026 年提出并开源的通用智能体验证框架。它针对长时程 coding、机器人和医疗智能体中‘能生成不等于能判断’的瓶颈，把验证从一次性的离散打分改造成可以扩展的计算轴：读取评分 token 的完整 logprob 分布并求期望得到连续奖励；对同一候选重复评估以降低方差；把整体质量拆成可判定的标准以降低提示偏差；再用 Probabilistic Pivot Tournament 在近似 O(Nk) 的预算下从 N 条轨迹中选择最佳解。本文从背景问题、related work 差异、源码级实现、概率模型、缓存与多模态细节，到 coding/机器人/医疗/RL 的落地代码逐层拆解，并讨论它不能替代执行测试、如何防止 verifier 偏差、何时值得付出额外推理成本。"
---

LLM-as-a-Verifier（以下简称 LaV）是一个面向智能体轨迹的通用验证框架，代码仓库位于 [llm-as-a-verifier/llm-as-a-verifier](https://github.com/llm-as-a-verifier/llm-as-a-verifier)，论文为 *LLM-as-a-Verifier: A General-Purpose Verification Framework*[^1]。项目的关键判断是：预训练、后训练和 test-time scaling 都在扩展“生成”能力，但真正限制智能体系统上限的常常是“从多个尝试中可靠地选出正确尝试”。因此，验证本身应成为一个可以增加计算预算、研究缩放规律并工程化复用的轴。

## 1. 背景问题：生成能力有余，选择能力不足

### 1.1 智能体的成功概率被“选择器”卡住

长时程智能体通常会反复采样轨迹：每条轨迹包含思考、工具调用、代码编辑、环境观察和最终输出。即使单次 Pass@1 不高，N 次采样中往往已经存在一条正确轨迹；理想的 oracle selector 可以把 Terminal-Bench V2 的成功率推到接近 98.9%[^1]。现实系统没有 oracle，只能依赖一个 verifier 在候选之间做判断。若 verifier 把正确和错误轨迹判成同分，增加候选数量就只会增加成本，不会转化为成功率。

### 1.2 标准 LLM-as-a-Judge 的离散分数太粗

传统 LLM-as-a-Judge 通常让模型生成一个整数或等级，然后取该 token 作为分数。这相当于把模型在评分位置上的概率分布压缩成 argmax：例如 A=0.51、B=0.49 与 A=0.99、B=0.01 都只留下同一个离散答案。复杂代码或长轨迹的细微差异因此大量变成 ties，论文在 Terminal-Bench 上观察到离散 judge 约 27% 的比较出现并列[^1]。并列会让 Best-of-N 排序失去分辨率，也无法表达“模型很不确定”这一信息。

### 1.3 训练奖励模型的泛化与数据成本

ORM（Outcome Reward Model）和 PRM（Process Reward Model）通常依赖人工偏好、正确/错误标签或领域轨迹数据训练[^2][^3]。训练好的模型在数据分布内有效，但换到新的工具链、任务类型、模态或评价标准时需要重新收集数据。对 coding、医疗和机器人建立统一标签体系尤其昂贵；而且奖励模型可能被轨迹格式、表面模式或 reward hacking 误导。LaV 选择直接复用具有领域知识的通用 LLM，仅通过 criteria prompt 指定评价维度，不要求额外训练。

### 1.4 只看最终答案无法解释长轨迹

长时程任务的失败可能发生在中间步骤：改错文件、没有运行测试、在错误数据库副本上验证、机器人动作逐渐偏离目标等。只给最终结果的 ORM 不能帮助 harness 及时中止坏轨迹；只给局部动作的 PRM 又可能看不到全局约束。LaV 将完整 trajectory 作为输入，同时提供离线 `track` 和在线 `ProgressTracker`，把“当前状态是否已经满足隐藏 grader”变成每一步都可估计的连续信号。

## 2. 相比其他相关工作的差异与创新

| 方向 | 典型做法 | LaV 的差异 |
| --- | --- | --- |
| Test-time scaling | 增加 CoT、搜索、采样或 Best-of-N[^4][^5] | 把额外预算投入 verifier：扩大评分粒度、重复次数、criteria 数，并用 PPT 控制候选比较成本 |
| LLM-as-a-Judge | 生成离散整数/标签，或做单次 pairwise 判断[^6] | 读取评分位置的完整 logprob，计算连续期望；通过多标准和重复评估降低偏差与方差 |
| ORM/PRM | 用带标签轨迹训练 outcome/process reward model[^2][^3] | 零训练、跨领域、直接评估完整交互轨迹；同一分数既可做 outcome 选择，也可做 process 进度 |
| V1 等并行自验证 | 通过多次生成和验证提升结果[^7] | 重点研究 verifier 的可缩放性，并提出环形选 pivot 的 PPT，将比较从 O(N²) 降到 O(Nk) |
| 机器人奖励模型 | Robometer、TOPReward、RoboReward 等专门训练的视觉奖励模型[^8][^9][^10] | 用多模态 LLM 的图像/视频上下文和统一评分协议，在不重训的情况下迁移到机器人轨迹 |
| 规则/执行器验证 | 单元测试、编译器、仿真器等可验证奖励 | LaV 不替代硬验证；它补充“质量、根因、过程是否可信”等难以写成规则的软判断 |

创新可以概括为四点。

1. **概率化评分而非离散标签。** 设评分 token 集合为 $V_{score}={v_1,ldots,v_G}$，每个 token 映射到标量 $\phi(v_g)$。对任务 $x$、标准 $c$、轨迹 $\tau$，计算

   $$R(x,\tau)=\frac{1}{CK}\sum_{c=1}^{C}\sum_{k=1}^{K}\sum_{g=1}^{G}p_\theta(v_g\mid x,c,\tau)\phi(v_g).$$

   这保留了模型的不确定性，$G$ 越大，分辨率越细。

2. **三个验证缩放轴。** 增加 $G$ 可降低并列、拉开正负样本间隔；增加重复次数 $K$ 相当于对独立评估取均值、降低方差；把整体任务拆成 root cause、code review、verification 等 $C$ 个标准，可减少一个长 prompt 中的注意力竞争和评价偏置。论文的消融显示三者在受控预算下都带来稳定收益[^1]。

3. **Probabilistic Pivot Tournament（PPT）。** 完整 round-robin 要比较 $\binom{N}{2}$ 对。PPT 先在随机 Hamilton 环上比较 N 对，利用每个候选一次处于 A 位、一次处于 B 位抵消位置偏差；再选 ring 得分最高的 k 个 pivot，让其余候选只与 pivot 比较，并用 Bradley–Terry 概率聚合软胜负。比较数为 $N+k(N-k)+\binom{k}{2}$，固定 k 时近似 O(Nk)。

4. **一个信号服务三种用途。** 连续奖励可用于 Best-of-N 选择、在线进度监控和 RL dense reward。论文报告 Terminal-Bench V2 86.5%、SWE-Bench Verified 78.2%、RoboRewardBench 87.4%（trajectory preference accuracy）和 MedAgentBench 73.3%[^1]。

## 3. 底层技术原理与源码实现

仓库的核心代码集中在 `llm_verifier/`：`fine_grained_reward.py` 负责调用模型、读取 logprob、期望分数和缓存；`pivot_tournament.py` 负责候选选择；`progress.py` 负责离线/在线进度；`prompts.py` 负责 criteria 文件解析；`benchmarks.py` 只保存数据加载与运行配置。

### 3.1 评分 token 与期望值

实现使用 20 个有序字母 A-T，而不是数字。A 映射为 20 分，T 映射为 1 分，同时接受大小写和可能的 `>A`、` A` tokenizer 变体。`extract_score` 在 `<score_A>` 或 `<score_B>` 后的位置读取 top-logprobs，对出现的有效 token 做 softmax 归一化并求期望，再线性归一化到 [0,1]；找不到 logprob 时才退化为解析文本中的字母，仍找不到则返回 0.5。

这一步和普通 judge 的根本区别是：并不只取概率最高的字母。例如有效 token 的概率为 A:0.45、B:0.35、F:0.20，期望值会落在“高分但不确定”的中间位置；另一条轨迹若 A:0.55、T:0.45，离散 argmax 也许相同，但期望值会明显更低。评分分布因而提供了可校准的软信号。

### 3.2 Prompt、criteria decomposition 与槽位去偏

每次 pairwise 评估的 prompt 包含任务描述、轨迹 A/B、20 级评分量表和一个 criteria。criterion 被故意放在 prompt 尾部，让同一对长轨迹在不同标准、不同重复评估间共享前缀，命中后端 prefix cache。内置 criteria 文件示例包括：

- SWE-Bench：`root_cause` 判断是否修改了真实故障路径，`code_review` 检查语义/接口/回归，`verification` 只相信实际命令输出；
- Terminal-Bench：规格满足、输出匹配、错误信号；
- MedAgentBench：FHIR 查询参数准确性、响应与答案一致性、`FINISH([...])` 格式。

重复评估的奇数次会交换 A/B 轨迹，最终再映射回候选索引。这样可抵消模型把 A 位天然看得更好或更差的 position bias。PPT 的 ring 也用同样思想保证每个候选在两个槽位各出现一次。

### 3.3 从连续分数到 pairwise 偏好

对同一任务中两条轨迹得到 $R_a,R_b\in[0,1]$ 后，源码采用 Bradley–Terry 映射：

$$p(a\succ b)=\sigma(R_a-R_b)=\frac{1}{1+e^{-(R_a-R_b)}}.$$

如果 $R_a=R_b$，两者各获得 0.5 的软胜；差距越大，胜率越接近 1。比较结果累计到每个候选的 win mass $w_i$ 和比较次数 $c_i$，以 $w_i/c_i$ 作为最终平均偏好。相比只记“赢/输”，软概率不会丢失不确定性。

### 3.4 PPT 的具体流程

```text
输入：N 条轨迹，标准集合 C，重复次数 K，pivot 数 k
1. 随机打乱索引，构造 N 条环边 (i_t, i_(t+1) mod N)
2. 评估环边，累计 w_i/c_i，取均值最高的 k 条为 pivots
3. 生成所有 (non-pivot, pivot) 与 pivot-pivot 对
4. 对每个方向、criterion、repeat 读取连续分数并转成 Bradley–Terry 概率
5. 合并 ring 与 pivot round，返回 argmax_i(w_i/c_i)
```

实现中的 pair 是有方向的：`(a,b)` 与 `(b,a)` 使用不同缓存键；`rep % 2 == 1` 时交换 prompt 槽位但把分数写回原候选顺序。默认 `pivots=2`，N 较大时成本近似线性；增加 k 会提高召回但也增加 API 调用。论文在 N=20 的实验中报告 k=9 已接近 full round-robin，而查询对数显著更少[^1]。

### 3.5 缓存、并发与后端适配

框架支持 Vertex Gemini、DeepSeek 以及任何能返回 token-level logprobs 的 OpenAI-compatible 服务（vLLM、SGLang 等）。`LazyClient` 延迟创建客户端，使完全命中缓存的重跑不需要 API key。`score_directed_pairs` 只提交 PPT 实际需要的 directed pairs，并把每个 criterion/repeat 的结果写入 JSON。

缓存优化有两个关键点：一是 criterion 位于 prompt 尾部，二是先对每个独特 `(task, slot-A, slot-B)` 前缀发起 warm-up，完成后再并发 fan-out。README 在 Terminal-Bench 2.1 报告缓存命中率约 78.4%，未缓存输入 token 减少约 3.4 倍。`USAGE` 是线程安全的进程级计数器，记录 input、cached input、output 和 reasoning token，避免只按请求数估算成本。

### 3.6 多模态与进度跟踪

所有入口都接受本地路径、URL 或 bytes 图像。对于机器人，帧会按顺序附在 verifier message 中；`ProgressTracker.update(step, images=...)` 会在轨迹中插入 `[Image i attached]` 标记，并在后续更新中保留全部历史。视觉帧因此成为轨迹状态的一部分，而不是一次性截图。

`track` 对完整轨迹一次生成多个 checkpoint 的评分，复杂度是 O(K) 次 verifier 调用而不是 O(TK)；`ProgressTracker` 则每来一步就只把当前前缀发给模型，不能偷看未来。进度 prompt 明确要求“相信观察到的输出，不相信 agent 的自述”，并允许分数回退：错误操作后分数应下降，而不是被步骤数量或努力程度奖励。

## 4. 适用场景与边界

### 4.1 适合使用的情况

1. **有多个候选轨迹且单次生成昂贵。** 例如 coding agent 为同一 issue 生成 3-10 个 patch，或者规划 agent 产生多条工具调用路径，此时 LaV 的选择收益大于额外 verifier 成本。
2. **正确性包含软标准。** 根因是否真正修复、验证是否方法学严谨、翻译是否保留风格、医疗查询是否与问题一致，这些难以只靠字符串或单元测试判断。
3. **长轨迹需要在线止损。** 进度分数连续偏低时可以提前终止 rollout、重新采样或切换策略，节省后续工具调用。
4. **跨模态任务。** 机器人、GUI、图像编辑等可把帧与文字轨迹一起交给多模态 verifier。
5. **强化学习需要 dense reward。** 当环境只在最终成功时给 sparse reward，可将每步/每帧的 verifier 分数作为 shaping 信号；论文在 LIBERO 的 SAC 和 MATH 的 GRPO 上观察到样本效率提升[^1]。

### 4.2 不应直接依赖的情况

- 存在可靠、便宜、完备的硬验证器时（编译、单元测试、形式化证明、数据库约束），硬信号应作为主判定，LaV 只做补充排序；
- verifier 看不到完成任务所需的外部状态，或轨迹缺少命令输出、文件 diff、环境观测；此时模型只能猜；
- 领域知识高度专业且基础模型明显不具备时，criteria prompt 不能替代领域训练；
- 对安全、医疗、金融等高风险动作，连续分数不是授权信号，必须设置规则门禁、人工复核和可审计证据；
- 模型 API 不提供 logprobs。论文给出的折中办法是让 GPT-5.5 先生成 reasoning，再交给能返回 logprobs 的 Gemini 读取评分分布，但这会增加系统复杂度且不等价于原生方案[^1]。

主要风险包括 verifier 自身偏差、criteria 泄漏 ground truth、轨迹过长导致注意力稀释、重复评估的相关噪声、以及把模型“看起来完成”误当成真实完成。因此生产系统仍应把测试输出、执行器状态和 LaV 分数一起记录。

## 5. 具体应用：从安装到代码

### 5.1 安装与后端配置

```bash
pip install llm-verifier
# 或从源码安装
pip install -e .
```

任选一个支持 logprobs 的后端。以 OpenAI-compatible 的本地 vLLM 为例：

```bash
vllm serve Qwen/Qwen3.5-9B --port 8000
$env:OPENAI_BASE_URL = "http://localhost:8000/v1"
$env:OPENAI_API_KEY = "EMPTY"
```

也可以设置 `DEEPSEEK_API_KEY` 使用 `deepseek-v4-flash`，或设置 Vertex 的 `VERTEX_API_KEY` 使用 Gemini。DeepSeek 的 reasoning 与输出预算可通过 `DEEPSEEK_EFFORT`、`DEEPSEEK_MAX_TOKENS` 调整；预算过小会导致模型在输出评分标签前耗尽 token。

### 5.2 Coding agent 的 Best-of-N

下面的代码用两个标准选择多个修复轨迹。标准应描述**可从轨迹观察到的证据**，避免把“成功标签”直接写进 prompt。

```python
import llm_verifier

problem = "修复 utils.py 中失败的测试，并解释根因。"
candidates = [traj_1, traj_2, traj_3, traj_4, traj_5]

result = llm_verifier.select(
    problem=problem,
    candidates=candidates,
    criteria={
        "根因": "是否定位到产生问题的真实代码路径，而不是只绕过样例？",
        "验证": "是否运行了相关测试，并展示与任务成功条件直接对应的输出？",
    },
    n_evaluations=4,  # K：每个标准重复评估 4 次
    pivots=2,          # k：控制 PPT 的比较预算
    seed=0,
    cache="cache/utils_fix.json",
)

print(result.index, result.best)
print(result.ranking, result.scores)
print(result.n_comparisons)
```

`result.index` 是输入列表中的获胜索引，`result.scores` 是聚合后的平均偏好，不是成功概率。真正落地时应在选中后再次运行测试或部署前检查，并把 verifier 评分与硬验证结果做一致性审计。

### 5.3 在线监控与提前止损

```python
from llm_verifier import ProgressTracker

tracker = ProgressTracker(problem, n_evaluations=4)
for step in agent_stream():
    score = tracker.update(step.text, images=step.frames)
    print(f"step={len(tracker.steps)} score={score:.3f}")
    # 例：连续多步低于阈值时中止并重新采样
    if len(tracker.steps) >= 8 and score < 0.05:
        abort_rollout()
        break
```

离线评估已完成的轨迹则使用：

```python
result = llm_verifier.track(
    problem=problem,
    steps=recorded_steps,
    checkpoint_steps=[1, 3, 5, 8, 10],
    n_evaluations=4,
)
print(result.steps, result.scores, result.final)
```

阈值必须用历史成功/失败轨迹校准，不能直接照搬示例中的 0.05。进度分数是代理指标，不能替代最终验收。

### 5.4 自定义 benchmark 的 criteria 文件

仓库约定 `criteria/<name>.md`：

```markdown
# My Tool Agent

## Ground Truth Note
只相信命令的实际输出，不相信 agent 的“已完成”声明。

## Criteria

### specification {#specification}
是否满足任务中所有输入、输出和文件路径约束？

### verification {#verification}
是否在真实目标环境执行了可复现的验证，并保留关键输出？
```

然后可以直接调用 `llm_verifier.select(..., criteria="my_tool_agent")`。如果需要复现实验，参考 `scripts/run.py`：loader 读入带有 ground-truth reward 的轨迹，先处理所有 swing tasks，再按 ring pass 和 pivot rounds 两阶段评分，最后报告 Pass@1、LaV 和 Oracle 三组指标。

### 5.5 把分数接入 RL

在机器人或推理训练中，可以把 LaV 分数作为额外的 dense reward，但要控制权重和调用频率。一个简化的 shaping 形式是：

$$r'_t=r^{env}_t+\lambda\,(s_t-s_{t-1}),$$

其中 $s_t$ 是截至 t 步的 verifier 进度，$\lambda$ 控制软奖励相对环境奖励的影响。实践中应固定策略、优化器和随机种子，对 sparse baseline、只用硬奖励和 LaV shaping 做多次对照；还要监控 reward hacking，例如模型学会生成更容易被 verifier 判高分的叙述，却没有改善真实任务状态。

## 6. 复盘：如何正确理解这个项目

LaV 最重要的贡献不是“再训练一个更大的 judge”，而是把验证过程本身变成可扩展、可测量的推理系统：用 logprob 期望保留不确定性，用 repeated evaluation 做方差缩减，用 criteria decomposition 做复杂度分解，用 PPT 把有限预算集中在最有希望的候选上，再把同一连续信号复用于选择、进度和 RL。它解决的是“候选已经存在但无法可靠挑选和监控”的系统问题，而不是让一个弱模型凭空获得领域真值。

工程上最稳妥的组合是：**硬验证负责可证伪事实，LaV 负责软质量与过程判断，人工负责高风险裁决**。只有当轨迹携带了足够的观察证据、criteria 可操作、后端能提供 logprobs 且额外 API 成本可接受时，LaV 才能把 test-time sampling 的潜在收益转化成实际成功率。

## 参考文献

[^1]: Jacky Kwok, Shulu Li, Pranav Atreya, Yuejiang Liu, Yixing Jiang, Chelsea Finn, Marco Pavone, Ion Stoica, Azalia Mirhoseini. *LLM-as-a-Verifier: A General-Purpose Verification Framework*. arXiv:2607.05391, 2026. [论文](https://arxiv.org/abs/2607.05391)；[代码](https://github.com/llm-as-a-verifier/llm-as-a-verifier)。
[^2]: Karl Cobbe et al. *Training Verifiers to Solve Math Word Problems*. arXiv:2110.14168, 2021.
[^3]: Hunter Lightman et al. *Let's Verify Step by Step*. arXiv:2305.20050, 2023.
[^4]: Charlie Snell, Jaehoon Lee, Kelvin Xu, Aviral Kumar. *Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters*. arXiv:2408.03314, 2024.
[^5]: Hengyuan Hu et al. *V1: Unifying Generation and Self-Verification for Parallel Reasoners*. arXiv:2603.04304, 2026.
[^6]: Lianmin Zheng et al. *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena*. arXiv:2306.05685, 2023.
[^7]: Harman Singh et al. *V1: Unifying Generation and Self-Verification for Parallel Reasoners*. arXiv:2603.04304, 2026.
[^8]: S. Chen et al. *TOPReward: Token Probabilities as Hidden Zero-Shot Rewards for Robotics*. arXiv:2602.19313, 2026.
[^9]: Tianyi Lee et al. *RoboReward: General-Purpose Vision-Language Reward Models for Robotics*. arXiv:2601.00675, 2026.
[^10]: A. Liang et al. *Robometer: Scaling General-Purpose Robotic Reward Models via Trajectory Comparisons*. arXiv:2603.02115, 2026.
