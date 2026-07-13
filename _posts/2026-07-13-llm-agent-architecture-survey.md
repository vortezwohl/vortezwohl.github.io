---
layout: post
toc: true
title: "LLM Agent 设计架构综述：从 ReAct 与 Workflow 到规划、搜索、记忆、多 Agent 协作与场景选型"
categories: AI
tags: [LLM, Agent, Multi-Agent, ReAct, Workflow, Code Agent, Computer Use]
author:
  - vortezwohl
  - 吴子豪
---

> 本文整理截至 **2026 年 7 月 13 日**可公开查证的学界研究与业界实践。它讨论的是“如何组织具有目标、状态、工具、反馈与控制边界的 LLM Agent 系统”，不把单纯的提示词技巧、单次 RAG、固定提示词链或底层机器人控制器误当成 Agent 架构。研究论文的实验结论通常受模型、工具、基准和成本约束，不能直接外推为生产结论。

## 先看结论

如果只记住几条原则，应记住下面这些：

1. **ReAct 只是控制循环，不是 Agent 架构的全集。** 它解决“观察、思考、行动、再观察”的在线决策问题；复杂系统还需要规划、搜索、记忆、验证、权限、停止条件和审计。
2. **Workflow 与 Agent 的差别在控制权。** 固定 DAG、路由和提示词链中，下一步主要由程序决定；Agent 中，模型或 Agent 控制器会根据环境状态动态决定下一步、工具和重试路径。两者可以组合，生产系统常常是“确定性 Workflow 包住局部 Agent”。
3. **先做单 Agent，再证明必须拆多 Agent。** 多 Agent 并不天然更聪明；它同时放大上下文丢失、通信开销、状态冲突、共同幻觉、调试难度和成本。只有工具、权限、上下文或专业边界确实不同，才值得拆分。
4. **真实反馈优先于 LLM 自评。** 测试、编译器、数据库执行结果、页面状态、原始图像、结构化 schema、人工审批，通常比“另一个 LLM 说看起来正确”可靠。
5. **场景决定架构。** 标准 OCR、常规翻译和 FAQ 往往先用 Workflow；代码、浏览器、研究、复杂客服等开放任务才更需要 Agent；机器人还要接入传统规划、控制和安全层，不能等同于 Computer Use。
6. **生产可靠性主要来自边界，而不是更长的提示词。** 工具契约、状态机、预算、权限、可回滚性、验证器、trace、失败恢复和人工接管，是 Agent 成功的工程基础。

一个实用的默认方案通常是：

```text
确定性工作流 / 状态机
  + 单 Agent 工具循环
  + 必要的计划与重新规划
  + 结构化任务状态与检索记忆
  + 确定性验证器
  + 高风险动作的人审、预算和审计
```

而不是一开始就搭建大量“互相聊天”的角色 Agent。

## 一、什么算 Agent，什么不算

### 1. Agent 的最小闭环

从系统角度，一个 LLM Agent 至少应具备如下闭环：

```text
目标
  -> 读取环境或状态
  -> 决定下一步
  -> 调用工具或执行动作
  -> 获得环境反馈
  -> 更新状态、继续、修正、终止或求助
```

这里的关键不是模型是否输出了长篇“思考”，而是：

- 下一步是否会根据真实状态动态变化；
- 是否能调用外部工具、改变外部状态或读取环境；
- 是否存在反馈和失败恢复；
- 是否有完成条件、预算、权限和停止条件。

OpenAI 将 Agent 概括为：LLM 管理工作流执行和动态决策，并按需使用工具与外部系统交互。Anthropic 也把预定义代码路径的 Workflow 与模型动态主导过程的 Agent 区分开来。两种定义的共同点，是**控制流与环境交互**，不是“提示词写得像人在思考”。

### 2. 不应混淆的概念

| 概念 | 是否通常构成完整 Agent | 原因 |
|---|---:|---|
| 单次问答、摘要、分类、翻译 | 否 | 没有持续状态、工具行动和反馈循环。 |
| Few-shot、角色提示词、CoT | 否 | 主要是提示词或推理呈现方式。 |
| RAG | 通常否 | 检索是能力部件；只有动态决定何时检索、如何验证和后续行动时，才进入 Agent 系统。 |
| 固定 Prompt Chaining / DAG | 通常是 Workflow | 后续节点由开发者预先决定。 |
| 固定状态机中的 LLM 节点 | 介于两者之间 | 系统整体是受控 Workflow；节点可能是局部 Agent。 |
| ReAct + 工具 + 环境反馈 | 是 | 具备在线观察—行动—反馈循环。 |
| Planner + Executor + Verifier | 是 | 具备动态执行、验证和重新规划能力。 |
| 多个 persona 同时回答 | 不一定 | 若没有独立状态、职责、通信和合成机制，只是多次采样。 |

### 3. 一个可操作的判断问题

在设计前先问：**如果把 LLM 换成固定规则，系统的下一步路径是否仍然完全确定？**

- 若基本确定，优先把它当 Workflow 设计；
- 若必须根据工具结果、环境变化、未知文件、网页状态、用户澄清或失败日志实时决定下一步，才需要 Agent；
- 若高层是固定流程、局部节点需要动态探索，则采用“Workflow 包裹 Agent”通常最稳妥。

## 二、Agent 架构不是单一谱系，而是九个可组合维度

把 Agent 只分为 Workflow 和 ReAct 会遗漏大部分真实设计空间。更完整的视角是九个正交维度：

| 维度 | 要回答的问题 | 常见选项 |
|---|---|---|
| 控制循环 | Agent 如何推进？ | ReAct、状态机、事件驱动、Plan-and-Execute |
| 规划 | 是否先拆任务？ | 即时决策、一次性计划、分层计划、滚动重规划 |
| 搜索 | 是否同时探索多条路径？ | 单路径、树搜索、图搜索、MCTS、回溯 |
| 记忆 | 如何保留状态和经验？ | 工作记忆、RAG、情景/语义/程序记忆、技能库 |
| 工具 | 如何发现和调用能力？ | 工具路由、工具检索、代码执行、浏览器/桌面接口 |
| 验证 | 如何判断动作成功？ | 环境反馈、测试、规则、LLM Judge、人工 |
| 学习 | 如何从失败中改进？ | 反思、经验压缩、技能沉淀、离线训练 |
| 协作 | 是否拆成多个主体？ | Supervisor、Handoff、流水线、黑板、辩论 |
| 治理 | 如何控制风险？ | 权限、预算、审批、回滚、审计、隔离 |

因此，一个生产 Agent 的描述应当像这样：

```text
单 Agent；滚动计划；ReAct 工具循环；任务状态机；
代码库检索 + 结构化工作记忆；测试为主验证；
失败后局部反思；写操作需人审；最大 20 步；全链路 trace。
```

这比简单说“这是 ReAct Agent”更有工程意义。

## 三、单 Agent 架构

### 1. 反应式工具循环：ReAct、Tool Calling、CodeAct、Computer Use

最常见的单 Agent 架构是：

```text
Observe -> Decide -> Act -> Observe -> ... -> Finish
```

ReAct 让推理与行动交替出现；现代产品中常以函数调用、浏览器操作、终端命令、代码执行或桌面鼠标键盘事件呈现。CodeAct 可以理解为“把代码和执行环境作为主要行动语言”的 Agent；Computer Use 则以屏幕、DOM、可访问性树或窗口状态作为观察。

**优势：**

- 路径可根据真实环境动态变化；
- 实现和调试相对直接；
- 适合工具调用、客服、浏览器、数据分析、轻量代码任务；
- 便于插入最大步数、人工确认和日志。

**局限：**

- 容易短视，缺乏整体任务结构；
- 易重复查询或循环；
- 长任务上下文膨胀；
- 一次错误观察会污染后续决策；
- 如果工具回馈不精确，模型很难自我纠正。

**适用：** 步数不可预测、环境反馈频繁、单步可验证且动作大多可撤销的任务。

### 2. Plan-and-Execute：规划、执行与重新规划分离

结构如下：

```text
目标
  -> Planner 生成计划
  -> Executor 执行一个或多个步骤
  -> Verifier 检查结果
  -> Replanner 按需修订剩余计划
```

它适合长任务：先把目标分解为可验证切片，再按环境反馈滚动修订，而不是每一步都从头思考全部问题。

**常见变体：**

- Plan-and-Solve；
- Planner / Worker；
- ReWOO；
- 分层任务网络（HTN-like）；
- 把高层计划表示为 DAG，而不是线性清单。

ReWOO（Reasoning Without Observation）将“生成工具调用计划”“执行工具”“最终求解”解耦，目标是减少每次观察结果都塞回大推理上下文的成本。它特别适合可预先识别信息需求、工具执行可批量化的任务；但环境高度变化、必须看一步走一步的网页任务并不总是适配。

**设计要点：** 初始计划不是合同。必须定义何时重规划，例如工具返回空结果、前置条件不成立、验证失败、发现新的依赖、成本接近上限。

### 3. 分层规划：从目标到子目标再到原子动作

层次结构通常是：

```text
总目标
  -> 子目标
      -> 任务
          -> 原子工具动作
```

例如一个代码修复任务：

```text
修复缺陷
  -> 理解复现条件
  -> 定位调用链
  -> 设计最小补丁
  -> 修改代码
  -> 运行局部测试
  -> 复查 diff 与回归风险
```

高层负责目标、优先级、预算和重规划；低层负责具体文件、命令和工具。优点是可将局部失败限制在子树中，而不推翻整个任务。缺点是层次越多，信息压缩和任务转交越可能丢失关键约束。

### 4. 搜索型 Agent：ToT、GoT、LATS、MCTS 风格规划

线性 ReAct 只走一条路径：

```text
A -> B -> C -> D
```

搜索型 Agent 会保留并评估多条候选路径：

```text
        -> B1 -> C1
A ->
        -> B2 -> C2 -> D
        -> B3
```

常见形式：

- Tree-of-Thoughts：分支、打分、剪枝和回溯；
- Graph-of-Thoughts：允许分支合并和中间结论复用；
- Monte Carlo Tree Search：在探索与利用间分配预算；
- LATS（Language Agent Tree Search）：将语言 Agent、环境行动、价值评估和树搜索结合。

需要明确：**ToT/GoT 原始上首先是推理或搜索方法，不自动等于 Agent。** 当树节点对应真实工具动作、可观察环境状态、可验证结果和回溯策略时，它们才成为 Agent 的行动搜索架构。

**适用：** 网页导航、复杂规划、游戏、网络安全、候选代码方案探索等高分支任务。

**不适用或须谨慎：** 有不可逆外部动作、无法复制环境、成本极高的任务。支付、邮件发送、生产数据库写入等场景，不宜用未经约束的树搜索反复“试错”。

### 5. 反思与自我修复：Reflexion、Critic-Actor、Self-Debug

典型循环：

```text
Actor 执行
  -> Evaluator 评估
  -> Reflector 归因失败
  -> 更新任务策略或经验
  -> 重试 / 终止 / 升级人工
```

Reflexion 的重要点在于：不更新模型权重，而把失败经验转化为可检索的语言记忆，以影响后续尝试。代码场景中常表现为“运行测试 -> 读取失败日志 -> 提出假设 -> 最小修复 -> 再测”。

反思可以发生在：

- **行动前：** 检查前置条件、权限和可逆性；
- **行动后：** 对失败结果归因；
- **任务后：** 将可复用经验抽取为技能或规则。

**风险：** LLM 反思本身也可能幻觉。应优先让反思绑定真实证据，例如测试日志、HTTP 状态、DOM 变化、编译器错误、原图坐标，而不是接受“我认为问题出在 X”的自述。

### 6. Evaluator-Optimizer：生成、评估、修订

它与反思相近，但重点是明确的质量优化：

```text
Generator -> Candidate -> Evaluator -> Critique / Score -> Revision
```

评估器可以是：

- 单元测试、编译器、JSON Schema、正则规则；
- 双语对齐、术语表、一致性检查；
- 专项 OCR 或事实核验模型；
- LLM Judge；
- 人类专家。

最适合“输出可迭代改善、且质量标准能部分显式表达”的任务，如文学翻译、代码、报告、长文创作、结构化抽取。它不适合在没有外部评价准则时无限循环润色。

### 7. 世界模型与模拟驱动规划

这类 Agent 不只看当前观察，还维护对环境转移的近似模型：

```text
当前状态
  -> 预测多个行动后果
  -> 选择计划
  -> 在真实环境执行
  -> 用反馈修正预测
```

它在游戏、仿真、资源调度、科学实验和具身任务中有价值。需要注意，LLM 生成的“常识性世界模型”不能取代真实环境约束；生产中常需要结合规则引擎、仿真器、数据库、传统规划器和安全约束。

### 8. 记忆增强 Agent：不是一个向量库就够了

Agent 的记忆至少可以分为：

| 记忆类型 | 内容 | 常见用途 |
|---|---|---|
| 工作记忆 | 当前目标、最近观察、临时变量 | 单次任务推进 |
| 情景记忆 | 历史任务、行动、反馈与结果 | 从经历中避免重复失败 |
| 语义记忆 | 实体、事实、关系和约束 | 企业知识、研究、长期助手 |
| 程序记忆 | 可复用步骤、工具操作、技能 | 代码、浏览器、运维、游戏 |
| 用户记忆 | 偏好、权限、历史选择 | 个性化助手与客服 |
| 失败记忆 | 失败模式、修复经验、禁忌 | 重试和风险控制 |

RAG 主要解决“从外部取回相关信息”，但还需要回答：何时写入、何时检索、如何处理过期或冲突记忆、如何隔离用户数据、如何让错误记忆不永久污染系统。

Voyager 提出的自动课程、可执行技能库和长期积累，展示了“程序记忆/技能库”对开放式环境探索的价值。对代码和浏览器 Agent 而言，技能更应看作可审查、可版本化、带前置条件和验证步骤的操作资产，而不是一段模糊提示词。

### 9. Router、Tool Selector 与动态能力加载

一个 Agent 往往不是直接选择“答案”，而是先选择能力：

```text
任务 -> 选择模型 / 工具 / 技能 / 子流程 -> 执行 -> 验证
```

常见路由依据：

- 任务类别；
- 风险等级；
- 用户权限；
- 工具适配度；
- 成本和延迟预算；
- 上下文长度；
- 置信度。

Router 可以是规则、分类模型、LLM、混合策略。高风险操作应将“是否允许”从 LLM 判断中剥离，由确定性授权层决定。

### 10. 状态机、事件驱动与人类接管

在客服、订单、金融、审批等领域，最可靠的设计常不是完全自治，而是：

```text
明确状态机
  + LLM 负责理解、填槽、解释和局部决策
  + 规则负责合法迁移、权限、金额、审计和回滚
```

人类介入分为两类：

- **Human-in-the-loop：** 高风险动作前必须审批；
- **Human-on-the-loop：** Agent 常规自主执行，但人可以监控、暂停、接管。

应显式设置：成功条件、失败条件、重试上限、超时、最大成本、可逆性等级、人工升级阈值和审计记录。

## 四、多 Agent 架构

多 Agent 的核心价值是**拆分真正不同的上下文、工具、权限、专业能力或并行工作**，而不是让同一个模型换几个角色名重复回答。

### 1. Supervisor / Manager：主管—专家

```text
                 Supervisor
              /      |       \
       Search Agent  Code Agent  Database Agent
```

主管负责目标分解、路由、上下文裁剪、结果汇总、预算和终止；专家负责有限职责、专用工具和结构化输出。

这是当前企业系统中最常见、也最容易治理的多 Agent 结构。优点是控制集中、审计清晰、权限可分；缺点是 Supervisor 容易成为瓶颈和单点失误来源。

**适用：** 企业客服、研究助理、数据分析、代码工程、多工具办公流程。

### 2. Handoff / Swarm：去中心化交接

```text
通用接待 Agent -> 订单 Agent -> 技术支持 Agent -> 人工
```

当前 Agent 把控制权转交给更合适的 Agent。它适合对话型系统和职责自然切换的业务；但必须定义交接协议、上下文摘要、权限继承和“何时交还控制权”，否则容易出现循环转交和信息丢失。

### 3. Hierarchical Multi-Agent：多层团队

```text
总协调 Agent
  -> 项目经理 Agent
      -> 实现 Agent
      -> 测试 Agent
      -> 评审 Agent
```

它是单 Agent 分层规划在多主体上的扩展。Magentic-One 等研究系统采用编排者统筹不同专项 Agent，在任务进度、失败恢复和重新规划上进行协调。

**适用：** 长周期工程、复杂研究、网络安全、跨系统业务流程。

**代价：** 层级带来摘要损失、响应变慢、调试困难。层数应由任务边界支付，不能为了组织感而无限加层。

### 4. Pipeline / Assembly Line：角色流水线

```text
需求 -> 设计 -> 实现 -> 测试 -> 评审 -> 发布
```

每个 Agent 的输入输出是约定工件，而不是开放式闲聊。MetaGPT 用软件公司的 SOP 组织角色与中间产物；ChatDev 使用角色对话链模拟设计、编码和测试协作。

**适用：** 具有稳定阶段边界的内容生产、翻译审校、文档生成、软件交付。

**风险：** 上游错误会向下游传播。因此每个阶段需要验收条件、工件版本与回退点。

### 5. Parallel Experts：并行分工与投票

```text
             -> Expert A
任务 ->      -> Expert B -> Aggregator
             -> Expert C
```

两类最常见：

- **Sectioning：** 不同 Agent 处理不同子问题，例如文档的版面、表格、事实和风格；
- **Voting：** 多个 Agent 独立处理同一问题，再投票或裁决。

并行适合可独立的子任务，也可降低单点遗漏；但“多个同模型、同提示、同上下文”的投票独立性很弱，可能只是放大共同偏差。

### 6. Debate / Critique：辩论、红队与裁决

```text
Proposer -> Critic / Red Team -> Judge -> Revision
```

适用于代码安全、事实核验、法律/合规分析、重要研究结论和高风险方案。有效性依赖证据和裁决标准，而不取决于“辩论轮数”：若没有可核验事实，多个 Agent 可能只是更有说服力地重复错误。

### 7. Mixture-of-Agents：分层聚合

MoA 的核心不是简单投票，而是后层 Agent 读取前层多个 Agent 的输出并进行综合：

```text
Layer 1: 多个独立候选
Layer 2: 读取 Layer 1 的候选并改进
Layer 3: 汇总生成最终结果
```

它适合高质量文本、推理和研究摘要；缺点是上下文迅速膨胀，且前层错误可能被集体吸收。对有强确定性验证器的任务，直接用验证器筛选往往比堆叠 LLM 更可控。

### 8. Blackboard / Shared Workspace：共享工件而非长对话

```text
共享工作区
  - 任务状态
  - 证据与来源
  - 中间文件
  - 已做决策
  - 测试和错误
  - 待办与责任归属
```

多个 Agent 通过读写结构化状态、版本化文档、代码仓库或任务表协作。对代码、研究、写作、复杂文档处理尤其重要，因为“直接把长对话传给下一个 Agent”难以审计且易丢信息。

关键设计问题：状态字段所有权、并发冲突、版本、过期信息、证据溯源、访问控制、任务完成定义。

### 9. Market / Contract-Net、动态生成与社会模拟

这些架构更前沿，生产采用相对少：

- **Market / Auction：** 任务发布后由 Agent 报价、声明能力或竞争执行权；
- **Dynamic Spawning：** 主 Agent 按任务临时创建子 Agent；
- **Role-playing Society：** 通过角色、通信和组织规则模拟群体协作，例如 CAMEL；
- **Competitive / Adversarial Team：** 攻防、红蓝对抗或压力测试。

它们对大规模分布式任务、模拟、研究和安全测试有价值，但需要成熟的能力描述、信誉、权限、成本控制和状态一致性设计。普通业务系统通常不应首先选择它们。

## 五、单 Agent 与多 Agent 的选型边界

### 优先单 Agent

优先单 Agent，若：

- 工具数量有限且职责相近；
- 需要统一维护长上下文；
- 任务不能有效并行；
- 延迟、成本和可调试性重要；
- 角色差异只存在于提示词而非工具、权限或评价标准；
- 一个明确的 Planner、Executor、Verifier 即可闭环。

### 考虑多 Agent

考虑拆多 Agent，若至少满足一项：

- 子任务可真正并行；
- 专家需要不同工具、模型、权限或上下文；
- 单 Agent 经常误选工具或被上下文淹没；
- 需要独立的审查、红队或合规边界；
- 需要把写操作与验证操作隔离；
- 任务天然存在稳定工件接口和责任边界。

### 一个实用的判断公式

不要问“多 Agent 会不会更强”，而应问：

```text
拆分后的收益
  是否大于
通信开销 + 上下文损失 + 状态一致性成本 + 调试复杂度 + token/延迟成本
```

若答案无法用评估数据证明，就先保留单 Agent。

## 六、按应用场景选架构

### 1. 机器翻译

| 子场景 | 推荐设计 | 不宜优先使用 |
|---|---|---|
| 简短通用翻译 | 单模型或固定 Workflow | 自治 Agent |
| 术语/法律/技术翻译 | 术语库 RAG + 一致性校验 + 审校循环 | 无验证的自由改写 |
| 文学翻译 | Generator-Critic、风格/人物/术语记忆、人工编辑 | 无限自我润色 |
| 长篇翻译 | 章节 Pipeline + 全局实体/术语/风格记忆 + 文档级复核 | 逐段独立翻译 |

翻译的关键不是“是否用了 Agent”，而是：原文对齐、术语一致性、跨段人物与语气一致、专名管理、可追溯修订。文学翻译适合 Evaluator-Optimizer；标准化技术翻译常常先用确定性术语和格式 Workflow。

### 2. OCR、票据和复杂文档理解

| 子场景 | 推荐设计 |
|---|---|
| 标准表单、票据、证件 | 检测 -> OCR -> 规则解析 -> Schema 校验的确定性 Pipeline |
| 表格、合同、扫描长文 | 版面/表格/文本/字段抽取分工 + 原图证据 + 低置信复核 |
| 非标准文档调查 | 多模态工具 Agent，动态裁剪、旋转、放大、复读局部区域 |

OCR 系统应保存原图、页码、坐标、行列关系、字段置信度和纠正历史。只输出一段“看起来正确的文本”不足以支撑审计。Agent 的价值在于决定何时重试局部、调用哪种解析器、如何处理异常版面；标准识别流程不必强行 Agent 化。

### 3. 网文、长篇创作与编辑

推荐采用“主笔 + 结构化世界状态 + 可选编辑室”的组合：

```text
世界观 / 人物关系 / 时间线 / 伏笔表 / 已发布章节
  -> 主笔 Agent 生成章节
  -> 一致性与节奏检查
  -> 人类作者审批
  -> 写入版本化记忆
```

可选角色包括人物一致性、设定、节奏、语言编辑、读者模拟，但它们应读取同一个版本化黑板，而不是依靠互相聊天。最关键的能力是情景记忆、语义世界模型和工件版本控制。已发布正文的修改应当生成 diff 并交由人审。

### 4. Chatbot、客服与业务操作

建议：

```text
入口 Router
  -> FAQ / 知识库 Agent
  -> 账户与订单 Agent
  -> 技术支持 Agent
  -> 高风险流程状态机
  -> 人工接管
```

FAQ 多数只需 RAG/Workflow；当系统需要查询订单、修改地址、退款、创建工单时，才进入 Agent 和权限治理问题。必须做身份验证、最小权限、金额阈值、确认步骤、操作审计与人工升级。对话“自然”不能替代状态“合法”。

### 5. Code Agent

代码任务的核心是 Agent-Computer Interface（ACI），而不只是提示词。一个可信的最小闭环应是：

```text
Issue / 需求
  -> 仓库检索和阅读
  -> 计划最小改动
  -> 局部编辑
  -> 运行特定测试 / lint / build
  -> 分析失败日志
  -> 最小修复
  -> 检查 diff、回归风险和范围
```

推荐组合：Plan-and-Execute + CodeAct/ReAct + 测试验证 + 任务状态 + 必要的反思。多 Agent 只在代码探索、实现、测试、安全审查确实需要独立上下文或并行时使用。多个 Agent 同时无锁写同一工作区通常是坏设计。

### 6. Browser Use 与 Computer Use

浏览器任务的观察来源可包括 DOM、URL、可访问性树、截图和网络状态；桌面任务还要面对窗口、像素、焦点、弹窗、鼠标坐标和文件系统。

推荐闭环：

```text
观察页面/桌面
  -> 识别状态与目标元素
  -> 选择下一原子动作
  -> 执行
  -> 验证页面/窗口状态确实变化
  -> 必要时回退、重新规划或人工确认
```

Browser Use 更适合 ReAct + Planner + 动作验证；高分支任务可局部使用搜索。Computer Use 的主要难点往往是视觉 grounding 和实际执行，而不只是语言规划。对发送邮件、提交表单、购买、删除文件等不可逆操作必须提高审批等级。

### 7. 数据分析、SQL 与企业知识

推荐：

```text
问题 -> Schema / 语义层检索 -> 查询计划 -> 生成 SQL
  -> 静态检查 -> 受限执行 -> 结果解释 -> 复核
```

必须区分只读与写入 SQL，限制扫描范围、执行成本和数据权限。LLM 自评不能证明 SQL 正确；数据库执行结果、统计复算、时间范围和字段语义验证更关键。

### 8. 科学研究与深度研究

推荐 Supervisor 或单 Agent 分层规划，并使用共享证据黑板：

```text
问题 -> 文献检索 -> 证据归档 -> 假设 -> 实验 / 代码 -> 分析
  -> 反驳与复核 -> 报告
```

必须区分事实、推断、假设和未验证结论；记录来源、版本、失败实验与数据处理过程。多 Agent 的价值主要在并行检索、交叉审查和专项分析，而不是让它们“投票决定真相”。

### 9. 网络安全

授权场景下，适合分层规划、专项工具 Agent、红蓝对抗和严格沙箱。高层规划器协调侦察、漏洞专项检查和结果验证；所有写操作和风险动作应运行在授权环境，受网络边界、命令白名单、审计和人工审批约束。不能把“更会探索”误当成“可在任何环境自主运行”。

### 10. 机器人与具身智能

机器人与数字 Agent 有共同的高层任务规划需求，但不是同一层问题：

```text
LLM Agent：目标理解、任务分解、技能选择
  -> 传统规划器：路径、碰撞、资源和约束
  -> 技能策略：抓取、导航、操作
  -> 低层控制器：实时动作
  -> 传感器与状态估计：视觉、力觉、位置
```

真实机器人需要连续控制、动力学、时延、安全和不可逆物理后果。LLM 不应直接承担低层电机控制；其合理位置通常是高层语义规划、工具选择和人机交互。Voyager 这类虚拟具身 Agent 的技能学习有启发性，但不能直接等同于现实机器人部署。

## 七、生产系统的工程底座

### 1. 工具契约

工具是确定性系统与非确定性模型之间的边界。每个工具应至少说明：

- 名称和目的；
- 参数类型、默认值和约束；
- 读/写副作用；
- 权限要求；
- 成功和失败的结构化返回；
- 可逆性、幂等性和重试规则；
- 分页、截断与数据脱敏策略。

工具返回应该让 Agent 能定位失败，而不是只给“操作失败”。

### 2. 状态、工件与可恢复性

不要只依赖聊天历史。任务状态至少应记录：

```text
目标、当前阶段、已执行动作、关键观察、证据、产物版本、
未决问题、失败原因、下一步、预算、权限状态、完成判定。
```

长任务应设置 checkpoint；不可逆动作前保存可恢复状态；多 Agent 使用明确的共享数据模型和责任归属。

### 3. 验证优先级

从可靠性角度，通常应优先：

```text
真实环境状态
  > 确定性测试 / 规则 / Schema
  > 专项模型或外部验证服务
  > 有证据约束的 LLM Judge
  > Agent 自我评价
```

举例：

- 代码：测试、编译、静态分析优先于“代码看起来合理”；
- 浏览器：目标 URL、DOM 状态、下载文件存在优先于“我已完成”；
- OCR：原图坐标和字段校验优先于“识别文本通顺”；
- 翻译：术语表、对齐和人工抽检优先于“译文优美”；
- SQL：受限执行和结果复算优先于“查询意图正确”。

### 4. 权限与人工接管

应把权限判断从语言模型中剥离：

- LLM 可以提出动作建议；
- 策略层决定该动作是否被允许；
- 高风险动作必须确认或审批；
- 需要记录谁批准、执行了什么、影响了什么。

典型高风险动作：资金、生产配置、数据删除、外发消息、权限变更、医疗建议、法律承诺、安全测试和外部发布。

### 5. 评估与可观测性

每种架构都应被同一组指标约束：

- 任务成功率与部分成功率；
- 真实错误率和高风险误操作率；
- 平均步骤数、工具调用数、token 与时间成本；
- 重试率、人工升级率、回滚率；
- 计划错误、工具选择错误、执行错误、验证错误的占比；
- 对不同输入、环境变化和边界条件的稳定性。

没有评估集和 trace 的“架构升级”，难以证明比普通 Workflow 更有效。

## 八、一个实用的架构选择流程

设计新系统时，可以按以下顺序决定：

1. **先界定目标和副作用。** 输出文本、读取信息、还是修改真实世界状态？
2. **先判断是否必须 Agent。** 若路径可预定义，优先 Workflow；若需根据未知反馈动态行动，才加 Agent。
3. **选择最小控制循环。** 从单 Agent 工具循环或状态机开始，不先建多 Agent 社会。
4. **定义真实验证器。** 在实现模型提示之前先决定成功如何被观测。
5. **补齐状态与记忆。** 将任务状态、工件和证据外部化，不能只依赖上下文窗口。
6. **为失败建立闭环。** 明确重试、重新规划、回滚和人工升级条件。
7. **最后才判断是否拆多 Agent。** 只按工具、权限、上下文、并行度和验证边界拆分。
8. **用评估结果控制复杂度。** 新增一个 Agent、搜索分支或记忆层，必须带来可量化收益。

## 九、架构速查图

```text
LLM Agent 系统
├── 单 Agent
│   ├── 控制：ReAct / Tool Calling / CodeAct / 状态机
│   ├── 规划：Plan-and-Execute / ReWOO / 分层规划
│   ├── 搜索：ToT / GoT / MCTS / LATS / 回溯
│   ├── 改进：Reflexion / Self-Debug / Evaluator-Optimizer
│   ├── 记忆：RAG / 情景 / 语义 / 程序 / 技能库
│   ├── 路由：模型 / 工具 / 技能 / 权限路由
│   └── 治理：预算 / 审批 / 审计 / 回滚
└── 多 Agent
    ├── Supervisor / Manager
    ├── Handoff / Swarm
    ├── Hierarchical Teams
    ├── Pipeline / Assembly Line
    ├── Parallel Experts / Voting
    ├── Debate / Critique / Red Team
    ├── Mixture-of-Agents
    ├── Blackboard / Shared Workspace
    ├── Market / Contract-Net
    ├── Dynamic Spawning
    └── Role-playing / Competitive Society
```

## 十、参考资料与延伸阅读

### 业界工程资料

- OpenAI, *A practical guide to building agents*：<https://openai.com/business/guides-and-resources/a-practical-guide-to-building-ai-agents/>
- Anthropic, *Building effective agents*：<https://www.anthropic.com/engineering/building-effective-agents>
- Anthropic, *Writing tools for agents*：<https://www.anthropic.com/engineering/writing-tools-for-agents>
- Google, *Agent Development Kit documentation*：<https://google.github.io/adk-docs/>
- LangGraph, *Multi-agent concepts*：<https://langchain-ai.github.io/langgraph/concepts/multi_agent/>
- Microsoft, *AutoGen documentation*：<https://microsoft.github.io/autogen/stable/>
- AWS, *Amazon Bedrock multi-agent collaboration*：<https://aws.amazon.com/about-aws/whats-new/2025/03/amazon-bedrock-multi-agent-collaboration/>

### 基础论文与研究系统

- Yao et al., *ReAct: Synergizing Reasoning and Acting in Language Models*：<https://arxiv.org/abs/2210.03629>
- Xu et al., *ReWOO: Decoupling Reasoning from Observations for Efficient Augmented Language Models*：<https://arxiv.org/abs/2305.18323>
- Shinn et al., *Reflexion: Language Agents with Verbal Reinforcement Learning*：<https://arxiv.org/abs/2303.11366>
- Yao et al., *Tree of Thoughts: Deliberate Problem Solving with Large Language Models*：<https://arxiv.org/abs/2305.10601>
- Besta et al., *Graph of Thoughts: Solving Elaborate Problems with Large Language Models*：<https://arxiv.org/abs/2308.09687>
- Zhou et al., *Language Agent Tree Search Unifies Reasoning Acting and Planning in Language Models*：<https://arxiv.org/abs/2310.04406>
- Wang et al., *Voyager: An Open-Ended Embodied Agent with Large Language Models*：<https://arxiv.org/abs/2305.16291>
- Li et al., *CAMEL: Communicative Agents for “Mind” Exploration of Large Scale Language Model Society*：<https://arxiv.org/abs/2303.17760>
- Hong et al., *MetaGPT: Meta Programming for A Multi-Agent Collaborative Framework*：<https://arxiv.org/abs/2308.00352>
- Qian et al., *ChatDev: Communicative Agents for Software Development*：<https://arxiv.org/abs/2307.07924>
- Wang et al., *Mixture-of-Agents Enhances Large Language Model Capabilities*：<https://arxiv.org/abs/2406.04692>
- Yang et al., *SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering*：<https://arxiv.org/abs/2405.15793>
- Wang et al., *OSWorld: Benchmarking Multimodal Agents for Open-Ended Tasks in Real Computer Environments*：<https://arxiv.org/abs/2404.07972>
- Microsoft, *Magentic-One: A Generalist Multi-Agent System for Solving Complex Tasks*：<https://arxiv.org/abs/2411.04468>

## 结语

Agent 架构的关键不是给模型增加多少“人格”或写多长提示词，而是让系统在目标不确定、环境可变、工具复杂、风险不同的条件下，仍然能做到：**知道下一步为什么做、如何验证是否做对、失败后如何收敛、越权前如何停下、长期运行后如何保持状态一致。**

从这个视角看，ReAct、工作流、规划、搜索、记忆、反思和多 Agent 并不是互相替代的标签，而是可以按任务风险和验证能力组合的工程部件。最好的设计通常不是最自治、最复杂的设计，而是能用最小复杂度稳定完成任务、并能向人清楚解释其边界与证据的设计。
