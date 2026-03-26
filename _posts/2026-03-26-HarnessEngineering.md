---
layout: post
toc: true
title: "Harness Engineering: 反馈闭环和熵治理, 构建可控的 LLM Agent"
categories: Agent
tags: [LLM, agent, agentic-ai, agi, ai]
author:
  - vortezwohl
  - 吴子豪
---
Harness Engineering（驾驭工程）是 OpenAI 在 2026 年 2 月正式提出的面向 AI Agent 时代的新型软件工程方法论$^{[1]}$，"Harness"本意是马具 (如缰绳、马鞍)，把马的力气引到正确方向上。而 LLM 就像一匹蛮力十足但方向感不太行的马，跑得快但容易跑偏。其核心理念为 "Humans Steer, Agents Execute"（人类掌舵，智能体执行）。它彻底重构软件开发流程，将工程师角色从代码编写者转变为环境设计师、意图规范者和反馈回路构建者，通过精密的 Harness 引导 AI 智能体自主、可靠地完成软件工程任务。

> harness [ˈhɑːnɪs] n.马具 v.利用

## 什么是 Harness Engineering

Harness Engineering 并不是凭空出现的，它是 Prompt Engineering 和 Context Engineering 的自然延伸。三者构成嵌套关系：

![alt text](/images/HarnessEngineering/image.png)

Harness Engineering 指为 AI 智能体搭建包含明确约束、可用工具链、自动验证标准和反馈闭环的独立运行环境$^{[2]}$，使 Agent 在无人值守情况下依然能自主、高质量地完成开发任务。它不是 AI 模型本身，而是围绕模型的一整套控制系统，如同马具之于骏马 —— 既释放其全部潜力，又确保方向可控。

- **核心理念:** 人类掌舵，Agent 划桨

    工程师工作重心从编写代码转向三大核心任务：

    - **设计环境**: 构建完整的开发生态系统，包括代码仓库结构、工具链、权限边界和执行沙箱.

    - **明确意图**: 通过声明式提示（Declarative Prompts）而非命令式指令，定义系统目标与验收标准.

    - **构建反馈闭环**: 建立自动化的检测 - 修复机制，使 Agent 能自我纠错、持续改进.

## 为什么需要 Harness Engineering

### LLM 模型能力不是瓶颈

- **Can.ac 实验:** 仅改变 Harness 的工具格式（编辑接口），就在 16 个模型上显著提升了编码基准分数。效果最显著的 Grok Code Fast 1 从 6.7% 跃升至 68.3%——没有修改任何模型权重.

- **LangChain 实验:** 仅通过 Harness 改进，在 Terminal Bench 2.0 上从第 30 名跃升至第 5 名，同一模型提升了 13.7 分。

这些结果表明，在纠结模型选择之前，先审视 Harness 设计能获得更高的 ROI。

**OpenAI 团队说得很直接:** 真正卡你的不是 Agent 写代码的能力，而是围绕它的结构、工具和反馈机制跟不上。基础设施才是瓶颈，而非智能水平。

### Code Agent 的典型失败模式

Anthropic 在做长时间运行 Code Agent 的过程中，总结了 Agent 常见的失败: 

1. **One-shotting (试图一步到位):** Agent 倾向于一次做完所有事情，结果在实现进行到一半时上下文窗口就耗尽了。下一个会话启动时看到的是半成品、没有文档的代码，只能花大量时间猜测之前发生了什么并试图恢复工作状态。

2. **过早宣告胜利:** 在项目后期，当部分功能已经完成后，Agent 会环顾四周，看到已有进展就直接宣布任务完成——即使还有大量功能未实现。

3. **过早标记功能完成:** 在没有明确提示的情况下，Agent 写完代码就标记为“完成”，却没有做端到端测试。单元测试或 curl 命令通过了不代表功能真正可用。

4. **环境启动困难:** 每次新会话启动时，Agent 需要花费大量 token 弄清楚如何运行应用、如何启动开发服务器，而不是把时间花在实际开发上。

### 有效上下文窗口远比想象中小

Dex Horthy 有个很实用的经验观察$^{[6]}$：上下文填得越满，LLM 输出质量越差。

以 168K token 的上下文窗口为例，大约用到 40% 就开始走下坡路了:

**Smart Zone (上下文窗口利用率低于 40%):** 聚焦、准确的推理。Agent 拥有相关、精炼的信息。

**Dumb Zone (上下文窗口利用率超过 40%):** 幻觉、循环、格式错误的工具调用、低质量代码。更多 token 反而损害性能。

## Harness Engineering 方法论

Harness Engineering 是为 LLM Agent 做 “可控长时执行” 的系统工程，核心公式：

$$
Agent = LLM + Harness
$$

Harness 则是 `任务调度` + `上下文治理` + `评估反馈` + `状态持久化` + `工具链`.

综合 OpenAI、Anthropic、Carlini、Huntley、Horthy 等五个独立团队的实践，四种设计反复出现并形成收敛: 即 `上下文治理` `Agent 专业化` `状态持久化` `结构化执行`

- `上下文治理`: Agent 应该**恰好获得当前任务所需的上下文**, 不多不少, 使用分层上下文和渐进式披露.

- `Agent 专业化`: **专注于特定领域**, 拥有受限工具的 Agent 优于拥有全部权限的通用 Agent.

- `状态持久化`: 进度持久化在文件系统上, 而非上下文窗口中, **记忆独立于会话本身**.

- `结构化执行`: **将思考与执行分离**, 研究和规划在受控阶段进行, 验证通过自动化反馈完成.

    研究和规划在受控阶段进行，执行基于验证过的计划，验证通过自动化反馈（测试、Linter、CI）和人类审查完成。所有团队都施加了刻意的执行序列：`理解 → 规划 → 执行 → 验证`。

Harness 的设计务必遵循以下五大原则:

1. **设计环境, 而非编写代码:** 工程师的工作转向为 Agent 准备高效运行的环境。当 Agent 卡住时，不是"更加努力"，而是诊断"缺少什么能力"并让 Agent 自己构建该能力。

2. **机械化地执行架构约束:** 实验团队为每个领域定义依赖方向——Types → Config → Repo → Service → Runtime → UI——并用自定义 Linter 和结构测试自动检测违规。文档中记录是不够的；如果不能机械化地强制执行，Agent 就会偏离。

3. **将代码仓库作为唯一事实源:** 写在 Slack 讨论或 Google Docs 中的知识对 Agent 来说等于不存在。所有团队知识都作为版本控制的制品放置在仓库中。

4. **将可观测性连接到 Agent:** 实验团队将 Chrome DevTools 连接到运行时，使 Agent 能够捕获 DOM 快照和截图。通过赋予查询日志和指标的能力，"将启动时间降至 800ms 以下"变成了可度量的目标。

5. **熵治理:** 最初团队每周五花 20% 的时间手动清理"AI Slop"（低质量生成物）。这后来被自动化为 Codex 运行的后台任务——清理吞吐量与代码生成吞吐量成正比扩展。


## 参考文献

[[1](https://openai.com/index/harness-engineering/)] OpenAI Research Team. Harness engineering: leveraging Codex in an agent-first world. *Openai official website*, 2026.

[[2](https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents)] Anthropic Research Team. Effective harnesses for long-running agents. *Anthropic official website*, 2025.

[[3](https://www.anthropic.com/engineering/building-c-compiler)] Nicholas Carlini. Building a C Compiler with Claude. *Anthropic official website*, 2026.

[[4](https://martinfowler.com/articles/exploring-gen-ai/harness-engineering.html)] Birgitta Böckeler. Harness Engineering. *martinfowler.com*, 2026.

[[5](https://martinfowler.com/articles/exploring-gen-ai/context-engineering-coding-agents.html)] Birgitta Böckeler. Context Engineering for Coding Agents. *martinfowler.com*, 2026.

[[6](https://github.com/humanlayer/12-factor-agents)] Dex Horthy. 12 Factor Agents: Build Reliable LLM Applications. *Github*, 2025.
