---
layout: post
toc: true
title: "论文精读：LLMs Get Lost in Evolving User Intent"
categories: AI
tags: [AI, LLM, Agent, Paper, Benchmark]
author:
  - vortezwohl
excerpt: >-
  这篇论文研究的不是模型会不会做题，而是模型能否在多轮协作里持续跟住“用户现在真正想要什么”。作者没有重新手搓一个昂贵的多轮数据集，而是把已有单轮 benchmark 反向展开成可验证的 evolving-intent 对话：用户会逐步补条件、推翻旧条件，甚至中途切换任务。实验显示，许多在单轮上很强的模型，一旦进入这种动态协作环境就会明显“迷路”；最致命的并不是缺少上下文，而是任务切换后的信念状态更新失败。对工程实践来说，这篇论文最重要的启发不是再堆一点提示词，而是显式维护当前意图状态、把任务切换当成一级事件、用结构化 recap 替代被动回看整段 transcript，并建立专门覆盖 reveal、revision、switch 的多轮回归评测。全文的价值在于，它把“Agent 为什么老是越聊越偏”这件事，从抱怨变成了一个能系统构造、自动验证、稳定复现的问题。  
---

## 一、这篇论文到底在研究什么

这篇论文研究的核心问题，不是大语言模型会不会解题，也不是 Agent 会不会调工具，而是更具体的一件事：

> 当用户意图在多轮对话中不断演化时，模型能不能始终对齐“当前这一轮真正要完成的目标”？

现实里的协作任务几乎从来不是单轮静态请求。用户往往会一边看中间结果一边补充限制条件、修正前文说法，甚至在同一个会话里切到一个相关但不同的新任务。论文把这类现象统一称为 evolving user intent，并认为这是协作型 LLM/Agent 的关键能力边界之一。作者的基本判断是：现有主流 benchmark 大多仍是“问题一次给全、答案一次算完”的静态设定，因此会系统性高估真实协作场景中的模型能力。[^1]

这篇论文的目标不是提出一个新模型，而是提出一套可规模化、可自动验证、可以复用现有 benchmark 的评测构造方法，并用这套方法回答两个问题：

1. 单轮表现很强的模型，在多轮意图演化下会掉多少？
2. 模型到底更怕哪一类意图变化：补信息、改信息，还是换任务？[^1]

## 二、论文试图解决的难题是什么

这项研究要解决的难点，并不是“没人知道多轮对话更复杂”，而是如何把这个复杂性做成一个工程上可落地、可稳定评估的问题。

如果直接构造多轮数据集，会立刻遇到几类成本：

- 标注成本高：需要人工写多轮对话、维护状态一致性、给出每轮可验证答案；
- 分布不稳：不同标注者写出的对话风格、隐含假设、难度分布都可能飘；
- 验证困难：多轮任务常常依赖 LLM-as-judge，结果不够稳定；
- 可扩展性弱：很难快速把方法迁移到数学、SQL、搜索、代码修复等不同任务上。[^1]

论文的关键洞察是：

> 最昂贵的并不是“问题本身”，而是“可靠的 verifier”。

像 GSM8K、BIRD-SQL、BrowseComp+、SWE-Bench Verified 这类 benchmark 已经有现成样本、现成答案、现成评测器。与其重新造一个多轮 benchmark，不如把这些单轮样本“抬升”为多轮 evolving-intent 场景，同时保住原来的验证协议。只要最终一轮仍然收敛回原始单轮问题，整个系统就能继续沿用原 benchmark 的 verifier。[^1][^2]

这也是本文最有价值的方法论贡献：它没有把问题变成“另一个更花钱的数据工程”，而是把问题重写成“如何借已有 verifier 构造动态对话”。

## 三、作者如何形式化“意图演化”

论文把第 \(t\) 轮用户意图形式化为一个四元组：

\[
I_t = (f_t, C_t, C_t^{rev}, y_t)
\]

其中：

- \(f_t\)：当前任务或函数，例如“查餐馆”“生成 SQL”“修复 bug”；
- \(C_t\)：当前任务需要满足的参数或约束集合；
- \(C_t^{rev}\)：到目前为止已经向 Agent 披露过的约束子集；
- \(y_t\)：在当前任务定义下可验证的答案。[^1]

在这个定义上，作者给出三类最关键的意图转移：

### 1. Argument Reveal

用户意图本身不变，只是逐步补充条件。例如先说“帮我找餐馆”，下一轮再补一句“要纽约、要素食”。这类变化主要挑战模型的持续收集能力。[^1]

### 2. Argument Revision

用户前面已经说过的条件被后续消息推翻或修正。例如先说“纽约”，后来改成“布鲁克林”；先说“保留这个接口”，后来改成“这个接口可以删”。这类变化要求模型主动废弃旧假设，而不是在新旧条件上混合执行。[^1]

### 3. Function Switch

用户从当前任务切到另一个相关但不同的任务，但会继承部分上下文。例如先让系统“找餐馆”，接着说“那就帮我订这家”；或者先让 Agent 定位一个 bug，再要求它基于刚找到的模块顺手补测试。这类变化不是简单补信息，而是目标函数本身换了。论文后续最重要的实验发现之一，就是模型最容易死在这里。[^1]

这个建模的价值在于，它把“需求变了”从一句模糊抱怨，变成了可以被程序操纵、组合和验证的状态转移系统。

## 四、方法创新：如何把单轮 benchmark 变成多轮 evolving-intent benchmark

论文最聪明的地方，是没有正向手写多轮对话，而是从“最终正确答案已知”的单轮样本出发，反向构造历史。

### 1. 先把原始单轮样本锚定为最终真实意图

原始 benchmark 里的一条样本，天然就是一个“最后一轮已经说清楚全部要求”的终态。作者把它当作对话终点，再往前倒推需要出现哪些 reveal、revision、switch。这样一来，多轮对话再复杂，最终仍然会落回原 benchmark 定义好的目标。[^1]

### 2. Intent extraction：从单轮样本里抽取任务和参数

作者先利用 LLM 从原始单轮样本中抽取“任务函数”和“参数集合”，例如把一道数学题、一条 SQL 任务或一个 SWE 问题抽象成结构化意图表示。抽取结果会经过可执行约束与一致性检查，不合格样本会被 rejection sampling 掉。[^1][^2]

### 3. Counterfactual generation：构造“被修正”的旧条件

为了模拟 revision，系统会为原参数生成看起来合理、但最终会被推翻的 counterfactual 值。例如城市、限制条件、表字段偏好、修复路径等都可以先给一个历史版本，再在后续轮次修正。[^1][^2]

### 4. Predecessor generation：构造“前一个任务”

为了模拟 function switch，系统还会生成一个与终态任务共享上下文、但目标不同的 predecessor function。这个 predecessor 不是随便编一个不相干任务，而是必须与最终任务存在可继承上下文，否则就不能体现真实协作里“换任务但信息仍部分相关”的难点。[^1][^2]

### 5. Turn scheduling：先排状态变化，再渲染自然语言

论文没有直接让模型“一把写完整段对话”，而是先确定各轮发生什么状态跳转，再用规则模板与语言自然化模块把它渲染成用户话术。这种先计划、后自然化的设计，显著降低了多轮对话中状态非法、逻辑断裂、前后矛盾的问题。[^1][^2]

### 6. 保持原始 verifier 不变

因为最终一轮仍然锚定到原始任务，所以论文可以继续使用 GSM8K 的答案校验、BIRD-SQL 的执行验证、BrowseComp+ 的原评测协议，以及 SWE-Bench Verified 的既有验证流程。这个设计让多轮数据集不再依赖脆弱的主观评分，是整篇论文最硬的工程基础。[^1][^2]

## 五、开源项目到底提供了什么

这篇论文不仅给出方法，还放出了完整仓库 `microsoft/evolving-intent`。仓库更像一个 benchmark construction + evaluation framework，而不只是附几份脚本。[^2]

### 1. `intent_construction/`

这个目录对应论文中的 intent extraction、counterfactual generation、predecessor generation 流程。它解决的是“如何从单轮题目得到结构化意图和历史意图候选”的问题。[^2]

### 2. `situated_simulation/`

这个目录用于把结构化状态转移编排成多轮对话，包括 turn scheduler、rule-based renderer，以及可选的自然化步骤。换句话说，这部分负责“如何把状态机变成人类像样会说的话”。[^2]

### 3. `evaluation/`

这个目录负责真正跑实验，支持 GSM8K、BIRD-SQL、BrowseComp+、SWE-Bench Verified 等任务域，并保留原始 benchmark 的评测逻辑。SWE 子目录里还提供了 mini-swe-agent v2 相关流程，用于研究代码代理在多轮意图演化下的失真方式。[^2]

### 4. 仓库的定位

这个项目并不是简单地发布一份固定死的数据文件，而是给出一套“如何再生这类 benchmark”的过程性资产。仓库公开了所需目录、配置、任务子集和脚本接口，强调的是可复现的构造流程，而不是把所有生成中间结果都硬编码成一个静态语料包。[^2]

从工程角度看，这也更合理：因为 evolving-intent benchmark 的价值不只是这一次实验结果，而是未来可以被迁移、变体化、内化进团队回归评测。

## 六、实验到底发现了什么

### 1. 单轮很强，不代表多轮还能跟住用户意图

论文最直接的结论是：一旦进入多轮意图演化场景，模型表现会系统性下降，而且下降幅度不小。作者在四类任务上做了统一评测：GSM8K、BIRD-SQL、BrowseComp+、SWE-Bench Verified。[^1]

以论文中的结果为例，在“每种 transition 各出现两次”的设定下，GPT-5.5 这类强模型在多个任务上的表现均较单轮基线明显下降；其他模型在 SQL、搜索等任务上掉得更狠。论文借此说明：静态 benchmark 评到的主要是“在需求给全时求解”的能力，而不是“在需求持续变化时重建当前目标”的能力。[^1]

这个发现本身并不反直觉，但论文把它做成了跨任务域、同协议、可自动验证的定量结论，这点很重要。

### 2. Reveal 和 Revision 没那么致命，Function Switch 才是真正的深水区

论文做了逐轮 intent tracking 分析。方法很直接：在每轮结束时，让模型显式回答“当前用户意图是什么”，再用独立 judge 进行打分。结果显示：

- 对 Argument Reveal，模型几乎总能跟住；
- 对 Argument Revision，模型也大体能更新；
- 对 Function Switch，准确率下降最明显，而且随着 switch 次数增加持续恶化。[^1]

这意味着模型真正的病灶，不只是“没吸收到新信息”，而是“换任务以后没能正确重建新的信念状态”。工程上，这比“上下文窗口不够长”更值得警惕，因为它对应的是状态管理缺陷，而不是单纯容量缺陷。

### 3. Switch 之后继续叠加更新，模型更容易彻底迷路

论文进一步发现：如果只是刚发生 task switch，模型有时还能勉强跟住；但如果 switch 之后又继续 reveal 或 revise，新旧状态叠在一起，性能会进一步恶化。[^1]

这说明 function switch 的难点不仅是“识别用户换任务了”，而是：

- 哪些旧上下文仍然有用；
- 哪些旧约束已经作废；
- 新任务该继承什么；
- 新任务后续补充的条件应该覆盖哪里。

换句话说，真正难的是“上下文重构”，不是“上下文读取”。

### 4. 简单 recap 有帮助，但不足以回到单轮水平

作者试了两类缓解方式：

- Prompt recap：提示模型先回顾并整理当前上下文；
- Oracle recap：直接提供当前正确意图摘要。[^1]

结果两者都能提升表现，尤其在 function switch 上更明显；但即使给了 oracle 级 recap，模型也通常回不到原始单轮基线。这个结果的含义很强：

> 问题不只是“模型没看见”，而是“模型在冲突或残留上下文里无法稳定执行当前意图”。

也就是说，单纯让模型“多看一遍历史”不是解药；更重要的是，把当前意图从混杂 transcript 中提纯成结构化工作状态。

### 5. 任务越难，多轮动态带来的惩罚越大

作者在 BIRD-SQL 上用 hint 控制题目难度后发现：难题在多轮演化环境下受到的额外惩罚，显著大于单轮环境。[^1][^6]

这个结论很贴近真实产品：真正会交给 Agent 的，常常不是玩具任务，而是原本就复杂、需要外部知识、依赖上下文累积的难题。也就是说，多轮动态会放大真实难题的失败率，而不是只在学术玩具上“略掉几分”。

### 6. 在代码代理场景里，更多工具预算不一定带来更高成功率

SWE-Bench Verified 实验还有一个很实用的发现：在多轮动态场景里，模型往往把大量预算消耗在搜索、浏览、探索上，而不是稳定地进入执行阶段。[^1][^2][^5]

这说明对于代码代理而言，问题不是“不会搜”，而是搜到一半已经不确定现在到底该为哪个目标服务，结果就是：

- 工具调用变多；
- 搜索行为变多；
- 真正落到有效修改和验证上的动作变少；
- 整体路径越来越像“不断重定位”，而不是“沿清晰意图推进”。

这对做 coding agent 的团队很关键：如果系统在多轮需求变化后开始疯狂 `grep/find/read/search`，那不一定是它更认真，很可能是它已经丢了当前任务定义。

## 七、与相关工作的关系

这篇论文并不是凭空出现的，它站在几类相关工作之上，但推进方向相当明确。

### 1. 它直接继承并扩展了《LLMs Get Lost In Multi-Turn Conversation》

前作《LLMs Get Lost In Multi-Turn Conversation》已经指出：把原始完整指令拆成多轮对话后，模型会因为早期假设、信息披露顺序和记忆偏置而显著退化。[^3]

但那篇工作更集中在“逐步给信息”这一类多轮困难。相比之下，本文的推进主要有三点：

- 把问题从 underspecification 扩展到 revision 和 function switch；
- 把多轮掉点从“现象描述”推进为“按 transition 类型定位病灶”；
- 把评测统一扩到数学、SQL、搜索、代码代理等多个任务域，并尽量保留原始 verifier。[^1][^3]

可以说，这篇论文不是推翻前作，而是把“模型会在多轮里迷路”这件事做成了更接近真实协作的系统研究。

### 2. 《Intent Mismatch Causes LLMs to Get Lost in Multi-Turn Conversation》更像是对根因的解释

另一条相关工作《Intent Mismatch Causes LLMs to Get Lost in Multi-Turn Conversation》认为，多轮失败的根因未必只是模型能力不足，更可能是用户表达、模型理解和执行状态之间发生了 pragmatic mismatch。该工作提出通过中介层先显式化意图，再把结果交给下游执行模型。[^4]

这和当前论文并不冲突，反而互相支持：

- 当前论文把现象测清楚了：尤其是 function switch 最伤；
- Intent mismatch 方向则给出一种解释：模型没有稳定维持“当前意图的显式表示”。[^1][^4]

如果把两者合在一起看，工程结论会更清晰：真正值得投资源的，不是再堆一点全文提示词，而是做 intent mediation、state extraction、recap 和 switch-aware orchestration。

### 3. 与 τ²-Bench 这类环境型 benchmark 互补

τ²-Bench 关注的是对话代理与用户在共享环境中的协作控制问题，更强调“用户和 agent 共同作用于同一环境”时的策略与协调。[^7]

而本文更关注另一条轴：用户意图本身在对话中演化时，模型是否能稳定更新信念状态。两者的关系更像互补：

- τ²-Bench 偏环境协作；
- 本文偏意图演化与状态更新。

如果一个团队真在做通用 Agent，这两类 benchmark 都应该看：前者帮助发现环境协调问题，后者帮助发现意图漂移问题。

## 八、工程实践上最值得落地的经验

论文本身是 benchmark 研究，但它对工程实践的启发相当直接。

### 1. 不要把“当前意图”只埋在原始 transcript 里

如果系统完全依赖模型从整段聊天记录里自行推断“现在到底该做什么”，那么一旦出现 revision 或 switch，就很容易把旧目标、旧限制、过期假设混进当前计划。论文已经证明，这类错误不是偶发现象，而是系统性弱点。[^1]

更合理的做法是维护一份显式状态，例如：

- `current_goal`
- `active_constraints`
- `superseded_constraints`
- `inherited_context`
- `open_questions`
- `latest_user_decision`

这份状态应该独立于原始 transcript，被规划器和执行器直接消费。

### 2. 把 task switch 当成一级事件处理

Function Switch 是最伤模型的一类变化，因此工程上绝对不能把它当成普通 follow-up message。

更稳的做法应该是：

- 检测是否发生 goal switch；
- 明确哪些上下文要继承、哪些要失效；
- 清空或重建旧计划；
- 重新估算工具预算与验证路径；
- 在执行前显式确认当前目标描述。[^1]

如果系统没有这套 switch-aware 逻辑，就会出现典型问题：还在用旧任务的观察结果和旧假设，为新任务继续执行。

### 3. recap 要结构化，而不是让模型“自己回顾一下”

论文里 prompt recap 有帮助，但 oracle recap 更有效，这说明 recap 不是可有可无的心理安慰，而是应该被工程化成明确的数据结构。[^1]

建议 recap 至少包含：

- 当前目标是什么；
- 哪些条件仍然有效；
- 哪些条件已被修正；
- 当前目标继承了哪些历史上下文；
- 还有哪些缺失信息需要补充。

也就是说，真正有价值的 recap 不是“请结合上下文重新思考”，而是“这是经状态整理后的当前任务定义，请据此继续”。

### 4. 长上下文不是解药，脏上下文反而会变成毒药

很多团队遇到多轮错乱，第一反应是扩大上下文窗口、保留更多 tool traces、塞入更长的工作记忆。但这篇论文的多项分析都说明，如果系统没有选择性保留与选择性遗忘机制，更多上下文只会带来更多竞争注意力。[^1]

特别是在 function switch 后，旧任务的残留信息越多，模型越可能被锚定在错误目标上。

### 5. 对代码代理，要单独统计“探索”和“执行”

SWE 场景提示我们：多轮动态问题经常表现为“工具调用看上去很多，但有效执行很少”。因此，做 coding agent 时应显式监控：

- 搜索类调用占比；
- 编辑类调用占比；
- 验证类调用占比；
- 每次 switch 后到第一次有效执行之间的延迟；
- revision 后是否仍沿用旧计划。[^1][^2][^5]

只有这样，团队才能区分“模型在认真搜”和“模型已经失去当前意图，只能反复重定位”。

### 6. 回归测试必须覆盖 reveal / revision / switch 三类多轮模式

如果团队只测单轮最终答案，或者只测 reveal 型多轮补条件，那基本测不出最危险的问题。更合理的最小回归集合至少应该包含：

- 逐步补条件；
- 中途改条件；
- 任务切换；
- switch 后再继续 reveal/revision；
- 执行前 recap 与执行后行为的一致性检查。[^1]

否则系统在真实用户手里很容易出现“第一轮很聪明，第三轮已经偏题”的典型故障。

### 7. 如果要做后训练，多轮纠错数据比更多静态题更值钱

这篇论文虽然不直接做训练，但它暗示了一个明确方向：如果目标是提升协作型 Agent，而不是单轮答题机，那么比起继续堆静态单轮题，更值得补的是：

- 用户中途补充条件的数据；
- 用户修正旧条件的数据；
- 任务切换后的状态重建数据；
- 基于历史对话提炼“当前意图摘要”的监督数据。[^1]

因为系统失败的主要位置，不在“求解器完全不会做题”，而在“拿着旧题设去做新题”。

## 九、这篇论文的边界与局限

这篇工作非常强，但它不是完整复刻真实世界对话分布。它至少有以下边界：

### 1. 用户语言仍偏规整

尽管作者做了自然化，但这类 benchmark 的用户表达依然比真实产品中的口语、省略、错字、情绪化输入规整得多。现实世界里的切换往往没有那么干净，甚至不会显式说“改一下”。[^1]

### 2. 中间轮的验证不如终态验证硬

论文最强的地方是保住了最终一轮的原始 verifier，但中间轮状态的一致性仍然包含 LLM 辅助抽取与筛选步骤，因此并不是每个中间状态都像终态那样硬验证。[^1]

### 3. 单轮锚定策略天然偏向“最终仍可回收为一个原任务”

这种构造方法非常适合已有 benchmark 的再利用，但它也意味着：真实世界那种任务边界模糊、目标会持续扩展、最后根本不收敛为原问题的复杂协作过程，并没有被完全覆盖。[^1]

### 4. 它更像评测放大镜，而不是完整解决方案

这篇论文把问题揭示得非常清楚，但并没有给出一个已经被充分证明有效的通用系统设计。它更像一面高分辨率镜子：帮你看见 Agent 真实会死在哪里。真正的工程修复，还需要结合 state management、recap、plan reset、tool budget control 等系统设计。

## 十、我的总体判断

如果只把这篇论文当成“又一个 benchmark paper”，会低估它的价值。它真正重要的地方在于：

1. 它把“Agent 为什么越聊越偏”从经验抱怨变成了可构造、可验证、可复现的问题；[^1]
2. 它证明最危险的意图变化不是单纯补信息，而是 task switch 及其后续连锁更新；[^1]
3. 它说明简单 memory / recap 虽然有帮助，但不能替代显式的意图状态管理；[^1]
4. 它给出了一个非常值得工程团队内化的方法论：不要只盯单轮 benchmark，要建立自己的 evolving-intent regression suite。[^1][^2]

如果把这篇论文落到一句工程建议上，那就是：

> 协作型 Agent 的核心不只是“能不能算”，而是“能不能在需求变化时持续维护正确的当前任务定义”。

这件事做不好，再大的上下文、再多的工具、再强的单轮基座模型，都会在真实多轮交互里被放大成稳定失误。

## 参考文献

[^1]: Prasann Singhal, Yutong Deng, Abid Hussain, Yanhao Zhang, Tianyu Lu, Amos Azaria, Erik Hemberg, Daniel Su, Yue Wang, et al. *LLMs Get Lost in Evolving User Intent*. arXiv:2607.20734, 2026. [https://arxiv.org/pdf/2607.20734](https://arxiv.org/pdf/2607.20734)

[^2]: Microsoft. *evolving-intent*（官方代码仓库）. [https://github.com/microsoft/evolving-intent](https://github.com/microsoft/evolving-intent)

[^3]: Tianyu Lu, Yutong Deng, Jianian Zhang, et al. *LLMs Get Lost In Multi-Turn Conversation*. arXiv:2505.06120, 2025. [https://arxiv.org/abs/2505.06120](https://arxiv.org/abs/2505.06120)

[^4]: Shalev Lifshitz, Asaf Lumer, Or Honovich, et al. *Intent Mismatch Causes LLMs to Get Lost in Multi-Turn Conversation*. arXiv:2602.07338, 2026. [https://arxiv.org/abs/2602.07338](https://arxiv.org/abs/2602.07338)

[^5]: John Yang, Carlos E. Jimenez, Alexander Wettig, et al. *SWE-Bench Verified: Can Models Fix Real-World Python Bugs at Scale?* arXiv, 2024. [https://arxiv.org/abs/2408.06292](https://arxiv.org/abs/2408.06292)

[^6]: Wenhu Chen, Hexiang Hu, Lingfan Yu, et al. *BIRD: A Trustworthy Benchmark for Text-to-SQL in Realistic Big Databases*. NeurIPS Datasets and Benchmarks, 2023. [https://arxiv.org/abs/2305.03111](https://arxiv.org/abs/2305.03111)

[^7]: Ziyi Zhu, Ofir Press, Yi Zhang, et al. *τ²-Bench: Evaluating Conversational Agents in a Dual-Control Environment*. arXiv:2506.07982, 2025. [https://arxiv.org/abs/2506.07982](https://arxiv.org/abs/2506.07982)

[^8]: Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, et al. *Training Verifiers to Solve Math Word Problems*. arXiv:2110.14168, 2021. [https://arxiv.org/abs/2110.14168](https://arxiv.org/abs/2110.14168)
