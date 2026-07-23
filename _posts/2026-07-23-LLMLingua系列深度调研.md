---
layout: post
toc: true
title: "LLMLingua 深度调研：从困惑度压缩到长上下文、蒸馏分类与越狱意图提取"
categories: AI
tags: [LLM, Prompt Compression, RAG, Long Context, Security, LLMLingua]
author:
  - vortezwohl
  - 吴子豪
excerpt: "LLMLingua 不是单一的删词工具，而是一条持续演化的提示词压缩研究线：初代以小型因果语言模型的困惑度作为信息重要性近似，通过预算控制、示例筛选和迭代 token 压缩降低黑盒大模型的输入成本；LongLLMLingua 再将问题相关性、动态预算和文档重排加入长上下文 RAG，试图同时降低噪声与缓解 lost in the middle；LLMLingua-2 则以 GPT-4 蒸馏的抽取式数据训练双向编码器，把压缩改写为保留/删除分类问题；SecurityLingua 将同一机制转用于从越狱包装中抽取真实意图，并通过 system prompt 激活目标模型既有的安全护栏。本文逐篇对照论文和官方实现，解释其研究背景、数学目标、关键算法、代码路径、实验结果、可落地边界与尚未解决的局限。"
---

> 本文基于截至 **2026 年 7 月 23 日**可公开访问的论文、Microsoft/LLMLingua 仓库[^6] `main` 分支和随附训练脚本撰写。文中“论文结果”均指作者在特定模型、数据集、压缩率和硬件条件下的报告值，不应直接外推为任意生产输入上的无损承诺。

## 先看结论

LLMLingua 的关键价值不在于把自然语言“写短”，而在于把 prompt 视为可以删除冗余 token 的**抽取式传输表示**。它不修改目标 LLM 的权重，因而适合 API 形式的黑盒模型；代价是压缩器自身仍要做推理，且删除后的文本往往不再适合人类阅读。

这条研究线可以概括为四个阶段：

1. **LLMLingua（EMNLP 2023）**：使用小型因果语言模型的 PPL 估计 token 重要性，以粗到细流程压缩 CoT、ICL、对话与摘要 prompt。
2. **LongLLMLingua（ACL 2024）**：面向 RAG 和长上下文，引入 query-aware 文档评分、对比困惑度、动态文档预算和重排，目标是让关键事实更密集、更靠前或靠后。
3. **LLMLingua-2（ACL Findings 2024）**：不再把单向 PPL 当作训练免费但天然正确的信号；利用 GPT-4 蒸馏数据训练双向 Transformer token classifier，显著减小压缩器延迟。
4. **SecurityLingua（CoLM 2025）**：将“保留什么”从语义信息重要性改为真实意图，提取越狱 prompt 内被角色扮演、编码或噪声掩盖的恶意请求，并把结果作为 system prompt 的安全提示。

实际选型应当区分场景：离线或可复用的通用文本压缩优先考虑 LLMLingua-2；问题依赖的多文档 QA/RAG 需要 LongLLMLingua 的 query-aware 粗粒度筛选；初代 LLMLingua 更适合作为可解释的 PPL 基线；SecurityLingua 是 guardrail 的前置增强器，而不是独立的安全判定器。

## 一、研究背景：为什么要压缩 prompt

### 1. 长 prompt 是能力手段，也是成本和性能问题

CoT、few-shot ICL、RAG、对话历史、代码仓库上下文和 Agent 工具轨迹都在拉长输入。输入 token 会增加 API 账单与 prefill 延迟；对于自部署 Transformer，注意力和 KV cache 的存储、访问也会加重。更长并不总是更好：无关检索片段会分散注意力，关键证据放在中间时，模型常出现 [lost in the middle](https://arxiv.org/abs/2307.03172) 式位置偏差。

量化、模型剪枝、蒸馏、KV cache 优化和高效注意力属于“改模型或运行时”的路线，但 API 用户通常没有权重和缓存访问权。提示词压缩把优化位置前移到请求发送之前：只要能决定原 prompt 中哪些 token 必须留下，就能同时减少费用、延迟和上下文窗口压力。

### 2. 同行工作与 LLMLingua 的定位

LLMLingua 的直接前身是 Selective-Context[^5]。它使用小语言模型的自信息或 PPL，在句子、短语或 token 层面删除低信息单元。LLMLingua 继承“自然语言冗余、语言模型可以充当压缩器”的假设，但指出两个缺陷：一次性估计 token 分数忽略了前文已经删掉后的条件分布；压缩小模型的偏好未必与实际调用的目标 LLM 一致。

相邻方案可以分为四类：

- **示例/文档选择**：动态 ICL、BM25、dense retrieval、reranker 直接选择少量 demonstration 或文档。它保留了句子完整性，但容易丢掉被选择文档中的局部证据。
- **生成式摘要和记忆**：用另一个 LLM 总结历史或上下文。文本更可读，却要多一次昂贵调用，还可能改写、幻觉或遗漏 QA 所需的精确实体。
- **软提示与 prompt tuning**：把长输入压成少量连续向量或特殊 token。效率很高，但通常绑定模型和任务，且经常要求训练或访问目标模型。
- **模型内部优化**：量化、稀疏注意力、KV eviction/cache compression。它们与文本级压缩互补，但不适用于闭源 API 的通用部署。

SecurityLingua 的相关工作则来自 LLM jailbreak 防御：安全微调、PPL filter、[SmoothLLM](https://arxiv.org/abs/2310.03684)、Erase-and-check、安全解码、意图分析和多 Agent 审核。它们常见的矛盾是防御效果、额外采样/调用成本、误拒正常请求和用户体验之间的取舍。

## 二、统一问题：保留输出行为，同时缩短输入

将原 prompt 写成 $x$，压缩 prompt 写成 $\tilde{x}$，目标 LLM 分别生成 $y$ 和 $\tilde{y}$。初代论文将目标写成：在尽量减小 $\lVert\tilde{x}\rVert$ 的同时，使 $P(y \mid x)$ 与 $P(\tilde{y} \mid \tilde{x})$ 的距离尽可能小，原文以 KL divergence 描述这种行为保持。

压缩率容易产生术语混乱。论文中的 compression rate 通常是 $\tau = \|\tilde{x}\| / \|x\|$，而 README 与函数输出常把 `ratio` 写成原长度除以压缩后长度，即 $1 / \tau$。因此“4x 压缩”表示压缩后大约只剩四分之一 token，不是保留四倍 token。

这里的“faithful”也有两层含义：

- 对下游任务 faithful：目标模型答案与原 prompt 时尽量一致。
- 对原文本 faithful：压缩文本只由原 token 的子序列组成，不新增事实。

初代优先追求前者；LLMLingua-2 和 SecurityLingua 明确把后者放入数据与模型设计中。

## 三、LLMLingua：粗到细的困惑度压缩

原始论文是 LLMLingua: Compressing Prompts for Accelerated Inference of Large Language Models[^1]。它把 prompt 划分为 instruction、demonstrations 和 question 三类组件，并依次执行预算控制、迭代 token 压缩与分布对齐。

### 1. 预算控制器：先保结构，再删 token

instruction 与 question 直接决定输出，而多个 few-shot demonstration 往往存在冗余，因此不能对全部 token 使用同一压缩率。给定总预算，算法先为 instruction 和 question 预留较高保留率，再由长度守恒推导 demonstration 的可用预算。

随后用小语言模型计算每个 demonstration 的 PPL，按高 PPL 优先加入，直到达到 demonstration 预算。这里的直觉是：高 PPL 内容对该模型更不容易预测，因而更可能承载有区分度的信息。极端压缩时，示例级选择也比把所有示例都削成残句更能保留推理结构。

论文在实验中将 instruction、question 的预设保留率分别设为 0.85 和 0.9，并以 granular control coefficient 调节示例层预算。这个设计不是通用最优值，而是“先保护任务定义和待答问题”的工程先验。

### 2. ITPC：用已压缩前缀重新估计后续 token

朴素方法会在完整原文上一次性给每个 token 打分，然后删除低 PPL token。这隐含了“删除不改变之后 token 重要性”的假设。LLMLingua 的 Iterative Token-level Prompt Compression（ITPC）将文本切为若干段：

```text
原 prompt -> 预算控制后的 prompt
          -> segment 1 压缩 -> 得到 compressed segment 1
          -> segment 2 在 compressed segment 1 条件下重新评分
          -> ... -> 拼接压缩结果
```

对第 $j$ 段，模型用前面已经压缩的段作为上下文，计算本段 token 的条件概率，并依据当前段的预算从 PPL 分布中得到阈值。高于阈值的 token 被保留。这不能消除所有跨段依赖，却比静态全局分数更接近最终压缩文本上的条件分布。

### 3. 分布对齐：小压缩器与大目标模型并非同一个模型

用 GPT-2 或 LLaMA-7B 给 token 打 PPL，但实际接收 prompt 的可能是 GPT-3.5、Claude 或其他黑盒模型。两者预测分布存在差异。论文使用目标 LLM 生成的 Alpaca 指令响应来 instruction-tune 小模型，尝试让压缩器对“目标模型会在意什么”更敏感。

这一步是原始框架中最现实也最脆弱的部分：它需要针对目标模型取得代表性数据；目标模型升级、任务迁移或语言变化时，原有对齐可能失效。

### 4. 初代实验结论

作者在 GSM8K、BBH、ShareGPT、Arxiv-March23 上，用 GPT-3.5-Turbo-0301 和 Claude-v1.3 作为目标 LLM。代表性结果包括：

- GSM8K 的 quarter-shot 约 20x 压缩下，LLMLingua EM 为 77.33，原 full-shot 为 78.85；同一条件下 Selective-Context 为 44.20。
- GSM8K 的 half-shot 条件下，LLMLingua 使用约 14x 压缩仍得到 77.41 EM。
- ShareGPT、Arxiv-March23 的 BLEU、ROUGE、BERTScore 表明，它在相同 token 约束下通常优于句子选择和 Selective-Context。
- 论文据当时 GPT-3.5 价格估算，在四个数据集上都有输入/输出费用下降。

这些结果说明“难读的 token 子序列”仍可能足以触发大模型恢复推理链，而不是证明 20x 对所有事实抽取、代码或多语言任务都安全。

## 四、LongLLMLingua：把压缩改造成 query-aware 长上下文优化

LongLLMLingua[^2] 的目标不只是减少 token，而是提高 prompt 中**与当前问题相关的信息密度**，并对抗关键证据落在中间位置时的性能衰减。它在初代框架上增加四个部分。

### 1. 问题感知的文档级筛选

对每篇候选文档 $d_k$，它不直接使用 $P(d_k \mid q)$，而是计算问题及限制语句在给定文档后的条件 PPL：

$$
r_k = -\frac{1}{N_c}\sum_i \log P(q_i + \text{restrict} \mid d_k)
$$

直觉是：真正相关的文档应让“问题以及只能在给定材料中回答”的限制语句更容易预测。限制语句用于抑制小模型仅凭自身常识回答问题所造成的错误相关性。按 $r_k$ 排名后，只将高分文档送入细粒度压缩。

这与 BM25、embedding 相似度不同：后者比较 query 与文档表面或向量相似性，LongLLMLingua 试图以小语言模型的条件生成能力衡量“文档能否支持这个问题”。

### 2. 问题感知的 token 级对比困惑度

仅把 question 放在文档前面并不可靠，因为相关 token 在已知 question 后可能反而变得低 PPL，难以和冗余 token 拉开距离。因此论文定义：

$$
s_i = \operatorname{PPL}(x_i \mid x_{<i}) - \operatorname{PPL}(x_i \mid q, x_{<i})
$$

该值表示 question 的加入使 token 预测分布发生多大变化。论文将它推导为条件 pointwise mutual information 的等价形式；高分 token 往往聚集在回答问题所需的事实附近。细粒度阶段据此替代普通 PPL 分数。

### 3. 动态预算与文档重排

文档级相关性既用于“留不留”，也用于“留多少”。高相关文档得到更高 token 预算，低相关文档被更激进压缩。随后按重要性重排文档，将关键内容移到模型更容易利用的位置，从而缓解 lost in the middle。

重排不是自然语言语义的无害操作：对于严格时间线、程序执行顺序、法律条款顺序，文档整体重排可能改变阅读语境。论文在 LooGLE 的时间线任务上报告仍有收益，但工程中仍应按任务语义决定是否允许重排。

### 4. 子序列恢复

token 删除会损坏人名、地名、数字和组织名。模型生成答案后，算法寻找回答中与压缩 prompt 匹配的最长子串，再映射回原 prompt 中相应的最短连续子序列，以恢复原始拼写。例如压缩结果中的截断人名可以在最终答案里被扩展回原文全名。

它只能恢复源自 prompt 的 copied span，不能修复模型没有使用的事实、错误推理或原文没有出现的新信息。

### 5. LongLLMLingua 的实验结论

论文以 GPT-3.5-Turbo-0613、LongChat-13B-16k 为目标模型，LLaMA-2-7B-Chat 为压缩器，在 NaturalQuestions、LongBench、ZeroSCROLLS、MuSiQue、LooGLE 上评估。

- NaturalQuestions 中，作者报告 GPT-3.5-Turbo 以约四分之一 token 获得最高 21.4% 性能提升；表中 4x 设置的重排结果为 75.5，对应原 prompt 的 63.1。
- 约 10k token prompt、2x 到 6x 压缩时，报告端到端加速为 1.4x 到 2.6x。这里已经包含压缩器开销，因而比只比较目标 LLM prefill 更有意义。
- LooGLE 上报告 94.0% 成本下降；LongBench 消融显示移除 query-aware 文档筛选、细粒度对比 PPL、动态预算、重排或子序列恢复都会降低结果。
- 但收益在文档 QA 与 synthetic task 最明显，摘要和部分代码/顺序任务的提升更有限，不能把“压缩后可能更好”泛化为所有长上下文问题。

## 五、LLMLingua-2：从困惑度启发式到蒸馏式 token 分类

LLMLingua-2[^3] 的核心判断是：因果小模型的单向 PPL 既没有直接学习“压缩后仍完成任务”的目标，也看不到 token 两侧的完整上下文。因此采用“生成教师数据，再训练抽取式学生模型”的方案。

### 1. 数据蒸馏：强约束 GPT-4 只删除，不改写

作者用 MeetingBank 的会议记录作为原文，让 GPT-4 按以下约束压缩：只删除不重要词；不重排；不改词；不使用缩写或 emoji；不添加新词或符号。目标不是写出优美摘要，而是让 GPT-4 自己能从压缩文本尽可能重构原文。

长文先切成不超过 512 token、尽量以句号结尾的块，再分别压缩。固定全局压缩率没有被强制给教师模型，因为不同句子的信息密度不同；教师可以对高密度句子少删，对寒暄、重复发言多删。

### 2. 自动标注与质量控制

教师即使被约束也可能改词形、重排或引入词。论文把压缩文本的每个词对齐回原文，采用词形还原和以此前匹配位置为中心的双向滑动窗口；被匹配到的原词标为 `preserve`，其他为 `discard`。

两项质量指标用于滤除坏样本：

- **Variation Rate（VR）**：压缩文本中不在原文出现的词占比；高值意味着教师可能幻觉或改写。
- **Alignment Gap（AG）**：Hitting Rate 与 Matching Rate 之差；理想对齐为零，高值说明“压缩词看似在原文中，但自动标签没有准确匹配”。

作者过滤 VR 最高 5% 和 AG 最高 10% 的样本。这个步骤很关键：LLMLingua-2 的“faithful”承诺依赖训练标签质量，而标签并非人工金标。

### 3. 双向编码器与保留策略

输入 $x = \{x_i\}_{i=1}^N$ 经 Transformer encoder 得到 $h_i$，线性层和 softmax 输出每个 token 的两类概率：

$$
h = f_\theta(x), \qquad p(x_i) = \operatorname{softmax}(W h_i + b)
$$

训练采用 token 级交叉熵。推理时取每个词被预测为 `preserve` 的概率，保留概率最高的 $\tau N$ 个词并保持原顺序。主体模型为 XLM-RoBERTa-large（355M），小模型为 multilingual BERT（110M）。

这个设计有三个直接收益：双向上下文可以判断一个词在整句中的作用；分类比 7B 级因果语言模型计算 PPL 更快；只保留原文词使其不会凭空添加新事实。它仍不能保证所有语法依赖、数值关系或跨块依赖被保留。

### 4. 实验结论与边界

论文在 MeetingBank、LongBench、ZeroSCROLLS、GSM8K、BBH 上测试，并报告：

- MeetingBank 中约 3.1x 压缩时，LLMLingua-2 QA EM 为 86.92，原文为 87.75；小模型为 85.82。相较 Selective-Context 和初代 LLMLingua，QA 和摘要指标都明显更高。
- 作者报告压缩器本身比既有 PPL 路线快 3x 到 6x；在 2x 到 5x 压缩范围内，端到端延迟为 1.6x 到 2.9x 加速。
- XLM-RoBERTa-large 压缩器在该实验中峰值 GPU 显存约 2.1GB，初代 LLMLingua 的 LLaMA-2-7B 约 16.6GB，Selective-Context 约 26.5GB。
- 它只用英文 MeetingBank 蒸馏训练，却在中文 LongBench 上优于初代；但 5x 时平均为 38.1，仍低于原文 42.5，说明跨语言泛化不是无损。
- 与 LongLLMLingua 的文档级 query-aware 预算结合后，NaturalQuestions 4x 设置的结果达到 71.9，优于纯 task-agnostic LLMLingua-2 的 46.2，但略低于完整 LongLLMLingua 的 75.5。

## 六、SecurityLingua：用“压缩意图”激活安全护栏

SecurityLingua[^4] 将 LLMLingua-2 的抽取分类器改造成 jailbreak 防御器。它观察到攻击 prompt 常以“已故祖母”“虚构小说”“角色扮演”“忽略此前指令”等外层噪声包装核心恶意指令；普通语义压缩可能仍保留这些诱导语言，而专门训练的压缩器应抽出真实意图。

### 1. 运行时流程

```text
用户原始 prompt
  -> SecurityLingua：抽取连续的真实意图 token
  -> 将意图附在 system prompt 中
  -> system prompt + 完整原始 prompt 一起交给目标 LLM
```

关键点是**原用户 prompt 不被替换**。例如，压缩器抽取“制定在线销售非法物品的计划”，系统提示将这一意图显式呈现给目标模型，再由目标模型自身的安全对齐决定拒答。与直接 filter 相比，这降低了对良性 query 的误拒风险；与 Rewrite/Erase-and-check 相比，不需要多次修改或采样原 prompt。

### 2. 数据、模型与阈值

SecurityLingua 构造约 221K 样本，约 122K 良性、100K 恶意，混合 Alpaca、OpenHermes、Disaster-tweet、WildJailbreak、JailbreakV-28K、JailbreakHub。数据由两类流程构成：

- 对长/复杂 query 做**意图压缩**，只保留真实请求。
- 对短 query 做**扩写**，人为加入角色、个人、虚构或非正式上下文；原短 query 作为监督目标。

为绕过通用教师模型对恶意文本的拒绝，论文按 GPT-4o、Mistral-Large、Uncensored-LLaMA2-72B 顺序级联生成，失败才向更少审查的模型回退。之后仍使用词形归一化、窗口匹配、VR 和 AG 过滤构造 token 标签。

架构为预训练 Transformer encoder 加线性 token classifier，与 LLMLingua-2 相同；不同点是运行时固定保留阈值为 0.5，而不是按目标压缩率保留 top-k。固定阈值反映其目标是提取“足够明确的意图”，而不是满足某个 token 预算。

### 3. 安全实验结论

作者在 JailbreakBench 的 PAIR、GCG、JB-Chat、RS 攻击上测试 Llama-2-7B-chat、GPT-3.5、GPT-4：

- 无防御平均攻击成功率为 35%；SecurityLingua 为 1%。论文表中 Erase-and-check 为 4%，JClassifier 为 6%，SmoothLLM 为 7%。
- SecurityLingua 平均额外约 32 token，约为原 prompt 的 11%，单次 forward 平均增加 25ms。论文报告 SmoothLLM 约增加 4,260 token，Erase-and-check 约 9,000 token；后两者还要检查多种扰动版本。
- 在 GPT-4 的 ARC、GPQA、MMLU、GSM8K 上，SecurityLingua 平均分为 72.0，高于无防御时的 69.7，拒答率为 0%。这支持“显式意图有时能帮助理解 query”的解释，但并不等价于它对所有正常任务都提升。

### 4. 必须明确的安全边界

SecurityLingua 不会自行阻断攻击，也不具备形式化鲁棒性证明。它依赖三项外部前提：目标模型已经安全对齐；调用方能够控制 system prompt；目标模型会正确遵循“识别到恶意意图则拒答”的系统策略。对没有 guardrail 的模型、系统消息权限被攻击者影响的环境、跨语言或新型多模态 jailbreak，不能仅凭论文结果宣称安全。

## 七、官方代码如何实现这些论文机制

仓库的核心入口是 [`llmlingua/prompt_compressor.py`](https://github.com/microsoft/LLMLingua/blob/main/llmlingua/prompt_compressor.py) 中的 `PromptCompressor`。

### 1. 模型加载与三条执行路径

构造器根据 Hugging Face `config.architectures` 选择 `AutoModelForCausalLM` 或 `AutoModelForTokenClassification`。默认模型名仍是 `NousResearch/Llama-2-7b-hf`，默认设备为 `cuda`；`use_llmlingua2=True` 和 `use_slingua=True` 会初始化分类器路径。

- **初代/Long 路径**：`get_ppl()` 前向执行因果模型，使用逐 token cross-entropy 作为 PPL；可传入 `past_key_values`，让后续 segment 基于已压缩前缀继续计算。
- **LLMLingua-2 路径**：`TokenClfDataset` 将块 pad/truncate 到 512，分类输出经 softmax，标签 1 的概率被视为保留概率。
- **SecurityLingua 路径**：复用分类器后端；内部 `__compress()` 对 `use_slingua` 采用固定 0.5 阈值。

### 2. 结构化输入与保护 token

`structured_compress_prompt()` 支持 `<llmlingua>` 标签，每段可配置 `rate` 和 `compress=False`。`compress_json()` 通过 [`utils.py`](https://github.com/microsoft/LLMLingua/blob/main/llmlingua/utils.py) 把 JSON 键、括号、逗号等结构放进不可压缩段，只压缩允许的 value，再解析回 JSON。

实现还支持 `force_tokens`、新增 `[NEWi]` special token 映射、数字保留和分段比例；这些不是论文主算法，却是让抽取式压缩在实际 prompt/JSON 中不至于立即破坏格式的必要工程补充。

### 3. LongLLMLingua 的源码映射

`get_rank_results()` 中的 `get_distance_longllmlingua()` 对每份文档计算条件 PPL，实际限制语句为：`We can get the answer to this question in the given documents.`。`rank_method="longllmlingua"`、`condition_in_question`、`condition_compare=True`、`reorder_context="sort"` 和 `dynamic_context_compression_ratio` 分别控制 query-aware 评分、对比、重排和动态预算。

所以 README 中的 LongLLMLingua 示例并非只切换一个模型名，而是显式开启了问题条件、排序和预算相关参数；漏掉这些参数就退化为较接近初代的压缩行为。

### 4. LLMLingua-2 与 SecurityLingua 的数据脚本

[`experiments/llmlingua2`](https://github.com/microsoft/LLMLingua/tree/main/experiments/llmlingua2) 包含数据采集、训练和评估脚本。SecurityLingua 的 [`label_word.py`](https://github.com/microsoft/LLMLingua/blob/main/experiments/securitylingua/label_word.py) 用 spaCy lemma 和前后窗口匹配给原词标注；[`filter.py`](https://github.com/microsoft/LLMLingua/blob/main/experiments/securitylingua/filter.py) 依次按 variation rate、alignment gap 的 90 分位过滤。

源码与论文总体一致，但使用时应注意实现细节会决定实际效果：默认 CUDA、tokenizer 差异、512 token 分块、词与子词概率合并、OpenAI `tiktoken` 用于目标 token 预算，都会使“请求的压缩率”与最终目标模型 token 数有偏差。

## 八、局限性与后续研究方向

### 1. 困惑度不是语义重要性的充分统计量

初代假设“高 PPL token 更重要”。这对罕见实体、算术数字、代码符号可能有帮助，但高 PPL 也可能只是拼写噪声；低 PPL 的否定词、条件词、关系词却可能决定答案。LLMLingua-2 已通过学习式双向分类修正这一点，但它把问题转化为“教师模型在 MeetingBank 上倾向保留什么”。下一步需要面向可验证下游目标、不同语言和不同数据类型的更可靠监督。

### 2. 抽取式 faithful 不等于任务 faithful

只删除原词避免了生成式幻觉，却可能删除语法桥梁、指代、运算符或跨句依赖。压缩文本对 LLM 可用不代表它对人类可读；压缩后答案相同也不代表所有中间事实都被保存。未来可结合结构感知单元，例如句法依存、表格单元、代码 AST、JSON schema 与实体原子性约束，而不是只在 token/词层选取。

### 3. 端到端加速存在临界点

压缩器不是免费前处理。初代和 LongLLMLingua 要运行小因果模型，短 prompt、低端 GPU 或无法批处理时，压缩耗时可能超过目标模型节省的 prefill。LLMLingua-2 显著减轻这一问题，但仍需依据实际模型、硬件、输入长度和并发量测量 break-even point。生产系统应记录压缩器耗时、输入长度、目标模型首 token 延迟、输出质量和回退率，而不是只看 token 账单。

### 4. 长上下文 query-aware 方法依赖相关性估计

LongLLMLingua 的优势来自“问题感知”，也因此依赖小模型能正确评估文档与问题的关系。多跳推理、时间顺序、法律条件、全局代码依赖和答案不在单一文档的场景，文档筛选或重排都可能删除关键桥接信息。后续可研究多跳证据图、保守的多样性约束、答案可证性检查和在输出置信度低时的自动回退。

### 5. SecurityLingua 的数据、覆盖面和系统权限

安全压缩数据由多个教师模型生成，恶意样本又天然受到审查和拒绝影响；这可能带来教师偏差、覆盖缺口和数据投毒风险。攻击者还会持续适配公开防御。后续评测应覆盖自适应攻击、间接 prompt injection、多语言、视觉/音频输入、工具调用链和不同 system prompt 权限模型。

最重要的是，SecurityLingua 应部署在独立的输入验证、工具权限、数据访问控制、输出审核和可审计日志之中。把提取出的意图写进 system prompt 只是增强目标 LLM 判断的一个信号，不能替代安全边界。

## 九、实践建议

1. **先定义损失函数和回退条件。** QA 看答案/证据召回，代码看编译与测试，JSON 看 schema，摘要看人工或任务指标；不能只看压缩率。
2. **通用语料先离线压缩。** 若同一文档会服务多个 query，LLMLingua-2 的 task-agnostic 属性可避免每个 query 重算；但要保留原文索引和可回退版本。
3. **RAG 使用两层策略。** 先用 LongLLMLingua 的 query-aware 文档预算，再以 LLMLingua-2 或保守 token 压缩细化；对数字、实体、标题、引用、代码块和 JSON key 设置强制保留。
4. **以真实端到端指标选阈值。** 分别统计压缩器耗时、目标模型延迟、token 成本、任务质量和失败案例，按业务的质量下限选择压缩率。
5. **安全场景不要只部署分类器。** SecurityLingua 的意图应作为策略引擎的输入之一；高风险操作仍需确定性授权、工具沙箱、敏感数据控制和人工升级路径。

## 参考资料

[^1]: Huiqiang Jiang et al. [LLMLingua: Compressing Prompts for Accelerated Inference of Large Language Models](https://aclanthology.org/2023.emnlp-main.825/), EMNLP 2023.
[^2]: Huiqiang Jiang et al. [LongLLMLingua: Accelerating and Enhancing LLMs in Long Context Scenarios via Prompt Compression](https://aclanthology.org/2024.acl-long.91/), ACL 2024.
[^3]: Zhuoshi Pan et al. [LLMLingua-2: Data Distillation for Efficient and Faithful Task-Agnostic Prompt Compression](https://aclanthology.org/2024.findings-acl.57/), Findings of ACL 2024.
[^4]: Yucheng Li et al. [SecurityLingua: Efficient Defense of LLM Jailbreak Attacks via Security-Aware Prompt Compression](https://arxiv.org/abs/2506.12707), CoLM 2025.
[^5]: Yucheng Li et al. [Compressing Context to Enhance Inference Efficiency of Large Language Models](https://aclanthology.org/2023.emnlp-main.400/), EMNLP 2023.
[^6]: [Microsoft/LLMLingua 官方仓库](https://github.com/microsoft/LLMLingua)，包括 `PromptCompressor`、LLMLingua-2 和 SecurityLingua 训练脚本。