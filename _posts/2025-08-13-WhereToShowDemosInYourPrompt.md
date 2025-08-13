---
layout: post
toc: true
title: "Paper 追踪: Where to show Demos in Your Prompt: A Positional Bias of In-Context Learning (上下文学习中示例在提示中的位置会影响生成性能)"
categories: NLP
tags: [AI, LLM, Agent, NLP]
author:
  - vortezwohl
  - 吴子豪
---
来自美国马里兰大学的一项研究揭示的 DPP 偏差 (Demo’s Position in Prompt bias) 表明，prompt 中示例的 “空间位置” 是 ICL (In-Context Learning) 性能的关键调节因素。这一发现不仅推动了对 LLMs 上下文利用机制的理解，更为实践中提升 prompt 有效性提供了可操作的指南 ——prompt 工程需从 “内容设计” 扩展到 “结构优化”，结合模型特性和任务类型动态调整示例位置。

Paper: [**Where to show Demos in Your Prompt**: A Positional Bias of In-Context Learning](https://doi.org/10.48550/arXiv.2507.22887)

## 过去 In-Context Learning 所面临的问题

In-Context Learning（ICL）是大语言模型（LLMs）通过在 prompt 中嵌入少量示例（demos）实现少样本学习的核心能力，但此前研究已发现其存在显著局限性：

1. **对示例细节敏感**: 示例的顺序, 数量或选择的微小变化可能导致性能剧烈波动. 例如, 2022 年 ACL 所收录的一项研究 (Lu et al. 2022) 发现示例的顺序调整可使推理任务的准确率波动 ±15%, 而 EMNLP 于 2022 年所收录的一项研究 (Min et al. 2022) 则表明语言模型可能依赖示例与用户查询 (Query) 间的表面词汇重叠, 而非真正学习语义映射.

2. **鲁棒性不足**: 语言模型的表现易受 prompt 模板措辞 (Cho et al. 2024), 示例格式 (Kim et al. 2022) 等表面因素影响, 难以稳定复现结果, 这挑战了 "语言模型能够真正从上下文学习" 的假设.

3. **研究空白**: 现有研究聚焦于示例内容, 顺序或模板设计, 但示例在 prompt 中的空间位置 (特别是相对于系统提示和用户消息的位置), 对 ICL 的影响尚未被系统性探索, 这一研究空白限制了我们对 ICL 机制的全面理解.

## 本篇 Paper 的新发现以及实验验证

本文首次提出了 DPP 偏差概念 (即 Demo's Position in Prompt bias), 并验证了示例在 prompt 中的位置 (与内容无关) 会显著影响语言模型的预测准确性和稳定性.

> 此处的预测 (Predict) 并非针对分类任务, 在机器学习语境下, 深度神经网络进行前向传播的过程被称为"预测"或"推理".

**其核心发现可以归纳为**: DPP 偏差指当示例块在 prompt 中的位置（系统提示 / 用户消息的开头或结尾）变化时，模型准确率可能波动达 20%，且近半数预测结果会翻转，这种偏差与示例内容无关，仅由空间位置导致.

文中定义了 4 种典型的示例位置: 

|位置|解释|
|--|--|
SSP|示例位于系统提示的最开头
ESP|示例位于系统提示的结尾
SUM|示例位于用户提示的开头
EUM|示例位于用户提示的结尾

针对以上 4 种典型位置, 研究团队进行了实验: 实验针对 10 个开源语言模型 (分别来自 Qwen, LLaMA, Mistral, Cohere 四大模型家族, 参数规模从 1.5B 到 72B 不等), 并设计了 8 个不同的任务 (分别属于文本分类, 文本问答, 文本摘要, 逻辑推理).

实验采用两个评估指标对结果进行量化评估:

- **ACCURACY-CHANGE**: 位于提示词不同位置的示例相对于零样本提示的准确率变化.

    > 零样本提示 (Zero-Shot Prompting) 是一种提示技术, 即不提供任何示例 (称样本) 的情况下, 仅通过自然语言描述, 引导语言模型生成符合语气的输出, 这种方法高度依赖语言模型自身在预训练过程习得的先验知识和泛化能力. (这篇文章所探讨的领域称为少样本提示, 即 Few-Shot Prompting)

- **PREDICTION-CHANGE**: 位于提示词不同位置的示例导致的预测结果翻转率.

**关键结论**: 

1. 示例放在系统提示词中 (SSP, ESP) 时, 模型输出最稳定且准确, 准确率较 SUM 位置有极大提升 (最高情况下提升 6%, MMLU 任务); 而示例放在 EUM 位置时, 在问答任务中导致了超过 30% 的预测翻转, 且准确率无法提升 (SQuAD 任务).

    > MMLU 即 Measuring Massive Multitask Language Understanding, 数据集: https://huggingface.co/datasets/openai/MMMLU

    > SQuAD 即 Stanford Question Answering Dataset, 数据集: https://huggingface.co/datasets/rajpurkar/squad

2. 从参数规模上看, 小型模型如 Qwen-1.5B 受 DPP 偏差的影响更大; 大型模型如 LLaMA3-70B 则更具鲁棒性, 但也仅限于简单任务, 其在复杂任务中仍受 DPP 偏差影响. 而从任务类型上看, 生成任务如文本摘要, 对示例位置最为敏感, EUM 位置甚至导致 LLaMA3-3B 在 CNN/DailyMail 任务中准确率由 49% 急降至 1%.

3. 没有绝对的万能位置, 最有位置因基线模型与任务而异. 例如, 小型语言模型在分类任务中偏好 SSP 和 ESP 位置, 而 LLaMA3-70B 在部分任务中偏好 SUM 位置. 但至少我们能够排除 EUM 位置 (也就是在用户提示的末尾, 示例置于用户提示末尾在多数情况下效果最差)

## 该发现对于业界的启发

1. 打破**格式无关**误区, prompt 的结构, 特别是示例的位置, 并非一个简单的提示词风格细节, 而是实际影响性能的核心因素, 需要将位置设计纳入提示词设计流程.

2. LLMs 对上下文的利用存在**位置偏好**, 可能与 Transformer Decoder 的自回归机制原理有关 (早期 Token 会持续参与后续生成过程的交叉注意力计算), 也可能和训练数据中的位置规律有关, 这为我们理解 AR 语言模型 "推理黑箱" 提供了新视角.

3. 对于小型语言模型, 优先考虑将示例置于系统提示部分 (SSP/ESP), 利用其对早期信息的高敏感性提升性能; 对于中大型模型, 可尝试将示例置于用户提示开头 (SUM), 如果追求更优性能, 还需针对具体任务进行测试. 在任务类型上, 针对问答和生成任务, 应该优先考虑把示例往前放, 利用早期位置强化任务指令与示例的关联.

4. 规避高风险位置, 避免把示例置于用户消息结尾 (EUM) 位置, 尤其在问答任务和生成任务中, 此位置容易导致预测不稳定且准确率下降.

我们还可以探索一些进阶的偏差缓解策略, 例如在微调阶段引入随机位置示例, 增强语言模型对示例位置变化的鲁棒性.

## 重要参考文献

**[1]** Yao Lu, Max Bartolo, Alastair Moore, Sebastian Riedel, Pontus Stenetorp. Fantastically Ordered Prompts and Where to Find Them: Overcoming Few-Shot Prompt Order Sensitivity. *Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, 2022.

**[2]** Sewon Min, Xinxi Lyu, Ari Holtzman, Mikel Artetxe, Mike Lewis, Hannaneh Hajishirzi, Luke Zettlemoyer. Rethinking the Role of Demonstrations: What Makes In-Context Learning Work?. *Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing*, 2022.

**[3]** Ikhyun Cho, Gaeul Kwon, Julia Hockenmaier. Tutor-ICL: Guiding Large Language Models for Improved In-Context Learning Performance. *Findings of the Association for Computational Linguistics: EMNLP 2024*, 2024.

**[4]** Hyuhng Joon Kim, Hyunsoo Cho, Junyeob Kim, Taeuk Kim, Kang Min Yoo, Sang-goo Lee. Self-Generated In-Context Learning: Leveraging Auto-regressive Language Models as a Demonstration Generator. *arXiv preprint*, 2022.
