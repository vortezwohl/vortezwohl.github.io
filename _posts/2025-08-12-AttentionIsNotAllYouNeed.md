---
layout: post
toc: true
title: "Paper 追踪: Attention is NOT all you need: Pure Attention Loses Rank Doubly Exponentially with Depth (深层自注意力网络的灾难性秩退化)"
categories: DL
tags: [Math, NLP, DeepLearning]
author:
  - vortezwohl
  - 吴子豪
---
自 2017 年 Transformer 提出以来，基于自注意力机制的模型已成为自然语言处理（NLP）、计算机视觉（CV）、语音识别等领域的核心架构。其成功被广泛归因于自注意力对长距离依赖的建模能力，但学界对其内在工作机制的理解仍不充分。该 Paper 聚焦一个关键问题：纯自注意力网络（仅堆叠自注意力层，移除跳跃连接和 MLP）的表达能力为何会随深度急剧下降？ 这一问题的本质是探索自注意力机制的 “归纳偏置”—— 即 **Transformer 模型在没有外部约束（如跳跃连接、MLP）时，天然倾向于学习何种模式**。此前相关研究多关注单个自注意力矩阵的秩特性或优化问题，而本文首次系统性分析了 “深层纯自注意力网络的输出是否会退化”，并试图解释：**为何完整的 Transformer（含跳跃连接和 MLP）能避免这种退化？**

Paper: [**Attention is Not All You Need**: Pure Attention Loses Rank Doubly Exponentially with Depth](https://doi.org/10.48550/arXiv.2103.03404)

## 相关工作

- **自注意力与 Transformer 基础研究**: 自注意力机制最初用于机器翻译 (Bahdanau et al., 2014), Transformer (Vaswani et al., 2017) 通过堆叠自注意力层和 MLP 实现了突破性性能. 此前的研究已探讨了自注意力与卷积的关系 (Cordonnier et al., 2020) , 注意力矩阵的低秩近似 (Wang et al., 2020) 等, 但未关注**深层网络的秩退化**.

- **跳跃连接 (残差链接)**: 跳跃连接源于 ResNet (He et al., 2016), 传统观点认为其主要作用是 "缓解梯度消失, 辅助梯度优化", 而本文则发现其另一核心作用: **阻止注意力网络的秩塌陷**, 这是对跳跃连接的全新理解.

- **网络秩退化**: NeurIPS 2020 的一篇文章 (Daneshmand et al., 2020) 发现随机初始化的线性 / ReLU 网络可能因批归一化 (Batch Norm) 避免秩塌陷, 但未涉及自注意力网络. 文章首次证明: 纯自注意力网络的秩退化速度远超传统网络, 且机制完全不同.

- **Transformer 变体**: 为降低自注意力的二次复杂度, 学界提出了低秩近似 (Wang et al., 2020), 稀疏注意力 (Child et al,. 2019) 等变体. 本篇文章则发现, 低秩近似可能加速秩塌陷, 而稀疏化则能够缓解退化, 为变体设计提供了理论依据.


## 核心发现

本文通过理论推导和实验验证, 得出以下结论:

1. **纯自注意力网络 (SANs) 存在秩塌陷现象**: 仅堆叠自注意力层 (无跳跃连接和 MLPs) 的网络, 其输出会以双指数速度 (因指数的指数增长) 收敛到秩 1 矩阵 (所有行均相同), 具体而言, 随深度 $L$ 增加, 输出与秩 1 矩阵的差距 (残差) 会以 $3^L$ 的速度衰减, 远快于传统随机矩阵乘积的线性收敛, 最终导致模型失去表达能力 (无法有效区分序列中的不同元素).

2. **MLP 可缓解秩塌陷但效果有限**: MLP 通过非线性变换调整 $Lipschitz$ 常数, 使残差收敛速度减慢 (约束中引入 $\lambda$ 因子), 但无法完全阻止退化. 此外, 过大的 $Lipschitz$ 常数可能降低模型的鲁棒性, 使其对输入的扰动更敏感.

3. **层归一化对秩塌陷无作用**: 层归一化仅对输入进行缩放和移位, 等价于对权重矩阵和偏置的重新参数化, 而不改变矩阵的秩特性, 因此和退化无关.

4. **自注意力网络是 "浅网络集成"**: 通过 "路径分解" (将多 Head 网络拆分为单 Head 路径的组合) 发现: 深层自注意力网络的表达能力主要依赖短路径 (长度 1 到 2), 长路径因秩塌陷几乎无贡献. 这与 ResNet 的浅网络集成特性类似, 但机制更极端.

## 对学/业界的启发

本文的发现对 Transformer 模型的设计 优化 应用提供了重要的指导:

1. **重新认识跳跃连接的价值**: 跳跃连接的核心作用不仅是 “辅助优化”，更在于 “维持模型表达能力”。这解释了为何移除跳跃连接的 Transformer 会完全失效，为轻量化模型设计（如减少层数但保留跳跃连接）提供了理论依据。

2. **指导高效 Transformer 变体设计**: 由于长路径贡献有限, 可通过以下方式优化模型:

    1. 减少深层注意力层的复杂度 (如稀疏化)

    2. 强化短路径的表达能力 (增加浅层 Head 数量)

    3. 避免过度依赖低秩近似 (可能加速秩塌陷)

3. **解释模型性能与深度的关系**: 文章解释了 "为什么 Transformer 并非越深越好", 过深的网络中长路径退化, 实际有效深度被限制在浅层, 这与实践中 "过深的 Transformer 易过拟合" 现象一致.

4. **启发新的归纳偏置研究**: 自注意力的 “token 均匀性偏置”（倾向于使所有 token 相同）是一把双刃剑：在需要全局一致性的任务（如句子分类）中可能有益，但在需要局部差异的任务（如命名实体识别）中需通过跳跃连接 / MLP 抵消。未来可设计更灵活的机制平衡这种偏置。

5. **推动对路径效率的关注**: 模型性能可通过 "激活短路径" 提升, 这为动态 Transformer (根据输入调整路径长度) 提供了研究基础.

## 重要参考文献

**[1]** Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention Is All You Need. *arXiv preprint*, 2017.

**[2]** Dzmitry Bahdanau, Kyunghyun Cho, Yoshua Bengio. Neural Machine Translation by Jointly Learning to Align and Translate. *arXiv preprint*, 2014.

**[3]** Jean-Baptiste Cordonnier, Andreas Loukas, Martin Jaggi. On the Relationship between Self-Attention and Convolutional Layers. *arXiv preprint*, 2019.

**[4]** Sinong Wang, Belinda Z. Li, Madian Khabsa, Han Fang, Hao Ma. Linformer: Self-Attention with Linear Complexity. *arXiv preprint*, 2020.

**[5]** Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun. Deep Residual Learning for Image Recognition. *arXiv preprint*, 2015.

**[6]** Hadi Daneshmand, Jonas Kohler, Francis Bach, Thomas Hofmann, Aurelien Lucchi. Batch normalization provably avoids ranks collapse for randomly initialised deep networks. *Advances in Neural Information Processing Systems*, 2020.

**[7]** Rewon Child, Scott Gray, Alec Radford, Ilya Sutskever. Generating Long Sequences with Sparse Transformers. *arXiv preprint*, 2019.
