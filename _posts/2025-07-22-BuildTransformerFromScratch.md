---
layout: post
toc: true
title: "从 0 开始, 构建 Transformer 模型."
categories: DL
tags: [Math, NLP, DeepLearning]
author:
  - vortezwohl
  - 吴子豪
---
在深度学习领域，特别是自然语言处理 (NLP) 任务中，Transformer 架构已经成为当前最先进的模型之一.Transformer 的核心创新在于其自注意力机制(Self Attention)，它允许模型在处理序列数据时动态地关注输入序列的不同部分，从而捕捉长距离依赖关系​.与传统的循环神经网络 (RNN) 和卷积神经网络 (CNN) 相比，自注意力机制在并行计算能力和长期依赖建模方面表现出显著优势.

## 基础知识

自注意力机制是 Transformer 架构的核心组成部分，它允许模型在处理序列数据时动态地关注输入序列的不同部分. 自注意力机制的基本思想是计算输入序列中每个位置与其他位置之间的关联程度，从而为每个位置生成一个**上下文向量**. 

### 基础自注意力计算

1. **获取 $QKV$ 权重矩阵**: 注意力模块中, 存在三个特别的全连接神经网络 (即 `Linear`), 即 `q_proj` `k_proj` `v_proj`, 这三个网络的作用是将输入向量 (通常是词嵌入序列) 分别投影到**查询空间** **键空间** 和 **值空间**.

    对应代码如下:

    ```python
    from torch import nn

    embed_dim = 512  # 假设词嵌入的维度是 512
    q_proj = nn.Linear(embed_dim, embed_dim)​
    k_proj = nn.Linear(embed_dim, embed_dim)​
    v_proj = nn.Linear(embed_dim, embed_dim)
    ```

    > `q_proj` `k_proj` `v_proj` 的权重由训练得到, 训练算法请参考[BERT预训练](https://vortezwohl.github.io/nlp/2025/04/30/%E6%B7%B1%E5%85%A5BERT.html#bert-%E9%A2%84%E8%AE%AD%E7%BB%83%E6%96%B9%E6%B3%95). 三个网络经训练得到的权重矩阵分别记为 $W_Q$ $W_K$ $W_V$.

2. **计算投影**: 分别计算输入序列对 $QKV$ 矩阵的乘积 (更一般地, 可以进行任意线性变换), 得到**查询向量** **键向量**和**值向量**.

    对于输入序列中的一个 Token 嵌入 $x$, 其投影向量计算如下:

    $$
    query = x \cdot W_Q \\
    key = x \cdot W_K \\
    value = x \cdot W_V
    $$

    其中, $query$ $key$ $value$ 分别是**查询向量** **键向量** 和**值向量**.

    对应代码如下:

    ```python
    query = q_proj(x)
    key = k_proj(x)
    value = v_proj(x)
    ```

3. **计算注意力分数**: 分别计算 **Token 之间**的注意力分数, 通常采用点积方式 (也就是 DotProductAttention, 即点积注意力).

    对于 $Token_1$, 其与 $Token_2$ 之间的注意力分数计算如下:

    $$
    attn_{12} = query_1 \cdot key_2
    $$

    对于 $Token_2$, 其与 $Token_1$ 之间的注意力分数计算如下:

    $$
    attn_{21} = query_2 \cdot key_1
    $$

    当然, 在实践中, 我们通常使用矩阵来进行批量计算, 可以将所有 Token 的 $query$ $key$ $value$ 进行有序堆叠, 分别得到矩阵 $Q$ $K$ $V$.

    然后我们可以通过矩阵乘法, 一次性得到挑战者 Token 和所有 Token 之间的注意力分数, 也就是注意力矩阵 ($ATTN$):

    $$
    ATTN = QK^T
    $$

    对应代码如下:

    ```python
    attn = torch.matmul(q, k.transpose(-2, -1))
    ```

4. **缩放注意力分数**: 为了**稳定梯度**, 在实践中我们通常会对注意力分数进行缩放, 例如**除以嵌入向量维度的平方根**.

    $$
    attn = \frac {attn} {\sqrt{d_k}}
    $$

    对应代码如下:

    ```python
    attn = attn / (embed_dim ** .5)
    ```

5. **将 $Logits$ 映射到概率域**: 缩放后的注意力分数可以被视作 $logits$, 通过应用 $softmax$ 算子, 将其映射到取值为 $[0, 1]$ 的概率域, 并将概率分布视作已有 Token 相对于挑战者 Token 的关联性权重分布.

    $$
    weights = [w_1, w_2, w_3, ..., w_n] = \text{Softmax}(attn)
    $$

    对应代码如下:

    ```python
    import torch.nn.functional as F

    weights = F.softmax(attn, dim=-1)
    ```

6. **计算上下文向量**: 使用关联性权重分布对已有 Token 的 $value$ 向量进行加权求和, 得到挑战者 Token 的上下文向量.

    $$
    context = \sum_{i=1}^n w_i \cdot value_i = weights * V
    $$

    对应代码如下:

    ```python
    context = torch.matmul(weights, v)
    ```

    最终，自注意力机制的最终计算结果是**上下文向量**，它综合了输入序列中所有位置的信息，其中每个位置的权重由注意力分数决定​.

...
