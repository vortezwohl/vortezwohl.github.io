---
layout: post
toc: true
title: "从 0 开始, 手搓一个 Transformer (Roformer) Encoder"
categories: DL
tags: [Math, NLP, DeepLearning]
author:
  - vortezwohl
  - 吴子豪
---
Transformer 架构凭借其强大的[注意力机制 (Attention)](
https://doi.org/10.48550/arXiv.1706.03762)，彻底改变了自然语言处理（NLP）领域的格局。与依赖序列顺序处理的 RNN 或受限于局部感受野的 CNN 不同，自注意力机制让模型能动态捕捉序列中任意位置的依赖关系，同时支持高效并行计算。本文将秉持 “从零开始” 的实践理念，逐步拆解 Transformer Encoder 的核心组件 —— 从自注意力机制的数学原理与代码实现，到位置编码（如 RoPE）如何注入序列位置信息，再到前馈神经网络的特征变换逻辑，最终手把手构建一个可运行的基础 Transformer Encoder，帮助读者深入理解这一经典架构的底层逻辑与工程实现细节.

## 基础知识

### 位置编码原理与计算

在 Transformer 架构中，自注意力机制本身并不具备对序列顺序的感知能力 —— 它仅通过序列中 Embedding 间的关联计算注意力，完全忽略了 Embedding 在序列中的位置先后。为弥补这一缺陷，位置编码（Positional Encoding）应运而生，其核心作用是将 Embedding 的位置信息以数学形式编码到输入嵌入中，确保模型能识别 “相同 Embedding 在不同位置具有不同含义” 这一规律.

点击查看[*位置编码的基础原理与计算式*](https://vortezwohl.github.io/nlp/2025/05/22/%E8%AF%A6%E8%A7%A3%E6%97%8B%E8%BD%AC%E4%BD%8D%E7%BD%AE%E7%BC%96%E7%A0%81.html).

### 基于 Torch 实现 RoPE 位置编码模块

代码源: [`deeplotx.nn.rope`](https://github.com/vortezwohl/DeepLoTX/blob/main/deeplotx/nn/rope.py)

引入必要依赖:

```python
from typing_extensions import override

import torch

from deeplotx.nn.base_neural_network import BaseNeuralNetwork
```

以下是一个标准 RoPE 编码[2]模块实现, 特征维度分为 2 组 (奇数组与偶数组), 基数为 10000:

```python
from typing_extensions import override

import torch

from deeplotx.nn.base_neural_network import BaseNeuralNetwork


class RoPE(BaseNeuralNetwork):
    def __init__(self, feature_dim: int, base: int = 10000, device: str | None = None, dtype: torch.dtype = torch.float32):
        super().__init__(in_features=feature_dim, out_features=feature_dim, model_name=None,
                         device=device, dtype=dtype)
        assert feature_dim % 2 == 0, f'feature_dim must be divisible by 2.'  # 特征维度必须是偶数
        self._base = base  # 基数选择 10000, 与 doi.org/10.48550/arXiv.2104.09864 一致
        self._num_groups = feature_dim // 2
        self._inv_freq = 1.0 / (theta ** (torch.arange(start=0, end=self._num_groups, step=1,
                                                       device=self.device, dtype=self.dtype).float() / self._num_groups))  # 计算逐维度的逆频率
        self.register_buffer('inv_freq', self._inv_freq)  # 将张量注册到缓冲区, 其不会参与反向传播

    @property
    def dim(self):
        return self._dim

    @property
    def base(self):
        return self._base

    def rotate_half(self, _t: torch.Tensor) -> torch.Tensor:
        return torch.cat((- _t[..., self._num_groups:], _t[..., :self._num_groups]), dim=-1)  # 将向量旋转 -90 度, 准确的说, 是将特征的实部与虚部进行交叉重组, 为 RoPE 提供交叉项

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ensure_device_and_dtype(x, device=self.device, dtype=self.dtype)  # 确保 x 和本模块的其他张量在同一设备且同为一种数据类型
        *other_dims, seq_len, feature_dim = x.shape  # x 的形状通常为 (batch_size, seq_len, feature_dim)
        assert feature_dim == self.in_features, f"feature_dim of x doesn't match with defined feature_dim {self.in_features}."
        t = torch.arange(start=0, end=seq_len, step=1, device=self.device, dtype=self.dtype)
        freq = torch.outer(t, self._inv_freq)  # 使用外积 (叉乘) 计算位置和逆频率的乘积, 形状为 (seq_len, self._num_groups)
        emb = torch.cat((freq, freq), dim=-1)  # 分别将各个向量的奇数维度位置编码向量与偶数维度位置编码向量进行拼接, 构建完整的位置编码矩阵, 形状为 (seq_len, feature_dim), 因为 feature_dim = 2 * self._num_group
        sin_emb, cos_emb = emb.sin(), emb.cos()  # 分别对位置编码矩阵应用 sin 和 cos 函数, 得到两个形状完全相同的矩阵
        return x * cos_emb + self.rotate_half(x) * sin_emb  # 将输入张量与余弦编码矩阵相乘, 将旋转 -90 度后的输入向量与正弦编码矩阵相乘, 最后相加, 得到被注入了相对位置信息的张量 x
```

> **逆频率 (inverse frequency)** 是计算 RoPE 位置编码的核心参数之一，它决定了不同维度特征的周期性变化速率. 逆频率的本质是频率的倒数，而频率与周期成反比 ($周期 = \frac {1} {频率}$), 因此, 逆频率和周期成正比, 逆频率越大, 则周期越大, RoPE 位置编码的旋转越缓慢, 而逆频率越小, 其旋转就越剧烈. 

> 在 RoPE 中, 不同维度会被分配不同的逆频率, 在低维特征上, RoPE 分配较小的逆频率, 所以在这些特征上位置编码的变化速度更快, 这样可以更好地捕获短期上下文, 而在高维特征上, RoPE 分配较大的逆频率, 所以位置编码变化的周期更大, 更缓和, 以捕获更长期的依赖.

### 基础自注意力计算

自注意力机制 (Self Attention) 是 Transformer 架构的核心组成部分[1]，它允许模型在处理序列数据时动态地关注输入序列的不同部分. 自注意力机制的基本思想是计算输入序列中每个位置与其他位置之间的关联程度，从而为每个位置生成一个**上下文向量**. 

自注意力的计算可以拆解为以下 6 个步骤.

1. **获取 $QKV$ 权重矩阵**: 注意力模块中, 存在三个特别的全连接神经网络 (即 `Linear`), 即 `q_proj` `k_proj` `v_proj`, 这三个网络的作用是将输入向量 (通常是词嵌入序列) 分别投影到**查询空间** **键空间** 和 **值空间**.

    对应代码如下:

    ```python
    from torch import nn

    embed_dim = 512  # 假设词嵌入的维度是 512
    q_proj = nn.Linear(embed_dim, embed_dim)​
    k_proj = nn.Linear(embed_dim, embed_dim)​
    v_proj = nn.Linear(embed_dim, embed_dim)
    ```

    > `q_proj` `k_proj` `v_proj` 的权重由训练得到, 训练算法请参考[*BERT预训练*](https://vortezwohl.github.io/nlp/2025/04/30/%E6%B7%B1%E5%85%A5BERT.html#bert-%E9%A2%84%E8%AE%AD%E7%BB%83%E6%96%B9%E6%B3%95)[3]. 三个网络经训练得到的权重矩阵分别记为 $W_Q$ $W_K$ $W_V$.

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

5. **将 $Logits$ 映射到概率域**: 缩放后的注意力分数可以被视作 [*$logits$*](https://vortezwohl.github.io/math/2025/07/17/%E4%BB%80%E4%B9%88%E6%98%AFlogit.html), 通过应用 $softmax$ 算子, 将其映射到取值为 $[0, 1]$ 的概率域, 并将概率分布视作已有 Token 相对于挑战者 Token 的关联性权重分布.

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

> 可以这样理解, 自注意力机制本质上是一种 "可学习的权重", 其本质作用是为序列中的某一项设置其余项对于该项的重要性权重.

> 自注意力的计算复杂度为 $O(n^2)$.

### 基于 Torch 实现自注意力模块

代码源: [`deeplotx.nn.attention`](https://github.com/vortezwohl/DeepLoTX/blob/main/deeplotx/nn/attention.py)

首先, 引入必要依赖:

```python
import torch
from torch import nn
from torch import softmax
```

以下是一个**简单自注意力模块**的实现, 它接受一个输入维度参数 `embed_dim`, 并使用 3 个 `Linear` 层分别计算 Token 嵌入的 $QKV$ 投影.

```python
class SelfAttention(nn.Module):
    def __init__(self, embed_dim: int, device: torch.device, dtype: torch.dtype):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.embed_dim = embed_dim
        self.q_proj = nn.Linear(in_features=embed_dim, out_features=embed_dim,
                                device=device, dtype=dtype)
        self.k_proj = nn.Linear(in_features=embed_dim, out_features=embed_dim,
                                device=device, dtype=dtype)
        self.v_proj = nn.Linear(in_features=embed_dim, out_features=embed_dim,
                                device=device, dtype=dtype)

    def _attention(self, x: torch.Tensor) -> torch.Tensor:
        # x (由 Token 嵌入堆叠而成的输入矩阵) 的形状是 (batch_size, seq_len, embed_dim)
        q = self.q_proj(x)
        k = self.k_proj(x)
        # attn (自注意力分数矩阵) 的形状是 (batch_size, seq_len, seq_len)
        attn = torch.matmul(q, k.transpose(-2, -1))
        # 对注意力分数进行缩放, 避免梯度不稳定
        attn = attn / (self.embed_dim ** 0.5)
        return softmax(attn, dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        v = self.v_proj(x)
        # 最后计算上下文向量, 形状为 (batch_size, seq_len, embed_dim)
        return torch.matmul(self._attention(x), v)
```

进行测试:

假设输入序列为三个嵌入向量: `[1, 1, 1, 1]` `[2, 2, 2, 2]` `[3, 3, 3, 3]`, 分别计算其自注意力权重和上下文向量:

```python
if __name__ == '__main__':
    self_attention = SelfAttention(4, torch.device('cpu'), torch.bfloat16)
    inputs = torch.tensor([
        [1, 1, 1, 1],
        [2, 2, 2, 2],
        [3, 3, 3, 3]
    ], dtype=self_attention.dtype, device=self_attention.device)
    attention_weights = self_attention._attention(inputs)
    context_vector = self_attention.forward(inputs)
    print('注意力权重分配: \n', attention_weights)
    print('自注意力上下文向量: \n', context_vector)
```

标准输出流:

```
注意力权重分配: 
 tensor([[0.2148, 0.3164, 0.4688],
        [0.1641, 0.2969, 0.5391],
        [0.1235, 0.2734, 0.6016]], dtype=torch.bfloat16,
       grad_fn=<SoftmaxBackward0>)
自注意力上下文向量: 
 tensor([[0.6992, 0.7578, 0.8672, 0.1216],
        [0.7461, 0.8164, 0.8906, 0.1309],
        [0.7812, 0.8672, 0.9062, 0.1387]], dtype=torch.bfloat16,
       grad_fn=<MmBackward0>)
```

在实践中, 特别是处理 seq2seq (序列到序列, 例如翻译, TTS, 文本摘要等任务) 任务时, 我们通常需要掩码机制, 以防止模型在训练/推理过程中看到未来的信息. 例如, 在语言建模任务中, 我们不希望模型在预测当前词时关注后续的词.

> 对语言模型来说, 后续的词作为训练数据是不可见的, 是它在考试过程中的"参考答案", 看到就相当于作弊了.

为了支持掩码机制, 我们需要改进**简单自注意力模块**以实现一个**掩码自注意力模块**, 通过引入一个 `mask` 参数, 实现在计算注意力分数后应用掩码:

```python
class SelfAttention(nn.Module):
    def __init__(self, embed_dim: int, device: torch.device, dtype: torch.dtype):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.embed_dim = embed_dim
        self.q_proj = nn.Linear(in_features=embed_dim, out_features=embed_dim,
                                device=device, dtype=dtype)
        self.k_proj = nn.Linear(in_features=embed_dim, out_features=embed_dim,
                                device=device, dtype=dtype)
        self.v_proj = nn.Linear(in_features=embed_dim, out_features=embed_dim,
                                device=device, dtype=dtype)

    def _attention(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        # x (由 Token 嵌入堆叠而成的输入矩阵) 的形状是 (batch_size, seq_len, embed_dim)
        q = self.q_proj(x)
        k = self.k_proj(x)
        # attn (自注意力分数矩阵) 的形状是 (batch_size, seq_len, seq_len)
        attn = torch.matmul(q, k.transpose(-2, -1))
        # 对注意力分数进行缩放, 避免梯度不稳定
        attn = attn / (self.embed_dim ** 0.5)
        # 在 softmax 之前应用掩码, 且掩码的形状必须和注意力分数矩阵相同
        attn = attn.masked_fill(mask == 0, -1e9) if mask is not None else attn
        return softmax(attn, dim=-1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        v = self.v_proj(x)
        # 最后计算上下文向量, 形状为 (batch_size, seq_len, embed_dim)
        return torch.matmul(self._attention(x, mask), v)
```

### 多头注意力计算 (并行多头)

多头注意力（Multi-Head Attention）是对基础自注意力机制的关键升级，其核心思想是将输入特征拆分到多个并行的 “注意力头” 中，让每个头独立学习不同子空间的注意力模式，最终通过拼接与线性变换融合多维度的关联信息。这种设计的优势在于：单一自注意力头可能仅捕捉到序列中某一类依赖关系（如语法关联或语义关联），而多头机制通过并行计算不同子空间的注意力分布，能同时挖掘 Token 间多样化的关联模式（例如长距离语义依赖、局部语法结构、上下文情感倾向等），显著提升模型对复杂序列特征的表达能力.

### 基于 Torch 实现并行多头注意力 (非标准实现)

代码源: [`deeplotx.nn.multi_head_attention`](https://github.com/vortezwohl/DeepLoTX/blob/main/deeplotx/nn/multi_head_attention.py)

引入必要依赖:

```python
from typing_extensions import override

import torch
from torch import nn

from deeplotx.nn.base_neural_network import BaseNeuralNetwork
from deeplotx.nn.attention import Attention
```

以下是一个并行多头注意力的变体实现 (与最初提出的注意力实现[1]有所不同， 也与当下常见的联合多头注意力不同):

```python
class MultiHeadAttention(BaseNeuralNetwork):
    def __init__(self, feature_dim: int, num_heads: int = 1, bias: bool = True, positional: bool = True,
                 proj_layers: int = 1, proj_expansion_factor: int | float = 1.5, dropout_rate: float = 0.02,
                 model_name: str | None = None, device: str | None = None, dtype: torch.dtype | None = None,
                 **kwargs):
        super().__init__(in_features=feature_dim, out_features=feature_dim, model_name=model_name,
                         device=device, dtype=dtype)
        self._num_heads = num_heads
        self.expand_proj = nn.Linear(in_features=feature_dim, out_features=feature_dim * self._num_heads, bias=bias,
                                     device=self.device, dtype=self.dtype)
        self.attn_heads = nn.ModuleList([Attention(feature_dim=feature_dim, bias=bias, positional=positional,
                                                   proj_layers=proj_layers, proj_expansion_factor=proj_expansion_factor,
                                                   dropout_rate=dropout_rate, device=self.device, dtype=self.dtype,
                                                   **kwargs) for _ in range(self._num_heads)])
        self.out_proj = nn.Linear(in_features=feature_dim * self._num_heads, out_features=feature_dim, bias=bias,
                                  device=self.device, dtype=self.dtype)

    @override
    def forward(self, x: torch.Tensor, y: torch.Tensor | None = None, mask: torch.Tensor | None = None) -> torch.Tensor:
        x = self.ensure_device_and_dtype(x, device=self.device, dtype=self.dtype)
        y = x if y is None else self.ensure_device_and_dtype(y, device=self.device, dtype=self.dtype)
        x, y = self.expand_proj(x), self.expand_proj(y)
        x_heads, y_heads = x.split(self.in_features, dim=-1), y.split(self.in_features, dim=-1)
        head_outs = [self.attn_heads[_](x=x_heads[_], y=y_heads[_], mask=mask) for _ in range(self._num_heads)]
        return self.out_proj(torch.concat(head_outs, dim=-1))
```

### 前馈神经网络

前馈神经网络 (Feed-Forward Network，FFN) 是 Transformer 架构中的另一关键组件，它通常被放置在自注意力模块之后, FFN 的核心功能是对序列中的每个位置进行独立的非线性变换，它不考虑 Token 之间的依赖关系 (这与自注意力机制形成互补)，而是专注于挖掘单个 Token 内部的高阶特征.

FFN 的结构并不复杂，通常由线性变换 (`nn.Linear`) 与非线性激活函数 (`ReLU` 及其变体) 组成，有时还会加入 `dropout` 机制以缓解过拟合. 其基本计算过程可以拆解为以下几个步骤：

1. **升维线性变换**: 通过一个权重矩阵将输入向量投影到更高维的特征空间.

    $$
    x = W_1 \cdot x + b_1
    $$

    其中, $W_1$ 是升维权重矩阵, $b_1$ 是线性偏置项.

2. **非线性变换**: 对升维后的向量应用非线性激活函数（通常是 `ReLU`），引入非线性特征变换能力，增强模型对复杂模式的捕捉能力.

    $$
    x = \text{max}(0, x)
    $$

    > 这里对向量 x 应用了简单 `ReLU` 函数.

3. **降维线性变换**: 通过第二个全连接层将高维特征向量进行压缩, 投影回原始特征空间, 以确保 FFN 的输入维度与输出维度是一致的.

    $$
    x = W_2 \cdot x + b_2
    $$

    其中, $W_2$ 是升维权重矩阵, $b_2$ 是线性偏置项.

在实际应用中，为了增强特征变换的深度，FFN 可以由多个相同结构的前馈层堆叠而成，形成多层 FFN. 每层独立进行升维、激活和降维操作，并通过残差连接保留原始输入信息，避免深层网络中的梯度消失问题.

### 基于 Torch 实现前馈神经网络

代码源: [`deeplotx.nn.feed_forward`](https://github.com/vortezwohl/DeepLoTX/blob/main/deeplotx/nn/feed_forward.py)

引入必要依赖:

```python
from typing_extensions import override

import torch
from torch import nn

from deeplotx.nn.base_neural_network import BaseNeuralNetwork
```

以下是一个较为完善的 FFN 实现, 其引入了残差连接 (Residual Connetion) 和 `dropout` 机制, 以适应更复杂的应用场景.

```python
class FeedForwardUnit(BaseNeuralNetwork):
    def __init__(self, feature_dim: int, expansion_factor: int | float = 2,
                 bias: bool = True, dropout_rate: float = 0.05, model_name: str | None = None,
                 device: str | None = None, dtype: torch.dtype | None = None):
        super().__init__(in_features=feature_dim, out_features=feature_dim, model_name=model_name, device=device, dtype=dtype)
        self._dropout_rate = dropout_rate
        self.fc1 = nn.Linear(feature_dim, int(feature_dim * expansion_factor), bias=bias,
                             device=self.device, dtype=self.dtype)
        self.fc2 = nn.Linear(int(feature_dim * expansion_factor), feature_dim, bias=bias,
                             device=self.device, dtype=self.dtype)
        self.parametric_relu_1 = nn.PReLU(num_parameters=1, init=5e-3,
                                          device=self.device, dtype=self.dtype)
        self.layer_norm = nn.LayerNorm(normalized_shape=self.fc1.in_features, eps=1e-9,
                                       device=self.device, dtype=self.dtype)

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ensure_device_and_dtype(x, device=self.device, dtype=self.dtype)
        residual = x
        x = self.layer_norm(x)
        x = self.fc1(x)
        x = self.parametric_relu_1(x)
        if self._dropout_rate > .0:
            x = torch.dropout(x, p=self._dropout_rate, train=self.training)
        return self.fc2(x) + residual


class FeedForward(BaseNeuralNetwork):
    def __init__(self, feature_dim: int, num_layers: int = 1, expansion_factor: int | float = 2,
                 bias: bool = True, dropout_rate: float = 0.05, model_name: str | None = None,
                 device: str | None = None, dtype: torch.dtype | None = None):
        if num_layers < 1:
            raise ValueError('num_layers cannot be less than 1.')
        super().__init__(in_features=feature_dim, out_features=feature_dim, model_name=model_name, device=device, dtype=dtype)
        self.ffn_layers = nn.ModuleList([FeedForwardUnit(feature_dim=feature_dim,
                                                         expansion_factor=expansion_factor, bias=bias,
                                                         dropout_rate=dropout_rate,
                                                         device=self.device, dtype=self.dtype)] * num_layers)

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ensure_device_and_dtype(x, device=self.device, dtype=self.dtype)
        for ffn in self.ffn_layers:
            x = ffn(x)
        return x
```

## 实现一个基础的 Transformer (Roformer 变体)

接下来，我将基于这些已实现的组件，按照 [*Roformer*](https://doi.org/10.48550/arXiv.2104.09864)[2] 的架构逻辑，逐步拼接出一个完整的基础 Transformer (Roformer) Encoder:

代码源: [`deeplotx.nn.roformer_encoder`](https://github.com/vortezwohl/DeepLoTX/blob/main/deeplotx/nn/roformer_encoder.py)

引入必要依赖：

```python
from typing_extensions import override

import torch
from torch import nn

from deeplotx.nn.base_neural_network import BaseNeuralNetwork
from deeplotx.nn.feed_forward import FeedForward
from deeplotx.nn.multi_head_attention import MultiHeadAttention
```

以下是一个集成了多头注意力、RoPE 位置编码以及 FFN 的 Roformer 编码器实现:

```python
class RoFormerEncoder(BaseNeuralNetwork):
    def __init__(self, feature_dim: int, attn_heads: int = 2, bias: bool = True,
                 ffn_layers: int = 1, ffn_expansion_factor: int | float = 2,
                 dropout_rate: float = 0.02, model_name: str | None = None,
                 device: str | None = None, dtype: torch.dtype | None = None, **kwargs):
        super().__init__(in_features=feature_dim, out_features=feature_dim,
                         model_name=model_name, device=device, dtype=dtype)
        self.attn = MultiHeadAttention(feature_dim=feature_dim, num_heads=attn_heads,
                                       bias=bias, positional=True,
                                       proj_layers=kwargs.get('attn_ffn_layers', 1),
                                       proj_expansion_factor=kwargs.get('attn_expansion_factor', ffn_expansion_factor),
                                       dropout_rate=kwargs.get('attn_dropout_rate', dropout_rate),
                                       device=self.device, dtype=self.dtype, **kwargs)
        self.ffn = FeedForward(feature_dim=feature_dim * 2, num_layers=ffn_layers,
                               expansion_factor=ffn_expansion_factor,
                               bias=bias, dropout_rate=dropout_rate,
                               device=self.device, dtype=self.dtype)
        self.layer_norm = nn.LayerNorm(normalized_shape=feature_dim, eps=1e-9,
                                       device=self.device, dtype=self.dtype)
        self.out_proj = nn.Linear(in_features=feature_dim * 2, out_features=feature_dim,
                                  bias=bias, device=self.device, dtype=self.dtype)

    @override
    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        x = self.ensure_device_and_dtype(x, device=self.device, dtype=self.dtype)
        if mask is not None:
            mask = self.ensure_device_and_dtype(mask, device=self.device, dtype=self.dtype)
        attn = self.attn(x=self.layer_norm(x), y=None, mask=mask)  # 归一化并计算自注意力分数
        x = torch.concat([attn, x], dim=-1)  # 将注意力分数与输入特征进行融合
        return self.out_proj(self.ffn(x))  # 将融合后的 Embedding 进行非线性变换, 然后通过线性变换映射回原始特征空间, 以确保输出和输入形状一致
```

测试一下这个编码器:

首先创建该编码器模型

```python
model = RoFormerEncoder(
    feature_dim=512,  # 嵌入维度 512 维
    attn_heads=12,  # 12 个注意力头
    bias=True,  # 引入偏置项
    ffn_layers=4,  # 4 层 FFN 非线性变换
    ffn_expansion_factor=4  # FFN 的缩放系数为 4
    )

print(model)  # 把模型结构打印出来看看
```

不得不说这个模型真大, 共 116,033,576 参数 (0.1B)

```
===============================================================
Model_Name: RoFormerEncoder
In_Features: 512
Out_Features: 512
Device: cpu
Dtype: torch.float32
Total_Parameters: 116033576
Trainable_Parameters: 116033576
NonTrainable_Parameters: 0
---------------------------------------------------------------
RoFormerEncoder(
  (attn): MultiHeadAttention(
    (expand_proj): Linear(in_features=512, out_features=6144, bias=True)
    (attn_heads): ModuleList(
      (0-11): 12 x Attention(
        (q_proj): FeedForward(
          (ffn_layers): ModuleList(
            (0): FeedForwardUnit(
              (up_proj): Linear(in_features=512, out_features=2048, bias=True)
              (down_proj): Linear(in_features=2048, out_features=512, bias=True)
              (parametric_relu): PReLU(num_parameters=1)
              (layer_norm): LayerNorm((512,), eps=1e-09, elementwise_affine=True)
            )
          )
        )
        (k_proj): FeedForward(
          (ffn_layers): ModuleList(
            (0): FeedForwardUnit(
              (up_proj): Linear(in_features=512, out_features=2048, bias=True)
              (down_proj): Linear(in_features=2048, out_features=512, bias=True)
              (parametric_relu): PReLU(num_parameters=1)
              (layer_norm): LayerNorm((512,), eps=1e-09, elementwise_affine=True)
            )
          )
        )
        (v_proj): FeedForward(
          (ffn_layers): ModuleList(
            (0): FeedForwardUnit(
              (up_proj): Linear(in_features=512, out_features=2048, bias=True)
              (down_proj): Linear(in_features=2048, out_features=512, bias=True)
              (parametric_relu): PReLU(num_parameters=1)
              (layer_norm): LayerNorm((512,), eps=1e-09, elementwise_affine=True)
            )
          )
        )
        (rope): RoPE()
      )
    )
    (out_proj): Linear(in_features=6144, out_features=512, bias=True)
  )
  (ffn): FeedForward(
    (ffn_layers): ModuleList(
      (0-3): 4 x FeedForwardUnit(
        (up_proj): Linear(in_features=1024, out_features=4096, bias=True)
        (down_proj): Linear(in_features=4096, out_features=1024, bias=True)
        (parametric_relu): PReLU(num_parameters=1)
        (layer_norm): LayerNorm((1024,), eps=1e-09, elementwise_affine=True)
      )
    )
  )
  (layer_norm): LayerNorm((512,), eps=1e-09, elementwise_affine=True)
  (out_proj): Linear(in_features=1024, out_features=512, bias=True)
)
===============================================================
```

现在我们创建一个用于测试的张量

```python
t = torch.randn(1, 32, 512)  # (batch_size, seq_len, feature_dim) 这里我创建了一个 batch 的 长度为 32 的 512 维 Embedding 序列
print(t)
```

```
tensor([[[-0.9948, -0.0349, -0.0292,  ...,  0.1072,  0.1905, -0.7041],
         [-0.5063, -0.5237,  0.8143,  ..., -0.5679,  0.3080,  0.4045],
         [-0.5734,  0.9023, -0.0459,  ...,  0.2895, -0.5912,  0.2613],
         ...,
         [-1.7784, -0.1019,  0.1402,  ...,  2.5297,  1.1557,  0.7828],
         [-1.3071,  0.4030, -0.2874,  ..., -1.0355, -1.3376,  0.3785],
         [-1.4015, -1.2260, -0.0717,  ...,  0.3206,  0.9351,  0.4492]]])
```

让我们用 `model` 算一算这个序列的自注意力加权特征

```python
emb = model.forward(t)
print(emb)
```

不难发现, 模型的输出形状 (`shape`) 和输入序列完全相同, 输出后的张量已经融合了自注意力分数, 其序列中的每一个 Embedding 都融合了自己的上下文信息

```
tensor([[[ 0.0977, -0.8709, -0.9439,  ..., -0.1505, -0.1445, -0.0918],
         [ 0.3352,  0.1858, -0.0060,  ..., -0.1324,  0.5277,  0.5881],
         [ 0.2196, -0.8635, -1.0969,  ..., -0.1946, -0.4823,  0.2941],
         ...,
         [ 0.2569, -0.1919,  0.2975,  ..., -0.3115,  0.6879,  0.1908],
         [-0.1365,  0.1804,  0.4099,  ...,  0.0668,  0.1690,  0.1809],
         [-0.1341, -0.1368, -0.3354,  ..., -0.2252, -0.1670,  0.8261]]],
       grad_fn=<ViewBackward0>)
```

## 参考文献

**[[1](https://doi.org/10.48550/arXiv.1706.03762)]** Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention Is All You Need. *arXiv preprint*, 2017.

**[[2](https://doi.org/10.48550/arXiv.2104.09864)]** Jianlin Su, Yu Lu, Shengfeng Pan, Ahmed Murtadha, Bo Wen, and Yunfeng Liu. RoFormer: Enhanced Transformer with Rotary Position Embedding. *arXiv preprint*, 2017.

**[[3](https://doi.org/10.48550/arXiv.1810.04805)]** Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. *arXiv preprint*, 2017.

**[[4](https://github.com/vortezwohl/DeepLoTX)]** Zihao Wu. DeepLoTX: Easy-2-use long text NLP toolkit. *GitHub repository*. https://github.com/vortezwohl/DeepLoTX (Accessed: 2025-07-29).