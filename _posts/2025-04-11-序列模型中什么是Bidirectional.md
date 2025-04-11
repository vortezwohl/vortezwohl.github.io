---
layout: post
toc: false
title: "序列模型中什么是 Bidirectional"
categories: NLP
tags: [DeepLearning, NLP]
author:
  - vortezwohl
  - 吴子豪
---
Bidirectional 是一种神经网络的结构特性，表示网络在处理序列数据时，同时考虑正向和反向的上下文信息。这种结构通常用于循环神经网络（RNN）及其变体（如 LSTM 和 GRU）中，以提高模型对序列数据的理解能力。

### 作用

- 正向处理：从序列的开头到结尾依次处理每个时间步。

- 反向处理：从序列的结尾到开头依次处理每个时间步。

- 合并结果：将正向和反向的输出结果合并（通常是拼接，但也可以是求和、平均等），作为最终的输出。

### 模型结构

```
输入序列: x₁, x₂, x₃, x₄, x₅, x₆, x₇, x₈

正向 RNN:
h₁ → h₂ → h₃ → h₄ → h₅ → h₆ → h₇ → h₈

反向 RNN:
h₈ ← h₇ ← h₆ ← h₅ ← h₄ ← h₃ ← h₂ ← h₁

最终输出: [h₁; h₈], [h₂; h₇], [h₃; h₆], [h₄; h₅], [h₅; h₄], [h₆; h₃], [h₇; h₂], [h₈; h₁]
```

[h₁; h₈] 表示将正向和反向的隐藏状态进行拼接。

通过这种方式，双向模型能够充分利用序列的完整上下文信息，从而在许多任务中表现出色。

### PyTorch 实现

```python
import torch
import torch.nn as nn

# 定义一个双向 GRU 模型
class BiGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(BiGRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bigru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )
        self.fc = nn.Linear(hidden_size * 2, output_size)  # 因为是双向，所以输出维度是 hidden_size * 2

    def forward(self, x):
        # 初始化隐藏状态
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        # 前向传播
        out, _ = self.bigru(x, h0)
        # 取最后一个时间步的输出
        out = self.fc(out[:, -1, :])
        return out

# 示例输入
input_size = 10
hidden_size = 20
num_layers = 2
output_size = 5
batch_size = 32
seq_length = 100
x = torch.randn(batch_size, seq_length, input_size)

# 创建模型并前向传播
model = BiGRU(input_size, hidden_size, num_layers, output_size)
output = model(x)
print(output.shape)  # 输出形状为 (batch_size, output_size)
```