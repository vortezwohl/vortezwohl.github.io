---
layout: post
toc: true
title: "Fill-In-the-Middle Completion：让大语言模型补全文本中间缺口"
categories: LLM
tags: [LLM, FIM, infilling, code-completion, DeepSeek, API, Python]
author:
  - vortezwohl
  - 吴子豪
  - Codex
---

FIM 是 `Fill-In-the-Middle` 的缩写，也常被叫作 `infilling`。和普通的 left-to-right 续写不同，它不是只看左侧上下文继续往后生成，而是同时利用“前缀 + 后缀”去补中间缺失的片段。对代码生成来说，这比单纯的续写更接近真实开发流程，因为真实编程往往不是从文件第一行一路写到最后一行，而是在已有文件中插入、修改、重构和补洞。如果把普通补全理解为“接着写”，那 FIM 更像“把中间空出来的部分补上”。这也是为什么 IDE 内联补全、自动修复 TODO、补全函数体、插入参数校验、补写测试用例等场景，天然更适合 FIM。

## 什么是 FIM Completion

给定一段文本或代码，我们把它拆成三部分：

$$
x = (x_{\text{prefix}}, x_{\text{middle}}, x_{\text{suffix}})
$$

普通自回归语言模型学习的是：

$$
p(x)=\prod_{t=1}^{n} p(x_t \mid x_{<t})
$$

也就是每次只根据左边已经出现的 token 预测下一个 token。

而 FIM 的目标变成了：

$$
p(x_{\text{middle}} \mid x_{\text{prefix}}, x_{\text{suffix}})
$$

也就是说，模型需要在“已知左上下文”和“已知右上下文”的条件下，把中间缺失的内容补出来。这种能力对代码尤其重要，因为代码的正确性常常同时依赖前面的变量定义、后面的返回语句、缩进层级、异常处理以及 API 调用方式。

## 技术原理

### 1. FIM 训练并不一定需要改模型结构

FIM 一个非常重要的工程结论是：**很多 decoder-only 自回归模型不需要改成 encoder-decoder，也不需要彻底换架构，只要在训练数据构造上做变换，就能学会中间补全。** OpenAI 在《Efficient Training of Language Models to Fill in the Middle》中系统证明了这一点$^{[4][5]}$。

核心思路是：从原始文本里随机截取一个中间片段，把它移动到序列尾部，并用特殊标记指示三段内容的角色。一个常见的表示方式是：

$$
\text{FIM}(x)=\langle PRE \rangle\ x_{\text{prefix}}\ \langle SUF \rangle\ x_{\text{suffix}}\ \langle MID \rangle\ x_{\text{middle}}
$$

模型仍然做标准 next-token prediction，只不过训练样本已经被重排过了。这样模型在看到前缀和后缀后，就学会了如何生成中间缺失的内容。

这类做法的工程价值很高：

- 不必为 FIM 单独设计一套全新推理架构。
- 原有 left-to-right 能力通常不会明显退化。
- 可以和现有代码模型、补全模型、IDE 工作流自然结合。

### 2. 为什么 FIM 比普通续写更适合代码编辑

在代码场景里，右侧上下文经常提供强约束。例如：

```python
def normalize_scores(scores):
    # middle
    return [x / total for x in scores]
```

这里即使左侧上下文不完整，右侧的 `return [x / total for x in scores]` 也强烈暗示中间应该定义 `total`，并处理零值或空输入。普通续写只能“从上往下猜”，而 FIM 可以利用右侧的 `return` 结构反推中间逻辑。

这也是 InCoder、Code Llama、DeepSeek-Coder、SantaCoder 等代码模型都把 infilling/FIM 作为核心能力的原因$^{[6][7][8][9]}$。

### 3. FIM 的常见训练与推理变体

FIM 并不是只有一种做法，常见差异主要出现在两处：

- **span 如何采样**：随机字符级切分、按 token 切分、按语法节点切分。
- **三段内容如何排序**：常见有 Prefix-Suffix-Middle（PSM）等格式。

较新的工作进一步指出，**“随便随机挖一段”并不总是最优**。如果在代码里按 AST 节点、表达式块、函数体等语法边界来挖空，训练出来的模型更贴近真实代码编辑分布。AST-FIM 在真实代码编辑任务上优于随机切分 FIM$^{[10]}$。

此外，Self-Infilling 还把 FIM 能力从“单次补洞”进一步扩展到“边生成边插空再回填”的非单调解码过程，用来提高长代码生成时的全局一致性$^{[11]}$。

## 为什么 FIM 对代码补全特别重要

真实软件开发里，很多任务都不是从空白文件开始，而是围绕已有代码做局部编辑。FIM 对这些任务非常契合：

- 在函数签名和 `return` 之间补完整个函数体。
- 在已有类中插入一个缺失方法。
- 在 `try:` 与 `except:` 之间补业务逻辑。
- 根据前后文补充 import、参数校验、日志和异常处理。
- 根据测试断言和函数签名，回填实现代码。
- 根据旧代码和新 API 调用方式，在中间插入迁移逻辑。

从论文结果和工业模型设计上看，FIM 已经不是代码模型的边缘能力，而是主流能力之一。DeepSeek-Coder 明确把 fill-in-the-blank 任务纳入预训练与评测中$^{[8]}$；Code Llama 明确提供 infilling-capable 变体$^{[9]}$；SantaCoder 和 InCoder 也都把 infilling 当作主要卖点$^{[6][7]}$。

## 应用场景

### 1. IDE 内联补全

这是最直接的场景。编辑器已经知道光标前后的代码，自然就能把它们作为 `prompt` 和 `suffix` 发送给 FIM 模型。相比普通补全，FIM 更容易补出和后续代码风格、变量名、返回值结构一致的内容。

### 2. 自动修复 TODO / FIXME / pass / ...

例如：

```python
def build_headers(api_key: str) -> dict[str, str]:
    # TODO
    return headers
```

这里右侧的 `return headers` 已经告诉模型中间应该产生一个 `headers` 字典。FIM 很适合这种“中间有洞，但右边答案结构已经暴露”的情况。

### 3. 代码重构和 API 迁移

当你已经确定函数开头和结尾结构，希望模型只插入中间迁移逻辑时，FIM 可以显著减少“模型把后续代码一并改写掉”的风险。

### 4. 模板化文档、配置和结构化文本生成

FIM 不只适用于代码，也适合：

- Markdown 模板中插入指定章节。
- SQL 模板中补齐 `WHERE` / `JOIN` 条件。
- YAML / JSON / TOML 配置中补某个字段块。
- 法务、合同、报表模板中的局部回填。

### 5. Agent 式代码编辑

在 AI coding agent 场景里，很多“精确局部修改”其实都可以抽象为：

```text
prefix + <gap> + suffix
```

此时 FIM 往往比长对话式 code generation 更稳定，因为任务边界更清楚，生成空间更小，也更利于做自动校验。

## DeepSeek FIM Completion API

截至 **2026-04-27**，DeepSeek 官方文档对 FIM Completion 的定义有几个要点$^{[1][2][3]}$：

- FIM 使用的是 **`POST /completions`**，不是 `/chat/completions`。
- 需要把 `base_url` 设为 **`https://api.deepseek.com/beta`**。
- 当前 FIM Completion 参考页列出的模型值是 **`deepseek-v4-pro`**。
- FIM 功能属于 Beta 能力。
- `prompt` 表示左侧上下文，`suffix` 表示右侧上下文。
- FIM 只支持**非思考模式**$^{[3]}$。

官方文档同时说明，DeepSeek API 与 OpenAI SDK 兼容，可以直接使用 `openai` Python SDK，只需要改 `base_url` 和模型名即可$^{[2]}$。

### 关键请求字段

- `model`：当前参考页列出为 `deepseek-v4-pro`$^{[1]}$
- `prompt`：补全缺口左侧内容
- `suffix`：补全缺口右侧内容
- `max_tokens`：中间补全的最大长度
- `temperature`：采样温度；DeepSeek 官方建议代码/数学场景可设为 `0.0`$^{[3]}$
- `stop`：停止序列
- `stream`：是否使用 SSE 流式返回

## Python 项目接入示例

### 1. 安装依赖与环境变量

```bash
pip install openai
```

```bash
set DEEPSEEK_API_KEY=your_api_key_here
```

如果是在 macOS / Linux：

```bash
export DEEPSEEK_API_KEY="your_api_key_here"
```

### 2. 最小可运行示例

下面的例子演示“在函数体中间补全缺失逻辑”：

```python
import os
from openai import OpenAI


client = OpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com/beta",
)

prefix = """def moving_average(nums: list[float], window: int) -> list[float]:
    if window <= 0:
        raise ValueError("window must be positive")

"""

suffix = """
    return result
"""

response = client.completions.create(
    model="deepseek-v4-pro",
    prompt=prefix,
    suffix=suffix,
    max_tokens=160,
    temperature=0.0,
    stop=["\n\nif __name__ == \"__main__\":", "\nclass "],
    stream=False,
)

middle = response.choices[0].text
print("=== middle ===")
print(middle)
print("=== merged ===")
print(prefix + middle + suffix)
```

这个请求的语义很直接：

- `prompt` 给出函数头和前置校验。
- `suffix` 给出函数最终必须回到的收尾结构。
- 模型只负责生成中间的缺失实现。

### 3. 封装成 Python 项目里的可复用函数

如果你要在项目里长期用 FIM，建议封装成一个小工具函数：

```python
import os
from openai import OpenAI


class DeepSeekFIMClient:
    def __init__(self, api_key: str | None = None) -> None:
        self._client = OpenAI(
            api_key=api_key or os.environ["DEEPSEEK_API_KEY"],
            base_url="https://api.deepseek.com/beta",
        )

    def complete(
        self,
        prefix: str,
        suffix: str,
        *,
        model: str = "deepseek-v4-pro",
        max_tokens: int = 256,
        temperature: float = 0.0,
        stop: list[str] | None = None,
    ) -> str:
        response = self._client.completions.create(
            model=model,
            prompt=prefix,
            suffix=suffix,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop,
            stream=False,
        )
        return response.choices[0].text


if __name__ == "__main__":
    fim = DeepSeekFIMClient()
    middle = fim.complete(
        prefix="def add(a: int, b: int) -> int:\n",
        suffix="\n    return result\n",
        max_tokens=64,
    )
    print(middle)
```

### 4. 在代码编辑流程中的典型用法

一个实用的工程模式是：

1. 先定位待编辑代码区域。
2. 取目标位置前面的文本作为 `prefix`。
3. 取目标位置后面的文本作为 `suffix`。
4. 调用 FIM 生成 `middle`。
5. 把 `prefix + middle + suffix` 拼回去。
6. 再做语法检查、单元测试、格式化和静态分析。

这比“把整个文件交给聊天模型，让它重写一遍”更稳，因为修改范围被限定在一个明确缺口里。

## 工程接入建议

### 1. 代码场景优先使用低温度

DeepSeek 官方参数说明里明确建议，代码/数学任务可将 `temperature` 设为 `0.0`$^{[3]}$。对 FIM 来说这尤其合理，因为中间缺口通常需要精确满足右侧约束，而不是追求发散创意。

### 2. 缺口不要切得过大

FIM 很适合补“局部中间片段”，不适合把半个仓库都挖空再让模型补。缺口越大，约束越弱，模型越容易退化回普通开放式生成。

一个常见经验是：优先让缺口对应一个局部可验证单元，例如：

- 一个函数体
- 一个 `if` 分支
- 一个辅助方法
- 一段 import / config block

### 3. `suffix` 应尽量保留强约束

右侧上下文不是可有可无的附加说明，它是 FIM 的核心信息源。像下面这些右侧线索都非常有价值：

- 最终 `return`
- 异常分支
- 变量名
- JSON / YAML 结构闭合
- 缩进层级
- 函数调用签名

### 4. 把 FIM 放进自动验证链路

FIM 只是生成方式，不是 correctness guarantee。真正稳妥的工程做法仍然是：

1. FIM 生成中间代码。
2. 执行语法检查。
3. 执行 formatter / linter。
4. 跑局部测试。
5. 失败时缩小缺口、降低温度或追加更明确上下文后重试。

## 局限与风险

FIM 很强，但它不是万能的。

### 1. 它不天然理解语法边界

如果训练时只是随机挖空，模型学到的缺口分布和真实开发编辑分布不一定一致。这也是 AST-FIM 一类工作的意义：用语法结构约束缺口采样$^{[10]}$。

### 2. 长距离依赖仍然困难

即使有前后文，模型也不一定能稳定处理跨很多文件、很多模块的依赖关系。FIM 更适合“局部但强约束”的任务，不是完整仓库级程序综合的银弹。

### 3. 工程上仍需校验

即使是 FIM，模型也可能：

- 生成语法正确但语义错误的代码
- 使用错误的 API
- 漏掉边界条件
- 产生与右侧上下文不完全匹配的局部变量

所以 FIM 的正确位置是：**更适合代码编辑的生成原语**，而不是“可以跳过验证”的理由。

## 总结

FIM Completion 的核心价值不在于“模型能从中间开始写字”这么简单，而在于它把代码生成从开放式续写，变成了**受前后文双向约束的局部补全问题**。这让它天然适合 IDE、agent、自动修复、局部重构和模板回填。

从研究上看，FIM 已经从基础数据变换，发展到结构感知训练、真实编辑评测和自回填解码；从工业上看，DeepSeek、Code Llama、DeepSeek-Coder、SantaCoder、InCoder 等主流代码模型路线都把它视为核心能力之一$^{[1][6][7][8][9][10][11]}$。

如果你在 Python 项目里接 DeepSeek FIM，当前最重要的实现细节只有三点：

1. 用 OpenAI 兼容 SDK。
2. `base_url` 指向 `https://api.deepseek.com/beta`。
3. 用 `prompt + suffix` 明确表达缺口左右文，并把生成结果放进自动验证链路。

## 参考文献

[[1](https://api-docs.deepseek.com/api/create-completion)] DeepSeek. Create FIM Completion (Beta). *DeepSeek API Docs*, accessed 2026-04-27.

[[2](https://api-docs.deepseek.com/)] DeepSeek. Your First API Call. *DeepSeek API Docs*, accessed 2026-04-27.

[[3](https://api-docs.deepseek.com/zh-cn/quick_start/pricing)] DeepSeek. 模型 & 价格. *DeepSeek API Docs*, accessed 2026-04-27.

[[4](https://api-docs.deepseek.com/quick_start/parameter_settings)] DeepSeek. The Temperature Parameter. *DeepSeek API Docs*, accessed 2026-04-27.

[[5](https://arxiv.org/abs/2207.14255)] Mohammad Bavarian, et al. Efficient Training of Language Models to Fill in the Middle. *arXiv*, 2022.

[[6](https://arxiv.org/abs/2204.05999)] Daniel Fried, et al. InCoder: A Generative Model for Code Infilling and Synthesis. *arXiv*, 2022.

[[7](https://arxiv.org/abs/2301.03988)] Loubna Ben Allal, et al. SantaCoder: don't reach for the stars! *arXiv*, 2023.

[[8](https://arxiv.org/abs/2401.14196)] Daya Guo, et al. DeepSeek-Coder: When the Large Language Model Meets Programming -- The Rise of Code Intelligence. *arXiv*, 2024.

[[9](https://arxiv.org/abs/2308.12950)] Baptiste Rozière, Jonas Gehring, et al. Code Llama: Open Foundation Models for Code. *arXiv*, 2023.

[[10](https://arxiv.org/abs/2506.00204)] Linyuan Gong, Yifang Chen, Arjun Guha, and Paulo Villegas. Structure-Aware Fill-in-the-Middle Pretraining for Code. *arXiv*, 2025.

[[11](https://arxiv.org/abs/2311.17972)] Lin Zheng, Jianbo Yuan, Zhi Zhang, Hongxia Yang, and Lingpeng Kong. Self-Infilling Code Generation. *arXiv*, 2024.
