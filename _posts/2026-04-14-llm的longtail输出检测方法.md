---
layout: post
toc: true
title: "LLM long-tail 输出检测算法设计"
categories: LLM
tags: [LLM, NLP, decoding, longtail, z-algorithm, kmp, harness-engineering]
author:
  - vortezwohl
  - 吴子豪
  - Codex
---

这里说的 longtail 输出，不是统计意义上的长尾分布，而是 LLM 生成到尾部时进入失控重复：同一句话、同一段标点、同一个 JSON 片段、同一个提示模板或者同一小段乱码被连续复制很多次。它看起来像“模型还在正常输出”，实际上已经从任务空间滑进了重复循环。工程上最危险的地方是，这种错误不一定会触发 HTTP 失败，也不一定会破坏纯文本类型约束；如果调用方只检查“非空字符串”或“请求成功”，长尾内容就会继续流入摘要、翻译、入库、消息发送和后续 agent 工具调用。解决它的关键不是相信模型自觉停止，而是在推理结果进入业务逻辑前做一层可解释、低成本的重复模式检测。

## 业界痛点

LLM 的长尾输出通常出现在开放式生成、长文本改写、翻译、代码补全、JSON/Markdown 结构化输出和多轮 agent 轨迹中。它的表象很多：有时是“好的，下面是……”无限重复；有时是 `}`、`</think>`、列表序号、分隔线反复出现；有时是一个看似合理的短句被拼接几十次；还有时是模型在接近 `max_tokens` 时没有收束，继续用高概率模板填满剩余 token。

这类问题的麻烦在于它介于“模型质量问题”和“工程故障”之间。

1. **它不总是语法错误**: 一个重复 40 次的短句仍然是合法字符串，一个重复字段的 Markdown 也可能被渲染出来，甚至一个重复片段拼出来的 JSON 可能在局部看起来是合法的。

2. **它会放大成本**: 长尾输出往往在尾部发生，如果没有流式截断或生成后检测，调用方已经为无效 token 付费；如果后续还有 LLM judge、embedding、翻译、入库或人工审核，成本会被继续放大。

3. **它会污染下游**: 摘要任务可能把重复尾巴当成事实，翻译任务可能把重复句子当成原文内容，agent 任务可能把重复工具调用当成可执行计划。对自动化系统来说，长尾输出不是“文风差一点”，而是需要阻断的异常结果。

4. **它很难靠单一提示词消除**: 提示词里写“不要重复”可以降低概率，但不能作为可靠约束。模型采样依然受 `temperature`、`top_p`、`max_tokens`、停止词、上下文长度、服务端解码实现和模型本身退化模式影响。更稳妥的做法是把它当成 harness engineering 问题：生成后必须检查，检查失败必须重试、降级或阻断。

## 如何检测

最朴素的办法是扫描文本里有没有连续重复片段。问题在于重复片段的长度事先不知道：可能是 1 个字符，也可能是 8 个字、20 个字符，甚至是一段 Markdown。`vortezwohl.nlp.RepeatPatternDetector` 的实现把这个问题拆成两个能力：

1. `detect(text)`：不知道重复模式是什么时，自动找出最长的连续重复片段。
2. `locate(text, pattern)`：已经知道某个模式时，定位它在文本里最长的连续重复区间。

底层返回值是一个很小的结构：

```python
PatternMatch(
    pattern="重复片段",
    repeat=42,
    start=128,
    end=380,
)
```

这里的 `pattern` 是被重复的最小候选片段，`repeat` 是连续重复次数，`start` 和 `end` 是原文里的字符区间。业务层不需要理解整个文本，只需要根据 `repeat`、片段长度和位置判断是否阻断。

### 用 Z 算法发现未知重复模式

Z 算法的核心是为字符串 `s` 计算一个 Z 数组：`z[i]` 表示从位置 `i` 开始的后缀与整个字符串前缀能匹配多长。常见字符串匹配场景会构造 `pattern + 分隔符 + text`，再用 Z 数组找出模式出现位置；Z 数组本身可以在线性时间内计算，因为它维护了一个最靠右的 Z-box，并复用窗口内已经算过的前缀匹配信息$^{[1]}$ $^{[2]}$。

在 longtail 检测里，我们不一定有外部给定的 `pattern`，所以实现对每个起点 `start` 取一个后缀 `suffix = text[start:]`，对这个后缀计算 Z 数组。然后枚举候选模式长度 `pattern_len`：

```text
suffix = text[start:]
pattern = suffix[0:pattern_len]

如果 suffix[pattern_len:] 的开头还能匹配 suffix 的前缀，
那么 z[pattern_len] 就表示第二段开始后还能连续匹配多少字符。

repeat_count = 1 + z[pattern_len] // pattern_len
```

举例说，后缀是 `abcabcabcx`，候选 `pattern_len = 3`，那么候选模式是 `abc`。`z[3] = 6`，因为从第 3 位开始的 `abcabc...` 和前缀 `abcabc...` 能匹配 6 个字符，所以重复次数是 `1 + 6 // 3 = 3`。

实际实现还做了几个工程化取舍：

- 支持 `ignore_case`，可以在不改变返回原文片段的情况下做大小写归一化匹配。
- 支持 `min_pattern_len` 和 `max_pattern_len`，用来降低过短模式误报，或者限制扫描成本。
- 候选排序优先选择重复次数更多的模式；重复次数相同时，选择覆盖字符更多的模式；覆盖长度也相同时，选择更短的基元模式；最后选择更靠前的起点。

这个检测器不是把整篇文本一次性做到严格线性时间。Z 算法本身对单个后缀是线性的，但外层还会枚举起点和模式长度，因此整体更接近二次扫描。这个取舍在 LLM 输出检测里是可接受的：生成文本长度通常被 `max_tokens` 限制，检测发生在调用边界，换来的是实现简单、行为可解释、误报容易调参。若要处理几十万字符级日志，则应该改成更专门的周期串或 suffix 结构算法。

### 用 KMP 定位已知重复模式

`locate(text, pattern)` 解决的是另一个问题：如果上游已经知道某个模式可疑，如何找到它最长的连续重复区间。这里实现用了 KMP。

KMP 的关键是先为 `pattern` 构造 LPS 表，也就是每个前缀位置上“最长 proper prefix 同时也是 suffix”的长度。匹配时一旦发生不一致，就不用把文本指针回退到朴素算法的下一个窗口，而是利用 LPS 把模式指针跳到可以继续比较的位置，因此整体复杂度是 `O(n + m)`$^{[3]}$ $^{[4]}$。

在当前实现里，KMP 先找出 `pattern` 在文本中的所有起点，然后把这些起点放进集合。接着按起点排序，只从连续重复链的第一个位置开始计数：

```text
step = len(pattern)
如果 start - step 也在 starts 中，说明当前位置不是链头，跳过。
否则从 start 开始，不断检查 start + step、start + 2 * step ...
直到下一段不再出现。
```

这种做法能区分“同一个短句在文章不同位置出现很多次”和“同一个短句在尾部连续重复很多次”。longtail 检测真正关心的是后者，因为连续重复才更像解码退化。

### 在 LLM 调用链路里使用

在 `any_llm.llm.LLM.__call__` 的实践里，检测器被放在 HTTP 调用成功、响应非空之后：

```python
pattern_match = repeat_pattern_detector(res)
if pattern_match:
    if pattern_match.repeat > int(os.getenv('LONG_TAIL_REPEAT_THRESHOLD', 32)):
        raise ValueError(
            'Long-tail pattern detected. Try reducing your `top_p` parameter.'
        )
```

这里默认阈值是 `LONG_TAIL_REPEAT_THRESHOLD=32`。它的语义很直接：如果任意连续重复模式超过阈值，就把这次 LLM 输出视为无效结果。由于外层重试装饰器会捕获 `ValueError`，这类异常可以进入统一重试逻辑，而不是把坏结果返回给业务层。

这个阈值不应该被理解为普适常数。不同任务要分开调：

- 结构化 JSON、代码、翻译、摘要：阈值可以更低，因为重复通常就是错误。
- 诗歌、歌词风格、表格、列表、测试样例生成：阈值要更保守，因为合法重复更多。
- 字符级检测容易命中标点和换行，可以提高 `min_pattern_len` 或在业务层忽略纯标点模式。
- 长文生成最好同时看 `repeat` 和重复片段覆盖长度。`"。" * 40` 与一个 50 字短句重复 6 次，风险形态不同。

也要承认这个算法的边界。它擅长发现逐字连续重复，不擅长发现语义重复，例如“我理解了 / 明白了 / 可以的”这种变体循环；它也不判断输出是否事实正确、格式是否完整、是否符合业务 schema。因此它应该和 JSON schema、正则结构检查、关键词黑名单、最大长度、流式 early stop、业务语义校验一起使用，而不是单独承担全部质量控制。

## 如何缓解

检测只能阻断坏输出，缓解要回到解码参数和调用策略。

1. **控制 `max_tokens`**: 长尾输出经常发生在模型已经回答完、但仍被允许继续生成的时候。对摘要、分类、抽取、短翻译这类任务，不要给一个过大的输出上限。能用 300 token 完成的任务，不应该默认给 4000 token。

2. **设置停止条件**: 对于结构化输出，可以用明确的 stop sequence、闭合标签或 JSON schema 解析作为终止依据。流式调用时，如果检测到同一片段开始连续重复，可以提前 abort，避免等到整个 `max_tokens` 用完。

3. **调整 `top_p`**: Top-P 采样会选择累积概率达到阈值 `P` 的最小候选集合，再在这个集合里采样；它本来就是为了在开放域生成中平衡多样性和退化问题而被广泛使用$^{[5]}$。如果长尾来自过宽的候选空间和低质量尾部 token，可以尝试降低 `top_p`，例如从 `1.0` 降到 `0.9`、`0.8`，让采样空间更集中。这也是当前 `any_llm` 实践里检测失败后给出的默认建议。

4. **调整 `temperature`**: Temperature 通过缩放 logits 改变概率分布：`T < 1` 会让分布更陡峭，输出更确定；`T > 1` 会让分布更平坦，输出更多样但也更不稳定$^{[5]}$。如果重复来自高温采样导致的跑偏，可以降低 temperature；如果重复来自极低温或贪心式的固定模板自循环，则可以小幅提高 temperature，或者配合 repetition penalty、frequency penalty、presence penalty 等服务端参数。不要机械地把所有任务都调成同一个温度。

5. **失败后改变参数重试，而不是原样重试**: 如果检测器已经证明某个参数组合产生了长尾，原样重试可能只是再次采样到同类坏结果。更好的策略是按任务类型选择降级路径：降低 `top_p`、收紧 `max_tokens`、加入 stop sequence、提高 `min_pattern_len` 后复检、改用更强模型，或者返回可解释错误。

6. **把 longtail 检测放到统一验证层**: 我的偏好是把它和“空响应检测、格式检测、schema 校验、业务语义校验”放在同一层：HTTP 成功只代表模型服务返回了东西，不代表输出可用。LLM 可能返回错误格式、错误事实、重复尾巴或半截 JSON；调用方必须把这些都当成不同的失败类型来处理。

一个可执行的默认策略可以是：

```text
1. 生成后检查空响应。
2. 检查重复模式，repeat > threshold 则判定 longtail。
3. 检查任务格式，例如 JSON schema、Markdown section、代码块闭合。
4. 失败时带上错误类型重试，并按错误类型调整参数。
5. 重试仍失败时返回结构化错误，不把坏文本交给下游。
```

这套策略的核心不是某个字符串算法有多聪明，而是“不信任 LLM 的最后一公里”。Z 算法和 KMP 只是让这层不信任变得便宜、确定、可解释。

## 参考文献

[[1](https://www.geeksforgeeks.org/dsa/z-algorithm-linear-time-pattern-searching-algorithm/)] GeeksforGeeks. Z algorithm (Linear time pattern searching Algorithm). *GeeksforGeeks*, n.d.

[[2](https://www.cnblogs.com/xxeray/p/z-algorithm.html)] XxEray. Z 算法/拓展 KMP 详解. *博客园*, n.d.

[[3](https://www.geeksforgeeks.org/dsa/kmp-algorithm-for-pattern-searching/)] GeeksforGeeks. KMP Algorithm for Pattern Searching. *GeeksforGeeks*, n.d.

[[4](https://drbtaneja.com/knuth-morris-pratt-kmp-algorithm/)] Balvinder Taneja. Knuth-Morris-Pratt (KMP) Algorithm. *Dr. Balvinder Taneja*, n.d.

[[5](https://vortezwohl.github.io/nlp/2025/08/18/%E5%9B%A0%E6%9E%9C%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B%E8%A7%A3%E7%A0%81%E7%AD%96%E7%95%A5.html)] vortezwohl. 文本生成算法中, 采样与解码的基本原理: Top-K, Top-P, Temperature, Beam Search. *vortezwohl.github.io*, 2025.

[[6](https://github.com/vortezwohl/MyToolSuite/blob/main/vortezwohl/nlp/repeat_pattern_detector.py)] vortezwohl. BasePatternDetector and RepeatPatternDetector source code. *MyToolSuite / GitHub*, 2026. See also: [base_pattern_detector.py](https://github.com/vortezwohl/MyToolSuite/blob/main/vortezwohl/nlp/base_pattern_detector.py).

[7] vortezwohl. LLM long-tail output detection practice in `LLM.__call__`. *any-llm-sdk / local source code*, 2026. `~\project\any-llm-sdk\any_llm\llm.py`.
