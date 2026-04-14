---
layout: post
toc: true
title: "LLM 生成(补全)阶段的 Neural Text Degeneration 检测算法设计"
categories: LLM
tags: [LLM, NLP, decoding, neural-text-degeneration, z-algorithm, kmp, repetition-penalty, min-p-sampling, harness-engineering]
author:
  - vortezwohl
  - 吴子豪
  - Codex
---

**Neural Text Degeneration（神经文本退化）** 现象由 Holtzman 等人在同名论文中提出, 论文中它被描述为语言模型解码时生成 bland、incoherent 或陷入 repetitive loops 的退化现象$^{[9]}$。本文聚焦其中最容易工程化检测的一类：LLM 生成到尾部时进入失控重复，同一句话、同一段标点、同一个 JSON 片段、同一个提示模板或者同一小段乱码被连续复制很多次。它看起来像“模型还在正常输出”，实际上已经从任务空间滑进了重复循环。工程上最危险的地方是，这种错误不一定会触发 HTTP 失败，也不一定会破坏纯文本类型约束；如果调用方只检查“非空字符串”或“请求成功”，退化内容就会继续流入摘要、翻译、入库、消息发送和后续 agent 工具调用。解决它的关键不是相信模型自觉停止，而是在推理结果进入业务逻辑前做一层可解释、低成本的重复模式检测。

## 业界痛点

Neural Text Degeneration 通常出现在开放式生成、长文本改写、翻译、代码补全、JSON/Markdown 结构化输出和多轮 agent 轨迹中。它的表象很多：有时是“好的，下面是……”无限重复；有时是 `}`、`</think>`、列表序号、分隔线反复出现；有时是一个看似合理的短句被拼接几十次；还有时是模型在接近 `max_tokens` 时没有收束，继续用高概率模板填满剩余 token。

这类问题的麻烦在于它介于“模型质量问题”和“工程故障”之间。

1. **它不总是语法错误**: 一个重复 40 次的短句仍然是合法字符串，一个重复字段的 Markdown 也可能被渲染出来，甚至一个重复片段拼出来的 JSON 可能在局部看起来是合法的。

2. **它会放大成本**: 退化输出往往在尾部发生，如果没有流式截断或生成后检测，调用方已经为无效 token 付费；如果后续还有 LLM judge、embedding、翻译、入库或人工审核，成本会被继续放大。

3. **它会污染下游**: 摘要任务可能把重复尾巴当成事实，翻译任务可能把重复句子当成原文内容，agent 任务可能把重复工具调用当成可执行计划。对自动化系统来说，Neural Text Degeneration 不是“文风差一点”，而是需要阻断的异常结果。

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

数学上，给定字符串 $$S=s_0s_1\cdots s_{n-1}$$，Z 数组定义为：

$$
Z[0]=n
$$

$$
Z[i]=\max\{\ell \mid 0 \leq \ell \leq n-i,\ S[0:\ell]=S[i:i+\ell]\},\quad 1 \leq i < n
$$

也就是说，$$Z[i]$$ 是后缀 $$S[i:n]$$ 与原串前缀 $$S[0:n]$$ 的最长公共前缀长度。朴素计算每个 $$Z[i]$$ 需要反复比较前缀，最坏会退化到 $$O(n^2)$$；Z 算法用一个右端点最远的匹配窗口 $$[L, R]$$ 缓存已经知道的匹配区间。当 $$i \leq R$$ 时，先复用窗口内的历史结果：

$$
Z[i] \leftarrow \min(R-i+1,\ Z[i-L])
$$

然后再从这个初值继续向右扩展：

$$
\text{while } i+Z[i]<n \text{ and } S[Z[i]]=S[i+Z[i]],\quad Z[i]\leftarrow Z[i]+1
$$

如果扩展后的新窗口超过旧的 $$R$$，就更新：

$$
L\leftarrow i,\quad R\leftarrow i+Z[i]-1
$$

对应到 `RepeatPatternDetector.z_algorithm()`，实现非常直接：

```python
@staticmethod
def z_algorithm(text: str) -> list[int]:
    n = len(text)
    if n == 0:
        return []
    z = [0] * n
    left = 0
    right = 0
    for i in range(1, n):
        if i <= right:
            z[i] = min(right - i + 1, z[i - left])
        while i + z[i] < n and text[z[i]] == text[i + z[i]]:
            z[i] += 1
        if i + z[i] - 1 > right:
            left = i
            right = i + z[i] - 1
    z[0] = n
    return z
```

在 Neural Text Degeneration 的重复模式检测里，我们不一定有外部给定的 `pattern`，所以实现对每个起点 `start` 取一个后缀 `suffix = text[start:]`，对这个后缀计算 Z 数组。然后枚举候选模式长度 `pattern_len`：

```text
suffix = text[start:]
pattern = suffix[0:pattern_len]

如果 suffix[pattern_len:] 的开头还能匹配 suffix 的前缀，
那么 z[pattern_len] 就表示第二段开始后还能连续匹配多少字符。

repeat_count = 1 + z[pattern_len] // pattern_len
```

举例说，后缀是 `abcabcabcx`，候选 `pattern_len = 3`，那么候选模式是 `abc`。`z[3] = 6`，因为从第 3 位开始的 `abcabc...` 和前缀 `abcabc...` 能匹配 6 个字符，所以重复次数是 `1 + 6 // 3 = 3`。

写成数学表达式就是：对原始文本 $$T$$、起点 $$a$$、候选模式长度 $$p$$，令后缀 $$U=T[a:|T|]$$，对 $$U$$ 计算 Z 数组，则连续重复次数为：

$$
r(a,p)=1+\left\lfloor \frac{Z_U[p]}{p}\right\rfloor
$$

只有当 $$r(a,p)\geq 2$$ 时，它才是一个有效重复候选。候选片段和区间为：

$$
\operatorname{pattern}(a,p)=T[a:a+p]
$$

$$
\operatorname{span}(a,p)=[a,\ a+p\cdot r(a,p))
$$

实际实现还做了几个工程化取舍：

- 支持 `ignore_case`，可以在不改变返回原文片段的情况下做大小写归一化匹配。
- 支持 `min_pattern_len` 和 `max_pattern_len`，用来降低过短模式误报，或者限制扫描成本。
- 候选排序优先选择重复次数更多的模式；重复次数相同时，选择覆盖字符更多的模式；覆盖长度也相同时，选择更短的基元模式；最后选择更靠前的起点。

这个检测器不是把整篇文本一次性做到严格线性时间。Z 算法本身对单个后缀是线性的，但外层还会枚举起点和模式长度，因此整体更接近二次扫描。这个取舍在 LLM 输出检测里是可接受的：生成文本长度通常被 `max_tokens` 限制，检测发生在调用边界，换来的是实现简单、行为可解释、误报容易调参。若要处理几十万字符级日志，则应该改成更专门的周期串或 suffix 结构算法。

### 用 KMP 定位已知重复模式

`locate(text, pattern)` 解决的是另一个问题：如果上游已经知道某个模式可疑，如何找到它最长的连续重复区间。这里实现用了 KMP。

KMP 的关键是先为 `pattern` 构造 LPS 表，也就是每个前缀位置上“最长 proper prefix 同时也是 suffix”的长度。匹配时一旦发生不一致，就不用把文本指针回退到朴素算法的下一个窗口，而是利用 LPS 把模式指针跳到可以继续比较的位置，因此整体复杂度是 `O(n + m)`$^{[3]}$ $^{[4]}$。

数学上，给定模式串 $$P=p_0p_1\cdots p_{m-1}$$，LPS 数组可以写成：

$$
\operatorname{lps}[i]=\max\{k \mid 0 \leq k < i+1,\ P[0:k]=P[i-k+1:i+1]\}
$$

这里的 $$k < i+1$$ 排除了整个字符串本身，所以它是 proper prefix。匹配文本 $$T$$ 时，设 $$i$$ 是文本指针，$$j$$ 是模式指针：

$$
T[i]=P[j]\Rightarrow i\leftarrow i+1,\ j\leftarrow j+1
$$

当 $$j=m$$ 时，说明在 $$i-m$$ 位置找到一次完整匹配，然后用 $$\operatorname{lps}[j-1]$$ 继续寻找重叠匹配：

$$
\operatorname{match\_start}=i-m,\quad j\leftarrow \operatorname{lps}[j-1]
$$

当 $$T[i]\neq P[j]$$ 时，如果 $$j>0$$，不回退文本指针，只回退模式指针：

$$
j\leftarrow \operatorname{lps}[j-1]
$$

如果 $$j=0$$，说明当前文本字符无法作为任何匹配前缀，文本指针前进：

$$
i\leftarrow i+1
$$

对应的程序实现分两段。先构造 LPS：

```python
@staticmethod
def __build_lps(pattern: str) -> list[int]:
    lps = [0] * len(pattern)
    length = 0
    i = 1
    while i < len(pattern):
        if pattern[i] == pattern[length]:
            length += 1
            lps[i] = length
            i += 1
        elif length > 0:
            length = lps[length - 1]
        else:
            lps[i] = 0
            i += 1
    return lps
```

再用 LPS 找出所有匹配起点：

```python
@staticmethod
def kmp_find_all(text: str, pattern: str) -> list[int]:
    if not pattern:
        raise ValueError('pattern must not be empty')
    lps = RepeatPatternDetector.__build_lps(pattern=pattern)
    positions: list[int] = []
    i = 0
    j = 0
    while i < len(text):
        if text[i] == pattern[j]:
            i += 1
            j += 1
            if j == len(pattern):
                positions.append(i - j)
                j = lps[j - 1]
        elif j > 0:
            j = lps[j - 1]
        else:
            i += 1
    return positions
```

在当前实现里，KMP 先找出 `pattern` 在文本中的所有起点，然后把这些起点放进集合。接着按起点排序，只从连续重复链的第一个位置开始计数：

```text
step = len(pattern)
如果 start - step 也在 starts 中，说明当前位置不是链头，跳过。
否则从 start 开始，不断检查 start + step、start + 2 * step ...
直到下一段不再出现。
```

这种做法能区分“同一个短句在文章不同位置出现很多次”和“同一个短句在尾部连续重复很多次”。Neural Text Degeneration 的重复模式检测真正关心的是后者，因为连续重复才更像解码退化。

从数学上看，设所有匹配起点集合为：

$$
A=\{i \mid T[i:i+m]=P\}
$$

其中 $$m=|P|$$。连续重复链的链头集合是：

$$
H=\{h \in A \mid h-m \notin A\}
$$

从链头 $$h$$ 出发，连续重复次数为：

$$
c(h)=\max\{q \mid q\geq 1,\ \forall 0\leq t<q,\ h+t\cdot m \in A\}
$$

`locate(text, pattern)` 返回的就是：

$$
h^*=\arg\max_{h\in H}(c(h), -h)
$$

也就是重复次数最多、重复次数相同时起点更靠前的连续区间。对应实现是：

```python
def __longest_contiguous_repeat_substring(
    self,
    text: str,
    pattern: str,
) -> PatternMatch | None:
    if not pattern:
        raise ValueError('pattern must not be empty')
    text = text or ''
    normalized_text = self._normalize(text=text)
    normalized_pattern = self._normalize(text=pattern)
    starts = set(self.kmp_find_all(text=normalized_text, pattern=normalized_pattern))
    if not starts:
        return None
    step = len(pattern)
    best: PatternMatch | None = None
    for start in sorted(starts):
        if start - step in starts:
            continue
        count = 1
        pos = start + step
        while pos in starts:
            count += 1
            pos += step
        matched_pattern = text[start:start + step]
        candidate = PatternMatch(
            pattern=matched_pattern,
            repeat=count,
            start=start,
            end=start + count * step
        )
        if best is None:
            best = candidate
            continue
        if count > best.repeat or (count == best.repeat and start < best.start):
            best = candidate
    return best
```

### Neural Text Degeneration 检测算法的数学表示和程序实现

把 Z 算法和候选排序合起来，`detect(text)` 实际是在求一个最优重复候选。设归一化后的文本为 $$T$$，长度为 $$n$$，最小候选模式长度为 $$p_{\min}$$，最大候选模式长度为 $$p_{\max}$$。如果没有显式传入 `max_pattern_len`，则：

$$
p_{\max}=\left\lfloor \frac{n}{2}\right\rfloor
$$

对每个起点 $$a$$，只要后缀长度还足够容纳两段最小模式，就继续扫描：

$$
n-a \geq 2p_{\min}
$$

对每个候选长度 $$p$$，有效范围是：

$$
p_{\min}\leq p\leq \min\left(p_{\max},\left\lfloor\frac{n-a}{2}\right\rfloor\right)
$$

重复次数仍然由该后缀的 Z 数组给出：

$$
r(a,p)=1+\left\lfloor \frac{Z_{T[a:n]}[p]}{p}\right\rfloor
$$

如果 $$r(a,p)<2$$，它不是连续重复候选；否则得到候选：

$$
M(a,p)=(T[a:a+p],\ r(a,p),\ a,\ a+p\cdot r(a,p))
$$

当前实现的候选排序 key 是：

$$
K(a,p)=\left(r(a,p),\ p\cdot r(a,p),\ -p,\ -a\right)
$$

因此最优匹配是：

$$
M^*=\arg\max_{(a,p)} K(a,p)
$$

业务层再用阈值 $$\tau$$ 判定是否出现重复型 Neural Text Degeneration：

$$
\operatorname{is\_degenerate}(x)=
\begin{cases}
\operatorname{true}, & M^*\neq \varnothing \land M^*.\operatorname{repeat}>\tau \\
\operatorname{false}, & \text{otherwise}
\end{cases}
$$

对应的核心实现如下$^{[6]}$：

```python
def __most_repeated_substring(self, text: str) -> PatternMatch | None:
    text = text or ''
    normalized_text = self._normalize(text=text)
    n = len(normalized_text)
    if n == 0:
        return None
    if self._max_pattern_len is None:
        max_pattern_len = n // 2
    else:
        max_pattern_len = min(self._max_pattern_len, n // 2)
    best: PatternMatch | None = None
    for start in range(n):
        suffix = normalized_text[start:]
        if len(suffix) < self._min_pattern_len * 2:
            break
        z = self.z_algorithm(text=suffix)
        upper = min(max_pattern_len, len(suffix) // 2)
        for pattern_len in range(self._min_pattern_len, upper + 1):
            repeat_count = 1 + z[pattern_len] // pattern_len
            if repeat_count < 2:
                continue
            pattern = text[start:start + pattern_len]
            end = start + pattern_len * repeat_count
            candidate = PatternMatch(
                pattern=pattern,
                repeat=repeat_count,
                start=start,
                end=end
            )
            if self.__is_better_match(candidate=candidate, best=best):
                best = candidate
    return best
```

排序函数也很关键。它不是简单地找最长片段，而是优先找重复次数最多的片段：

```python
@staticmethod
def __is_better_match(candidate: PatternMatch, best: PatternMatch | None) -> bool:
    if best is None:
        return True
    candidate_key = (
        candidate.repeat,
        len(candidate.pattern) * candidate.repeat,
        -len(candidate.pattern),
        -candidate.start
    )
    best_key = (
        best.repeat,
        len(best.pattern) * best.repeat,
        -len(best.pattern),
        -best.start
    )
    return candidate_key > best_key
```

### 在 LLM 调用链路里使用

在 `any_llm.llm.LLM.__call__` 的实践里，检测器被放在 HTTP 调用成功、响应非空之后。更推荐的命名方式是使用 Neural Text Degeneration 语义，并兼容旧的 `LONG_TAIL_REPEAT_THRESHOLD` 环境变量：

```python
threshold = int(os.getenv(
    'NEURAL_TEXT_DEGENERATION_REPEAT_THRESHOLD',
    os.getenv('LONG_TAIL_REPEAT_THRESHOLD', 32)
))
pattern_match = repeat_pattern_detector(res)
if pattern_match and pattern_match.repeat > threshold:
    raise ValueError(
        'Neural Text Degeneration pattern detected. '
        'Try reducing your `top_p` parameter.'
    )
```

这里默认阈值是 `32`。旧变量名 `LONG_TAIL_REPEAT_THRESHOLD` 是历史命名，更准确的语义是“重复型 Neural Text Degeneration 的重复次数阈值”：如果任意连续重复模式超过阈值，就把这次 LLM 输出视为无效结果。由于外层重试装饰器会捕获 `ValueError`，这类异常可以进入统一重试逻辑，而不是把坏结果返回给业务层。

这个阈值不应该被理解为普适常数。不同任务要分开调：

- 结构化 JSON、代码、翻译、摘要：阈值可以更低，因为重复通常就是错误。
- 诗歌、歌词风格、表格、列表、测试样例生成：阈值要更保守，因为合法重复更多。
- 字符级检测容易命中标点和换行，可以提高 `min_pattern_len` 或在业务层忽略纯标点模式。
- 长文生成最好同时看 `repeat` 和重复片段覆盖长度。`"。" * 40` 与一个 50 字短句重复 6 次，风险形态不同。

也要承认这个算法的边界。它擅长发现逐字连续重复，不擅长发现语义重复，例如“我理解了 / 明白了 / 可以的”这种变体循环；它也不判断输出是否事实正确、格式是否完整、是否符合业务 schema。因此它应该和 JSON schema、正则结构检查、关键词黑名单、最大长度、流式 early stop、业务语义校验一起使用，而不是单独承担全部质量控制。

### 重试机制的实现

重复型 Neural Text Degeneration 检测本身只负责把坏输出变成确定的失败信号。要让系统自动恢复，还需要把这个失败信号接入重试机制。`vortezwohl.func.Retry` 提供了两种边界校验方式$^{[8]}$：

1. `on_return(validator)`：函数正常返回，但返回值不满足校验器时重试。
2. `on_exceptions(*exceptions)`：函数抛出指定异常类型时重试，抛出其它异常时直接重新抛出。

`any_llm` 的 Neural Text Degeneration 检测走的是第二种：检测到重复模式超过阈值后抛出 `ValueError`，而 `LLM.__call__` 外层装饰器把 `ValueError` 纳入可重试异常集合$^{[7]}$。

```python
retry = Retry(max_retries=2, delay=True)


@retry.on_exceptions(ValueError, HTTPError, ConnectionError, SSLError, Timeout, ConnectTimeout, ReadTimeout)
def __call__(self, user_message: str, system_message: str | None = None, **kwargs):
    ...
    threshold = int(os.getenv(
        'NEURAL_TEXT_DEGENERATION_REPEAT_THRESHOLD',
        os.getenv('LONG_TAIL_REPEAT_THRESHOLD', 32)
    ))
    pattern_match = repeat_pattern_detector(res)
    if pattern_match and pattern_match.repeat > threshold:
        raise ValueError('Neural Text Degeneration pattern detected.')
    return res
```

数学上，可以把异常重试写成一个有限状态过程。设被包装函数为 $$f$$，可重试异常集合为 $$E$$，最大重试次数为 $$R$$。注意这里的 $$R$$ 是失败后的 retry 次数，不是总尝试次数；总尝试次数最多是：

$$
A_{\max}=R+1
$$

第 $$a$$ 次尝试的结果为：

$$
Y_a=f(x)
$$

如果 $$Y_a$$ 正常返回，则直接返回；如果抛出异常 $$e_a$$，判断：

$$
\operatorname{retryable}(e_a)=\exists E_i\in E,\ e_a \text{ is instance of } E_i
$$

当 $$\operatorname{retryable}(e_a)=\operatorname{false}$$ 时，异常立即向上抛出；当它为真且 $$a<R$$ 时进入下一次尝试；当它为真且 $$a=R$$ 时，抛出 `MaxRetriesReachedError`。

如果启用 `delay=True`，每次重试前会进入指数退避加随机抖动。源码中的 `sleep(retries, base=2., max_delay=600.)` 可以写成：

$$
k=\max(\operatorname{retries},1)
$$

$$
d=\min(2^k,600)
$$

$$
\operatorname{sleep\_time}=d+U(0.1,d)
$$

对应实现是：

```python
def sleep(retries: int, base: float = 2., max_delay: float = 600.):
    retries = max(retries, 1)
    delay = min(base ** retries, max_delay)
    time.sleep(delay + random.uniform(.1, delay))
    return
```

`on_exceptions()` 的核心逻辑是先调用一次 `validator()`，它负责执行真实函数并捕获异常；如果异常类型属于可重试集合，就返回 `need_retry=True`，否则直接抛出：

```python
def validator() -> tuple[tuple, bool, Any]:
    result = None
    need_retry = False
    error_type = None
    error_super_type = None
    error = None
    try:
        result = func(*_args, **_kwargs)
    except Exception as e:
        reraise = True
        for exception in _exceptions:
            if isinstance(e, exception):
                reraise = False
                need_retry = True
                error = e
                error_type = e.__class__.__name__
                error_super_type = exception.__name__
                break
        if reraise:
            raise e
    return (error, error_type, error_super_type), need_retry, result
```

然后根据 `max_retries` 决定是无限重试还是有限重试。有限重试的核心循环是：

```python
for retry_count in range(self._max_retries):
    if self._delay:
        sleep(retries=retry_count)
    (_error, _error_type, _error_super_type), _need_retry, _result = validator()
    if not _need_retry:
        return _result
    else:
        logger.debug(...)
raise MaxRetriesReachedError(
    retries=self._max_retries,
    message=f'{_error_type}({_error_super_type}) occurred: {str(_error)}\n'
            f'Returns: {_result}'
)
```

`on_return()` 的形式类似，只是失败信号来自返回值校验器而不是异常集合。设校验器为 $$g$$，则：

$$
\operatorname{valid}(Y_a)=g(Y_a)
$$

当 $$g(Y_a)=\operatorname{false}$$ 时重试，直到某次返回值通过校验，或者重试次数耗尽后抛出 `MaxRetriesReachedError`。这类模式适合“HTTP 成功但业务响应为空”“JSON 能解析但 schema 不合格”这类失败；重复型 Neural Text Degeneration 检测则更适合在业务函数内部抛出 `ValueError`，让它和 HTTP 错误、超时错误进入同一条异常重试链路。

## 如何缓解

检测只能阻断坏输出，缓解要回到解码参数和调用策略。Neural Text Degeneration 不是单一原因造成的，不能只靠一个参数兜底；更稳妥的做法是把长度约束、停止条件、截断采样、重复惩罚和失败重试组合起来。

1. **控制 `max_tokens`**: 退化输出经常发生在模型已经回答完、但仍被允许继续生成的时候。对摘要、分类、抽取、短翻译这类任务，不要给一个过大的输出上限。能用 300 token 完成的任务，不应该默认给 4000 token。

2. **设置停止条件**: 对于结构化输出，可以用明确的 stop sequence、闭合标签或 JSON schema 解析作为终止依据。流式调用时，如果检测到同一片段开始连续重复，可以提前 abort，避免等到整个 `max_tokens` 用完。

3. **调整 `top_p`**: Top-P / Nucleus Sampling 会选择累积概率达到阈值 `P` 的最小候选集合，再在这个集合里采样；它本来就是为缓解开放域生成中的 Neural Text Degeneration 而提出的采样策略之一$^{[5]}$ $^{[9]}$。如果退化来自过宽的候选空间和低质量尾部 token，可以尝试降低 `top_p`，例如从 `1.0` 降到 `0.9`、`0.8`，让采样空间更集中。这也是当前 `any_llm` 实践里检测失败后给出的默认建议。

4. **使用 Min-p Sampling**: Min-p Sampling 是一种动态截断策略，它不使用固定累计概率阈值，而是用当前最高概率 token 作为参照，只保留相对概率足够高的候选 token$^{[11]}$。设当前步归一化概率为 $$p_i$$，最高概率为 $$p_{\max}=\max_j p_j$$，Min-p 阈值为 $$\alpha$$，候选集合为：

$$
V_{\operatorname{min-p}}=\{i \mid p_i \geq \alpha \cdot p_{\max}\}
$$

然后在 $$V_{\operatorname{min-p}}$$ 上重新归一化并采样。它的直觉是：当模型很确定时，候选集合会更窄；当模型本来就不确定时，候选集合会保留更多合理分支。对于高温采样下的创意生成，它通常比单纯固定 `top_p` 更自适应，但仍然需要按任务调参。

5. **使用 Repetition Penalty**: Repetition Penalty 会对已经生成过的 token 降低再次出现的倾向，CTRL 论文中也使用了这种重复惩罚思路来减少退化重复$^{[10]}$。一种常见实现是在采样前修改 logits。设原始 logit 为 $$z_i$$，已生成 token 集合为 $$G$$，惩罚系数为 $$\theta \geq 1$$，则可以用下面的符号表示：

$$
z'_i=
\begin{cases}
z_i/\theta, & i\in G \land z_i>0 \\
z_i\cdot\theta, & i\in G \land z_i\leq 0 \\
z_i, & i\notin G
\end{cases}
$$

再用 $$z'_i$$ 进入 softmax 和后续采样。`repetition_penalty` 太低时几乎不起作用，太高时会压制必要复现，例如术语、变量名、表格列名和诗歌回环，因此它更适合作为重复退化的软约束，而不是替代输出校验。

6. **调整 `temperature`**: Temperature 通过缩放 logits 改变概率分布：`T < 1` 会让分布更陡峭，输出更确定；`T > 1` 会让分布更平坦，输出更多样但也更不稳定$^{[5]}$。如果重复来自高温采样导致的跑偏，可以降低 temperature；如果重复来自极低温或贪心式的固定模板自循环，则可以小幅提高 temperature，或者配合 Repetition Penalty、Min-p Sampling、frequency penalty、presence penalty 等服务端参数。不要机械地把所有任务都调成同一个温度。

7. **失败后改变参数重试，而不是原样重试**: 如果检测器已经证明某个参数组合产生了退化重复，原样重试可能只是再次采样到同类坏结果。更好的策略是按任务类型选择降级路径：降低 `top_p`、切换或收紧 `min_p`、提高 `repetition_penalty`、收紧 `max_tokens`、加入 stop sequence、提高 `min_pattern_len` 后复检、改用更强模型，或者返回可解释错误。

8. **把 Neural Text Degeneration 检测放到统一验证层**: 我的偏好是把它和“空响应检测、格式检测、schema 校验、业务语义校验”放在同一层：HTTP 成功只代表模型服务返回了东西，不代表输出可用。LLM 可能返回错误格式、错误事实、重复尾巴或半截 JSON；调用方必须把这些都当成不同的失败类型来处理。

一个可执行的默认策略可以是：

```text
1. 生成后检查空响应。
2. 检查重复模式，repeat > threshold 则判定重复型 Neural Text Degeneration。
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

[[6](https://github.com/vortezwohl/MyToolSuite/blob/main/vortezwohl/nlp/repeat_pattern_detector.py)] vortezwohl. BasePatternDetector and RepeatPatternDetector source code. *MyToolSuite / GitHub*, 2026.

[7] vortezwohl. Neural Text Degeneration repeat-pattern detection practice in `LLM.__call__`. *any-llm-sdk / local source code*, 2026. `~\project\any-llm-sdk\any_llm\llm.py`.

[[8](https://github.com/vortezwohl/MyToolSuite/blob/main/vortezwohl/func/retry.py)] vortezwohl. Retry source code. *MyToolSuite / GitHub*, 2026.

[[9](https://openreview.net/forum?id=rygGQyrFvH)] Ari Holtzman, Jan Buys, Li Du, Maxwell Forbes, and Yejin Choi. The Curious Case of Neural Text Degeneration. *ICLR*, 2020.

[[10](https://arxiv.org/abs/1909.05858)] Nitish Shirish Keskar, Bryan McCann, Lav R. Varshney, Caiming Xiong, and Richard Socher. CTRL: A Conditional Transformer Language Model for Controllable Generation. *arXiv*, 2019.

[[11](https://arxiv.org/abs/2407.01082)] Minh Nhat Nguyen, Andrew Baker, Clement Neo, Allen Roush, Andreas Kirsch, and Ravid Shwartz-Ziv. Turning Up the Heat: Min-p Sampling for Creative and Coherent LLM Outputs. *ICLR*, 2025.
