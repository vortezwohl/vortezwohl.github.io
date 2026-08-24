---
layout: post
toc: true
title: "Grammar-Constrained Decoding：让语言模型输出合法的魔术"
categories: AI
tags: [AI, LLM, Structured Generation, Grammar, Decoding, Agent, Function Calling]
author:
  - vortezwohl
  - 吴子豪
excerpt: "这是一篇从零开始理解 Grammar-Constrained Decoding（GCD，语法约束解码）的教程。文章先把大语言模型解释成“根据前文猜下一个词”的自动补词机，再用 JSON 小例子说明为什么提示词不能稳定保证格式；随后用铁路道岔、门卫和地图逐步引出 grammar、合法前缀、parser state、tokenizer、token mask 和状态转移。读者会看到一个 token 如何在采样前被允许或拒绝，理解格式稳定来自什么数学不变式，也会知道 GCD 不保证事实、语义、权限和工具决策。后半部分再介绍 FSM、CFG、JSON Schema、PICARD、Outlines、LM Format Enforcer、llama.cpp、XGrammar、vLLM、Structured Outputs、分布偏移、拒答设计、性能优化、生产分层和评测方法。"
---

> **给第一次接触这个主题的读者：不要一上来背公式。** 这篇文章安排成一堂循序渐进的课。你只需要先记住一句话：**模型负责“从允许的选项里挑哪个”，grammar 负责“哪些选项根本不能出现”。** 后面的数学、解析器、tokenizer 和工程框架，都是在精确描述这句话。
>
> 文中的公式全部使用 Markdown 数学格式：行内公式写成 `$...$`，独立公式写成 `$$...$$`。引用使用 `[^n]` 脚注。公开资料核验截至 2026 年 8 月；论文中的性能数字只对其特定实验条件负责。

## 0. 阅读引导

如果你完全不懂 GCD，建议按下面的顺序阅读，不要跳到实现框架：

1. **先懂模型在做什么**：模型不是一次性写完答案，而是一次只猜一个 token。
2. **再懂格式为什么会错**：模型有概率，没有外部规则的硬门禁。
3. **再懂 grammar 是什么**：它像一张规定“哪些完整字符串算合法”的地图。
4. **再懂 parser state**：解析器每读一个字符，都记住自己现在走到地图的哪一格。
5. **再懂 token 和字符的差别**：模型选择 token，但 grammar 通常描述字符或结构。
6. **最后看公式和工程**：mask、状态转移、缓存、GPU overlap 都是前面直觉的精确实现。

读完全文，你应该能自己回答五个问题：

- GCD 究竟在模型生成的哪一刻介入？
- 为什么它能把某些非法格式的概率变成零？
- tokenizer 为什么会让看似简单的约束变难？
- “格式合法”为什么不等于“答案正确”？
- 什么时候值得付出延迟和实现复杂度使用它？

## 1. 第一课：把大语言模型想成一个自动补词机

### 1.1 模型不是“直接输出一段话”

我们平时说“模型回答问题”，容易产生一个错觉：好像模型先在脑中写好完整答案，然后一次性把答案交出来。实际运行方式更像手机输入法的自动补全，只是规模大得多：

1. 读取输入和已经生成的内容；
2. 在词表中给下一项打分；
3. 选出一个 token；
4. 把这个 token 接到末尾；
5. 再重复一次。

这里的 token 可以是一个字、一个词、半个词、标点，甚至一小段常见字符串。给定输入 $x$ 和前缀 $y_{<t}$，模型每一步都在估计：

$$
y_t \sim P_{\mathrm{LM}}(y_t\mid x,y_{<t})
$$

**人话翻译**：在已经看到的内容后面，词表里的每个候选有多大可能成为下一个 token。

模型先产生 logits $z_t$，再转换成概率。logits 只是“分数”，分数越高，通常越容易被选中；它还没有表示任何外部 grammar 的许可：

$$
P_{\mathrm{LM}}(y_t=v\mid x,y_{<t})=\operatorname{softmax}(z_t)_v
$$

### 1.2 一个很小的例子

假设我们要求模型只输出下面这种 JSON：

~~~json
{"name": "Alice", "age": 20}
~~~

普通模型可能输出正确版本，也可能输出：

~~~text
当然可以，下面是结果：
{"name": "Alice", "age": "twenty",}
~~~

这段输出对人类也许“看得懂”，但机器会发现至少有四个问题：

- JSON 前面有解释文字；
- age 应该是数字，却变成字符串；
- 末尾有多余逗号；
- 模型可能漏掉 required 字段，或者提前结束。

**关键点**：模型不是“不知道 JSON 长什么样”，而是它每一步只在最大化语言概率，没有被一个外部裁判告知“这条路以后必须还能完成为指定 JSON”。

### 1.3 提示词为什么不是硬保证

提示词可以说“只输出 JSON，不要解释”，这会提高正确格式的概率。但它仍然只是影响 $P_{\mathrm{LM}}$：

- 它不能让非法 token 的概率严格变成零；
- temperature、采样随机性、长输出和复杂嵌套会重新放大错误；
- 模型可能把“内容正确”看得比“严格遵守格式”更重要；
- 当 schema 要求一个模型没有证据的字段时，模型可能编造一个看似合规的值。

所以我们需要一个不依赖模型自觉的外部机制。这就是 GCD。

## 2. 第二课：grammar 就是一张“合法字符串地图”

### 2.1 先不谈代码，只谈“哪些句子算合法”

把所有允许的完整输出放进一个集合，叫作语言 $L$。这里的“语言”不一定是中文，也可以是 JSON、SQL、Python 或某种工具协议。

例如，下面的规则规定 status 只能是 ok 或 error：

~~~text
输出必须是：
{"status": "ok"}
或
{"status": "error"}
~~~

那么合法集合 $L$ 里只有两条完整字符串。grammar $G$ 是描述这个集合的规则，通常写成正则、EBNF、CFG 或 JSON Schema；由 $G$ 描述出的集合记为 $L(G)$。

可以把它想成一张地图：

- 完整合法输出是地图上的终点；
- 每个字符或 token 是一步；
- 一条从起点走到终点的路线就是一个合法输出；
- 走进死胡同的路线不能继续。

### 2.2 为什么要看“合法前缀”

生成到一半时，字符串通常还不是完整 JSON。例如：

~~~text
{"status": "
~~~

它缺少值、引号和右括号，看起来“不完整”，但并不应该被判错，因为它仍有机会补成合法结果。

因此 GCD 不问“现在是不是完整终点”，而问“现在是不是仍在通往某个终点的路上”。所有可以继续补全为合法字符串的前缀组成：

$$
\operatorname{Prefix}(L(G))=\{p\mid\exists s,\;ps\in L(G)\}
$$

**逐字翻译**：如果存在某个后缀 $s$，把它接到前缀 $p$ 后面就能得到合法完整字符串，那么 $p$ 就是合法前缀。

对上面的例子：

- `{"status": "` 是合法前缀，因为后面可以接 `ok"}`；
- `{"status": "pend` 不是合法前缀，因为 enum 中没有以 pend 开头的值；
- `{"status": 123` 不是合法前缀，因为 grammar 规定这里必须是字符串。

### 2.3 parser 是“读地图的门卫”

grammar 只是规则文本，真正执行规则的是 parser（解析器）。parser 不需要每次从头阅读整篇输出；它会维护一个很小的状态，告诉自己：

- 现在是否刚打开了对象；
- 接下来是在等 key、冒号、value 还是右括号；
- 哪些 required 字段已经出现；
- 当前是否正在字符串内部；
- 当前 enum 已经匹配到哪几个字符；
- 当前数组或嵌套对象的栈深度是多少。

把 parser 想成填写表格的门卫：

- 刚进门：只能写左大括号；
- 看到 key 后：只能写冒号；
- 看到冒号后：只能写符合该字段类型的 value；
- 所有 required 字段完成后：才允许右大括号和 EOS。

这个“门卫此刻站在哪一步”就是 parser state。

## 3. 第三课：用一个具体例子走一遍 parser state

### 3.1 目标 grammar

我们定义一个极小的输出协议：

~~~text
{"status": "ok"} 或 {"status": "error"}
~~~

现在从空输出开始，观察“允许什么”的变化：

| 当前前缀 | 门卫状态 | 下一步允许的内容 |
| --- | --- | --- |
| 空字符串 | 等待对象开始 | `{` |
| `{` | 等待固定 key | `"status"` |
| `{"status"` | 等待冒号 | `:` |
| `{"status":` | 等待字符串值 | `"ok"` 或 `"error"` 的开头 |
| `{"status": "o` | enum 已匹配 `o` | 只能继续 `k` |
| `{"status": "ok"` | 值完成 | `}` |
| `{"status": "ok"}` | 接受状态 | EOS |

这张表就是 GCD 的直觉核心：**允许集合不是固定的，它随着前缀变化。**

### 3.2 模型偏爱非法值怎么办

假设当前前缀是：

~~~text
{"status": "
~~~

模型给出三个候选：

| 候选 | 模型原始分数 | grammar 是否允许 |
| --- | ---: | --- |
| ok | 8.2 | 是 |
| error | 7.9 | 是 |
| pending | 9.1 | 否 |

普通解码可能选 pending，因为它分数最高。GCD 会先把 pending 从候选表中划掉，再让模型在 ok 和 error 中选择。模型依然有选择权，只是选择范围被缩小了。

### 3.3 这个例子中“稳定”到底稳定了什么

它稳定的是：

- 不会出现未知 status 值；
- 不会在 status 后面写数字；
- 不会在对象结束前随便 EOS；
- 不会缺少右引号或右括号。

它没有稳定：

- 用户是否真的应该得到 status=ok；
- status=ok 是否符合外部事实；
- 这段 JSON 是否对应正确的业务对象。

这就是“格式正确”和“任务正确”的边界。

## 4. 第四课：token、字符和 tokenizer 为什么让事情变复杂

### 4.1 grammar 常按字符描述，模型却按 token 选择

初学者很容易以为模型每次选择一个字符：`{`、`"`、`a`、`g`。实际上模型的词表可能包含：

~~~text
" Alice"
",\n  \"age\""
~~~

一个 token 可以覆盖多个字符、一个字段，甚至跨越多个 grammar 状态。tokenizer 是把文本切成 token 的规则；BPE、Unigram、SentencePiece 和 byte-level tokenizer 的切法各不相同。

因此，grammar 引擎不能简单地说“下一个字符允许冒号，所以允许所有以冒号开头的 token”。它必须检查 token 解码后的**全部字符**拼接起来是否仍然是合法前缀。

### 4.2 prefix tree 求交：像两位门卫一起放行

LM Format Enforcer 的思路很适合用一个比喻理解：

- 字符 parser 是第一位门卫，知道 grammar 当前允许哪些字符；
- tokenizer prefix tree（trie）是第二位门卫，知道词表里真实存在哪些完整 token；
- 只有同时通过两位门卫的 token 才能放行[^3]。

算法可以拆成五步：

1. 从 trie 根节点开始；
2. 取 token 的第一个字符交给 parser 检查；
3. 若允许，就沿 trie 继续看下一个字符；
4. 只要某个字符会让 parser 进入非法状态，就剪掉整条 token 分支；
5. 走到 token 末尾时，将这个真实 token 加入 allowed 集合。

这解释了三个工程事实：

- 换 tokenizer，allowed token 集合可能变化；
- 同一个 grammar 在不同模型上可能一个能生成、另一个 dead-end；
- 空格、换行、Unicode、UTF-8 半字节和特殊 token 必须单独测试。

### 4.3 一个容易踩的坑：字符上可行，token 上不可行

假设 grammar 需要输出字符 `é`，字符级 parser 认为它合法，但某个 byte-level tokenizer 在当前状态下没有可以安全发出的 token，或者只能先发一个会导致非法 UTF-8 的半字节。此时“理论 grammar 可行”不等于“这个模型的 token 词表可行”。

所以实际系统的兼容单位不是单独的 grammar，而是：

$$
\text{模型} + \text{tokenizer} + \text{grammar backend}
$$

## 5. 第五课：把直觉翻译成 GCD 的数学算法

### 5.1 第一步：模型先给整张候选表

当前前缀记为 $p_t$，模型对词表 $V$ 中每个 token 给出 logit $z_t(v)$。此时模型还没有受到 grammar 约束。

### 5.2 第二步：解析器计算允许集合

token $v$ 解码成字符串 $\operatorname{decode}(v)$。只有当拼接后的新前缀仍然可以完成为合法输出时，才允许它：

$$
A(s_t)=\{v\in V\mid p_t+\operatorname{decode}(v)\in\operatorname{Prefix}(L(G))\}
$$

**人话翻译**：把词表里的每个候选 token 试着接上去；会把路线带进死胡同的，不放进候选池。

### 5.3 第三步：把不允许的分数改成负无穷

$$
z'_t(v)=
\begin{cases}
 z_t(v), & v\in A(s_t)\\
 -\infty, & v\notin A(s_t)
\end{cases}
$$

为什么是 $-\infty$？因为 softmax 中：

$$
\exp(-\infty)=0
$$

所以非法 token 的概率变成严格的零，而不是“概率变小一点”。这一步发生在采样之前，是 GCD 能提供硬格式边界的根本原因。

### 5.4 第四步：在合法候选中照常采样

mask 后的概率是：

$$
P'(y_t=v\mid p_t)=\frac{\exp z'_t(v)}{\sum_{u\in V}\exp z'_t(u)}
$$

接下来仍可以使用 greedy、temperature、top-k、top-p 或 beam search。区别只有一个：它们看到的词表已经被 grammar 过滤过。

### 5.5 第五步：消费 token，更新状态

选择 token $y_t$ 后，parser 读取它的全部解码字符，并更新状态：

$$
s_{t+1}=\delta\left(s_t,\operatorname{decode}(y_t)\right)
$$

然后回到第一步，直到 parser 进入 accepting state，并且 EOS 被允许。

### 5.6 完整伪代码

~~~text
state = grammar.start()
prefix = prompt

while not grammar.accepting(state):
    logits = model(prefix)
    allowed = grammar.allowed_tokens(state, tokenizer)
    if allowed is empty:
        fail("grammar_dead_end")
    logits[all_tokens - allowed] = -inf
    token = sampler(logits)
    prefix += tokenizer.decode(token)
    state = grammar.advance(state, tokenizer.decode(token))

allow EOS only when grammar.accepting(state)
return prefix
~~~

真实系统会缓存 parser state、压缩 mask、增量更新解析栈，不会每一步都从头解析整段文本；但逻辑顺序就是这五步。

## 6. 第六课：为什么它能保证格式——一个不变式证明

这部分第一次读可以慢一点。我们只证明“格式路径不会被走坏”。

### 6.1 证明需要的三个前提

1. 初始前缀 $p_0$ 是 grammar 的合法前缀；
2. parser 正确计算 $A(s_t)$；
3. 每一步只从 $A(s_t)$ 里选择 token，接受状态才允许 EOS。

### 6.2 逐步不变式

假设第 $t$ 步之前：

$$
p_t\in\operatorname{Prefix}(L(G))
$$

因为只允许 $y_t\in A(s_t)$，根据 $A(s_t)$ 的定义：

$$
p_{t+1}=p_t+\operatorname{decode}(y_t)\in\operatorname{Prefix}(L(G))
$$

这说明下一步仍然在“可抵达终点的道路”上。初始时道路合法，每一步都不离开道路，因此所有中间前缀都合法可扩展。

### 6.3 什么时候得到完整合法输出

当 parser 进入 accepting state，表示当前前缀本身已经属于 $L(G)$。如果这时才允许 EOS，那么最终输出满足：

$$
y\in L(G)
$$

所以稳定性不是玄学，也不是“模型被训练得更听话”，而是一个解码时的候选空间不变式。

### 6.4 证明的边界

这个证明不能替你检查实现是否正确。若 parser 有 bug、tokenizer 解码不一致、特殊 token 漏配、schema 被错误转换或系统在 dead-end 时偷偷放宽 mask，证明前提就失效。即使证明成立，也只说明输出属于 grammar 语言，不说明内容真实或动作安全。

## 7. 第七课：FSM、CFG、JSON Schema 到底有什么区别

### 7.1 正则和 FSM：一张有限状态交通图

正则表达式通常可以编译成有限状态机。机器只需记住有限个状态，例如“已经读了四位年份”“正在等待连字符”“正在读取两位月份”。适合：

- 日期、邮箱、电话号码；
- 固定格式 ID；
- 有限分类标签；
- 简单标记语言。

FSM 的优势是快、状态小、容易缓存；缺点是对任意深度嵌套不自然。

### 7.2 CFG：带解析栈的嵌套地图

上下文无关文法用产生式表达嵌套：

~~~ebnf
query ::= "SELECT" columns "FROM" table
columns ::= column ("," column)*
~~~

SQL、代码、表达式和 DSL 常需要 CFG，因为括号、块、函数调用和优先级可以递归嵌套。CFG parser 通常维护解析栈，开销比 FSM 大，但表达能力更强。

### 7.3 JSON Schema：把结构要求写成机器可读协议

JSON Schema 不只是“JSON 外壳”，它还可以指定：

- 对象有哪些字段；
- 哪些字段 required；
- 字段是 string、number、boolean 还是 array；
- enum 允许哪些值；
- 是否禁止未知字段；
- 数组元素和嵌套对象如何组织。

不同后端会把 schema 编译为内部 grammar。不要误以为所有 JSON Schema 关键字在所有后端中都等价支持；生产前必须验证支持矩阵[^4]。

### 7.4 GBNF、Outlines、PICARD 等名字如何记

可以用一句话记忆：

- **PICARD**：把增量 SQL parser 放进生成循环[^6]；
- **Outlines**：把 regex/JSON/grammar 编译成生成器[^11]；
- **LM Format Enforcer**：字符 parser 与 tokenizer trie 求交[^3]；
- **llama.cpp GBNF**：本地模型通过 grammar 文件过滤 token[^2]；
- **XGrammar**：为 CFG/JSON Schema 和高吞吐服务优化 parser、缓存与 GPU 协同[^7]；
- **vLLM**：提供 choice、regex、JSON、grammar 等接口，可接 xgrammar 或 guidance[^5]；
- **OpenAI Structured Outputs**：把服务端 grammar 约束封装成 API[^4]。

## 8. 第八课：GCD 能保证什么，不能保证什么

### 8.1 可以保证的“形式事实”

在 grammar 可满足、parser 正确、tokenizer 对齐且没有截断的前提下，GCD 通常可以保证：

- JSON 可解析；
- 引号、括号、逗号和转义处于合法路径；
- required 字段存在；
- enum 来自指定集合；
- 输出符合 regex/CFG/EBNF；
- SQL 或 DSL 满足声明的语法；
- 工具名来自 allowlist；
- 工具参数符合受支持 schema；
- EOS 只在接受状态出现。

### 8.2 不能保证的“世界事实”

GCD 不能自动证明：

- 字段内容真实或有证据；
- SQL 查询到了正确表和正确答案；
- 代码没有 bug、漏洞或资源耗尽；
- 数字满足业务上下界、币种和单位；
- 工具调用是否应该发生；
- 模型是否应该拒答；
- 输出是否符合用户真实意图；
- 动态权限、租户隔离和状态一致性。

看下面这个结果：

~~~json
{"temperature": 999}
~~~

它可能完全符合“字段是 number”的 schema，但业务上显然不合理。**语法 validator 只能说“形状对了”，不能说“含义对了”。**

## 9. 第九课：为什么“强制格式”有时反而降低答案质量

### 9.1 局部 mask 做了什么

最简单的 GCD 每一步只看当前候选是否合规：

$$
P_{\mathrm{local}}(y_t=v\mid p_t)\propto P_{\mathrm{LM}}(y_t=v\mid p_t)\mathbf{1}[v\in A(s_t)]
$$

其中指示函数 $\mathbf{1}[\cdot]$ 的意思是：条件为真取 1，否则取 0。

理想情况下，我们想要的是“在所有完整合法答案中，模型原本最偏爱的答案仍然最可能”：

$$
P_{\mathrm{ideal}}(y\mid x,\;y\in L(G))
$$

两者不完全相同，因为当前合法 token 的未来可能完全不同：

- token A 现在分数高，但会把 parser 带到很窄的死胡同；
- token B 现在分数稍低，却有很多自然、完整的后续路径。

普通局部 mask 未必知道这些未来差异。

### 9.2 grammar-induced distribution shift

因此 GCD 可能造成：

- 低概率但语法合法的词；
- 不自然空格、字段顺序和标点风格；
- 被迫填入 schema 要求但证据不足的字段；
- “格式化幻觉”：结构完整，却让错误内容显得更可信；
- 拒答下降：grammar 没有 unknown/refuse 出口时，模型只能编造合法值。

Grammar-Aligned Decoding 研究把这种现象称为 grammar alignment 问题，并提出 ASAp，通过估计未来可完成性，使输出更接近“模型分布条件于 grammar”的理想目标[^8]。

工程上应记录：原始 top-1 是否被屏蔽、选中 token 的原始 log-probability、合法候选数量、forced-token ratio 和 dead-end 位置。

### 9.3 工具调用中的 abstention

工具调用有两个不同问题：

1. 如果调用，参数格式对不对？
2. 现在到底应该调用、直接回答，还是拒绝？

GCD 很擅长第一个问题，却可能影响第二个问题。预印本 *Repair, Not Improvement: Decomposing Constrained Decoding in Tool-Call Abstention* 报告：约束明显修复了工具格式，但枚举和停止 token 约束可能改变调用/不调用决策，在部分条件下总体决策效果下降[^10]。

因此不能用 100% parse rate 代替 tool selection accuracy、abstention correctness 和安全执行评测。

## 10. 第十课：一次完整的 GCD 生成，像什么

可以把一次生成想成“机器人在有围栏的迷宫里走路”：

1. **模型**看到当前所在位置，给每条可能道路打分；
2. **grammar parser**检查每条道路是否仍在合法地图内；
3. **tokenizer trie**确认这条道路确实对应词表中的一个完整 token；
4. **mask**关闭越界道路；
5. **采样器**在剩余道路中按模型分数选择一条；
6. **状态更新**把机器人移动到下一格；
7. 到达终点后才允许 EOS。

对应的工程循环是：

~~~text
GPU：计算 logits
  ↓
grammar engine：根据 parser state 计算 allowed token mask
  ↓
mask：非法 token 的 logit = -inf
  ↓
sampler：temperature / top-k / top-p / beam
  ↓
parser：消费选中的 token，更新 state
  ↓
接受状态？是则允许 EOS，否则继续
~~~

## 11. 第十一课：tokenizer 求交的工程过程

### 11.1 两个问题必须同时回答

对一个候选 token $v$，系统必须同时回答：

- grammar 允许它的所有字符按当前状态进入吗？
- tokenizer 真的把这些字符作为一个完整 token 提供了吗？

只有两个答案都为“是”，$v$ 才属于 $A(s_t)$。

### 11.2 trie 求交的逐步过程

1. 从 tokenizer trie 根节点开始；
2. 取 token 的第一个字符，交给字符 parser；
3. 若 parser 允许，继续沿 trie 走；
4. 任意字符导致 grammar 非法，就剪掉整个分支；
5. 到达一个完整 token 节点时，把 token 放入 allowed 集合；
6. 选中后消费全部字符并推进 parser。

这就是为什么不能只检查 token 的第一个字符，也不能把“字符集合”直接当成“token 集合”。

### 11.3 必须测试的边界

生产系统必须测试：

- 前导空格和换行；
- JSON 字符串转义；
- Unicode 和 surrogate；
- byte-level BPE 的半个 UTF-8 字节；
- SentencePiece 词首标记；
- BOS、EOS、stop 和工具分隔符；
- grammar 需要但 tokenizer 不含的字符；
- encode/decode 的不可逆归一化。

理论上 grammar 可行但词表没有可用 token，就会出现 dead-end。

## 12. 第十二课：工程系统怎样把它跑起来

### 12.1 编译阶段

首次使用一个 grammar 或 schema 时，通常要：

1. 解析 JSON Schema、regex 或 CFG；
2. 检查不可达规则、左递归、未定义符号和冲突；
3. 规范化 grammar，展开或拒绝不支持的 schema；
4. 编译成 DFA、解析表、栈机器或后端专用状态；
5. 根据 tokenizer 做 token 可达性预处理；
6. 初始化 parser state、mask cache、schema/tokenizer 指纹。

静态 schema 应预编译缓存；动态 grammar 必须把输入上下文纳入缓存键。缓存键至少包含 model/tokenizer 标识、backend 版本和 schema hash。

### 12.2 每个请求保存什么

每条请求至少需要保存：

- parser state 或解析栈；
- 已生成 token 和字符/字节前缀；
- batch/beam 分支索引；
- EOS、stop 和最大长度状态；
- grammar/backend 版本；
- tokenizer 指纹；
- schema hash。

batch 中的请求不能共享可变 parser state；beam 分叉时必须复制或持久化栈。流式输出还要区分“中间片段已经发出”和“完整 grammar 已接受”。

### 12.3 mask 与 top-k/top-p 的顺序

常见顺序是：

1. 模型产生 logits；
2. grammar mask 把非法 token 设为 $-\infty$；
3. 再做 temperature、top-k、top-p、重复惩罚；
4. 采样并推进 parser。

通常先做 grammar mask，可以避免 top-k 预先截掉唯一合法 token；但具体后端可能不同，必须以框架实现和回归测试为准。

### 12.4 dead-end：没有任何 token 可以走

如果 allowed 集合为空，常见原因是：

- grammar 不可满足；
- tokenizer 缺少需要的字符；
- Unicode 或空白归一化不一致；
- EOS 处理错误；
- schema 转换丢分支；
- 特殊 token 漏配；
- parser/backend 有 bug。

必须显式失败，不能悄悄关闭 grammar 继续生成：

~~~json
{
  "error_type": "grammar_dead_end",
  "step": 37,
  "parser_state": "object_required_fields",
  "schema_hash": "...",
  "tokenizer_hash": "...",
  "backend": "xgrammar"
}
~~~

## 13. 第十三课：为什么约束引擎会影响延迟

每生成一个 token，约束引擎可能要做 parser 状态推进、token 合法性判断、mask 构造和 tokenizer 对齐。复杂 CFG、长 schema、大词表、高并发和长输出会增加 CPU 工作。

常见优化包括：

- token 预检查：预先分类与上下文无关的 token；
- 状态缓存：缓存 parser state 到 allowed token 的映射；
- 持久化解析栈：beam/batch 分支共享不可变栈片段；
- bitset 或稀疏索引压缩 mask；
- CPU/GPU overlap：GPU 生成当前 token 时预计算下一状态 mask；
- schema cache：静态 schema 编译一次，多请求复用；
- 按 grammar/backend/长度组织 batch；
- 编译期和首 token 阶段尽早发现不可达分支。

XGrammar 的 context-independent token 预检查、持久化栈和 GPU overlap 正是针对这些瓶颈设计的[^7]。论文报告的约 100 倍是 grammar-engine 基准中的特定结果，不等于所有端到端服务都能提升 100 倍。

## 14. 第十四课：应用场景，先判断“结构问题”是否值得约束

### 14.1 Function calling / tool calling

~~~json
{
  "tool": "search_orders",
  "arguments": {
    "user_id": "u123",
    "limit": 10
  }
}
~~~

GCD 可以保证工具名 allowlist、参数可解析、required 存在、类型正确、未知字段被拒绝。它适合数据库查询、订单系统、CRM、UI 操作和 agent 工作流。

但是否调用工具、调用时机、权限、幂等性、金额上限和人工确认仍属于策略层。高风险动作至少经过 grammar、schema、domain validator、policy gate 和执行前确认。

### 14.2 RAG 与信息抽取

合同、发票、简历、病历、日志和知识库问答可以返回：

~~~json
{
  "answer": "...",
  "citations": ["doc-12", "doc-19"],
  "uncertainty": "insufficient_evidence"
}
~~~

GCD 保证字段形状，降低下游解析成本；但 citation 是否支持 answer、金额是否来自原文、confidence 是否校准，仍需证据对齐和独立 validator。

### 14.3 Text-to-SQL、代码和 DSL

CFG 可以表达关键字顺序、嵌套、优先级、函数参数和终止条件，适合 SQL、代码、配置文件、游戏脚本、机器人命令和 API DSL。

SQL 仍需检查数据库方言、表/列存在性、只读事务、权限和执行计划；代码仍需编译、静态分析、测试和沙箱。

### 14.4 分类与路由

当输出只能是 refund、shipping 或 technical_support 时，choice/enum 比提示词可靠。生产 enum 应包含 other、unknown、insufficient_evidence 或 needs_human，否则模型在边界样本上会被迫选一个错误但合法的标签。

### 14.5 UI、工作流与 agent 协议

模型生成组件树、表单、流程节点或状态机事件时，schema/grammar 能保护协议边界；权限、资源存在性、状态一致性和版本兼容仍由业务层负责。

## 15. 第十五课：格式正确为什么不等于任务正确

把系统分成五层最容易理解：

~~~text
模型概率
  ↓
Grammar：能否继续组成合法字符串
  ↓
Schema：字段、类型、enum、required 是否满足
  ↓
Domain validator：值域、跨字段关系、证据是否合理
  ↓
Policy gate / sandbox：权限、风险、限额、审计和实际执行
~~~

工具调用至少经过：

1. grammar：输出是合法 JSON/工具协议；
2. schema：工具名、参数类型和 required 正确；
3. 业务校验：用户有权限、订单存在、金额合理；
4. 策略决策：是否应该调用、是否需要人工确认；
5. 安全执行：沙箱、限额、幂等键、重放保护和审计日志。

失败时拒绝执行并返回原因，不能因为“模型被 grammar 卡住”就自动放宽 grammar。对证据不足的情况，schema 应显式允许拒答或不确定状态；否则 required 字段会把不确定性变成格式化幻觉。

## 16. 第十六课：如何评测，而不是被“100% 合规”骗到

JSONSchemaBench 使用约 10,000 个真实 JSON Schema，从约束合规效率、约束覆盖度和生成质量三个维度比较多个系统[^9]。生产评测至少有四组：

### 16.1 合规

parse rate、schema validation rate、required recall、enum violation、unknown property、重复字段、非法 escape、EOS 合法率、dead-end rate、截断率、refusal 格式正确率。

### 16.2 任务质量

task accuracy、exact match、SQL execution accuracy、编译和单元测试通过率、抽取 precision/recall、事实性、groundedness、citation entailment、拒答正确率、tool selection accuracy 和危险动作拦截率。

### 16.3 性能

grammar compile latency、first-token latency、每 token mask latency、端到端 p50/p95/p99、tokens/s、吞吐、batch scaling、CPU/GPU、内存和 cache hit rate。

### 16.4 分布与自然度

token log-probability、forced-token ratio、原始 top-1 被屏蔽比例、选中 token 与原始 top-1 的分数差、近似 KL、长度/字段顺序偏差、空白风格、拒答率、幻觉率和人工偏好。

**100% schema compliant 但任务准确率下降，不叫成功。**

## 17. 第十七课：三分钟复习卡片

### 17.1 一句话版

GCD 在模型每一步采样前运行 grammar parser，把所有会导致非法前缀的 token 的 logit 设为 $-\infty$，因此模型只能在合法候选中选择。

### 17.2 四个关键词版

- **模型**：给每个 token 概率；
- **grammar**：定义哪些完整字符串合法；
- **parser state**：记录当前前缀走到规则的哪一步；
- **mask**：把不允许的 token 概率变成零。

### 17.3 一个公式版

$$
A(s_t)=\{v\in V\mid p_t+\operatorname{decode}(v)\in\operatorname{Prefix}(L(G))\}
$$

只要记得这句话就够了：**把 token 接上去以后，如果还存在某种补全方法能得到合法完整输出，就允许；否则拒绝。**

### 17.4 一个边界版

GCD 保证“像不像合法协议”，不保证“协议里的内容是否真实、合理、获授权、值得执行”。

## 18. 第十八课：生产清单

### Schema 设计

- 分离语法必须满足与业务可以拒绝；
- 为不确定、无证据、无需动作提供出口；
- 限制递归深度、数组长度、字符串长度和总 token 数；
- 只使用目标 backend 支持的 schema 子集；
- 动态依赖字段交给 domain validator；
- 高风险工具使用最小参数集合和安全默认值。

### 解码与运行时

- 明确 grammar mask 与 top-k/top-p 的顺序；
- 明确 EOS、stop、最大长度和空白策略；
- 静态 schema 预编译；
- 缓存键包含 model、tokenizer、backend、schema hash；
- batch/beam 维护独立 parser state；
- allowed token 为空时返回可诊断错误；
- 流式 API 不把未闭合片段标为最终成功。

### 可观测性

记录 parser state、每步合法 token 数、屏蔽比例、原始 top-1 是否被屏蔽、选中 token 原始概率、forced-token ratio、dead-end 位置、schema/tokenizer/backend 版本、业务失败原因和执行结果。

## 19. 参考文献与进一步阅读

[^1]: Saibo Geng, Martin Josifoski, Maxime Peyrard, Robert West. *Grammar-Constrained Decoding for Structured NLP Tasks without Finetuning*. EMNLP 2023. [arXiv:2305.13971](https://arxiv.org/abs/2305.13971).

[^2]: ggml-org. *llama.cpp GBNF Guide*. [GitHub](https://github.com/ggml-org/llama.cpp/blob/master/grammars/README.md).

[^3]: Noam Gat. *LM Format Enforcer*. [GitHub](https://github.com/noamgat/lm-format-enforcer).

[^4]: OpenAI. *Structured model outputs*. [Documentation](https://platform.openai.com/docs/guides/structured-outputs).

[^5]: vLLM Project. *Structured Outputs*. [Documentation](https://docs.vllm.ai/en/latest/features/structured_outputs.html).

[^6]: Torsten Scholak, Nathan Schucher, Dzmitry Bahdanau. *PICARD: Parsing Incrementally for Constrained Auto-Regressive Decoding from Language Models*. EMNLP 2021. [arXiv:2109.05093](https://arxiv.org/abs/2109.05093).

[^7]: Yixin Dong et al. *XGrammar: Flexible and Efficient Structured Generation Engine for Large Language Models*. MLSys 2025. [arXiv:2411.15100](https://arxiv.org/abs/2411.15100).

[^8]: Kanghee Park et al. *Grammar-Aligned Decoding*. NeurIPS 2024. [arXiv:2405.21047](https://arxiv.org/abs/2405.21047).

[^9]: Saibo Geng et al. *JSONSchemaBench: A Rigorous Benchmark of Structured Outputs for Language Models*. 2025. [arXiv:2501.10868](https://arxiv.org/abs/2501.10868).

[^10]: Janghoon Lee. *Repair, Not Improvement: Decomposing Constrained Decoding in Tool-Call Abstention*. 预印本，2026-08-14。 [arXiv:2608.13959](https://arxiv.org/abs/2608.13959).

[^11]: Outlines Developers. *Structured generation and grammar documentation*. [Documentation](https://dottxt-ai.github.io/outlines/latest/reference/generation/).

[^12]: Guidance Developers. *Guidance / llguidance*. [GitHub](https://github.com/guidance-ai/llguidance).
