---
layout: post
toc: true
title: "Grammar-Constrained Decoding：让语言模型输出合法的魔术"
categories: AI
tags: [AI, LLM, Structured Generation, Grammar, Decoding, Agent, Function Calling]
author:
  - vortezwohl
  - 吴子豪
excerpt: "Grammar-Constrained Decoding（GCD，语法约束解码）是在大语言模型的自回归解码循环中接入增量解析器：模型继续负责在候选中选择内容，语法引擎则根据当前前缀和 tokenizer 的真实 token 边界，计算仍有可能完成为合法字符串的 token 集合，并在采样前把所有非法 token 的 logit 置为负无穷。本文从普通解码为何不稳定、Prefix(L(G))、FSM/CFG/JSON Schema、subword tokenizer 求交和 mask 数学定义出发，解释 PICARD、Outlines、LM Format Enforcer、llama.cpp GBNF、XGrammar、vLLM 与 OpenAI Structured Outputs 的实现取舍；进一步讨论格式保证与语义正确的边界、grammar-induced distribution shift、Grammar-Aligned Decoding/ASAp、tool-call abstention、性能和 dead-end 风险，并给出 function calling、RAG、信息抽取、SQL、代码、DSL、UI 协议与安全工作流的分层设计和评测清单。"
---

> 本文是一份面向工程师和研究人员的 GCD 全景讲义。核心结论是：**模型给概率，grammar 给边界；约束发生在采样前，而不是输出后的字符串清洗。** 因此 GCD 可以把“尽量输出 JSON”升级成“无法输出不符合 grammar 的 token”。但它只负责可解析性和结构层约束，不能替代事实核验、业务规则、权限策略、拒答设计和执行前安全门禁。公开资料核验截至 2026 年 8 月；论文中报告的数字属于特定模型、数据集和后端，不能未经复现实验直接泛化。

## 一、先把问题说清楚：普通解码为什么会失稳

### 1.1 自回归模型的默认目标

给定输入 $x$ 和已经生成的前缀 $y_{<t}$，语言模型在整个词表 $V$ 上给出下一 token 的分布：

$$
y_t \sim P_{\mathrm{LM}}(y_t \mid x, y_{<t})
$$

实现上，模型先输出 logits 向量 $z_t\in\mathbb{R}^{|V|}$，再由 softmax、temperature、top-k、top-p、beam search 或 greedy 规则选出 token。这个分布目标是“下一个 token 在语言和上下文中有多可能”，而不是“这个 token 是否会让最终字符串属于某个外部 grammar”。

模型可能学会 JSON、SQL 和 Python 的常见写法，却没有一个自动绑定到当前请求的、可证明的外部语法状态。因此同一个 prompt 可能得到：

~~~json
{"name": "Alice", "age": 20}
~~~

也可能得到：

~~~text
Here is the JSON:
{"name": "Alice", "age": "twenty",}
~~~

典型失败包括：

- 缺少 required 字段，或者字段出现两次；
- 逗号、引号、括号不配对；
- enum 产生集合之外的值；
- 数字、布尔值和字符串类型混淆；
- 在 JSON 前后输出解释文本、Markdown 围栏或多余 token；
- SQL/代码在语法上无法解析；
- 提前生成 EOS，或者在没有停止条件时无限延长。

提示词、few-shot 和重试可以改变概率质量，却不能从数学上让非法序列的概率严格为零。GCD 的切入点是把外部形式语言直接放进采样环节。

### 1.2 “格式稳定”究竟指什么

需要区分至少四个层级：

1. **词法层**：字符、转义、空白和 token 边界合法；
2. **语法层**：字符串能被 JSON、SQL、CFG 或 DSL parser 接受；
3. **结构层**：对象字段、数组元素、required、enum 和类型符合 schema；
4. **语义/业务层**：事实有证据、数值合理、动作有权限且确实应该执行。

GCD 主要覆盖前两层，某些 JSON Schema 后端也覆盖第三层的一部分。第四层必须由独立 validator、检索证据、策略引擎或执行沙箱负责。

## 二、形式语言：解析器真正维护的是什么

### 2.1 Language 与 grammar

把允许的完整输出写成一个形式语言 $L$。grammar $G$ 描述的语言记为 $L(G)$。解码时不能只问“当前字符串是不是一个完整合法对象”，因为生成通常停在半个 token、半个字符串或未闭合括号上。真正需要的是**可扩展前缀**：

$$
\operatorname{Prefix}(L(G))=\{p\mid\exists s,\;ps\in L(G)\}
$$

若 $p\in\operatorname{Prefix}(L(G))$，则它虽然可能尚未完成，但至少存在某个后缀 $s$ 可以把它补成合法输出。比如 `{"status": "` 不是完整 JSON，却是合法前缀；`{"status": 1` 在状态枚举 grammar 下则不是合法前缀。

这一定义给出了 GCD 的安全边界：候选 token 必须让新前缀仍属于 $\operatorname{Prefix}(L(G))$，而不能只检查 token 本身是否“看起来合法”。

### 2.2 FSM、CFG 与解析栈

不同约束表示对应不同的状态复杂度：

- **正则表达式**可以编译成 NFA/DFA/FSM，适合日期、邮箱、电话号码、固定 ID、有限分类标签；
- **CFG/EBNF**通过终结符、非终结符和产生式描述嵌套结构，适合 SQL、程序代码、数学表达式、XML 子集和 DSL；
- **JSON Schema**通常被转换为专用 JSON grammar，并额外维护对象字段集合、required/optional、数组索引、值类型、enum、转义和嵌套栈；
- **输入依赖 grammar**会根据数据库 schema、检索证据或当前工具上下文动态缩小可生成语言[^1]。

有限状态约束只需保存一个状态编号；CFG 约束通常还要保存解析栈，状态可以表示为“栈顶符号、剩余产生式、已消费的终结符和局部语义标记”。这也是 CFG 比简单 FSM 更难做高吞吐的原因。

### 2.3 JSON Schema 不是完整业务规则

Schema 可以表达 type、required、enum、additionalProperties、数组和对象嵌套等结构要求，但不同后端支持的 JSON Schema 只是子集。递归引用、oneOf/条件逻辑、动态依赖字段、复杂数值约束可能被拒绝、近似转换或运行时不支持。

即使约束引擎接受了：

~~~json
{"temperature": 999}
~~~

它仍可能违反业务范围。因而要把“语法必须满足”和“业务可以拒绝”分开设计。OpenAI 文档明确区分 JSON mode 与 Structured Outputs：前者重点是可解析 JSON，后者针对受支持的 JSON Schema 子集提供 schema 合规，并用独立字段表达拒答[^4]。

## 三、GCD 的核心算法：从 mask 到状态转移

### 3.1 合法 token 集合

设当前已生成文本前缀为 $p_t$，解析器状态为 $s_t$，token $v$ 经 tokenizer 解码后的字符串为 $\operatorname{decode}(v)$。第 $t$ 步的合法 token 集合定义为：

$$
A(s_t)=\{v\in V\mid p_t+\operatorname{decode}(v)\in\operatorname{Prefix}(L(G))\}
$$

这里的“合法”是**拼接后仍可完成**，不是 token 自身属于某个字符集合。若 grammar 已进入完整接受状态，EOS 也必须被视为一种特殊的可接受 token；否则模型会在合法对象之后继续写内容。

### 3.2 对 logits 做硬屏蔽

模型输出原始 logits $z_t(v)$ 后，约束引擎构造：

$$
z'_t(v)=\begin{cases}z_t(v), & v\in A(s_t)\\-\infty, & v\notin A(s_t)\end{cases}
$$

随后在 $z'_t$ 上应用采样器：

$$
P'(y_t=v\mid p_t)=\frac{\exp z'_t(v)}{\sum_{u\in V}\exp z'_t(u)}
$$

因为非法 token 的 $\exp(-\infty)=0$，它们在这一步的概率严格为零。temperature、top-k、top-p、重复惩罚和 beam search 都只能在合法集合内继续发挥作用。

### 3.3 增量状态更新

采样得到 $y_t$ 后，解析器消费其完整解码文本，并执行状态转移：

$$
s_{t+1}=\delta\left(s_t,\operatorname{decode}(y_t)\right)
$$

重复以下循环：

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

工程实现通常不会真的把完整 prefix 每次重新解析；parser state、token 字节串和栈会增量更新，mask 也可缓存。

### 3.4 为什么格式会变稳定：逐步不变式

若初始前缀 $p_0$ 属于 $\operatorname{Prefix}(L(G))$，且每步只选择满足

$$
p_{t+1}=p_t+\operatorname{decode}(y_t)\in\operatorname{Prefix}(L(G))
$$

的 token，那么所有中间前缀都不会离开可完成路径。当 parser 进入接受状态，且只允许合法 EOS 时，最终输出 $y$ 满足：

$$
y\in L(G)
$$

这就是格式稳定的来源：**非法分支在生成前被置零**，不是生成后用正则替换，也不是模型突然变得更聪明。证明成立的前提是 grammar、parser、tokenizer 和 EOS 处理实现正确；它不自动证明业务语义正确。

## 四、tokenizer 是关键难点：字符语言与 token 词表如何求交

### 4.1 为什么 token 级过滤不能只看首字符

现代模型通常使用 BPE、Unigram 或 byte-level tokenizer。一个 token 可能包含前导空格、多个 JSON 字符或一段 SQL：

~~~text
" Alice"
",\\n  \\"age\\""
~~~

因此不能只检查 token 的第一字符，也不能假定每个 token 对应一个 grammar terminal。必须检查整个 prefix + decode(token) 是否仍在前缀语言中。不同模型、不同 tokenizer，即使使用同一个 JSON Schema，也可能得到不同的合法 token mask。

### 4.2 tokenizer prefix tree 求交

LM Format Enforcer 的典型路线是把 tokenizer 词表构造成 prefix tree（trie），并与字符级 parser 求交[^3]：

1. parser 根据当前状态给出允许的下一个字符集合或字符转移；
2. trie 从根向下遍历 token 的字符路径；
3. 只保留在每个字符位置都能被 parser 接受的分支；
4. 遍历到一个完整 token 时，把该 token 加入允许集合；
5. 选中 token 后，parser 消费其全部字符，进入新状态。

这种求交同时解决了 subword 边界、前导空格、Unicode、byte token 和多个 grammar 边界被一个 token 跨过的问题。它也解释了为什么“模型 + tokenizer + grammar backend”应被视为一个兼容性单元：换 tokenizer 可能改变可生成性、空白风格、mask 大小和 dead-end 位置。

### 4.3 词法细节与特殊 token

生产实现必须显式测试：

- 前导空格和换行是否属于 grammar；
- JSON 字符串中的转义和 Unicode surrogate；
- byte-level BPE 的半个 UTF-8 字节；
- SentencePiece 的词首标记；
- BOS、EOS、工具调用分隔符和 stop token；
- tokenizer 是否包含 grammar 要求的每个字节；
- decode/encode 是否存在不可逆归一化。

任何字符级 parser 的“接受”都必须映射到真实词表中的至少一个 token，否则理论上可行的 grammar 在具体模型上仍会无合法 token。

## 五、从论文到社区实现：主要路线与取舍

| 实现/系统 | 约束表示 | 核心机制 | 典型适用 |
| --- | --- | --- | --- |
| PICARD | 增量 SQL parser | 每步试探 token，拒绝 parser 不接受的 token | text-to-SQL |
| Outlines | regex、JSON、CFG | 编译生成器/FSM，并在推理时过滤 | Python 离线与服务推理 |
| LM Format Enforcer | JSON Schema、regex | 字符 parser 与 tokenizer prefix tree 求交 | Transformers、vLLM 集成 |
| llama.cpp GBNF | GBNF/EBNF | 本地 grammar 文件和运行时 token 过滤 | 本地模型、端侧 JSON/代码 |
| XGrammar | CFG、JSON Schema | token 预检查、持久化栈、CPU/GPU overlap | 高吞吐推理服务 |
| vLLM Structured Outputs | choice、regex、JSON、grammar、structural tag | 接入 xgrammar 或 guidance 后端 | 批量在线服务 |
| OpenAI Structured Outputs | JSON Schema 受限子集 | API 服务端编译 grammar 并执行 mask | 托管 API |

### 5.1 PICARD：解析器直接参与 SQL 解码

PICARD（Parsing Incrementally for Constrained Auto-Regressive Decoding）把 SQL parser 放进自回归循环：对每个候选 token 做增量解析试探，只保留 parser 能接受的候选。它在 Spider、CoSQL 等 text-to-SQL 任务中展示了约束对执行准确率和语法错误率的改善[^6]。其思想很通用，但 parser 必须匹配目标数据库方言；“SQL 语法合法”仍不等于表名存在、权限允许或查询结果正确。

### 5.2 Outlines、LM Format Enforcer 与 llama.cpp

Outlines 把正则、JSON 和 grammar 编译成可复用的生成器，强调 API 易用性和有限状态约束[^11]。LM Format Enforcer 通过 trie 与字符 parser 求交保留空格、字段顺序和可选字段自由度[^3]。llama.cpp 使用 GBNF 描述 JSON、棋谱、特殊 token 区段和自定义语言，适合本地推理[^2]。

它们共同的设计原则是：编译约束一次，解码阶段增量推进；但对递归 grammar、复杂 schema、动态 grammar 和 tokenizer 特性支持不同，不能只比较“是否支持 JSON”。

### 5.3 XGrammar 与服务端后端

XGrammar 面向大词表、复杂 CFG 和高并发服务，核心优化包括：

- 预先检查 context-independent tokens，把不会受当前 parser 状态影响的 token 预处理；
- 运行时只处理 context-dependent tokens；
- 使用持久化 parser stack 减少重复构造；
- 将 CPU grammar 工作与 GPU 推理重叠；
- 为 batch 中每条请求保存独立状态。

论文报告在特定 grammar-engine 基准上最高约 100 倍加速[^7]，但这不是端到端延迟自动降低 100 倍；真实结果还取决于模型推理占比、batch、schema 复杂度和缓存命中率。vLLM 当前提供 choice、regex、JSON、grammar 和 structural tag 等结构化输出入口，并可使用 xgrammar 或 guidance 后端[^5]。

### 5.4 OpenAI Structured Outputs 与服务端封装

托管 API 把 grammar 编译、tokenizer 对齐和 mask 执行隐藏在服务端。应用方仍需阅读支持的 schema 子集、拒答字段语义、strict 模式、流式输出和错误处理文档[^4]。API 层的“schema 合规”并不意味着它替你完成领域验证或高风险动作审批。

## 六、格式保证的边界：能保证什么，不能保证什么

### 6.1 通常可以强保证的内容

在 parser 与 tokenizer 实现正确、grammar 可满足且没有中途截断的前提下，GCD 可以保证：

- JSON 可解析；
- 引号、括号、逗号和转义处于合法路径；
- required 字段存在；
- enum 来自指定集合；
- 输出符合 regex/CFG/EBNF；
- SQL 或 DSL 满足声明的语法；
- 工具名来自 allowlist；
- 工具参数符合受支持 schema；
- EOS 只在接受状态出现。

### 6.2 不能自动保证的内容

GCD 不会自动证明：

- 字段内容真实或有证据；
- SQL 查到了正确表和正确答案；
- 代码没有 bug、漏洞或资源耗尽；
- 数字满足业务上下界、币种和单位；
- 工具调用是否应该发生；
- 模型是否应该拒答；
- 输出是否符合用户真实意图；
- 动态权限、租户隔离和状态一致性。

把语义错误塞进 grammar 往往会得到巨大、脆弱且难维护的 grammar；更好的做法是 grammar 保持协议级，业务 validator 负责值域与跨字段关系。

## 七、被忽略的核心问题：局部 mask 会改变分布

### 7.1 局部条件化

最简单的 token mask 产生的局部分布是：

$$
P_{\mathrm{local}}(y_t=v\mid p_t)\propto P_{\mathrm{LM}}(y_t=v\mid p_t)\mathbf{1}[v\in A(s_t)]
$$

它只在当前一步把非法 token 的质量归零并重新归一化。理想目标更接近在所有完整合法序列上的条件分布：

$$
P_{\mathrm{ideal}}(y\mid x,\;y\in L(G))
$$

两者不同的原因是：当前合法 token 可能把生成带到一个未来几乎无法完成的分支；另一个当前分数略低的 token 可能拥有大量高质量完成路径。局部 mask 通常没有把“未来可行性”和“后续模型概率”完全纳入当前选择。

### 7.2 Grammar-induced distribution shift

因此 GCD 可能带来低概率但语法合法的词、不自然空格、被迫字段值、格式化幻觉和拒答下降。Grammar-Aligned Decoding 把这种现象称为 grammar alignment 问题，并提出 ASAp，通过估计未来可完成性，使输出更接近模型分布条件于 grammar 的理想目标[^8]。

工程上至少应记录“原始 top-1 是否被屏蔽”“选中 token 的原始 log-probability”“合法候选数量”和“forced-token ratio”，用来发现约束正在多大程度改变模型行为。

### 7.3 工具 abstention 的特殊风险

工具调用不仅有“调用参数是否合法”，还有“应该调用还是应该直接回答/拒答”。最新预印本 *Repair, Not Improvement: Decomposing Constrained Decoding in Tool-Call Abstention* 报告：约束解码显著修复了工具调用格式，但枚举和停止 token 约束可能改变调用/不调用的决策，在部分条件下总体决策效果下降[^10]。这提醒我们不能用 100% parse rate 代替 tool selection、abstention correctness 和执行安全评测。

## 八、常见应用场景：为什么这些地方经常使用 GCD

### 8.1 Function calling / tool calling

~~~json
{
  "tool": "search_orders",
  "arguments": {
    "user_id": "u123",
    "limit": 10
  }
}
~~~

GCD 能把工具名限制在 allowlist，保证参数可解析、required 存在、类型和 enum 正确、未知字段被拒绝。它特别适合数据库查询、订单系统、CRM、UI 操作和 agent 工作流。

但是否调用工具、调用时机、用户权限、幂等性、金额上限和人工确认仍属于策略层。高风险动作至少要经过 grammar、schema、业务 validator、authorization/policy gate 和执行前确认。

### 8.2 RAG 与信息抽取

合同、发票、简历、病历、日志和知识库问答常需要稳定返回：

~~~json
{
  "answer": "...",
  "citations": ["doc-12", "doc-19"],
  "uncertainty": "insufficient_evidence"
}
~~~

GCD 可保证字段和数组形状，降低下游解析成本；但 citation 是否真的支持 answer、amount 是否来自原文、confidence 是否校准，必须由证据对齐、字段级 validator 和独立评测处理。

### 8.3 Text-to-SQL、代码与 DSL

CFG 可以表示关键字顺序、嵌套、优先级、函数参数和终止条件，适合 text-to-SQL、SQL 修复、JSON 查询语言、配置文件、编译器/接口 DSL、游戏脚本和机器人控制命令。

SQL 还需要数据库方言、表/列存在性检查、只读事务、权限隔离和执行计划限制；代码还需要编译、静态分析、测试和沙箱。语法约束只是第一道门。

### 8.4 分类、路由与有限选择

当输出只能是 `refund`、`shipping` 或 `technical_support` 时，choice/enum 约束比提示词可靠。生产 enum 应包含 `other`、`unknown`、`insufficient_evidence` 或 `needs_human` 等出口，否则模型在边界样本上会被迫选择一个错误但合法的标签。

### 8.5 UI、工作流与 agent 协议

模型生成组件树、表单定义、工作流节点、状态机事件或 agent message 时，schema/grammar 能保护协议边界：前端只接收已知组件类型，工作流只接受允许的状态转移，消息只携带声明过的字段。权限、状态一致性、资源存在性和版本兼容仍需业务层处理。

## 九、工程实现：从 grammar 编译到采样循环

### 9.1 编译阶段

请求到来前或首次使用 schema 时，系统通常：

1. 解析 JSON Schema、regex 或 CFG；
2. 检查不可达规则、左递归、未定义符号和冲突；
3. 规范化 grammar，展开或拒绝不支持的 schema 结构；
4. 编译成 DFA、解析表、栈机器或后端专用状态；
5. 根据 tokenizer 做 token 可达性预处理；
6. 初始化 parser state、mask cache 和 schema/tokenizer 指纹。

静态 schema 应预编译并缓存；动态 grammar 必须把输入上下文纳入缓存键。缓存键至少包含 model/tokenizer 标识、grammar backend 版本和 schema hash，否则换 tokenizer 或后端后可能复用错误 mask。

### 9.2 请求状态与 batch

每条请求至少维护 parser state 或解析栈、已生成 token 序列、batch/beam 分支索引、EOS/stop/最大长度状态、grammar/backend 版本、tokenizer 指纹和 schema hash。batch 中的请求不能共享可变 parser state；beam search 分叉时必须复制或持久化栈，回溯时恢复对应状态。流式输出还要区分“当前片段已发出”和“完整 grammar 已接受”。

### 9.3 mask、采样与停止顺序

一个常见循环是：

1. GPU 计算 logits；
2. grammar engine 计算 allowed token mask；
3. 将非法 logits 置为 $-\infty$；
4. 应用 temperature、top-k/top-p、重复惩罚和采样；
5. 消费选中 token，推进 parser；
6. 只有在 accepting state 才开放 EOS/stop；
7. 记录指标并继续下一步。

通常先 mask 再做 top-k/top-p，避免 top-k 预先截掉唯一合法 token；但后端可能采用不同顺序，必须以框架实现和回归测试为准。若 allowed 集合为空，应立即报告 grammar_dead_end，而不是放宽 grammar 或无条件采样。

### 9.4 dead-end 的诊断对象

无合法 token 的原因包括 grammar 不可满足、tokenizer 缺字符、Unicode/空白归一化不一致、EOS 处理错误、schema 转换丢失分支、特殊 token 漏配或 parser/backend bug。建议返回结构化错误，而非截断成伪完整 JSON：

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

## 十、性能：约束引擎为什么会成为系统问题

每生成一个 token，约束引擎可能需要 parser 状态推进、token 合法性判断、mask 构造和 tokenizer 对齐。复杂 CFG、长 schema、大词表、高并发和长输出会增加 CPU 工作，甚至让 grammar engine 成为 decode 瓶颈。

常见优化路线：

- token 预检查：把与上下文无关的 token 预先分类；
- 状态缓存：缓存 parser state 到 allowed token 的映射；
- 持久化解析栈：beam/batch 分支共享不可变栈片段；
- mask 压缩：使用 bitset、稀疏索引或 GPU-friendly 表示；
- CPU/GPU overlap：GPU 生成当前 token 时预计算下一状态 mask；
- schema cache：静态 schema 编译一次，多请求复用；
- 批处理分桶：按 grammar/backend/长度相近程度组织 batch；
- 早期失败：在编译期和首 token 阶段发现不可达分支。

XGrammar 的 context-independent token 预检查、持久化栈和 GPU overlap 正是针对这些瓶颈设计的[^7]。端到端收益必须同时报告 grammar-engine latency、first-token latency、每 token mask latency、吞吐、batch scaling 和内存，而不能只引用单项 microbenchmark。

## 十一、正确的系统分层：grammar 不等于 validator

推荐把结构化生成放在以下流水线上：

~~~text
模型概率
  ↓
Grammar：前缀是否仍可完成、字符串是否可解析
  ↓
Schema：字段、类型、enum、required 是否满足
  ↓
Domain validator：值域、跨字段关系、证据和状态是否合理
  ↓
Policy gate：权限、租户隔离、风险等级和是否允许动作
  ↓
Execution sandbox：只读、限额、超时、幂等、审计
~~~

以工具调用为例：

1. grammar：输出是合法 JSON/工具协议；
2. schema：工具名、参数类型和 required 正确；
3. 业务校验：用户有权限、订单存在、金额和时间范围合理；
4. 策略决策：是否应该调用、是否需要人工确认；
5. 安全执行：沙箱、限额、幂等键、重放保护和审计日志。

失败时应拒绝执行并返回原因，不能因为“模型被 grammar 卡住”就自动放宽 grammar。对于证据不足，schema 应显式允许拒答或不确定状态；把字段标成 required 却没有可靠来源，往往会把不确定性转化成格式化幻觉。

## 十二、评测：不能只看 schema compliance

JSONSchemaBench 使用约 10,000 个真实 JSON Schema，从约束合规效率、约束覆盖度和生成质量三个维度比较 Guidance、Outlines、llama.cpp、XGrammar、OpenAI、Gemini 等系统[^9]。生产评测至少包括四组：

### 12.1 合规指标

- parse rate、schema validation rate；
- required recall、enum violation、unknown property；
- 类型错误、重复字段、非法 escape；
- EOS 合法率、dead-end rate、截断率；
- refusal/abstention 格式正确率。

### 12.2 任务质量指标

- task accuracy、exact match；
- SQL execution accuracy、编译通过率和单元测试通过率；
- 抽取字段 precision/recall、事实性和 groundedness；
- citation entailment、拒答正确率；
- tool selection accuracy、参数正确率和危险动作拦截率。

### 12.3 性能指标

- grammar compile latency；
- first-token latency；
- 每 token mask latency；
- 端到端 p50/p95/p99 延迟；
- tokens/s、请求吞吐和 batch scaling；
- CPU/GPU 占用、内存、cache hit rate。

### 12.4 分布与自然度指标

- token log-probability；
- forced-token ratio；
- 原始 top-1 被屏蔽比例；
- 选中 token 与原始 top-1 的分数差；
- 近似 KL、长度偏差、字段顺序偏差和空白风格；
- 约束前后拒答率、幻觉率和人工偏好。

“100% schema compliant 但任务准确率下降”不能判定为成功。约束效果必须和质量、自然度、拒答与性能一起看。

## 十三、与其他策略的关系

| 方案 | 主要保证 | 优点 | 局限 |
| --- | --- | --- | --- |
| Prompt engineering | 概率性格式倾向 | 灵活、无运行时 parser | 没有硬保证 |
| JSON mode | 可解析 JSON | 接入简单 | 不保证给定 schema |
| GCD/Structured Outputs | grammar/schema 合规 | 生成时阻断非法 token | parser 成本、分布失真、schema 子集 |
| Parse + retry | 事后修复 | 可叠加已有 API | 浪费 token，语义问题仍在 |
| Fine-tuning | 输出风格和领域适配 | 可学习复杂习惯 | 训练、更新、遗忘和数据成本 |
| Domain validator | 业务合法性 | 能阻止危险值和状态错误 | 不能替代 grammar |
| Compiler/type checker | 程序级合法性 | 对代码、SQL 很强 | 只适用于可执行形式 |

稳妥的组合通常是：

$$
\text{Prompt}+\text{Grammar}+\text{Schema Validator}+\text{Domain Validator}+\text{Policy Gate}+\text{Retry/Fallback}
$$

不要把 grammar 当成唯一可靠性层，也不要让 parse retry 掩盖 grammar 设计不完整。

## 十四、生产清单与故障复盘

### Schema 设计

- 分离“语法必须满足”和“业务可以拒绝”；
- 为不确定、无证据、无需动作提供显式出口；
- 限制递归深度、数组长度、字符串长度和总 token 数；
- 只使用目标 backend 明确支持的 schema 子集；
- 动态依赖字段交给 domain validator；
- 对高风险工具定义最小参数集合和安全默认值。

### 解码与运行时

- 明确 grammar mask 与 top-k/top-p 的顺序；
- 明确 EOS、stop、最大长度和空白策略；
- 静态 schema 预编译；
- 缓存键包含 model、tokenizer、backend、schema hash；
- batch/beam 为每条分支维护独立 parser state；
- allowed token 为空时返回可诊断错误；
- 流式 API 不把中间未闭合片段标记为最终成功。

### 可观测性

至少记录 parser state 或栈摘要、每步合法 token 数和屏蔽比例、原始 top-1 是否被屏蔽、选中 token 的原始概率、forced-token ratio、dead-end 位置、schema/tokenizer/backend 版本，以及 domain validator、policy gate 和执行结果。

### 故障复盘顺序

1. 先确认是 tokenizer、grammar、采样顺序还是业务 validator 失败；
2. 保存失败前缀、parser state、schema hash 和 logits 摘要；
3. 用最小 grammar 重现；
4. 修复后同时回归合规、任务质量和性能；
5. 不要在 dead-end 后自动放宽约束并继续执行高风险动作。

## 十五、研究前沿与开放问题

1. **从格式正确走向分布正确**：局部 mask 不是理想条件采样，需要未来可行性估计和 grammar alignment[^8]。
2. **从单请求约束走向高吞吐服务**：parser cache、持久化栈、batch 调度和 GPU overlap 成为系统问题[^5][^7]。
3. **从静态 schema 走向输入依赖 grammar**：数据库 schema、检索证据和工具上下文会动态决定可行输出空间[^1]。
4. **从 JSON 到可靠 abstention**：工具调用需要同时评测“调用、拒绝调用、请求澄清”三个分支，不能只看参数格式[^10]。
5. **从简单 demo 到真实基准**：复杂度、覆盖度、效率、自然度和任务质量必须同时测[^9]。
6. **从 grammar 到可证明协议**：未来系统会组合 grammar、类型系统、权限策略、执行沙箱和审计，把“能生成”推进到“能安全执行”。

## 十六、最终心智模型

- **模型给概率，grammar 给边界**：模型仍负责在合法候选中选择内容。
- **约束发生在采样前**：非法 token 的 logit 被置为 $-\infty$，不是生成后清洗。
- **前缀可完成性是核心**：判断的是 $p+\operatorname{decode}(v)$ 是否仍在 $\operatorname{Prefix}(L(G))$。
- **tokenizer 是一等公民**：字符 parser 与 subword 词表必须求交。
- **格式保证不等于任务正确**：schema、业务、事实、权限和安全仍需独立验证。
- **拒答是 grammar 设计的一部分**：没有 unknown、refuse 或 insufficient_evidence，约束可能把不确定性变成幻觉。
- **性能是系统问题**：编译、mask、缓存、batch 和 GPU overlap 决定生产可用性。

最准确的比喻是：GCD 是“解码层的形式协议执行器”。它把结构化输出从软约定升级为硬边界，但不替代模型本身的知识与推理，也不替代证据检索、业务规则、权限控制、拒答机制和人类责任边界。

## 参考文献

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
