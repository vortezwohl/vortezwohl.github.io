---
layout: post
toc: true
title: "MemPalace: 启发式记忆检索与本地原文记忆系统"
categories: Agent
tags: [LLM, memory, rag, chromadb, agent]
author: vortezwohl
---

MemPalace 是一个很容易被 README 叙事带偏的项目。它把自己的世界观包装成 palace metaphor: wing、hall、room、closet、drawer，听起来像一套全新的记忆架构；但如果真正顺着源码读下去，会发现它最重要的设计并不神秘，反而相当朴素：**尽可能保留原文，再用轻量结构化和元数据过滤把原文变得可找回。**$^{[1][2]}$

换句话说，它并不是先让 LLM 决定什么值得记忆，再把结果压缩成 summary 存起来；它采取的是近似相反的路线：先把 code、docs、聊天记录中的 verbatim 文本切成小块存入本地向量库，结构化信息只负责导航、过滤和上下文预算控制，而不是替代原文本身。这种路线的优点是保真，缺点是很多“智能”能力都必须退回到启发式。MemPalace 的真正价值，恰恰就在于它把这些启发式系统化成了一套可运行、可维护的本地 memory harness。

> MemPalace 的核心不是“发明了新的记忆模型”，而是“在本地原文存储之上，做了足够克制的结构化检索工程”。

## 调研方法

本文基于对 `mempalace` 仓库的静态源码阅读完成，重点查看了 CLI 入口、项目与对话两条 ingest 链路、搜索层、分层记忆层、知识图谱层，以及实体、房间、查询清洗等启发式模块。本文**没有执行测试**，所有判断都来自代码实现本身，而非运行时结果。

## MemPalace 到底在做什么

从代码实现看，MemPalace 可以压缩成一条很清晰的数据流：

```text
本地文件 / 聊天导出
    -> 规范化 normalize
    -> 启发式分段 chunk
    -> 赋予 wing / room 等元数据
    -> ChromaDB 中的 drawer 文本块
    -> 语义检索 + 元数据过滤
    -> Layer0/1/2/3 分层召回
```

并行地，项目还维护了一套单独的 SQLite knowledge graph：

```text
结构化事实
    -> entities + triples
    -> valid_from / valid_to 时态过滤
    -> KG 查询与失效旧事实
```

这里最关键的一点是：**drawer 才是主记忆载体，knowledge graph 只是旁路结构化层，AAAK 只是可选压缩表达层。** README 里后来也明确承认，默认 benchmark 获胜模式是 raw verbatim text，而不是 AAAK。$^{[2]}$

## 核心设计: Store Everything, Then Make It Findable

MemPalace README 里最准确的一句，反而是最朴素的一句：`store everything, then make it findable`。$^{[2]}$ 从代码看，这句话并不是 marketing slogan，而是几乎贯彻到每个模块的真实约束。

### 1. 原文优先，摘要靠后

项目文件 ingest 的主逻辑在 `miner.py`。它不会先提炼摘要，也不会先做语义抽取，而是直接读取文本，按固定窗口切块，再把每个 chunk 作为 document 写进 ChromaDB。每个 drawer 带的 metadata 主要是：

- `wing`
- `room`
- `source_file`
- `chunk_index`
- `added_by`
- `filed_at`
- `source_mtime`

这说明它把“记忆”定义成了**带来源与位置标签的原文片段**，而不是“从原文中抽出来的结论”。$^{[3]}$

### 2. 结构化只是导航层

当用户查询时，`searcher.py` 并没有做复杂的多路召回或 rerank，而是直接调用 Chroma 的 `query_texts + where filter`。所谓 palace 结构，本质上就是给向量检索加上 `wing/room` 过滤条件。$^{[4]}$ 这也是 README 后来承认“metadata filtering 是标准 ChromaDB 特性，不是新 retrieval moat”的原因。$^{[2]}$

### 3. 上下文预算通过层次管理

`layers.py` 实现了典型的 4-layer memory stack：

- `Layer0`: 读取 `~/.mempalace/identity.txt`
- `Layer1`: 从 palace 中挑 top drawers，拼成 wake-up context
- `Layer2`: 按 wing/room 做按需召回
- `Layer3`: 全量语义搜索

这并不是新检索算法，而是一种上下文管理策略：**把“记忆是否存在”与“当前应该注入多少记忆”分开处理。**$^{[5]}$

## 具体实现一：项目文件如何被挖进 palace

### Gitignore 不是交给 Git，而是自己实现

`miner.py` 里写了一个 `GitignoreMatcher`，自己解析每一级目录里的 `.gitignore`，支持：

- anchored rule
- dir-only rule
- negation rule
- 多层级 matcher 叠加
- 最后匹配覆盖前面匹配

同时它还有 `include_ignored` 机制，允许显式把被忽略的目录或文件重新纳入扫描。$^{[3]}$ 这类代码没有“算法美感”，但对本地长期记忆系统很关键，因为用户往往真正想记住的内容恰好在 `docs/`、`generated/`、导出文件或缓存目录里。

### 房间分配是轻量启发式路由

单个文件会被路由到一个 room，优先级很直接：

1. 路径中的目录名命中 room 名或关键词
2. 文件名命中 room 名
3. 内容前 2000 字里的关键词计数最高者
4. fallback 到 `general`

这一层没有 embedding classifier，也没有 AST 分析，纯粹是路径与关键词打分。其目标不是“绝对准确分类”，而是给向量检索加一个足够有用的预过滤维度。$^{[3]}$

### Chunking 策略非常保守

项目文件分块参数写死在 `miner.py`：

- `CHUNK_SIZE = 800`
- `CHUNK_OVERLAP = 100`
- `MIN_CHUNK_SIZE = 50`

算法本身也很简单：先按长度截断，再尽量回退到双换行或单换行边界。也就是说，它本质是**带重叠的字符窗口切块**，只是尽量不把自然段切碎。$^{[3]}$

### 增量更新的关键细节

`palace.py` 里的 `file_already_mined()` 会利用 `source_mtime` 判断文件是否需要重挖。真正值得注意的是，文件修改后并不是对已有向量做原地更新，而是先按 `source_file` 删除旧 drawers，再重新插入新 chunks。源码注释明确写了这样做是为了避开 Chroma/hnsw 更新路径的稳定性问题。$^{[3][6]}$

这一点很说明问题：MemPalace 的“算法优势”并不只在 retrieval，而在于它为了让本地 memory 长期可用，做了很多底层稳定性工程。

## 具体实现二：对话记忆如何被标准化和抽取

### 先统一 transcript 格式

`normalize.py` 支持的输入格式不少：

- Claude Code JSONL
- OpenAI Codex CLI JSONL
- ChatGPT conversations JSON
- Claude.ai JSON
- Slack JSON
- 已有 transcript 或普通文本

它最终会尽量转成一种统一的 transcript 格式：用户消息前用 `>` 标记，后面紧跟 assistant 回复。$^{[7]}$ 这个格式选择非常务实，因为下游 chunker 就不需要理解各种上游 schema 了。

### 默认模式是 exchange pair

`convo_miner.py` 的默认 chunking 不是按 token window，而是按 exchange pair：

```text
> 用户一轮
AI 回复若干行
```

如果文本里至少有 3 行 `>` 开头，就按 turn 切；否则回退到段落切分。$^{[8]}$ 这意味着它把“一问一答”视为对话记忆的自然最小单元，而不是句子或固定 token 窗口。

### 话题分类仍然是关键词启发式

默认 convos 模式下，room 由 `technical / architecture / planning / decisions / problems` 五组 topic keywords 计分选出。$^{[8]}$ 也就是说，这里没有 conversation topic model，只有受控词表。

### General 模式才是 5 类记忆抽取

如果用户传 `--extract general`，就进入 `general_extractor.py`。这个模块是 MemPalace 里很有代表性的“启发式智能”：

1. 先按段落或说话轮次分段
2. 尽量过滤代码行，只保留 prose
3. 用五组 regex marker 对每段打分
4. 根据 sentiment 和是否包含 resolution 做消歧
5. 将段落判为 `decision / preference / milestone / problem / emotional`

其中一个比较漂亮的规则是：如果一段首先像 `problem`，但同时出现了 `fixed / solved / it works / figured out` 这类 resolution signal，它会被改判为 `milestone`。$^{[9]}$ 这不是复杂机器学习，但很符合工程语料的实际分布。

## 具体实现三：检索层其实很简单

如果只看主搜索链路，MemPalace 的检索算法几乎可以用一句话说完：

> 用 ChromaDB 做语义检索，再用 wing/room 做元数据过滤。

`searcher.py` 里没有 query decomposition、没有 hybrid BM25、没有 multi-vector index、没有 learning-to-rank。$^{[4]}$ 项目的强项来自两个地方：

- 原文保留度高
- 检索前搜索空间已经被 palace taxonomy 缩小

这一点也解释了为什么它的 README 一边强调 benchmark，一边又必须在后续说明里修正很多“算法上很新”的表述。真正有效的东西，很多时候恰好不是新东西，而是把已有机制拼得足够实用。

## 具体实现四：分层记忆 stack 是上下文管理，不是神经记忆

`Layer1` 经常被包装成“essential story”。源码里它的做法其实很透明：

- 批量把 drawers 拉出来
- 看 metadata 中是否有 `importance / emotional_weight / weight`
- 取前 `MAX_DRAWERS = 15`
- 总长度控制在 `MAX_CHARS = 3200`
- 按 room 分组，截短显示

这更像是一个启发式 recap builder。$^{[5]}$ 一个很现实的限制是：默认 miner/convo_miner 并不会积极写入 `importance`，所以很多数据在 `Layer1` 里最终会退化为默认权重 `3`。也就是说，wake-up 质量取决于上游有没有额外提供权重元数据。

## 具体实现五：知识图谱是旁路系统，不是自动主链

`knowledge_graph.py` 单独维护了一个 SQLite 时态三元组图：

- `entities`
- `triples`
- `valid_from`
- `valid_to`
- `confidence`
- `source_closet`
- `source_file`

它支持：

- `add_triple`
- `invalidate`
- `query_entity`
- `timeline`
- `stats`

设计上是相当干净的 temporal KG。$^{[10]}$ 但问题也很明确：**它不是从 project/convo miner 自动抽出来的默认产物。** 当前主链里，它更多是通过 MCP write tools 手工或半手工维护的结构化记忆后端。$^{[11]}$

所以如果把整个系统类比成人类记忆，drawer 是 episodic memory，KG 更像 declarative fact store；二者并存，但默认使用频率并不对等。

## 实体检测、房间检测与查询清洗：MemPalace 的启发式三件套

### 实体检测：宁可保守，不乱认

`entity_detector.py` 的候选提取策略非常保守：

- 抓 capitalized proper nouns
- 多词专有名词单独计数
- 至少出现 3 次才进入候选
- 默认每个文件只读前 5KB
- 默认最多读 10 个文件

接着再用 person signals 和 project signals 分类。更重要的是，它要求“至少两种不同的人类信号类别”才把候选判成 person，否则宁可落到 `uncertain`。$^{[12]}$ 这套策略明显是在对抗代码仓库里海量伪实体。

### 房间检测：文件系统即 taxonomy

`room_detector_local.py` 基于一个 70+ 项的 `FOLDER_ROOM_MAP`，先从顶层目录名推 room，不够再看文件名模式。$^{[13]}$ 这不是 semantic clustering，而是把已有项目结构尽量无损地转成 palace taxonomy。

### 查询清洗：处理 agent 的 system prompt 污染

这个项目里最有“harness engineering”味道的模块，反而是 `query_sanitizer.py`。它处理的是 agent 常见失误：把一大段 system prompt 或 wake-up context 拼到搜索 query 前面，导致 embedding 检索近乎失效。

它的处理顺序是：

1. 短 query 直接 passthrough
2. 从长文本里找最后一个问句
3. 否则取最后一个有意义句子
4. 最后 fallback 到尾部截断

阈值也写得很直白：

- `SAFE_QUERY_LENGTH = 200`
- `MAX_QUERY_LENGTH = 500`
- `MIN_QUERY_LENGTH = 10`

这是一个非常工程化的补丁，但对 agent memory system 很重要，因为真实失效点往往不在 embedding 本身，而在**上游 agent 传进来的 query 已经坏掉了**。$^{[14]}$

## AAAK 的真实位置：可选有损表达层

AAAK 可能是这个项目最容易被误解的部分。早期叙事里，它像是一种“closet memory language”；而从当前代码与 README 更正来看，它的真实位置已经清楚很多：

- 默认存储不是 AAAK，而是 raw text
- AAAK 不是 lossless compression，而是 lossy structured summary
- 它主要做实体缩写、topic 词提取、关键句摘取、emotion/flag 标记

`dialect.py` 的实现也证明了这一点：`compress()` 只是把一段文本压成类似 `header + entities + topics + key sentence + emotion + flag` 的紧凑表示。$^{[15]}$ 它更像一种给 LLM 看的 shorthand，而不是底层索引结构。

## 对 README 叙事的一个代码层判断

如果只看 README，很容易以为 MemPalace 的核心创新是：

- halls / tunnels 导航
- closet + AAAK 压缩
- palace structure 带来的 retrieval boost

但如果只看代码，我会给出更克制的结论：

1. **主干价值在 raw ingest + metadata filtering + context layering。**
2. **graph、AAAK、KG 都是附加能力，不是默认主链的唯一关键点。**
3. **项目真正成熟的部分是大量防御性工程，而不是 flashy algorithm。**

例如 `palace_graph.py` 需要 `hall/date` metadata 才能充分发挥作用，但默认 miner 并不会系统地产出这些字段；`Layer1` 依赖 importance weight，但默认 ingest 也很少写；KG 设计完整，但默认不是自动抽取。$^{[5][10][16]}$ 这类“概念比落地丰满”的落差，在 agent tooling 项目里很常见。

## 我的结论

MemPalace 不是一个“新一代记忆模型”，而是一个相当典型、但做得不差的**本地记忆检索 harness**。它真正可取的地方有三点：

1. **原文优先。** 不让摘要提前替代记忆。
2. **启发式克制。** 尽量只在 routing、classification、sanitization 上做轻量规则。
3. **工程防守强。** 很多功夫花在增量、过滤、稳定性、审计和 agent misuse 上。

如果把它放回更大的 Agent 工程语境中，我会说 MemPalace 的启发不是“如何发明更聪明的记忆”，而是：

> 当 LLM memory 问题还很不稳定时，最稳妥的路线往往不是让模型决定更多，而是让系统丢失更少、过滤更准、注入更省。

这也是我读完源码后，对它最认可的一点。

## 参考文献

[[1](https://github.com/milla-jovovich/mempalace)] milla-jovovich. MemPalace. *GitHub Repository*.

[[2](https://github.com/milla-jovovich/mempalace/blob/main/README.md)] milla-jovovich. README for MemPalace. *GitHub*.

[[3](https://github.com/milla-jovovich/mempalace/blob/main/mempalace/miner.py)] milla-jovovich. `mempalace/miner.py`. *GitHub*.

[[4](https://github.com/milla-jovovich/mempalace/blob/main/mempalace/searcher.py)] milla-jovovich. `mempalace/searcher.py`. *GitHub*.

[[5](https://github.com/milla-jovovich/mempalace/blob/main/mempalace/layers.py)] milla-jovovich. `mempalace/layers.py`. *GitHub*.

[[6](https://github.com/milla-jovovich/mempalace/blob/main/mempalace/palace.py)] milla-jovovich. `mempalace/palace.py`. *GitHub*.

[[7](https://github.com/milla-jovovich/mempalace/blob/main/mempalace/normalize.py)] milla-jovovich. `mempalace/normalize.py`. *GitHub*.

[[8](https://github.com/milla-jovovich/mempalace/blob/main/mempalace/convo_miner.py)] milla-jovovich. `mempalace/convo_miner.py`. *GitHub*.

[[9](https://github.com/milla-jovovich/mempalace/blob/main/mempalace/general_extractor.py)] milla-jovovich. `mempalace/general_extractor.py`. *GitHub*.

[[10](https://github.com/milla-jovovich/mempalace/blob/main/mempalace/knowledge_graph.py)] milla-jovovich. `mempalace/knowledge_graph.py`. *GitHub*.

[[11](https://github.com/milla-jovovich/mempalace/blob/main/mempalace/mcp_server.py)] milla-jovovich. `mempalace/mcp_server.py`. *GitHub*.

[[12](https://github.com/milla-jovovich/mempalace/blob/main/mempalace/entity_detector.py)] milla-jovovich. `mempalace/entity_detector.py`. *GitHub*.

[[13](https://github.com/milla-jovovich/mempalace/blob/main/mempalace/room_detector_local.py)] milla-jovovich. `mempalace/room_detector_local.py`. *GitHub*.

[[14](https://github.com/milla-jovovich/mempalace/blob/main/mempalace/query_sanitizer.py)] milla-jovovich. `mempalace/query_sanitizer.py`. *GitHub*.

[[15](https://github.com/milla-jovovich/mempalace/blob/main/mempalace/dialect.py)] milla-jovovich. `mempalace/dialect.py`. *GitHub*.

[[16](https://github.com/milla-jovovich/mempalace/blob/main/mempalace/palace_graph.py)] milla-jovovich. `mempalace/palace_graph.py`. *GitHub*.
