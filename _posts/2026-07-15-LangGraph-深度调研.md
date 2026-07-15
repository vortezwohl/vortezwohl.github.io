---
layout: post
toc: true
title: "LangGraph 深度调研：从 Graph API、Pregel 执行引擎到 Checkpoint / Interrupt / Time Travel 的完整源码级理解"
categories: Agent
tags: [LangGraph, LLM, Agent, Workflow, Code Agent, LangChain]
author:
  - vortezwohl
  - 吴子豪
  - Codex
---

这篇文档的目标不是“带你快速上手 LangGraph”，而是**把 LangGraph 彻底剖开**：它解决什么问题、抽象边界在哪里、Graph API 和 Functional API 是如何落到同一套运行时上的、为什么它的核心不是 prompt 而是 runtime、为什么 checkpoint 不是可有可无的 memory、为什么 interrupt / resume / replay / time-travel 能成立，以及在真实工程里到底该怎么用、怎么避坑。

如果只用一句话概括 LangGraph，那么我会这样说：

**LangGraph 不是一个“现成 agent”，而是一个以 Pregel / Bulk Synchronous Parallel 为执行模型、以 channels 为状态存储单元、以 checkpoints 为可恢复执行基础设施的低层 agent / workflow orchestration runtime。**

换句话说，它最重要的价值不在于“帮你写一个 ReAct agent”，而在于它把下面这些在生产环境里真正麻烦的事情系统化了：

- 长时运行；
- 有状态执行；
- 中断与恢复；
- 人机协同；
- 部分失败后的续跑；
- 子图与多 actor 协作；
- 流式输出；
- 节点级重试、缓存、超时、错误处理；
- 把 agent 从“一个 prompt 循环”升级成“一个可追踪、可恢复、可验证的执行系统”。

本文内容基于我在 `2026-07-15` 对 `langchain-ai/langgraph` `main` 分支、官方文档站、公开 README、关键源码文件与相关子包元数据的逐层核对。本文中所有“当前版本”“当前文档主线”“当前 deprecated 状态”等表述，均以这一天公开可见的信息为准。

> 截至 `2026-07-15`，我核到 `libs/langgraph/pyproject.toml` 中的 Python 主包版本为 `1.2.9`。

---

## 一、为什么要认真学 LangGraph

很多人第一次接触 LangGraph，是因为它和 LangChain、LangSmith、Deep Agents 绑定得很近，于是容易产生一个错觉：

- LangChain 是“写 prompt 和 tool”
- LangGraph 是“再高级一点的 agent 框架”

这个理解不够准确。

更准确的分层应该是：

- **LangChain** 更偏向“组件生态 + 高层 agent 构造器 + 模型/工具集成层”；
- **LangGraph** 更偏向“运行时编排层”；
- **LangSmith** 更偏向“可观测性 / 评估 / 部署 / 管理层”；
- **Deep Agents** 更偏向“建立在 LangGraph 之上的更高层现成 agent runtime”。

因此，LangGraph 的位置并不是“比 LangChain 更强”，而是“LangChain agent 想在生产场景里可靠，就需要一套更底层的 runtime，而 LangGraph 就是这层 runtime”。

它最适合解决的问题通常具备下列一个或多个特征：

- 任务不是一次 prompt 就能做完，而是有多步状态推进；
- 需要在中间步骤插入人工审查或人工输入；
- 需要跨多轮对话或多次调用保留执行状态；
- 某些步骤失败后，希望恢复时只重做必要部分，而不是整条链从头跑；
- 需要子图、分支、汇合、并行 fan-out、map-reduce；
- 需要边跑边输出 token、状态、任务事件、自定义事件；
- 需要给节点配置 retry、timeout、cache、error handler；
- 需要把 agent 当作“系统”而不是“单个 prompt 函数”来维护。

如果你做的是下面这类场景，LangGraph 很可能不是第一选择：

- 单次调用即可完成的简单文本生成；
- 没有状态、没有恢复、没有复杂工具协作的轻量流程；
- 只想尽快得到一个能用的常规 tool-calling agent。

这种情况下，更高层的 `langchain.agents.create_agent` 往往更直接。

---

## 二、必须先建立的总心智模型

理解 LangGraph，最重要的不是背 API，而是先建立一套正确心智模型。

### 1. LangGraph 不是“函数调用链”，而是“图 + 执行引擎”

很多 Python workflow 库本质上都是：

1. 调这个函数；
2. 结果传给下一个函数；
3. 失败就抛异常；
4. 成功就结束。

LangGraph 不是这个模型。

它更像是：

1. 图里有一组节点；
2. 节点不直接互相调用，而是通过状态 channel 读写；
3. 系统按 super-step 批次执行可运行节点；
4. 每个 step 的写入统一在 step 结束时应用；
5. 下一个 step 再根据哪些 channel 变化了，决定哪些节点要被触发；
6. 整个过程可以在任意 checkpoint 点持久化、恢复、回放。

所以真正的 mental model 不是：

`node_a() -> node_b() -> node_c()`

而是：

`一组节点围绕共享状态与消息通道，被一个 step-based runtime 调度执行`

### 2. LangGraph 的状态不是一个普通 dict，而是一组 channels

从用户视角看，你给 `StateGraph` 传的是一个 `TypedDict`、Pydantic Model 或 dataclass。

但从内部实现看，LangGraph 会把 state schema 拆成多个 key，并为每个 key 绑定一个 channel。

这意味着：

- 每个 key 都有自己的存储语义；
- 每个 key 都有自己的 update 语义；
- 每个 key 都有自己的 checkpoint 语义；
- 多节点并发写同一个 key 时，是否可合并，取决于对应 channel；
- 所谓“状态恢复”，实际上是在恢复一组 channel。

### 3. LangGraph 的真正核心是 Pregel runtime

用户常用的是：

- `StateGraph`
- `@entrypoint`
- `ToolNode`
- `create_react_agent`

但这些都不是根。

真正的根是 `pregel/main.py` 里的 `Pregel` 及其配套的 `_loop.py`、`_algo.py`、`_runner.py`、`_checkpoint.py`。

如果不理解 Pregel runtime，就无法真正理解下面这些现象：

- 为什么同一步里节点看不到彼此的新写入；
- 为什么 `Send` 可以 fan-out；
- 为什么中断恢复时节点会从头执行；
- 为什么成功节点在失败恢复时通常不会重跑；
- 为什么 time-travel 需要 fork checkpoint；
- 为什么 `DeltaChannel` 会让 checkpoint 不是完整状态快照。

### 4. LangGraph 的“可恢复执行”不是 marketing，而是底层协议

很多框架说自己支持 memory / persistence，本质只是把最后一轮消息存一下。

LangGraph 不是这样。

它保存的是：

- checkpoint id；
- parent checkpoint；
- channel values；
- channel versions；
- versions seen；
- pending writes；
- step metadata；
- delta channel 历史；
- interrupt / resume 信息。

这已经不是“聊天记忆”了，而是**执行恢复协议**。

---

## 三、仓库结构总览

截至 `2026-07-15`，我核到 `libs/langgraph/langgraph` 主包一级模块大致如下：

```text
langgraph/
  _internal/
  channels/
  func/
  graph/
  managed/
  pregel/
  stream/
  callbacks.py
  config.py
  constants.py
  errors.py
  runtime.py
  types.py
  typing.py
  warnings.py
```

可以按功能把它们压缩成四层：

### 1. 定义层

- `graph/`
- `func/`

负责让开发者声明“图长什么样”“节点是什么”“输入输出长什么样”。

### 2. 执行层

- `pregel/`

负责真正跑图。

### 3. 状态层

- `channels/`
- `managed/`
- `types.py`

负责状态、通道、控制信号与运行时类型。

### 4. 横切基础设施层

- `runtime.py`
- `stream/`
- `callbacks.py`
- `errors.py`
- `_internal/`

负责注入、流式、生命周期事件、内部配置、序列化、timeout、scratchpad 等。

如果只给出一张“最重要模块图”，应该是这样：

```text
StateGraph / entrypoint
        |
        v
   compile/build
        |
        v
      Pregel
        |
  +-----+-------------------+
  |         |        |      |
  v         v        v      v
loop      algo    runner  checkpoint
  |         |        |      |
  +-----> channels <-+------+
            |
            v
        graph state
```

---

## 四、顶层公共 API 到底提供了什么

### 1. `graph/__init__.py`

LangGraph Graph API 暴露的公共面很薄：

- `StateGraph`
- `START`
- `END`
- `add_messages`
- `MessagesState`
- `MessageGraph`

这说明官方有意把高层 Graph API 保持克制：**重点不是暴露很多花哨对象，而是让大部分复杂度藏在编译和运行时里。**

### 2. `func/__init__.py`

Functional API 主要暴露：

- `@task`
- `@entrypoint`

这是一套更像“普通 Python 工作流”的前端 DSL。

### 3. `prebuilt`

`langgraph-prebuilt` 子包暴露高层预制件：

- `ToolNode`
- `ValidationNode`
- `InjectedState`
- `InjectedStore`
- `ToolRuntime`
- `create_react_agent`

但要注意一个重要现实：

截至 `2026-07-15`，我核对到的源码里，`create_react_agent` 已经显式标为 **deprecated**，迁移方向是 `langchain.agents.create_agent`。因此，今天研究 LangGraph，不应把 `create_react_agent` 误当成主干，而应该把它当成“基于 LangGraph 运行时实现的一层预制 agent”。

---

## 五、Graph API：`StateGraph` 是怎么把业务代码编译成运行时对象的

`graph/state.py` 是全库最值得反复读的文件之一。

### 1. `StateGraph` 的本质：builder，不是 executor

源码里写得很清楚：`StateGraph` 是 builder class，本身不能执行，必须先 `compile()`。

这点非常关键，因为它决定了 Graph API 的职责边界：

- 负责收集节点、边、状态 schema、node defaults；
- 负责做静态校验；
- 负责把高层声明编译成低层 `PregelNode + channels + writers`；
- 不负责真正运行。

### 2. state schema 如何变成 channels

`StateGraph.__init__()` 会调用 `_add_schema()`，进一步调用 `_get_channels()`。

其逻辑大致是：

1. 读取 schema 的 type hints；
2. 对每个字段调用 `_get_channel(name, annotation)`；
3. 如果 `Annotated[..., reducer]`，则创建聚合 channel；
4. 如果是特殊 managed value 注解，则登记为 managed；
5. 否则默认落到 `LastValue`。

这意味着 LangGraph 不是“你返回 dict，系统帮你 merge 一下”，而是：

**系统在 compile 前就已经决定了每个 state key 的 update semantics。**

### 3. `add_node()` 的真正作用

`add_node()` 做的不是简单注册函数，而是创建一个 `StateNodeSpec`，其中会保存：

- `runnable`
- `metadata`
- `input_schema`
- `retry_policy`
- `cache_policy`
- `error_handler_node`
- `defer`
- `timeout`
- `ends`

也就是说，节点不是“一个函数”，而是“一个带有运行时策略的执行规格”。

### 4. edge 与 conditional edge 最后都会变成写 channel 的规则

这是理解 LangGraph 控制流的关键。

#### 普通 edge

普通 `add_edge(a, b)` 的效果不是“a 执行完马上调用 b”，而是：

- a 执行完后会向某个内部 branch channel 写入；
- b 订阅这个 channel；
- 下一 super-step 再触发 b。

#### conditional edge

`add_conditional_edges()` 会把路由函数包装成 `BranchSpec`。

`BranchSpec.run()` 里会把 path 函数的返回结果变成：

- node name
- 或 `Send`
- 或 END

然后再转成 `ChannelWrite`。

因此，LangGraph 的路由本质也是“状态驱动 + 通道触发”，而不是命令式函数跳转。

### 5. `compile()` 到底编出来了什么

`StateGraph.compile()` 最终返回的是 `CompiledStateGraph`，而其底层继承的就是 `Pregel`。

可以把 compile 结果理解为：

```text
用户定义的状态图
  ->
一组 PregelNode
  +
一组 channels
  +
触发关系与写入规则
  +
checkpoint/store/cache/interrupt 配置
  ->
可执行 Pregel 图
```

这也是为什么说 Graph API 只是 DSL，真正的行为都由 runtime 决定。

---

## 六、Functional API：另一套 DSL，不是另一套运行时

很多人第一次看到 `@entrypoint` 和 `@task`，会以为 LangGraph 其实有两套运行机制。不是。

### 1. `@task`

`@task` 的语义是：

- 把一个普通函数或 async 函数包装成 LangGraph task；
- 允许附加 `retry_policy`、`cache_policy`、`timeout`；
- 在运行时返回 future，而不是立即同步返回结果；
- 只能在 `entrypoint` 或 graph 节点环境中安全使用。

这使它很像“可被 runtime 跟踪和重试的异步工作单元”。

### 2. `@entrypoint`

`@entrypoint` 看上去像“把一个函数升级成 workflow”，底层其实是直接构造了一个极简 `Pregel`：

- `START` channel 接输入；
- 节点就是你那个函数；
- `END` channel 存输出；
- `PREVIOUS` channel 存下一次运行可读的 previous；
- 如果返回 `entrypoint.final(value=..., save=...)`，则 `value` 和 `save` 分流到不同 channel。

所以 Functional API 的本质，不是“绕开 Graph API”，而是“用函数式书写风格直接生成 Pregel 图”。

### 3. Functional API 什么时候更适合

适合：

- 主流程本身比较线性；
- 并发点主要通过 task future 管理；
- 你想保留普通 Python 控制流的可读性；
- 不想显式维护节点和边的名称。

不适合：

- 复杂显式分支；
- 多子图组合；
- 需要将控制流结构高度可视化；
- 需要大量复用节点图片段。

---

## 七、Pregel：LangGraph 的真正执行模型

`pregel/main.py` 的文档字符串已经把全局模型讲得很直接：

- actor model
- channels
- Bulk Synchronous Parallel

### 1. 最重要的三阶段

每个 super-step 包含三个阶段：

1. `Plan`
2. `Execution`
3. `Update`

这三阶段是 LangGraph 一切“看起来反直觉行为”的来源。

### 2. 为什么同一步里节点看不到彼此的新写入

因为 step 内部的写入不是立即可见的。

在 `Execution` 阶段，节点只是把写入塞进 `task.writes` / `pending_writes`。

真正应用到 channels，要等到 step 结束的 `Update` 阶段，由 `apply_writes()` 统一完成。

因此：

- step `N` 的节点看到的是 step `N-1` 已稳定的 channel 状态；
- step `N` 内节点之间不会看到彼此刚写的结果；
- step `N` 的写入只会影响 step `N+1` 的触发和读取。

这就是 BSP 模型的本质。

### 3. `PregelNode` 不是“节点函数”，而是“节点执行描述”

`pregel/_read.py` 里的 `PregelNode` 包含：

- `channels`
- `triggers`
- `mapper`
- `writers`
- `bound`
- `retry_policy`
- `cache_policy`
- `timeout`
- `tags / metadata`
- `is_error_handler`
- `error_handler_node`
- `subgraphs`

也就是说，运行时眼里的 node 已经不是业务函数本身，而是一个 **可执行 actor 定义**。

### 4. `NodeBuilder` 是更低层的直接 Pregel 构造器

Graph API 通常不会让你直接碰 `NodeBuilder`，但它揭示了 Pregel runtime 想要的最小元素：

- 订阅哪些 channel；
- 触发条件是什么；
- 执行什么 runnable；
- 向哪些 channel 写什么；
- 携带什么 policy。

这非常有助于反向理解 `StateGraph.compile()` 在做什么。

---

## 八、一次真实运行到底发生了什么

这一节最重要。我们不讲抽象，只讲时序。

下面以 `graph.stream(...)` 或 `graph.invoke(...)` 为例，按源码还原一次真实运行。

### Step 0：入口准备

`Pregel.stream()/astream()` 会做这些事：

- 整理 `stream_mode`、`interrupt_before/after`、`durability`、`subgraphs`；
- 构建 callback manager；
- 构建 messages/tool stream handler；
- 构建 `Runtime`；
- 注入 `context / store / stream_writer / control / server_info`；
- 根据 `version="v1"` 或 `"v2"` 决定流输出格式；
- 创建 `SyncPregelLoop` 或 `AsyncPregelLoop`；
- 创建 `PregelRunner`；
- 在 `while loop.tick():` 中逐步推进。

### Step 1：进入 loop，加载 checkpoint

`SyncPregelLoop.__enter__()` / `AsyncPregelLoop.__aenter__()` 会首先决定“从哪里开始跑”：

- 没有 checkpointer：相当于无状态执行；
- 指定了 `checkpoint_id`：精确加载这个 checkpoint；
- 如果有 `ReplayState`：说明当前是子图 replay，需要特殊加载；
- 否则：加载 thread 的最新 checkpoint；
- 如果 thread 从未跑过：构造 synthetic empty checkpoint。

这一步之后，loop 会得到：

- `checkpoint`
- `checkpoint_metadata`
- `checkpoint_pending_writes`
- `prev_checkpoint_config`
- `checkpoint_id_saved`

### Step 2：hydrate channels

接着调用 `channels_from_checkpoint()`：

- 普通 channel：直接 `spec.from_checkpoint(...)`
- `DeltaChannel`：如果当前 checkpoint 没有完整值，则通过 `saver.get_delta_channel_history()` 沿 ancestor chain 查 seed + writes，再回放恢复

这一点非常重要：

**LangGraph 读取 checkpoint 时，不一定是在读一份完整快照。**

### Step 3：`_first()` 决定是新输入还是恢复

`_first()` 是整个恢复语义的分水岭。

它先判断：

- 当前是否有历史 `channel_versions`
- 输入是否是 `None`
- 输入是否是 `Command`
- 是否与上次 `run_id` 相同
- 子图是否带 `CONFIG_KEY_RESUMING`

从而得到：

- `is_resuming`
- `is_time_traveling`

然后有三条路径：

#### 路径 A：`Command` 输入

`Command` 会先经 `map_command()` 变成 writes：

- `goto` -> `TASKS` 或 `branch:to:*`
- `resume` -> `RESUME`
- `update` -> 普通 state writes

如果 `resume` 是多 interrupt id map，还会额外写进 `CONFIG_KEY_RESUME_MAP`。

#### 路径 B：恢复执行

如果是恢复：

- 会把 `versions_seen[INTERRUPT]` 更新到当前 channel version；
- 如果是 time-travel，则可能先补一个 `source="fork"` checkpoint；
- 然后立刻 emit 一次当前 values。

#### 路径 C：全新输入

如果是新输入：

- `map_input()` 把输入映射到 input channels；
- `apply_writes()` 立即应用到 live channels；
- 必要时把 delta-channel 输入写入 pending writes；
- 立即 `_put_checkpoint({"source":"input"})`

所以首轮输入本身就产生 checkpoint，这不是附加能力，而是主流程一部分。

### Step 4：`tick()` 计算下一步 task 集合

进入主循环后，`loop.tick()` 做的第一件事是调用 `prepare_next_tasks()`。

这个函数会综合：

- 当前 checkpoint
- pending writes
- process/node 集合
- channels
- managed values
- step / stop
- updated_channels
- `trigger_to_nodes` 优化映射

产出下一 super-step 的 `PregelExecutableTask` 字典。

task 来源有两类：

- `PULL`：普通由 channel 更新触发的节点
- `PUSH`：由 `Send` 或 functional push 生成的任务

### Step 5：中断前检查

如果配置了 `interrupt_before`，`tick()` 会调用 `should_interrupt()`。

它不是简单“命中节点名就停”，而是：

1. 判断自上次 interrupt 以来是否有 channel version 发生变化；
2. 再看当前将执行的 task 是否落在 interrupt 集合里。

只有两个条件都满足，才会触发 interrupt。

### Step 6：runner 执行 task

`PregelRunner.tick()/atick()` 只执行 `not t.writes` 的任务。

这句话非常关键。

意味着：

- cache 命中的任务可能已经有 writes，不再执行；
- 从 checkpoint 恢复出成功结果的任务已有 writes，不再执行；
- 失败或中断的任务因为没有成功业务 writes，仍会被执行；
- 如果有 error handler，恢复时会把原失败节点标成已处理，再新建 handler task。

### Step 7：task 内部如果产生 `Send`，会动态扩展当前 task 集

当节点内部通过 `Send` 或 functional call 产生新任务时，runner 会调用 `loop.accept_push()`。

`accept_push()` 会：

- 基于当前 checkpoint id、task path、call index 生成新 task id；
- 通过 `prepare_single_task()` 把它编成一个新的 `PregelExecutableTask`；
- 放回 `self.tasks`；
- 如有必要，对它恢复已有 pending writes；
- 让 runner 在本轮或后续调度它。

### Step 8：`commit()` 把结果写入 pending writes

每个 task 完成后，runner 的 `commit()` 会区分四种情况：

#### 1. 正常完成

- 如果没有 writes，会补 `NO_WRITES`
- 正常把 task writes 保存到 checkpointer pending writes

#### 2. `interrupt()`

- 保存 `INTERRUPT`
- 必要时也保存 `RESUME`

#### 3. 普通异常

- 保存 `ERROR`

#### 4. 有 error handler 的异常

- 保存 `ERROR`
- 额外保存 `ERROR_SOURCE_NODE`

这个 `ERROR_SOURCE_NODE` 很重要，它就是恢复时“不要再跑原节点，转去跑 handler”的证据。

### Step 9：`after_tick()` 统一应用所有 writes

`after_tick()` 是整个 super-step 真正完成的时刻。

它做的事：

1. 汇总当前 step 全部 task writes；
2. 记录哪些 delta channel 本步用了 overwrite；
3. 调用 `apply_writes()` 更新 channels；
4. 发 `values` 输出；
5. 如果是 exit durability，额外累积 delta writes；
6. 清空 pending writes；
7. 关闭 replay 状态；
8. `_put_checkpoint({"source":"loop"})`
9. 检查 `interrupt_after`

所以从“系统状态变更”视角看，**一个 super-step 的完成点不是 task 返回时，而是 `after_tick()` 结束时。**

### Step 10：退出与异常清理

loop 退出时会经过 `_suppress_interrupt()`：

- 在 `durability="exit"` 模式下补写 exit delta writes；
- 持久化最终 checkpoint 与 pending writes；
- 如果是顶层 graph 的 `GraphInterrupt`，抑制异常向外层失控传播，并把 interrupt 转成最终输出 / 流事件 / lifecycle event。

---

## 九、`apply_writes()`：LangGraph 状态机的核心转移函数

`pregel/_algo.py` 里的 `apply_writes()` 是最核心的状态转移函数之一。

### 1. 它做了什么

可以把它理解成：

`本 step 的所有 task writes -> 新的 channel 状态 -> 新的 channel versions -> 新的 updated_channels`

### 2. 为什么要先按 task path 排序

源码里先按 task path 排序，目的是保证确定性。

因为多个并发任务写同一 channel 时，如果不固定顺序，即便 reducer 是纯函数，也可能造成：

- 调试难度上升；
- snapshot 不稳定；
- 回放行为不稳定；
- trace 对不齐。

### 3. `versions_seen` 的意义

每个节点并不是“只要触发器 channel 有值就跑”，而是“只要触发器 channel 的版本比它上次见过的新，就跑”。

这正是：

- `channel_versions`
- `versions_seen`

这两套结构存在的意义。

### 4. channel `consume()` / `finish()`

除了普通 `update(vals)` 外，`apply_writes()` 还会调用：

- `consume()`
- `finish()`

这允许某些 channel 在：

- 被读取过后
- 图即将结束时

改变自己的内部状态。

这就是为什么 channel 不是普通 dict value，而是主动对象。

---

## 十、Task、Call、Send、Command：LangGraph 控制流的四个关键类型

### 1. `PregelExecutableTask`

这是“被 runtime 真正调度执行的对象”。

它除了 node input 外，还携带：

- config
- writes deque
- cache key
- retry policy
- timeout
- task path
- subgraphs

### 2. `Call`

`Call` 是 functional/task push 机制里用来描述“待执行函数调用”的对象，里面会带：

- `func`
- `input`
- `retry_policy`
- `cache_policy`
- `callbacks`
- `timeout`

### 3. `Send`

`Send` 是 LangGraph 最有辨识度的控制流原语之一。

它表示：

- 不是直接去调用目标节点；
- 而是向未来 step 派发一个带自定义输入的任务包。

所以 `Send` 更像：

- actor model 里的 message / packet；
- 而不是普通函数跳转。

这使得 LangGraph 很自然地支持：

- fan-out
- map-reduce
- orchestrator-worker
- 每个子任务携带不同输入

### 4. `Command`

`Command` 则是更通用的“控制与更新复合体”。

它能同时表达：

- `update`
- `resume`
- `goto`
- `graph=parent`

因此，当节点需要同时：

- 改 state；
- 指定下一跳；
- 或把控制权上交父图；

就可以返回 `Command`。

---

## 十一、Checkpoint：LangGraph 为什么能恢复

### 1. checkpoint 不是最终结果快照

`langgraph-checkpoint` 的 `Checkpoint` 结构里最核心的字段有：

- `id`
- `ts`
- `channel_values`
- `channel_versions`
- `versions_seen`
- `updated_channels`

很多人只盯着 `channel_values`，这是不够的。

真正让 LangGraph 能恢复继续跑的是：

- channel 当前值
- channel 当前版本
- 每个节点已经看到过哪些版本

少任何一个，系统都不知道下一步该调度谁。

### 2. `CheckpointTuple`

恢复时真正读出来的是 `CheckpointTuple`，它在 checkpoint 外还额外带：

- `config`
- `metadata`
- `parent_config`
- `pending_writes`

这些字段分别支持：

- 按 thread / checkpoint 精确定位；
- 知道 checkpoint 的来源（input / loop / fork / update）；
- 回溯 ancestor chain；
- 恢复未完全收敛的一步执行。

### 3. `thread_id` 与 `checkpoint_id`

LangGraph 的恢复模型是：

- `thread_id`：一条执行历史链
- `checkpoint_id`：这条链上的某个状态点

因此：

- 普通继续跑：只带 `thread_id`
- 精确 replay/time-travel：带 `thread_id + checkpoint_id`

### 4. `pending_writes` 的意义

这部分必须彻底搞懂。

假设一个 super-step 有 3 个节点并发执行：

- A 成功
- B 成功
- C 失败

如果没有 `pending_writes`，恢复时你只能把 A/B/C 全部重跑。

有了 `pending_writes` 后：

- A/B 的成功写入会被持久化；
- 恢复时 `_reapply_writes_to_succeeded_nodes()` 会把这些写入灌回内存 task；
- runner 看到 A/B 已经“有 writes”，就不会再执行它们；
- 只需重跑 C，或转去执行 C 的 error handler。

这就是 LangGraph “部分 super-step 恢复”的本质。

---

## 十二、Interrupt / Resume：人机协同的真实语义

### 1. `interrupt()` 到底做了什么

`types.py` 里的 `interrupt(value)` 逻辑非常值得逐行读：

1. 取当前 task 的 `PregelScratchpad`
2. 用 `interrupt_counter()` 拿到“本 task 第几个 interrupt”
3. 如果 scratchpad 里已经有对应 resume 值，则直接返回
4. 如果有全局 null resume 值，则消费并追加到当前 task resume 列表，再返回
5. 否则抛 `GraphInterrupt(Interrupt(...))`

这里最关键的是：

- **interrupt 不是保存执行栈**
- **interrupt 不是从异常点继续**
- **interrupt 恢复靠的是“节点重跑 + scratchpad 按顺序喂回 resume 值”**

### 2. 为什么节点恢复时会从头执行

因为 LangGraph 没有做 Python 调用栈序列化。

它的语义是：

- checkpoint 保存图级状态；
- 节点恢复时重进函数；
- 节点内部再通过 scratchpad 找回“这个 interrupt 位置该返回什么值”。

因此，下面这条规则必须牢记：

**interrupt 前的副作用必须幂等，或必须被外提到可缓存 / 可持久化的 task。**

### 3. 多 interrupt 的匹配规则

一个 task 内如果有多个 `interrupt()`，匹配规则是：

- 按出现顺序匹配 resume 值；
- 恢复时通过 `interrupt_counter()` 决定当前返回列表中的第几个值。

这意味着：

- 你不能随便改 interrupt 顺序；
- 你不能在旧 checkpoint 尚需恢复时重构该节点的 interrupt 布局；
- 多 interrupt 恢复如果有歧义，必须传 `interrupt_id -> value` 映射。

### 4. 为什么没有 checkpointer 就不能 resume

因为 `interrupt()` 恢复依赖：

- checkpoint namespace
- pending writes
- resume write
- task 级 scratchpad 重建

没有 checkpointer，这套数据根本无处可存。

---

## 十三、Time Travel：它不是简单“回到旧状态重跑”

很多人会把 time-travel 理解成：

- 取旧 state
- 再跑一次

LangGraph 不是这么做的。

### 1. `is_replaying` 与 `is_time_traveling`

源码里会区分：

- `is_replaying`
- `is_time_traveling`

并不是所有显式 checkpoint 重入都是 time-travel。

一个很典型的情况是：

- 客户端 resume 时显式带上当前 head 的 `checkpoint_id`

这时是 replay 风格进入，但不是 time-travel 分叉。

### 2. time-travel 为什么要清掉旧 `RESUME`

因为旧 checkpoint 中可能残留已解决过的 interrupt resume 值。

如果 time-travel 回到那个 checkpoint 后不清掉 resume：

- 节点中的 `interrupt()` 可能直接返回历史 resume
- 而不是重新抛出 interrupt

这会导致语义错乱。

所以源码里在 `is_time_traveling` 时会清理旧 `RESUME` writes。

### 3. 为什么 time-travel 要创建 fork checkpoint

这是恢复语义里最容易被忽略，但又最关键的一点。

当你从某个旧 checkpoint 回去重跑时，新的执行历史不应该覆盖旧历史，而应该形成新分支。

因此 LangGraph 会创建 `source="fork"` checkpoint。

这样后续：

- 新执行继续沿 fork 分支积累；
- 旧历史仍可保留；
- interrupt 后 resume 也不会误回到旧 head。

### 4. 子图 replay：`ReplayState`

子图 replay 的难点在于：

- 父图在 replay；
- 子图可能在循环中反复进入；
- 同一个 subgraph namespace 第一次进入应该回到 replay 前状态；
- 后续进入应该跟着新执行走。

`_internal/_replay.py` 的 `ReplayState` 专门解决这个问题：

- 第一次 visit 某个稳定 namespace：取 replay checkpoint 之前的最近子图 checkpoint
- 再次 visit：退回 normal latest checkpoint loading

这使“父图 replay + 子图重复执行”的语义变得可定义。

---

## 十四、DeltaChannel：为什么 checkpoint 不一定保存完整值

### 1. DeltaChannel 的设计动机

某些 channel 如果每步都存完整值，代价会很高。

例如：

- 消息列表很长；
- 某个聚合对象不断增长；
- 每步只有增量变化；

这时完整快照会浪费大量存储和 IO。

`DeltaChannel` 的思路是：

- 不必每步保存完整值；
- 可以只保存 delta writes；
- 隔一段时间再保存 snapshot seed；
- 恢复时沿 ancestor chain 把写入回放回来。

### 2. 这会带来什么工程后果

首先，checkpoint 读取不再是 O(1)。

其次，任何 saver 的：

- `copy_thread`
- `delete_for_runs`
- `prune`

都必须 delta-aware，否则会静默损坏恢复语义。

LangGraph 源码和 README 对这一点给了非常明确的 warning。

### 3. 什么时候 snapshot

`delta_channels_to_snapshot()` 的规则是：

- 更新次数达到 `snapshot_frequency`
- 或距上次 snapshot 的 super-step 数达到系统上限

满足任一条件就该 snapshot。

### 4. exit durability 下为什么更麻烦

在 `durability="exit"` 模式下，中间步骤可能不写 checkpoint。

这时 delta writes 会先累积起来，退出时：

- 先决定哪些 channel 需要 snapshot；
- 其余保留 delta writes；
- 必要时创建 stub / anchor checkpoint；
- 确保 writes durable 后 final checkpoint 才对读者可见。

这说明 LangGraph 对 checkpoint 可见性做过非常细致的设计，而不是简单“最后存一下”。

---

## 十五、Streaming：为什么说它不是附属功能，而是一套子系统

LangGraph 的流式能力分两层。

### 1. 传统 stream v1 / v2

`types.py` 里定义了这些 `stream_mode`：

- `values`
- `updates`
- `messages`
- `custom`
- `checkpoints`
- `tasks`
- `debug`

含义大致是：

- `values`：每步后的完整输出视图
- `updates`：本步哪些节点写了什么
- `messages`：模型消息/分块事件
- `custom`：节点或工具主动发出的自定义流
- `checkpoints`：checkpoint 事件
- `tasks`：任务开始/结束事件
- `debug`：更完整的调试包装

`version="v2"` 则会把输出包装为结构化 `StreamPart`。

### 2. v3：`StreamMux + Transformer + RunStream`

更底层、更强的是 v3。

它由：

- `StreamMux`
- `StreamTransformer`
- `GraphRunStream` / `AsyncGraphRunStream`
- scoped child mux

组成。

这层能力的关键思想是：

- graph 只产生 protocol events
- transformer 把事件投影成不同视图
- 每个视图都是独立 projection
- 子图可以拥有自己的 scoped projection

这使得前端或观测系统可以同时订阅：

- state values
- token stream
- lifecycle events
- subgraph handles
- tools stream
- custom stream

而不是只能拿一条“混在一起”的日志。

### 3. `StreamMessagesHandler`

LangGraph 为 `stream_mode="messages"` 做了专门 callback handler：

- 收集 chat model token / chunk
- 收集 node 输出中的消息对象
- 处理 subgraph namespace
- v2 情况下避免 ToolMessage 在消息流里重复出现

这说明 LangGraph 的“消息流”不是单纯依赖模型 SDK，而是 runtime 主动统一塑形。

---

## 十六、Runtime 与注入：节点真正拿到的不是裸 state

`runtime.py` 的 `Runtime` 提供这些字段：

- `context`
- `store`
- `stream_writer`
- `heartbeat`
- `previous`
- `execution_info`
- `server_info`
- `control`

这意味着 LangGraph 节点并不只是：

```python
def node(state): ...
```

更推荐的长期写法其实是：

```python
def node(state, runtime: Runtime[Context]): ...
```

这样你能清楚区分：

- 什么属于业务状态
- 什么属于运行依赖
- 什么属于调试/观测信息
- 什么属于控制面

这比把数据库连接、用户上下文、流式 writer、控制信号都塞进 state 要干净得多。

---

## 十七、ToolNode 与 prebuilt：LangGraph 如何承接 tool-calling

`langgraph-prebuilt/tool_node.py` 是非常重要的参考实现。

### 1. `ToolNode` 不只是“批量执行工具”

它还处理：

- 工具参数校验；
- 并行 tool call；
- state 注入；
- store 注入；
- runtime 注入；
- 错误消息归一化；
- `Command` 返回值；
- 工具输出向消息流 / tools 流转发。

### 2. `InjectedState` / `InjectedStore` / `ToolRuntime`

这三个设计很漂亮：

- 模型只控制它应该控制的参数；
- 系统状态、长期存储、运行时信息由 runtime 注入；
- 不污染模型看到的 tool schema。

这比“把所有系统字段混进工具参数里再让模型自己别乱用”要可靠得多。

### 3. `create_react_agent` 的架构意义

尽管它已 deprecated，但它仍然很好地展示了一个关键思想：

**所谓 agent loop，不过是一个 LangGraph 图。**

里面通常包含：

- `agent` 节点：调用模型
- `tools` 节点：执行 tool calls
- 条件边：如果有 tool_calls 则回到 tools，否则结束

在 `version="v2"` 下，每个 tool call 甚至会被拆成单独 `Send("tools", ToolCallWithContext(...))`，这比“一条消息里顺序串行跑所有工具”更贴近 LangGraph 原生模型。

---

## 十八、LangGraph 里的错误处理不是“抛异常就完事”

LangGraph 的错误策略分层非常明确。

### 1. retry

适合：

- 瞬时网络错误；
- 限流；
- 短暂外部依赖抖动；

### 2. error handler node

适合：

- 某个节点失败后，需要用另一个节点做补偿或降级处理；
- 希望恢复时直接续跑 handler，而不是再跑原节点。

### 3. interrupt

适合：

- 需要用户、运营、审核员提供决策；
- 需要人工修复输入；
- 需要人工确认下一步动作。

### 4. bubble up

适合：

- 真正的未知错误；
- 当前图级别不该兜底；
- 应该向上层让调用方决定怎么处理。

这比很多“所有错误都统一重试或统一吞掉”的 agent 方案成熟得多。

---

## 十九、真实工程里该怎么建 LangGraph

这里给出一套我认为最稳妥的工程实践。

### 1. 先定义状态边界，再写节点

不要先写一堆 node，再边写边往 state 里塞字段。

更推荐：

1. 先列 state key
2. 决定哪些 key 是最后值，哪些 key 是聚合值
3. 决定哪些东西不该进 state，而该进 runtime/store
4. 再写节点

### 2. 把 state 保持为“原始事实”，不要提前拼 prompt

这是官方文档和工程经验都强烈支持的一条原则。

state 里应尽量放：

- 原始消息
- 原始检索结果
- 原始工具输出
- 决策标签
- 任务状态

而不是放：

- 已格式化的 prompt 大字符串
- 为某个特定模型拼好的中间文本

理由很简单：

- prompt 格式很容易变；
- state 是长期资产，prompt 通常是短期视图。

### 3. 把路由函数和业务执行函数分开

node 负责：

- 读取状态
- 做业务动作
- 产出更新

route / branch 函数负责：

- 判断下一步去哪

不要把“做事”和“决定去哪”混在一个超大 node 里。

### 4. 节点必须幂等

这不是建议，而是 LangGraph 场景里的硬要求。

尤其对下面这些节点：

- interrupt 前节点
- 可能 retry 的节点
- 可能 time-travel 后重跑的节点
- 依赖外部 API 的节点

必须要么：

- 幂等；
- 要么结果缓存；
- 要么副作用外提并带去重键。

### 5. 先用 InMemorySaver 开发，再切 Postgres

开发期：

- `InMemorySaver` 足够方便；

生产期：

- 应优先 `PostgresSaver`

因为生产里你真正要的是：

- durable execution
- 可恢复线程
- 可调试历史
- 可 time-travel

而不是只是“跑通一次”。

### 6. 给节点显式配置 retry / timeout / cache

不要依赖默认一把梭。

建议按节点类型分：

- LLM 调用节点：通常配 timeout，必要时配 retry
- 外部 HTTP 节点：通常配 retry + timeout
- 纯计算节点：通常无需 retry，但可考虑 cache
- 昂贵、确定性强的节点：适合 cache

### 7. 把 interrupt 当状态转换点，而不是 UI 事件

不要把 interrupt 理解成“弹个框”。

更准确地说，它是：

- 图执行暂停点
- 需要外部提供 `Command(resume=...)` 的状态门

前端 UI 只是它的一种消费方式。

### 8. 升级图结构时，把 checkpoint 当兼容面

如果你的 graph 已经在线上积累了 thread / checkpoint 历史，就必须谨慎升级：

- 不要轻易改 node name
- 不要轻易删 state key
- 不要轻易重排 interrupt 顺序
- 不要轻易改变同一 key 的 channel 语义

要采用：

- add-then-migrate-then-remove
- versioned state
- fork/staging 验证

---

## 二十、常见坑清单

### 1. 误把 LangGraph 当 prompt loop 库

后果：

- 不理解 checkpoint 和 channel 设计
- state 乱塞东西
- 一出恢复问题就完全解释不清

### 2. 在 interrupt 前做不可重入副作用

后果：

- resume 时副作用重复执行
- 业务幂等性崩掉

### 3. 多节点并发写同一 key，却没设计 reducer

后果：

- 最后写覆盖
- 丢结果

### 4. 把长期业务数据塞进 checkpoint，而不是 store

后果：

- thread 状态膨胀
- checkpoint 恢复成本变高
- 数据生命周期和执行生命周期耦合

### 5. 生产仍用 InMemorySaver

后果：

- 无法真正恢复
- 无法 time-travel
- 服务重启后历史消失

### 6. 以为 time-travel 就是“旧 checkpoint + resume”

后果：

- 旧 resume 污染新执行
- interrupt 不再重新触发

### 7. 自定义 saver 不理解 DeltaChannel

后果：

- prune/copy/delete 后历史静默损坏
- 读 state 结果不一致

### 8. 让节点既做业务又做复杂路由

后果：

- 图结构不可视化
- debug 困难
- 节点责任过重

### 9. 在 state 里保存 prompt 拼装结果

后果：

- 状态污染
- prompt 升级成本高
- 可读性下降

### 10. 不理解 `not t.writes` 对恢复的含义

后果：

- 看不懂为什么某些节点恢复时不执行
- 错误地以为 runtime 漏跑了节点

---

## 二十一、如果你要彻底吃透 LangGraph，建议这样读源码

这是我认为最有效的阅读顺序。

### 第一轮：建立骨架

1. `libs/langgraph/langgraph/graph/__init__.py`
2. `libs/langgraph/langgraph/graph/state.py`
3. `libs/langgraph/langgraph/func/__init__.py`
4. `libs/langgraph/langgraph/types.py`
5. `libs/langgraph/langgraph/runtime.py`

目标：

- 搞懂用户视角 API
- 知道 Graph API / Functional API / Runtime / types 是什么

### 第二轮：理解执行模型

1. `libs/langgraph/langgraph/pregel/main.py`
2. `libs/langgraph/langgraph/pregel/_read.py`
3. `libs/langgraph/langgraph/pregel/_write.py`
4. `libs/langgraph/langgraph/pregel/_algo.py`
5. `libs/langgraph/langgraph/pregel/_runner.py`
6. `libs/langgraph/langgraph/pregel/_loop.py`

目标：

- 搞懂 step 是怎么推进的
- 搞懂 task / writes / apply_writes / interrupt / restore

### 第三轮：理解状态与恢复

1. `libs/langgraph/langgraph/channels/base.py`
2. `libs/langgraph/langgraph/channels/binop.py`
3. `libs/langgraph/langgraph/channels/delta.py`
4. `libs/checkpoint/langgraph/checkpoint/base/__init__.py`
5. `libs/langgraph/langgraph/pregel/_checkpoint.py`
6. `libs/langgraph/langgraph/_internal/_scratchpad.py`
7. `libs/langgraph/langgraph/_internal/_replay.py`

目标：

- 搞懂 checkpoint 不是普通状态快照
- 搞懂 interrupt / time-travel / delta replay

### 第四轮：理解高层预制件

1. `libs/prebuilt/langgraph/prebuilt/tool_node.py`
2. `libs/prebuilt/langgraph/prebuilt/chat_agent_executor.py`

目标：

- 看清楚 agent 只是图的一种实现方式
- 理解 tool-calling 如何落到底层 runtime

### 第五轮：理解 streaming 与观测

1. `libs/langgraph/langgraph/stream/_mux.py`
2. `libs/langgraph/langgraph/stream/run_stream.py`
3. `libs/langgraph/langgraph/pregel/_messages.py`
4. `libs/langgraph/langgraph/callbacks.py`

目标：

- 理解 LangGraph 不只是“能 stream token”
- 而是有一整套 event projection 体系

---

## 二十二、最终结论：LangGraph 到底是什么

如果看完全文，还想把 LangGraph 压缩成最简洁的定义，我会给出下面这个版本：

### 1. 从抽象层说

LangGraph 是一个：

- 面向 stateful workflow / agent 的
- step-based graph runtime

### 2. 从执行模型说

LangGraph 是一个：

- 基于 Pregel / BSP
- 通过 channels 读写状态
- 通过 versions 追踪节点可见性
- 通过 tasks 调度 actor

的执行引擎。

### 3. 从生产能力说

LangGraph 是一个：

- 把 checkpoint、interrupt、resume、replay、time-travel、subgraph、streaming、retry、timeout、cache、error handler

统一纳入同一套 runtime 协议的 agent orchestration system。

### 4. 从工程价值说

LangGraph 最大的价值，不是替你写 agent，而是替你把 agent 从“一个脆弱的 prompt 循环”，升级成“一个可恢复、可调试、可演化、可长期维护的执行系统”。

因此，真正应该学会的不是：

- `add_node` 怎么写
- `create_react_agent` 怎么调

而是：

- 图如何编译成 Pregel；
- 状态如何通过 channel 存储与演化；
- 为什么 recovery 依赖 checkpoint + pending writes；
- 为什么 interrupt 是 replay-based 而不是 stack-based；
- 为什么 time-travel 必须 fork；
- 为什么 LangGraph 本质上是在做 runtime engineering。

一旦这一层吃透，你再看 LangChain、Deep Agents、Codex 类 runtime、Claude Code 类 harness、以及各种 agent framework，就会突然清楚很多：

**真正决定 agent 是否可靠的，往往不是模型有多聪明，而是 runtime 是否足够严谨。**

---

## 附：我本次调研时核对过的主要公开资料

### 官方仓库与 README

- [langchain-ai/langgraph GitHub 仓库](https://github.com/langchain-ai/langgraph)
- [仓库顶层 README](https://raw.githubusercontent.com/langchain-ai/langgraph/main/README.md)
- [Python 主包 README](https://raw.githubusercontent.com/langchain-ai/langgraph/main/libs/langgraph/README.md)
- [examples/README：官方已声明 examples 仅保留归档，不再更新](https://raw.githubusercontent.com/langchain-ai/langgraph/main/examples/README.md)

### 官方文档

- [LangGraph Overview](https://docs.langchain.com/oss/python/langgraph/overview)
- [Quickstart](https://docs.langchain.com/oss/python/langgraph/quickstart)
- [Graph API](https://docs.langchain.com/oss/python/langgraph/graph-api)
- [Functional API](https://docs.langchain.com/oss/python/langgraph/functional-api)
- [Persistence](https://docs.langchain.com/oss/python/langgraph/persistence)
- [Checkpointers](https://docs.langchain.com/oss/python/langgraph/checkpointers)
- [Interrupts](https://docs.langchain.com/oss/python/langgraph/interrupts)
- [Use Time Travel](https://docs.langchain.com/oss/python/langgraph/use-time-travel)
- [Streaming](https://docs.langchain.com/oss/python/langgraph/streaming)
- [Event Streaming](https://docs.langchain.com/oss/python/langgraph/event-streaming)
- [Use Subgraphs](https://docs.langchain.com/oss/python/langgraph/use-subgraphs)
- [Thinking in LangGraph](https://docs.langchain.com/oss/python/langgraph/thinking-in-langgraph)
- [Workflows and agents](https://docs.langchain.com/oss/python/langgraph/workflows-agents)
- [Application structure](https://docs.langchain.com/oss/python/langgraph/application-structure)
- [Test](https://docs.langchain.com/oss/python/langgraph/test)
- [Backward compatibility](https://docs.langchain.com/oss/python/langgraph/backward-compatibility)
- [Fault tolerance](https://docs.langchain.com/oss/python/langgraph/fault-tolerance)

### 关键源码

- [StateGraph](https://raw.githubusercontent.com/langchain-ai/langgraph/main/libs/langgraph/langgraph/graph/state.py)
- [BranchSpec](https://raw.githubusercontent.com/langchain-ai/langgraph/main/libs/langgraph/langgraph/graph/_branch.py)
- [Node typing / StateNodeSpec](https://raw.githubusercontent.com/langchain-ai/langgraph/main/libs/langgraph/langgraph/graph/_node.py)
- [Messages / add_messages](https://raw.githubusercontent.com/langchain-ai/langgraph/main/libs/langgraph/langgraph/graph/message.py)
- [Functional API](https://raw.githubusercontent.com/langchain-ai/langgraph/main/libs/langgraph/langgraph/func/__init__.py)
- [Pregel main](https://raw.githubusercontent.com/langchain-ai/langgraph/main/libs/langgraph/langgraph/pregel/main.py)
- [Pregel loop](https://raw.githubusercontent.com/langchain-ai/langgraph/main/libs/langgraph/langgraph/pregel/_loop.py)
- [Pregel algo](https://raw.githubusercontent.com/langchain-ai/langgraph/main/libs/langgraph/langgraph/pregel/_algo.py)
- [Pregel runner](https://raw.githubusercontent.com/langchain-ai/langgraph/main/libs/langgraph/langgraph/pregel/_runner.py)
- [Pregel checkpoint](https://raw.githubusercontent.com/langchain-ai/langgraph/main/libs/langgraph/langgraph/pregel/_checkpoint.py)
- [Pregel read](https://raw.githubusercontent.com/langchain-ai/langgraph/main/libs/langgraph/langgraph/pregel/_read.py)
- [Pregel write](https://raw.githubusercontent.com/langchain-ai/langgraph/main/libs/langgraph/langgraph/pregel/_write.py)
- [Pregel io](https://raw.githubusercontent.com/langchain-ai/langgraph/main/libs/langgraph/langgraph/pregel/_io.py)
- [Channel base](https://raw.githubusercontent.com/langchain-ai/langgraph/main/libs/langgraph/langgraph/channels/base.py)
- [Runtime](https://raw.githubusercontent.com/langchain-ai/langgraph/main/libs/langgraph/langgraph/runtime.py)
- [Stream mux](https://raw.githubusercontent.com/langchain-ai/langgraph/main/libs/langgraph/langgraph/stream/_mux.py)
- [Run stream](https://raw.githubusercontent.com/langchain-ai/langgraph/main/libs/langgraph/langgraph/stream/run_stream.py)
- [Messages stream handler](https://raw.githubusercontent.com/langchain-ai/langgraph/main/libs/langgraph/langgraph/pregel/_messages.py)
- [Graph lifecycle callbacks](https://raw.githubusercontent.com/langchain-ai/langgraph/main/libs/langgraph/langgraph/callbacks.py)
- [Scratchpad](https://raw.githubusercontent.com/langchain-ai/langgraph/main/libs/langgraph/langgraph/_internal/_scratchpad.py)
- [ReplayState](https://raw.githubusercontent.com/langchain-ai/langgraph/main/libs/langgraph/langgraph/_internal/_replay.py)
- [Types: `Send` / `Command` / `interrupt`](https://raw.githubusercontent.com/langchain-ai/langgraph/main/libs/langgraph/langgraph/types.py)

### Checkpoint 相关子包

- [checkpoint base](https://raw.githubusercontent.com/langchain-ai/langgraph/main/libs/checkpoint/langgraph/checkpoint/base/__init__.py)
- [InMemorySaver](https://raw.githubusercontent.com/langchain-ai/langgraph/main/libs/checkpoint/langgraph/checkpoint/memory/__init__.py)
- [PostgresSaver](https://raw.githubusercontent.com/langchain-ai/langgraph/main/libs/checkpoint-postgres/langgraph/checkpoint/postgres/__init__.py)
- [checkpoint conformance](https://raw.githubusercontent.com/langchain-ai/langgraph/main/libs/checkpoint-conformance/README.md)

### prebuilt 相关

- [langgraph-prebuilt README](https://raw.githubusercontent.com/langchain-ai/langgraph/main/libs/prebuilt/README.md)
- [ToolNode](https://raw.githubusercontent.com/langchain-ai/langgraph/main/libs/prebuilt/langgraph/prebuilt/tool_node.py)
- [create_react_agent](https://raw.githubusercontent.com/langchain-ai/langgraph/main/libs/prebuilt/langgraph/prebuilt/chat_agent_executor.py)

---

如果你在读这篇文章时已经准备自己上手实现一个真实系统，我建议先不要直接写“大而全 agent”，而是按下面顺序做最小闭环：

1. 先写一个带 `StateGraph + TypedDict + 2 个节点 + 1 条条件边` 的最小图。
2. 再加 `InMemorySaver`，确认同一 `thread_id` 下 state 会累计。
3. 再加一个 `interrupt()`，亲手走通 `Command(resume=...)`。
4. 再加一个 `Send` fan-out 场景，体验 task 不是普通函数调用。
5. 最后再引入 `ToolNode`、长流程、子图和 Postgres checkpointer。

按这个顺序，你会真正学会 LangGraph；反过来如果一上来就套 `create_react_agent`，通常只会学会“怎么调用一个 API”，而学不会“为什么这个 runtime 值得用”。
