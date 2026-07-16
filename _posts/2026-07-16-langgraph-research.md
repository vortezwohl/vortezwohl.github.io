---
layout: post
toc: true
title: "使用 LangGraph：用法与应用场景"
categories: AI
tags: [AI, Agent, LangChain, LangGraph, LLM, Workflow]
author:
  - vortezwohl
  - 吴子豪
---

基于 LangChain 官方 LangGraph 文档整理，回答三个问题：LangGraph 怎么用、适合什么场景，以及它和普通 ReAct Agent 到底是什么关系。

## 一、结论先行

**LangGraph 不是一个具体的 ReAct Agent，而是一个面向有状态 Agent 和长时间运行任务的编排框架与运行时。**

ReAct 是一种 Agent 控制策略，核心循环是：

```text
思考 → 选择工具 → 执行工具 → 观察结果 → 再思考
```

LangGraph 负责组织这类循环以及其他业务流程，包括状态保存、节点执行、条件路由、循环、并行、人工审批、暂停恢复、重试、故障处理、多 Agent 和长时间运行任务持久化。

因此，两者不是同一层次的概念：

```text
ReAct     = Agent 如何推理和选择行动
LangGraph = Agent 系统如何编排、执行、暂停、恢复和持久化
```

LangGraph 可以实现 ReAct，也可以实现完全不依赖 ReAct 的固定工作流，还可以构建“固定工作流 + ReAct 子 Agent”的混合系统。

官方文档：

- [LangGraph Overview](https://docs.langchain.com/oss/python/langgraph/overview)
- [Thinking in LangGraph](https://docs.langchain.com/oss/python/langgraph/thinking-in-langgraph)
- [Workflows and Agents](https://docs.langchain.com/oss/python/langgraph/workflows-agents)
- [LangGraph Quickstart](https://docs.langchain.com/oss/python/langgraph/quickstart)
- [Persistence](https://docs.langchain.com/oss/python/langgraph/persistence)
- [Interrupts](https://docs.langchain.com/oss/python/langgraph/interrupts)

## 二、LangGraph 解决什么问题

最简单的 LLM 程序只有一次调用：

```python
response = model.invoke(prompt)
```

简单的 ReAct Agent 通常是一个循环：

```python
while True:
    response = model.invoke(messages)

    if not response.tool_calls:
        return response

    tool_result = execute_tools(response.tool_calls)
    messages.append(response)
    messages.append(tool_result)
```

这种实现适合短任务，但复杂业务很快会遇到人工审批、外部 API 重试、进程崩溃恢复、固定步骤与动态步骤混合、并行执行、子任务拆分和中间状态审计等问题。

LangGraph 的价值不是让模型天然变得更聪明，而是把模型调用、工具调用和业务控制流放进一个可观察、可恢复、可干预的状态化执行系统中。它是低层编排框架，不会替开发者决定 Prompt、Agent 架构或推理策略。

## 三、核心抽象：State、Node、Edge

LangGraph 的基本编程模型可以概括为：

```text
State + Nodes + Edges + Runtime
```

### 3.1 State：共享状态

State 描述整个图执行期间需要传递的数据，通常使用 `TypedDict`、数据类或其他结构化 Schema 定义。

```python
from typing import TypedDict


class EmailState(TypedDict):
    email_content: str
    sender_email: str
    classification: dict | None
    search_results: list[str] | None
    draft_response: str | None
    approved: bool | None
```

每个节点读取当前状态，并返回需要更新的字段：

```python
def classify_email(state: EmailState) -> dict:
    """分类邮件并返回分类结果。"""
    classification = classify_with_llm(state["email_content"])
    return {"classification": classification}
```

State 建议保存原始、结构化的数据，而不是提前拼装好的 Prompt。结构化 State 更容易被不同节点复用，也更方便持久化、调试、测试和修改。

### 3.2 Node：可执行节点

Node 通常就是一个接收 State、执行一项工作的函数：

```python
def search_docs(state: EmailState) -> dict:
    """根据邮件分类结果检索知识库。"""
    query = state["classification"]["topic"]
    documents = search_knowledge_base(query)
    return {"search_results": documents}
```

Node 可以执行 LLM 调用、数据库查询、外部 API、文件处理、业务动作、人工输入等待或另一个子图。将操作拆成节点后，可以针对每个节点单独定义超时、重试、日志、权限和错误处理策略。

### 3.3 Edge：控制流和条件路由

固定流程可以直接连接节点：

```python
workflow.add_edge(START, "classify")
workflow.add_edge("classify", "search_docs")
workflow.add_edge("search_docs", "draft")
workflow.add_edge("draft", END)
```

动态路由可以根据状态选择下一步：

```python
def route_after_classification(state: EmailState) -> str:
    """根据邮件意图选择后续流程。"""
    intent = state["classification"]["intent"]

    if intent == "billing":
        return "human_review"
    if intent == "bug":
        return "create_ticket"
    return "search_docs"
```

```python
workflow.add_conditional_edges(
    "classify",
    route_after_classification,
)
```

也可以使用 `Command` 同时更新 State 和指定下一节点：

```python
from typing import Literal
from langgraph.types import Command


def classify_email(
    state: EmailState,
) -> Command[Literal["search_docs", "human_review", "create_ticket"]]:
    """完成邮件分类，并选择后续节点。"""
    classification = classify_with_llm(state["email_content"])

    if classification["intent"] == "billing":
        next_node = "human_review"
    elif classification["intent"] == "bug":
        next_node = "create_ticket"
    else:
        next_node = "search_docs"

    return Command(
        update={"classification": classification},
        goto=next_node,
    )
```

### 3.4 Compile：编译图

```python
from langgraph.graph import END, START, StateGraph


workflow = StateGraph(EmailState)
workflow.add_node("classify", classify_email)
workflow.add_node("search_docs", search_docs)
workflow.add_node("draft", draft_response)
workflow.add_edge(START, "classify")
workflow.add_edge("search_docs", "draft")
workflow.add_edge("draft", END)

app = workflow.compile()
```

执行：

```python
result = app.invoke(
    {
        "email_content": "我被重复扣款了",
        "sender_email": "user@example.com",
        "classification": None,
        "search_results": None,
        "draft_response": None,
        "approved": None,
    }
)
```

## 四、Graph API 和 Functional API

LangGraph 提供两种主要使用风格。

### 4.1 Graph API

Graph API 显式声明 State、Node、Edge、条件路由、起点和终点，适合复杂拓扑、分支循环、并行、多 Agent、子图以及需要团队 review 的流程。

### 4.2 Functional API

Functional API 更接近普通 Python 代码，适合已有函数、循环和条件组成的线性流程，迁移成本较低。

| 情况 | 建议 |
|---|---|
| 简单线性流程 | Functional API 或普通 Python |
| ReAct 工具循环 | Graph API 或预置 Agent |
| 复杂分支和循环 | Graph API |
| 多 Agent 和子图 | Graph API |
| 需要流程拓扑审查 | Graph API |
| 从已有 Python 函数逐步迁移 | Functional API |

## 五、如何在 LangGraph 中实现 ReAct

ReAct 的核心图结构是：

```text
START
  ↓
llm_call
  ├── 没有工具调用 → END
  └── 有工具调用 → tools
                         ↓
                      llm_call
```

代码骨架：

```python
from typing import Literal
from langgraph.graph import END, START, MessagesState, StateGraph


def llm_call(state: MessagesState) -> dict:
    """调用绑定了工具的模型。"""
    response = model_with_tools.invoke(state["messages"])
    return {"messages": [response]}


def should_continue(state: MessagesState) -> Literal["tools", END]:
    """判断是否继续调用工具。"""
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return END


builder = StateGraph(MessagesState)
builder.add_node("llm_call", llm_call)
builder.add_node("tools", tool_node)
builder.add_edge(START, "llm_call")
builder.add_conditional_edges(
    "llm_call",
    should_continue,
    ["tools", END],
)
builder.add_edge("tools", "llm_call")

agent = builder.compile()
```

这仍然是标准 ReAct：模型决定是否调用工具，工具结果回到消息状态，模型继续推理，直到没有工具调用。

LangGraph 增加的是工程能力：循环显式化、消息状态持久化、工具节点重试、工具调用前人工审批、最大步数限制，以及把 ReAct 循环嵌入更大业务流程。

## 六、典型应用场景

### 6.1 Prompt Chaining：固定的多阶段处理

```text
抽取信息 → 生成初稿 → 事实检查 → 润色 → 输出
```

适合文档生成、内容审核、翻译校对、信息抽取和代码生成后的检查。这类任务更像 Workflow，不要求模型自主规划全部路径。

### 6.2 Routing：意图路由

```text
用户问题
  ├── 退款 → 退款流程
  ├── 技术问题 → 技术支持流程
  ├── 账户问题 → 账户流程
  └── 投诉 → 人工升级
```

适合客服分流、多知识库问答、多模型选择、权限控制和风险等级路由。

### 6.3 Parallelization：并行检索和分析

```text
研究主题
  ├── 公司基本面
  ├── 行业数据
  ├── 竞争对手
  └── 新闻事件
          ↓
       综合报告
```

适合多源搜索、多文档分析、多专家 Agent、批量文件处理和多个独立 API 查询。

### 6.4 Orchestrator-Worker：动态拆分任务

```text
Orchestrator
  ├── Worker 1
  ├── Worker 2
  ├── Worker 3
  └── Worker N
          ↓
      Synthesizer
```

Orchestrator 动态生成子任务，Worker 各自执行，最后由 Synthesizer 合并。适合长报告、多文件代码修改、大型资料研究和动态数量的子任务。

### 6.5 Evaluator-Optimizer：生成、评估和修正

```text
Generator → Evaluator
                ├── 合格 → END
                └── 不合格 → Generator
```

适合 SQL 生成与校验、代码生成与测试、翻译质量优化、事实检查和文案审核。

### 6.6 Human-in-the-loop：人工审批

```text
查询信息 → 生成动作计划 → interrupt() → 人工审批 → 执行副作用
```

适合支付、退款、删除数据、发邮件、发布代码、权限修改、合规审批和高风险工具调用。

使用 `interrupt()` 时，暂停前节点在恢复时可能再次执行。因此，写入、扣款和发消息等副作用必须设计为幂等操作。

### 6.7 长时间运行和可恢复 Agent

```text
启动任务 → 等待外部系统 → 等待人工输入 → 恢复执行 → 完成
```

适合异步审批、长周期研究、复杂数据管道、自动化运维和跨多个外部系统的业务流程。

## 七、持久化：Checkpointer 和 Store

### 7.1 Checkpointer

Checkpointer 保存一个具体 Thread 的图状态，例如：

```text
thread_id = "refund-123"

第 1 步：识别订单
第 2 步：计算退款金额
第 3 步：等待人工审批
第 4 步：执行退款
```

流程在第 3 步暂停后，可以使用同一个 Thread 恢复。

### 7.2 Store

Store 保存跨 Thread 的长期数据，例如用户偏好、用户画像、组织配置、跨会话事实和共享知识。

可以这样区分：

```text
Checkpointer = 当前任务执行到哪一步
Store        = 用户或系统长期记住了什么
```

开发阶段可以使用内存实现；生产环境需要持久化后端，否则进程重启会丢失状态。

## 八、故障处理与副作用控制

生产级 Agent 需要对不同故障采用不同策略：

| 失败类型 | 处理策略 |
|---|---|
| 网络暂时失败 | 自动重试 |
| 限流 | 延迟后重试 |
| 工具参数错误 | 把错误反馈给模型修正 |
| 缺少用户输入 | 暂停并请求补充 |
| 权限不足 | 转人工或降级 |
| 重试耗尽 | 补偿、告警或人工介入 |
| 不可恢复错误 | 停止并保留现场 |

尤其要区分副作用：

```text
查询类操作：通常可以安全重试
写入类操作：需要幂等键
扣款类操作：必须防重复执行
发消息操作：需要去重或发送记录
```

LangGraph 应用应明确哪些节点可重试、哪些操作需要幂等、哪些失败可以补偿，以及哪些步骤必须人工确认。

## 九、和普通 ReAct Agent 的区别

| 维度 | 普通 ReAct Agent | LangGraph |
|---|---|---|
| 本质 | 一种 Agent 推理循环 | 编排框架和运行时 |
| 控制流 | 通常是 `while` 循环 | 图、条件边、动态跳转 |
| 推理方式 | LLM 主要决定工具路径 | 可由 LLM 决定，也可由代码决定 |
| 状态 | 消息列表或内存变量 | 显式 State Schema |
| 工作流 | 主要是工具调用循环 | 固定流程、分支、并行、循环、子图 |
| 人工审批 | 需要自行实现 | 原生支持 `interrupt()` |
| 进程恢复 | 通常需要自行实现 | Checkpointer 支持恢复 |
| 故障处理 | 主要写在循环中 | 节点级重试和错误分支 |
| 多 Agent | 手写 Agent 间调用 | 子图、Worker、动态分发 |
| 适用范围 | 短任务和轻量工具调用 | 生产级、有状态、长时间运行系统 |

普通 ReAct 的优点是简单、直接、上手快，适合单轮或短会话任务。LangGraph 的优点是状态、流程和生命周期显式，适合复杂业务；代价是需要设计 State、持久化、恢复和幂等性。

## 十、什么时候应该用哪一个

### 普通 ReAct 更合适

- 单轮或短会话；
- 工具数量少；
- 没有人工审批；
- 不需要中途恢复；
- 不需要复杂分支和并行；
- 失败后重新运行成本很低。

### LangGraph 更合适

- 有多步骤业务流程；
- 有明确的状态转移；
- 有多个分支和循环；
- 任务运行时间较长；
- 需要人工介入；
- 有高风险外部副作用；
- 需要重试、补偿和恢复；
- 需要并行 Worker 或多 Agent；
- 需要完整的执行轨迹和审计。

一个实用原则是：

> 能用确定性代码表达的控制流，不要全部交给模型；只有真正需要推理和不确定性判断的部分，才交给 Agent。

审批上限、权限规则、是否允许扣款和失败后是否重试，应该由代码或配置决定；搜索哪个知识库、如何组织检索结果和如何解释用户问题，才适合交给 Agent。

## 十一、企业级混合架构示例

以企业报销 Agent 为例：

```text
提交报销单
  ↓
解析票据
  ↓
字段校验
  ├── 缺少信息 → interrupt，要求员工补充
  ├── 小额合规 → 自动审批
  ├── 大额报销 → 人工审批
  └── 疑似异常 → 风险调查 Agent
                           ↓
                       调查结果
  ↓
财务系统入账
  ↓
通知员工
```

其中：

- `解析票据` 是 LLM 节点；
- `字段校验` 是确定性代码节点；
- `风险调查 Agent` 内部可以是 ReAct 循环；
- `人工审批` 使用 `interrupt()`；
- `财务系统入账` 是有副作用的 Action 节点；
- `Checkpointer` 保存当前报销流程；
- `Store` 保存员工长期偏好和组织配置；
- `RetryPolicy` 处理临时网络错误；
- 幂等键防止恢复执行时重复入账。

这个例子说明，LangGraph 的重点不是换一种方式写 Prompt，而是把 Agent 放到一套明确的业务执行边界中。

## 十二、最终判断

### LangGraph 是什么

LangGraph 是一个有状态的图执行框架、Agent 编排框架、工作流运行时和支持暂停恢复的执行层。它可以承载 ReAct、多 Agent 和混合工作流。

### LangGraph 不是什么

LangGraph 不是一个固定 Prompt，不是一种唯一的 Agent 推理方法，也不是一个只能做 ReAct 的黑盒 Agent。

### 一句话总结

> **ReAct 决定 Agent 如何“想和做”；LangGraph 决定整个 Agent 系统如何“流转、暂停、恢复、重试、分支、并行和持久化”。**

如果把普通 ReAct Agent 比作一个会自己规划路线的执行者，那么 LangGraph 更像是负责交通规则、任务状态、检查点、调度、审批和故障恢复的运行系统。