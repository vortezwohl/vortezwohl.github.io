---
layout: post
toc: true
title: "ML Agent：面向机器学习工程的 Hugging Face 代理运行时"
categories: Agent
tags: [LLM, Agent, Machine Learning, Hugging Face, MCP, LiteLLM, FastAPI, Tool Calling]
author:
  - vortezwohl
  - 吴子豪
  - Codex
---

读完 `ml-agent` 这个项目，最关键的感受是：它不是一个“把大模型接到几个 Hugging Face API 上”的简单聊天机器人，而是一个专门为机器学习工程任务设计的 agent runtime。它把用户输入、模型工具调用、人工审批、Hugging Face Jobs、Hub 仓库、文档检索、Paper 检索、MCP 工具、CLI 与 Web UI 都放进同一套异步状态机里。如果用一句话概括：**ML Agent 是一个以 LiteLLM function calling 为核心、以 Hugging Face 生态为执行平面的机器学习工程代理。**

它真正关心的问题不是“模型会不会回答”，而是：

- 模型如何可靠地研究当前 ML API，而不是靠过时记忆写代码；
- 模型如何检查数据集格式，而不是假设列名；
- 模型如何先在 sandbox 里验证训练脚本，再提交长时间 GPU job；
- 模型如何在需要花钱、写仓库、删分支、提交 job 时停下来等人审批；
- CLI、Web UI 和后端服务如何复用同一个 agent loop；
- MCP server 的工具如何被动态接入，并和内置工具一样提供给模型。

这类项目的核心价值不在某个单独工具，而在“把不可信的 LLM 输出包进一个可控 Harness 的机器学习工程流程”。

## ML Agent 在解决什么问题？

### 机器学习工程任务不是一次 LLM 回答

普通聊天机器人可以把用户问题发给模型，然后把回答显示出来。但机器学习工程任务通常不是这样。

比如用户说：

> fine-tune llama on my dataset

这句话背后至少包含这些步骤：

1. 确认用户到底要做 SFT、DPO、GRPO 还是别的训练；
2. 找当前版本 TRL / Transformers / PEFT 的正确示例；
3. 检查数据集字段是否匹配训练方法；
4. 选择模型、tokenizer、硬件、batch size、max length、timeout；
5. 写训练脚本；
6. 在 sandbox 或小样本上验证脚本；
7. 提交 Hugging Face Jobs；
8. 监控日志；
9. 确保训练产物 push 到 Hub，因为 job 文件系统是临时的；
10. 如果失败，读日志、定位错误、修复，而不是盲目重试。

这不是一次 completion 能稳定完成的工作。它需要一个 agent loop：

```text
LLM 思考
  -> 调工具
  -> 工具返回结构化结果
  -> 写入上下文
  -> 再调 LLM
  -> 继续调工具
  -> 直到没有工具调用，或者需要人工审批
```

ML Agent 的主循环正是这个形态。

### 模型的 ML 知识会过时

项目的 system prompt 写得很直接：模型对 TRL、Transformers、PEFT、Trackio 等 Hugging Face 库的内部知识可能是过时的。它会写错 import、写错 trainer 参数、假设错误的数据集 schema，或者忘记训练产物需要 push 到 Hub。

所以 ML Agent 的默认策略不是“让模型凭记忆写代码”，而是把研究工作前置：

- 用 `research` 子代理查论文、读方法章节、找训练 recipe；
- 用 `github_find_examples` 和 `github_read_file` 找真实可运行示例；
- 用 `explore_hf_docs` 和 `fetch_hf_docs` 查当前文档；
- 用 `hf_inspect_dataset` 检查数据集 schema 和样本；
- 用 `hf_papers` 做论文搜索、citation graph、paper details、section reading。

这体现了一个很重要的 agent 工程原则：**不要相信模型“知道当前 API”，要让它查证。**

### 训练 job 有成本、有风险、有状态

机器学习工程里的工具调用不是普通函数调用。提交 GPU job 会花钱；上传文件会覆盖仓库内容；删除分支、合并 PR、创建 repo 都有不可逆影响。

因此 ML Agent 的工具执行分为两类：

- 可自动执行的工具，例如文档查询、数据集检查、仓库读取；
- 需要审批的工具，例如 `hf_jobs` 的 run / uv，`sandbox_create`，`hf_repo_files` 的 upload/delete，`hf_repo_git` 的 delete/merge/create/update。

审批逻辑集中在 `_needs_approval()`，这使得“模型想做什么”和“系统允许它直接做什么”之间有一层明确边界。

这点和 Claude Code 里“不要信任 LLM，工具执行前必须过 harness”非常像。区别是 ML Agent 的风险模型更偏机器学习平台：GPU job、Hub repo、dataset/model artifact、sandbox、OAuth token。

## 整体架构

ML Agent 的架构可以压缩成这张图：

```text
CLI / Web UI
    |
    | user_input / exec_approval / compact / undo / shutdown
    v
submission_queue
    |
    v
Session + Agent Loop
    |
    | messages + tool specs
    v
LiteLLM acompletion()
    |
    | tool_calls[]
    v
ToolRouter
    |
    | built-in tools / MCP tools / sandbox / HF Jobs / docs / papers / GitHub
    v
tool results
    |
    v
ContextManager
    |
    v
event_queue -> CLI renderer / FastAPI SSE -> React UI
```

源码里最核心的几个文件是：

- `agent/main.py`：CLI 入口，负责交互输入、headless 模式、事件渲染、审批提示；
- `agent/core/agent_loop.py`：真正的 agent loop，负责调用 LLM、解析工具调用、审批、并发执行工具、继续循环；
- `agent/core/session.py`：会话状态，包括 event queue、context manager、pending approval、取消信号、session 轨迹保存；
- `agent/context_manager/manager.py`：上下文管理、system prompt 渲染、悬空 tool call 修补、undo、compact；
- `agent/core/tools.py`：工具注册、MCP client、工具 schema 转 OpenAI function calling 格式、工具执行路由；
- `backend/session_manager.py`：Web 多会话管理；
- `backend/routes/agent.py`：REST + SSE API；
- `frontend/src/lib/sse-chat-transport.ts`：前端把 AI SDK 消息转成后端 SSE 请求。

这个结构的关键点是：**CLI 和 Web UI 都不是核心，它们只是同一个 agent loop 的不同外壳。**

## CLI 接口如何使用

项目通过 `pyproject.toml` 暴露命令：

```toml
[project.scripts]
ml-agent = "agent.main:cli"
```

安装方式：

```bash
git clone git@github.com:huggingface/ml-agent.git
cd ml-agent
uv sync
uv tool install -e .
```

环境变量通常需要：

```bash
ANTHROPIC_API_KEY=<your-anthropic-api-key>
HF_TOKEN=<your-hugging-face-token>
GITHUB_TOKEN=<github-personal-access-token>
INFERENCE_TOKEN=<huggingface-router-token>
```

其中有一个容易混淆的点：

- `HF_TOKEN` 主要用于 Hugging Face Hub、OAuth、MCP Authorization、Jobs、Repo、Sandbox；
- `INFERENCE_TOKEN` 在源码里用于 Hugging Face Router 模型的 LiteLLM 调用。

### 交互模式

直接运行：

```bash
ml-agent
```

交互模式会：

1. 创建 `PromptSession`；
2. 获取或提示输入 HF token；
3. 加载 `configs/main_agent_config.json`；
4. 创建 `ToolRouter(config.mcpServers, hf_token=hf_token, local_mode=True)`；
5. 启动 `submission_loop()`；
6. 启动 CLI event listener；
7. 用户每输入一条消息，就封装成 `Operation(USER_INPUT)` 放进 `submission_queue`。

交互模式下可用命令：

```text
/help            Show this help
/undo            Undo last turn
/compact         Compact context window
/model [id]      Show available models or switch
/yolo            Toggle auto-approve mode
/status          Current model & turn count
/quit            Exit
```

`/yolo` 很危险，但对自动化有用。它会把 `config.yolo_mode` 切到 `True`，让 `_needs_approval()` 直接返回 false，后续工具调用不再等待人工确认。

### Headless 模式

传入 positional prompt 就进入 headless：

```bash
ml-agent "fine-tune llama on my dataset"
```

常用参数：

```bash
ml-agent --model anthropic/claude-opus-4-6 "your prompt"
ml-agent --max-iterations 100 "your prompt"
ml-agent --no-stream "your prompt"
```

headless 模式里源码会自动设置：

```python
config.yolo_mode = True
```

也就是说，headless 默认自动审批。这符合无人值守运行的需求，但也意味着它适合在受控环境中使用，不适合直接给高风险权限。

### CLI 与本地工具

CLI 创建 `ToolRouter` 时传入 `local_mode=True`。这会把默认 sandbox 工具替换成本地工具：

- `bash`
- `read`
- `write`
- `edit`

这些工具直接操作用户机器文件系统，而不是远程 sandbox。`read/write/edit` 还带了简单的防护：已有文件必须先 read，才能 write 或 edit。

这说明 ML Agent 的 CLI 模式更像一个本地编码/实验代理，而 Web 模式更倾向在 Hugging Face Space / remote sandbox 里运行。

## Web 接口如何使用

后端是 FastAPI，入口是：

```python
app = FastAPI(...)
app.include_router(agent_router)
app.include_router(auth_router)
```

开发时前端 Vite 代理：

```ts
server: {
  port: 5173,
  proxy: {
    '/api': {
      target: 'http://localhost:7860',
      changeOrigin: true,
      ws: true,
    },
    '/auth': {
      target: 'http://localhost:7860',
      changeOrigin: true,
    },
  },
}
```

后端启动：

```bash
cd backend
uvicorn main:app --host 0.0.0.0 --port 7860
```

前端启动：

```bash
cd frontend
npm install
npm run dev
```

### 会话创建

Web UI 先调用：

```http
POST /api/session
```

后端会：

1. 从 Authorization header 或 cookie 提取 HF token；
2. 创建 `submission_queue` 和 `event_queue`；
3. 创建 `ToolRouter` 和 `Session`；
4. 启动该 session 的后台 agent loop；
5. 返回 `session_id`。

`SessionManager` 支持多用户多会话，并有容量限制：

```python
MAX_SESSIONS = 50
MAX_SESSIONS_PER_USER = 10
```

### 提交消息与 SSE

Web 端最重要的接口是：

```http
POST /api/chat/{session_id}
Accept: text/event-stream
```

body 可以是普通消息：

```json
{
  "text": "train a classifier on this dataset"
}
```

也可以是审批结果：

```json
{
  "approvals": [
    {
      "tool_call_id": "call_xxx",
      "approved": true,
      "feedback": null,
      "edited_script": null
    }
  ]
}
```

这个接口的设计很实用：一次请求既提交操作，又返回该 turn 的 SSE 事件流。事件包括：

- `processing`
- `assistant_chunk`
- `assistant_message`
- `assistant_stream_end`
- `tool_call`
- `tool_output`
- `tool_log`
- `tool_state_change`
- `approval_required`
- `turn_complete`
- `error`
- `interrupted`
- `shutdown`

当收到 `turn_complete`、`approval_required`、`error`、`interrupted`、`shutdown` 这类 terminal event 时，SSE 结束。

前端 `sse-chat-transport.ts` 负责判断当前发送的是普通 user message，还是 AI SDK 工具审批 continuation。如果最后一条 assistant message 里有 `approval-responded` 状态的 dynamic-tool part，就组装成 `{ approvals }` 发给后端。

## MCP 接口如何使用

ML Agent 不是 MCP server，而是 MCP client。它把外部 MCP server 的工具动态加载进 `ToolRouter`，再以 OpenAI function calling 的形式提供给 LLM。

默认配置：

```json
{
  "mcpServers": {
    "hf-mcp-server": {
      "transport": "http",
      "url": "https://huggingface.co/mcp?login"
    }
  }
}
```

配置模型来自 FastMCP：

- `RemoteMCPServer`
- `StdioMCPServer`

HTTP / SSE / streamable-http server 可以这样写：

```json
{
  "mcpServers": {
    "my-remote-server": {
      "transport": "http",
      "url": "https://example.com/mcp",
      "headers": {
        "Authorization": "Bearer ${MY_TOKEN}"
      }
    }
  }
}
```

本地 stdio server 可以这样写：

```json
{
  "mcpServers": {
    "my-local-server": {
      "transport": "stdio",
      "command": "uvx",
      "args": ["my-mcp-server"],
      "env": {
        "TOKEN": "${MY_TOKEN}"
      },
      "cwd": "."
    }
  }
}
```

`load_config()` 会先加载项目根目录 `.env`，再加载当前工作目录 `.env`，然后递归替换 `${VAR}` 或 `${VAR:-default}`。

### MCP 工具如何进入 Agent

`ToolRouter` 初始化时：

```python
self.mcp_client = Client({"mcpServers": mcp_servers_payload})
```

如果有 `hf_token`，会自动给 MCP server 加：

```python
Authorization: Bearer <hf_token>
```

进入 async context 时：

```python
await self.mcp_client.__aenter__()
await self.mcp_client.initialize()
await self.register_mcp_tools()
```

`register_mcp_tools()` 调 `list_tools()`，把返回的 MCP tool 注册成：

```python
ToolSpec(
    name=tool.name,
    description=tool.description,
    parameters=tool.inputSchema,
    handler=None,
)
```

注意 `handler=None`。这表示该工具不是 Python 内置 handler，而是执行时走：

```python
result = await self.mcp_client.call_tool(tool_name, arguments)
```

MCP 返回值可能是：

- `TextContent`
- `ImageContent`
- `EmbeddedResource`

当前实现会把它们转成字符串塞回 LLM 上下文。图片和二进制资源还只是占位描述，没有真正多模态处理。

### MCP 工具屏蔽

源码里有一个禁用列表：

```python
NOT_ALLOWED_TOOL_NAMES = [
    "hf_jobs",
    "hf_doc_search",
    "hf_doc_fetch",
    "hf_whoami",
]
```

这说明项目有意让某些高价值或高风险工具使用本地内置实现，而不是让远程 MCP 工具覆盖。

这个设计很合理。MCP 扩展能力很强，但核心安全边界和平台关键操作最好留在本地 runtime 里。

## Agent 核心循环

核心函数是：

```python
Handlers.run_agent(session, text)
```

它的结构大致如下：

```text
add user message
emit processing

while iteration < max_iterations:
    compact if needed
    check doom loop
    messages = context_manager.get_messages()
    tools = tool_router.get_tool_specs_for_llm()
    response = litellm.acompletion(messages, tools, tool_choice="auto")

    if no tool calls:
        add assistant message
        emit turn_complete
        break

    parse tool calls
    add assistant message with tool_calls
    split tools into approval_required and non_approval

    execute non_approval tools concurrently
    add tool results to context
    emit tool_output

    if approval_required:
        emit approval_required
        session.pending_approval = tool_calls
        return
```

这个 loop 的重点不是“调用模型”，而是处理模型调用工具之后的一切边界问题。

### Streaming tool call 累积

流式模式下，LLM 返回的 tool call 不是一次性完整 JSON，而是 delta。源码用 `tool_calls_acc` 按 index 累积：

```python
tool_calls_acc[idx]["function"]["name"] += tc_delta.function.name
tool_calls_acc[idx]["function"]["arguments"] += tc_delta.function.arguments
```

流结束后再构造 `ChatCompletionMessageToolCall`。

这是一类 agent runtime 很容易写错的地方：如果只处理文本 streaming，不处理 tool call delta，就会在工具调用场景下丢状态。

### 工具调用参数校验

模型会把 JSON 参数写坏。ML Agent 做了两层处理：

1. `json.loads(tc.function.arguments)` 失败时，把该工具调用变成失败 tool result，告诉模型“参数坏了，请重试”；
2. `_validate_tool_args()` 专门检查一些工具的 `args` 字段是不是对象，避免模型把对象写成字符串。

这和 Claude Code 源码里“模型不擅长生成合法工具参数，所以必须验证”的思想一致。

### 审批机制

`_needs_approval()` 是权限边界。

需要审批的典型情况：

- `sandbox_create`：创建远程 sandbox；
- `hf_jobs` 的 `run` / `uv` / scheduled run；
- CPU job 默认也要确认，除非 `confirm_cpu_jobs=false`；
- `hf_private_repos` 的 `create_repo`；
- `hf_repo_files` 的 `upload` / `delete`；
- `hf_repo_git` 的 `delete_branch` / `delete_tag` / `merge_pr` / `create_repo` / `update_repo`。

如果 `config.yolo_mode=True`，则全部跳过审批。

审批时，agent loop 会先把需要审批的工具存在：

```python
session.pending_approval = {
    "tool_calls": [...]
}
```

然后返回，等待下一次 `EXEC_APPROVAL` operation。

审批恢复时，`exec_approval()` 会：

1. 根据 `tool_call_id` 建立 approval map；
2. 分出 approved tasks 和 rejected tasks；
3. approved 工具并发执行；
4. rejected 工具也写入一个 tool result；
5. 最后调用 `run_agent(session, "")`，让模型基于工具结果继续推理。

这个细节很重要：拒绝也必须写 tool result。否则上下文会出现 assistant tool_call 没有对应 tool message 的非法状态。

### 悬空 tool call 修补

`ContextManager.get_messages()` 里会调用 `_patch_dangling_tool_calls()`。

它会扫描最近 assistant message，如果发现有 tool_call 没有对应 tool result，就补一条：

```text
Tool was not executed (interrupted or error).
```

这也是典型 harness 工程。不是只有成功路径重要；中断、失败、异常退出时，message history 也必须保持 API 合法。

### Doom-loop 检测

`doom_loop.py` 做了简单但有用的重复工具调用检测：

- 连续 3 次相同工具 + 相同参数；
- 最近工具调用出现重复序列，例如 A/B/A/B。

检测到后，会向上下文注入系统提示，要求模型停止重复，换策略。

这不是高级 ML 算法，但对 agent 实际运行很重要。LLM 很容易在“查不到结果 -> 再查一次同样参数 -> 还是查不到”的循环里浪费 token。

### Context compaction

`ContextManager` 保存完整 messages，默认 system prompt 是 `system_prompt_v3.yaml`。当 `context_length > max_context` 时，`compact()` 会：

1. 保留 system message；
2. 保留第一条 user message；
3. 保留最近若干条消息；
4. 把中间历史交给 LLM 总结；
5. 用 summary 替换旧历史。

这里的实现比较朴素，但方向正确：agent 任务会很长，尤其是 ML 训练、调试、查论文、读日志，如果没有压缩机制，很容易撞 context window。

## ToolRouter：把所有能力统一成工具

`ToolRouter` 是这个项目的工具系统中心。它做了三件事：

1. 注册内置工具；
2. 注册 MCP 工具；
3. 把所有工具转换成 LLM 可用的 OpenAI tool schema。

内置工具大致包括：

- `research`：研究子代理；
- `explore_hf_docs` / `fetch_hf_docs` / `find_hf_api`：Hugging Face 文档和 API；
- `hf_papers`：论文、citation graph、paper details、section reading、snippet search；
- `hf_inspect_dataset`：数据集 schema 和样本检查；
- `plan_tool`：计划管理；
- `hf_jobs`：Hugging Face Jobs；
- `hf_repo_files`：Hub 文件 list/read/upload/delete；
- `hf_repo_git`：Hub 分支、tag、PR、repo 管理；
- `github_find_examples` / `github_list_repos` / `github_read_file`：GitHub 示例查找和读取；
- `bash/read/write/edit` 或 `sandbox_create/bash/read/write/edit`。

最终给 LLM 的 schema 是：

```python
{
    "type": "function",
    "function": {
        "name": tool.name,
        "description": tool.description,
        "parameters": tool.parameters,
    },
}
```

这使得内置工具和 MCP 工具在模型眼里没有差异。差异只存在于 `ToolRouter.call_tool()`：

- 有 handler：调用 Python handler；
- 没有 handler 且 MCP initialized：调用 MCP client；
- 否则返回 `MCP client not initialized`。

## 背后的 ML 技术栈

### LiteLLM：模型调用抽象

项目用 LiteLLM 的 `acompletion()` 统一调用不同模型：

- Anthropic 模型：例如 `anthropic/claude-opus-4-6`；
- Hugging Face Router 模型：例如 `huggingface/fireworks-ai/MiniMaxAI/MiniMax-M2.5`、`huggingface/novita/moonshotai/kimi-k2.5`、`huggingface/novita/zai-org/glm-5`。

对 Hugging Face Router 模型，源码没有直接使用 LiteLLM 默认的 `huggingface/` endpoint，而是改写成 OpenAI-compatible endpoint：

```text
huggingface/<router_provider>/<org>/<model>
    -> model = openai/<org>/<model>
    -> api_base = https://router.huggingface.co/<router_provider>/v3/openai
```

这样可以避开旧的 `api-inference.huggingface.co` 路径。

### Hugging Face Hub / Jobs / Spaces

ML Agent 的执行平面主要在 Hugging Face：

- `huggingface-hub` 用于 repo、job、Space、token、whoami；
- `datasets` 用于数据集检查；
- HF Jobs 用于训练、推理、数据处理、scheduled jobs；
- HF Spaces 用于 sandbox；
- Hub repo 工具用于保存训练脚本、模型、数据集、日志。

`hf_jobs` 工具描述里强制写入了不少 ML 工程经验：

- 训练前必须找当前示例；
- 必须检查 dataset format；
- 必须设置 `push_to_hub=True` 和 `hub_model_id`；
- job storage 是临时的，结束后文件会丢；
- 训练任务 timeout 不应使用默认 30 分钟；
- batch / ablation jobs 必须先跑一个，确认成功后再批量提交；
- OOM 时先调 batch size、gradient accumulation、gradient checkpointing 或升级 GPU，不要擅自把 full SFT 改成 LoRA。

这不是普通工具说明，而是把 ML 任务的工程规范写进工具接口。

### Research sub-agent

`research` 工具是一个独立上下文的子代理。它只拿一组 read-only / research 工具，例如：

- `read`
- `bash`
- `explore_hf_docs`
- `fetch_hf_docs`
- `find_hf_api`
- `hf_papers`
- `github_find_examples`
- `github_read_file`
- `hf_inspect_dataset`
- `hf_repo_files`

它的目标是：不要污染主 agent 的上下文窗口，把“查资料、读论文、找代码”这些工作压缩成一份 500-1500 字的研究结论，再交给主 agent。

这和 Claude Code 里的 explorer agent 模式很像。复杂任务中，研究和执行应该分离，否则主上下文会被大量文档、论文、代码搜索结果淹没。

### 文档与论文工具

项目内置了比较重的 ML research 工具：

- docs 工具会查 Hugging Face docs、OpenAPI spec，并用 Whoosh 做本地搜索索引；
- papers 工具接入 Hugging Face / Semantic Scholar / arXiv / ar5iv，用于论文搜索、citation graph、section reading；
- GitHub 工具会找真实 repo 示例，并读取具体文件。

这套组合反映了 ML Agent 的核心假设：

**机器学习代理要想写对代码，必须从论文、文档、真实示例和数据集样本里获得事实，而不是从模型参数里“回忆”。**

### 前端技术栈

Web UI 使用：

- React 18；
- Vite；
- TypeScript；
- MUI；
- Zustand；
- `ai` / `@ai-sdk/react`；
- React Markdown；
- syntax highlighter。

前端不是简单展示文本，它还需要管理：

- 多 session；
- SSE 流；
- tool call 状态；
- approval UI；
- edited script；
- reconnect；
- code/output panel；
- pending approval 恢复。

这说明 agent UI 的复杂度主要来自“工具状态”和“审批状态”，而不是聊天气泡本身。

## 和 Claude Code 的关系

ML Agent 的代码里多处注释提到 Codex / Claude Code 类似结构，例如：

- `submission_loop` 类似 Codex 主循环；
- `ToolRouter` 类似工具 router；
- `Session` 类似 agent session；
- `research` sub-agent 受 Claude Code code-explorer pattern 启发。

但它和通用 coding agent 的关注点不同。

Claude Code 的核心问题是：

```text
如何让不可信 LLM 在本地代码仓库里安全、可控、可验证地工作？
```

ML Agent 的核心问题是：

```text
如何让不可信 LLM 在 Hugging Face 机器学习平台上研究、写代码、提交训练、监控结果，并避免浪费算力或丢失产物？
```

两者都需要 harness，但边界不同。

| 维度 | Claude Code | ML Agent |
|---|---|---|
| 主要环境 | 本地代码仓库 | Hugging Face Hub / Jobs / Spaces / datasets |
| 核心工具 | read/edit/bash/git/test | docs/papers/datasets/jobs/hub/sandbox/github |
| 风险重点 | 文件破坏、命令执行、权限、上下文合法性 | GPU job 成本、repo 写入、训练产物丢失、过时 API、数据格式错误 |
| UI 形态 | 终端编码代理 | CLI + Web chat + SSE |
| 工程重点 | workspace trust、工具权限、patch、验证 | ML workflow guardrails、审批、job lifecycle、research-first |

所以 ML Agent 可以看成一种垂直领域 agent：它借鉴 coding agent runtime 的主循环，但把工具和规则换成 ML 工程专用。

## 使用时的关键注意点

### 不要把 headless 当安全模式

headless 模式会自动 `yolo_mode=True`。这意味着提交 job、上传文件等操作可能绕过交互审批。适合自动 benchmark 或受控环境，不适合高权限 token + 未审计 prompt。

### Hugging Face Router 需要 INFERENCE_TOKEN

配置里的模型可能是：

```json
"model_name": "huggingface/novita/moonshotai/kimi-k2.5"
```

但 `_resolve_hf_router_params()` 读取的是：

```python
_INFERENCE_API_KEY = os.environ.get("INFERENCE_TOKEN")
```

如果只设置 `HF_TOKEN`，Hub 工具可能能用，但模型推理未必能用。

### MCP 工具不是越多越好

MCP 可以动态扩展工具，但 ML Agent 已经屏蔽了一些关键工具名，说明作者意识到远程工具可能覆盖本地安全实现。

增加 MCP server 时，应关注：

- 工具名是否和内置工具冲突；
- 工具 schema 是否适合 LLM 调用；
- 是否需要 Authorization header；
- 返回内容是否会过大；
- 失败时是否给出模型可理解的错误；
- 是否会绕过本地审批逻辑。

### local mode 和 sandbox mode 行为不同

CLI 下 `local_mode=True`，`bash/read/write/edit` 直接操作本机。Web/backend 下默认不是 local mode，会使用 sandbox 工具。

这意味着同一个 prompt 在 CLI 和 Web UI 下，文件路径、执行环境、权限边界都不同。

system prompt 里也专门对 local mode 注入了说明：

```text
You are running as a local CLI tool on the user's machine.
There is NO sandbox...
Working directory: ...
Do NOT use /app/ paths...
```

### README 有编码显示问题

当前仓库里的 README 和部分源码注释在 Windows PowerShell 输出中出现了乱码显示，但实际源码结构清晰。写文档或二次开发时，应优先读源码，不要只依赖 README 的架构图。

## 我理解的 ML Agent

如果把 ML Agent 当成“LLM + Hugging Face API wrapper”，就会低估它。

更准确的理解是：

**它是一个机器学习工程代理的运行时骨架。**

它的核心不是某个模型，也不是某个工具，而是这套循环：

1. 用 system prompt 把 ML 工程规范写进 agent 行为；
2. 用 ToolRouter 把本地工具、HF 工具、GitHub 工具、论文工具、MCP 工具统一成 function calling；
3. 用 ContextManager 保证消息历史、tool call 配对和上下文压缩；
4. 用 approval_required 把高风险操作挡在人类审批前；
5. 用 Session 和 event_queue 支撑 CLI / Web / SSE 多种 UI；
6. 用 research sub-agent 降低主上下文污染；
7. 用 HF Jobs / Spaces / Hub 把 agent 的行动落实到真实 ML 平台上。

它背后的设计哲学很清楚：

**模型可以规划和生成，但不能被直接信任；机器学习任务可以自动化，但必须把研究、验证、审批、日志和产物保存纳入运行时。**

这也是当前 agent 工程里最值得学习的地方。真正可用的 agent，不是让模型“自由发挥”，而是让模型在一套明确的状态机、工具边界和验证流程里工作。
