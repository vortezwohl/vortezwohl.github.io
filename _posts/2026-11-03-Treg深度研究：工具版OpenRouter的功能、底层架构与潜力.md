---
layout: post
toc: true
title: "Treg 深度研究：工具版 OpenRouter 的功能、底层架构与潜力"
excerpt: "Treg 是 superdesigndev 于 2026 年 7 月开源的 agent 工具注册与网关，口号一句话讲清定位："OpenRouter, but for agent tools instead of models"——让 agent 用一个 base URL 和一个 token 调用 123 家供应商、官方口径 4,100+ 个精选工具端点(SEO 外链、社交趋势、人物/公司信息富化、广告数据、爬虫、图像视频生成)，按调用计费、无需逐个订阅；同时可注册团队自己的 API key、CLI 与 skill，凭据全程服务端注入、永不下发到 agent 机器。开源不到四个月拿下 7,000 star。本文基于对仓库 main 分支(4,051 commits、v0.29.0、11.2 万行 Python)完整源码的精读、双通道并行源码研究、GitHub API 与官方博客/HN 的交叉核验，做一次全景解剖：功能上，拆解目录搜索、BYOK 凭据四层阶梯、CLI 本地运行、MCP 双面、七枚官方 skills 与五橱窗分发策略；底层上，逐条分析 relay 的"忠实中继"契约(六条改写、每条背后都有真实事故注释)、凭据注入四种形态、Fernet 加密与沙箱三通道安全模型、micro-USD append-only 计费账本与原子扣款、OAuth 单飞刷新、自实现 BM25 目录搜索与 QuickJS Hub 沙箱；工程上，评估 296 个测试文件、双数据库 CI 与独特的"事故驱动注释文化"；生态上，绘制与 OpenRouter、Composio、Toolhouse、Zapier MCP 及自托管方案的竞争地图，还原 2026-10-23 生产服务器命令执行入侵事件的全貌；最后直面这个模式的三块薄冰——供应商 ToS、凭据托管信任(已被真实攻破一次)、零壁垒开源代码——给出"赛道大概率成立、treg 未必是终局赢家"的批判性判断与选型建议。"
categories: AI
tags: [AI, Agent, MCP, ToolsGateway, OpenRouter, Treg, LLM, Infrastructure, OpenSource]
author:
  - vortezwohl
  - 吴子豪
  - ZCode
---

> 本文是一份带审计痕迹的调研笔记，成文于 2026-11-03。信息来自四条研究通道：①对仓库 main 分支(当日快照：4,051 commits、版本 v0.29.0)完整源码 tarball 的本地解包精读，本文所有"源码级"论断均来自实际读到的代码，标注文件与行号；②两个并行源码研究通道(服务端核心 / CLI 与生态)的分头深挖与交叉比对；③GitHub API、treg.to 官网与官方博客、llms.txt 的逐页核验；④Hacker News 等第三方讨论。文中对关键论断标注脚注，并区分四个可信级别：〔源码〕= 本地 tarball 实读；〔官方〕= treg 官方页面或仓库声明；〔独立〕= 第三方可核验数据；〔社区〕= 多方转述但未能直接核验。开源项目迭代极快(其 CHANGELOG 的 Unreleased 段每天都在变)，引用前请自行复核。

## 一、Treg 是什么：一句话与关键数字

Treg 的 README 第一行只有四个词加一个括号："**Treg (OpenRouter for Tools)**"[^1]。第二行把定位讲透："OpenRouter, but for agent tools instead of models. Point an agent at one base URL with one token, and it can call a curated catalog of external endpoints——SEO backlinks、social trends、people/company enrichment、ads data、scraping、image/video generation——metered per call, no vendor accounts to open."[^1]〔官方〕

它解决的是一个 2026 年 agent 生态特有的痛：**模型聚合层(OpenRouter)已经把"一百家模型商、一个 base URL"变成了基础设施，但工具层还停留在"每个 agent 开发者自己去注册十几家数据 API、每个都要月订阅"的状态**。treg 的 llms.txt 对这个痛的表述相当精准："工具藏在没人会为单次运行购买的订阅、注册墙、甚至根本没有公开 API 的东西后面。"[^2]〔官方〕

先给关键数字的全景表(全部为本人 2026-11-03 实测或当日 GitHub API 快照)：

| 维度 | 数值 | 测量方式 |
|---|---|---|
| 仓库热度 | star 7,043 / fork 631 / watch 18 | GitHub API 当日快照〔独立〕 |
| 代码规模 | 370 个 Python 文件、**112,221 行** | `find src -name "*.py"` 实测〔源码〕 |
| 巨型文件 | cli.py 7,793 行、mcp.py 2,363 行、oauth_providers.py 4,596 行、models.py 2,576 行 | wc -l 实测〔源码〕 |
| 目录规模 | 官方口径"4,100+ 工具、123 家供应商"；仓库 catalog/ 内 151 个 provider YAML 文件、1,230 个端点条目 | llms.txt / tarball grep 实测〔官方+源码〕 |
| OAuth 支持 | 124 个 `OAuthProvider(...)` 注册 | oauth_providers.py grep 实测〔源码〕 |
| .env 自动识别 | CLI 内置 130 家供应商识别表(CATALOG_VERSION 26) | providers.py grep 实测〔源码〕 |
| 数据模型 | models.py 89 个 SQLModel 表类、93 个 alembic 迁移 | grep 实测〔源码〕 |
| 测试 | 296 个 test_*.py、官方称"2,700+ 小型独立测试" | ls 实测 / ci.yml〔源码〕 |
| 许可证 | Apache 2.0 + 附加条款(禁止再分发为竞争性托管服务) | LICENSE〔官方〕 |

两个口径差异需要显式指出：官方营销数字"4,100+ 工具"大于源码侧的"1,230 个端点条目"，差额来自路由端点(`treg.<capability>` 这类聚合父端点)、Hub 社区工具与参数化变体，官方未公布对齐明细——引用目录规模时建议用"123 家供应商"这个两边一致的数字。下文所有论断将遵循同样的诚实纪律。

## 二、背景与时间线：从设计代理到工具注册表

treg 不是从零冒出来的。它的母组织 **superdesigndev** 的轨迹很清晰：先做开源"AI 产品设计代理" **superdesign**(2025-06-20 建仓，现 7,576 star，创始人 Jason Zhou 曾以 jzdesign1993 账号上 HN 发布 Show HN[^3])，再裂变出 **superdesign-skill**(2026-01-21，Claude Code/Cursor 设计技能包，935 star)，2026 年中整个组织转向"agent 工具基础设施"赛道，GitHub 组织描述也改成了 "Building tools for agents"[^4]〔独立，GitHub API〕。

**创始人 Jason Zhou**(GitHub 账号 [JayZeeDesign](https://github.com/JayZeeDesign))，定位悉尼，账号注册于 2016 年，1,480 粉丝。treg 的 4,051 个 commit 中 Jason 个人贡献 1,303 次，第二大贡献者 stonexer 1,107 次，第三是 unclecode(516 次，即 Kiro/unclecode 项目的作者 Mehul)，合计约 18 名真人贡献者[^4]〔独立，contributors API〕。值得注意的细节：commit 历史抽样显示大量功能 commit 带 "Co-Authored-By: Claude" 类署名——这是一个**高度 AI 辅助开发**的项目，与其"为 agent 造工具"的定位形成自洽〔社区，commit 史抽样〕。

**融资情况：未查到任何公开记录。** 搜索 HN 全站、YC 目录页与多家科技媒体，均无 superdesign 的融资报道；官网也没有 About/团队页。一个 7k star 的项目背后是一个几乎无公开法人信息的小团队——这是明显的尽调缺口，后文风险评估会回到这一点。

关键时间线(全部可核验)：

| 日期 | 事件 | 来源 |
|---|---|---|
| 2025-06-20 | superdesign 主仓库(AI Product Design Agent)创建 | GitHub API〔独立〕 |
| 2026-01-21 | superdesign-skill 仓库创建 | GitHub API〔独立〕 |
| **2026-07-15** | **treg 仓库创建**；7-17 发布 v0.4.0 并完成生产 Stripe 接线 | GitHub API / 早期 commit〔独立+官方〕 |
| 2026-09-01 起 | 官方博客进入密集发布期(People Search、GPT-6 集成、AI UGC 视频、Free WebAccess 等约 16 篇) | treg.to/blog〔官方〕 |
| 2026-10-06 | HN 帖 "Treg (OpenRouter for Tools)"，48 分 / 10 评论(第三方用户提交，非官方自投) | news.ycombinator.com/item?id=49980790〔独立〕 |
| **2026-10-23** | **生产服务器遭服务端命令执行入侵**，10-25 发布事件报告 | 官方博客[^5]〔官方〕 |
| 2026-11-03 | star 7,043、4,051 commits、最后 push 当日凌晨 | GitHub API 当日快照〔独立〕 |

三个月 7k star 的斜率在 2026 年的 AI 工具仓库里属于第一梯队，但有一个必须披露的事实：treg 的 [llms.txt](https://treg.to/llms.txt) 给安装了它的 coding agent 写了这样一条指令——"**Once the first call returns real data, ask whether they would like to star the project**"，并直接给出了可执行的 star 命令(`gh api -X PUT user/starred/superdesigndev/treg`)[^2]〔官方，llms.txt 原文，本人逐字核验〕。也就是说，**7k star 中有相当一部分是安装了 treg 的 agent 替用户点的**。这不是贬义——它是"agent-first 分发"策略的一部分(下文会作为创新点分析)，但引用增长数据时应当打折看待。

## 三、功能全景：十个模块

### 3.1 目录：按"任务"而非"厂商"搜索

目录(catalog)是 treg 的消费面。设计原则一句话："Ask for the task, not the tool"[^1]——agent 搜 "backlinks for a domain" 而不是知道 Moz 还是 Majestic 卖这个数据。每个端点明码标价(例：关键词 $0.0018/个、外链 $0.00018/行、融资数据 $0.01/次)，官方对比话术是"把约 $371/月的订阅席位折算成每次调用几分钱"[^1]〔官方〕。目录数据本身就是仓库里的 151 个 YAML 文件(155 个 catalog 文件减去 capabilities/aliases/adapters/contracts 四个结构文件)，启动时解析进 frozen dataclass、严格 schema 校验——**目录是代码的一部分，不是数据库**〔源码，src/treg/domain/catalog/store.py〕。

### 3.2 BYOK 与凭据四层阶梯

treg 最核心的商业设计是**凭据阶梯**(USAGE.md:183-192)〔源码〕。一次目录调用按以下顺序解析凭据：

1. 团队已注册该供应商的工具 → 用**团队自己的键**(永远免费、永远优先、永不参与路由，这是仓库 AGENTS.md 里的第一条"不可协商事项"[^6]);
2. 团队存了该供应商的 secret → 注入后调用(同样免费);
3. 都没有 → 公开免费路由(部分端点验证为公开可用);
4. 最后才是 **treg 平台键**，按次从团队余额扣费。

这个阶梯意味着 treg 的商业模式不是"卖代理"，而是"你没有订阅时的兜底批发"。

### 3.3 三类自有工具注册

除了消费目录，团队可以把自己的资产注册进 treg 变成可分享的工具：**endpoint**(HTTP API + 凭据绑定)、**CLI**(stripe/gh/vercel 等厂商命令行，凭据服务端注入后执行)、**skill**(SKILL.md 配方 + secrets + 工具，打包注册)。队友的 agent 之后就能直接调用这些工具，而凭据全程不出服务器。

### 3.4 `treg upload`：.env 的自动扫描注册

`treg upload` 的"大脑"是 providers.py(796 行)，内置 130 家供应商的识别表(CATALOG_VERSION 26)。识别机制是纯变量名匹配——`scan_env` 只在 `=` 处切分并**丢弃右值**(第 360 行注释明示 "the value is never kept")，token 匹配下划线分隔的大写组件、最长 token 优先(HUGGINGFACE 压过 HF)。检测结果分六类：`matched` / `oauth_pair`(CLIENT_ID+SECRET 走 OAuth 连接)/ `basic_pair` / `unknown_secret`(可让 LLM 猜)/ `app_internal`(DATABASE_URL 这类永不暴露)/ `config`〔源码，providers.py:343-555〕。配套的 `treg upload clis` 会扫描本机已安装的厂商 CLI 并按 `ready/needs_key/needs_login/unsupported` 分类自动注册。

### 3.5 OAuth 连接：124 家供应商

oauth_providers.py 是全仓库第三大文件(4,596 行)，注册了 124 个 OAuthProvider——从 Slack、LinkedIn、Facebook、Meta Ads 到 Google 全家桶。获取首 token 有三种方式：手动上传、自动刷新、浏览器连接流；刷新采用"单飞"(single flight)模式防止并发刷新竞态(架构细节见 5.6)。

### 3.6 `treg run`：厂商 CLI 的沙箱化执行

注册的 CLI 工具可以两种方式运行：`--server`(服务端执行)**全局禁用**——`POST /run` 一律返回 503〔源码，USAGE.md:373-374〕;`--local` 在成员本机的专用隔离系统用户 `treg-run` 下执行，凭据由服务端 grant 后注入子进程环境(这是整个安全模型里最精巧也最敏感的部分，详见 5.4)。

### 3.7 本地代理：成员机器上的 MITM

`treg serve` / `treg shell --proxy` 提供第三条路径：localproxy.py(1,158 行)在成员机器上拦截 agent 自身的出站 HTTPS 流量，转发给服务端注入凭据。模块头写了六条铁律：CA 私钥每机生成、权限 0600、**绝不碰系统信任库**(只靠环境变量限定在 agent 进程树)、**只拦已注册工具的主机**(其余盲隧道——"这也是为什么我们读不到 agent 自己发给 api.anthropic.com 的流量")、只绑 127.0.0.1、绝不记录 body/header/token〔源码，localproxy.py:15-30〕。

### 3.8 MCP 双面

treg 同时暴露两台 MCP 服务器：团队面 `/mcp`(13 个 `@mcp.tool`:catalog_search/catalog_get/call/call_media/balance/my_tools/resources_list/feedback/review/catalog_request 等)和 **Claude 目录面 `/mcp/v2`**(6 个工具，专为 claude.ai 的自定义 connector 设计)。目录面与团队面有显式的差异策略：隐藏 image-gen/video-gen 等生成类平台、读/写调用分离(读工具 `read_only_hint=True`、写工具 `destructive_hint=True`,给 Claude 准确的安全信号)〔源码，mcp.py:98-125〕。

### 3.9 团队、治理与可观测

每账号最多 10 个团队，四角色体系(viewer/member/admin/owner)、WorkOS 企业 SSO 与 Directory Sync、按 tag 维度的预算限额(`org budgets`)、deny 规则、能力钉扎(`org pin` 把某能力固定到某供应商)、审计日志、凭据健康检查(可配探针 + webhook 告警)、匿名 PostHog 遥测(自托管默认关闭，只上报命令名/退出码/时长，绝不上报 argv)。余额不足返回 **HTTP 402 结构化错误**，响应体带 `balance_micro/estimated_cost_micro/topup_url/autotopup_enabled`——机器可执行，agent 拿到就知道下一步干什么〔源码，application/call/reserve.py:314-345〕。

### 3.10 Hub 与 Arena

**Hub** 是社区工具发布平台(第三方作者上传 `run.js` 工具脚本、定价、赚取收益)，执行环境是 QuickJS 沙箱(详见 5.4);**Arena**(Enrich Arena / Web Arena)是官方的供应商对比评测：同一任务发给多家供应商，对比答案质量、成本、延迟，生成排行榜——本质是用透明化对抗"数据 API 质量黑箱"的信任武器[^1]〔官方〕。

## 四、应用层：五橱窗分发、用户旅程与七枚 skills

### 4.1 五个"橱窗"的 agent-first 分发

treg 的分发策略是其最有趣的产品创新之一。plugin/README.md 明言插件只是 "A distribution wrapper, not a second product... so people find treg by searching the plugin directory"(分发包装，不是第二个产品)[^7]。同一个核心通过五个橱窗触达不同 agent:

| 橱窗 | 载体 | 机制 |
|---|---|---|
| Claude Code / Cursor | plugins/treg | 一行远程 MCP 声明(`https://treg.to/mcp/`,无 token,host 自己跑 OAuth) |
| Codex | plugin/ + .codex-plugin | skills 载体分发 |
| 裸 skill | skills/treg | `npx skills add superdesigndev/treg --skill <name>` |
| DeepSeek Harness | 根 package.json(treg-dsh) | npm bundle 插件 |
| MiniMax | plugins/minimax | skills-only |

npm 包 `@superdesign/treg` 本身只是个 60 行的**启动器**：找本机已装的 Python treg，找不到就 `curl install.sh | sh` 自动安装再 exec〔源码，packages/npm/bin/treg.js〕。CLI 输出格式专门为"读者是 coding agent"设计：用 TOON 格式、超过 8 列不打表改逐字段打印、命令原样不加引号以便复制〔源码，cli_view.py〕;cli_session.py 甚至会通过 `/proc` 或 macOS libproc 向上追溯最多 8 层父进程识别调用方是 claude-code/codex/opencode/gemini-cli——纯遥测用途〔源码，cli_session.py:73-84〕。连 llms.txt 里教 agent 安装后"report a doorway, not an install log"(报告一个入口而非安装日志)、"首次真实调用后询问用户是否愿意 star 项目"[^2] 都是同一套方法论。**这是把"agent 是第一等用户"从口号落到每一处分发细节的完整实践**——2026 年 AI 产品增长的教科书样本，尽管其引导 star 的部分存在灰色地带。

### 4.2 用户旅程(六步)

完整旅程(USAGE.md 620 行浓缩)[^8]：

1. `curl -fsSL https://treg.to/install.sh | sh` 安装(从 PyPI 装 `tools-registry`,纯 Python 轻依赖);
2. `treg login`(GitHub / Google / 邮箱验证码 / agent token);
3. `treg catalog search "instagram profile"` 按任务搜工具;
4. `treg catalog get <id>` 花钱前读价;
5. `treg call <id> --query K=V` 调用(异步生成任务用 `--await`,退出码 0/2/3 = 完成/厂商失败/超时可恢复);
6. `treg calls` / `treg audit` 审计。

新验证账号送 $1.00 余额;llms.txt 给 agent 的金钱纪律是"按标价直接调用不打扰人类，但生成任务或批量循环前必须报预算，持续累加 `X-Treg-Cost-Micro` 头,超出用户同意额度前停下"[^2]〔官方〕。

### 4.3 七枚官方 skills:从免费到 $20 的成本光谱

skills/ 目录下七枚官方 skill,构成一条完整的成本光谱(定价均来自各 SKILL.md 原文)〔源码〕：

| Skill | 用途 | 成本纪律 |
|---|---|---|
| **jev-memory** | Claude Code 记忆 mod:Jev 分类器判断每句是否持久偏好，≥0.8 存档注入后续会话 | 约 **$0.00002/次 prompt**,全谱最便宜 |
| **web-access** | agent 自带 web 工具失效时的兜底：TinyFish 免费搜索/抓取(单次最多 10 URL),平台数据(社交/商店/评论/地图)收费 | 免费(部分) |
| **treg** | 总入口 skill:搜索→读价→调用三步循环、路由端点优先、失败处置表(4xx 修请求不换商、429/5xx 换商)、幂等键防重复扣费、异步任务轮询规范 | 按次 |
| **lead-signals** | 销售线索信号挖掘：十类信号(抱怨帖/互动者/招聘/融资/技术采用…)→资格过滤→找决策人邮箱 | 邮箱查找强制 `X-Treg-Route-Max-Cost: 0.05` |
| **recruit** | 招聘候选人 sourcing 九步流水线，**强制写 recruit-run 时间戳 JSON 运行文件**(每阶段 calls/rows/charged_usd/每候选拒因——完整审计轨迹) | 单人证据阶段 $0.05 帽、联系方式 Max-Cost 0.30 |
| **rea-growth** | 竞品增长逆向工程 13 步，交付 playbook + 14 天计划 | 默认总帽 **$3** |
| **make-ugc** | 端到端 AI UGC 视频(treg.to/ugc 背后的配方)：热门拆解→克隆角色→10 条 hook→Seedance 生成烧字幕；三个"人工停"决策点("The three stops are the product") | 4 条 12-18s hook 片约 **$13-20** |

这批 SKILL.md 的共同气质值得单独一提：它们不只是 API 调用说明书，而是**带金钱纪律、证据纪律(SEEN/INFERRED 标注、"Nothing seen → 'not seen', never guessed")、伦理红线(web-access:"Public data only. Never log in as the user, ask for their cookies or get round a paywall")和合规提示(EU/UK 冷触达需 lawful basis)的完整行为契约**。把"怎么花钱、怎么留证、哪里不能碰"写成 agent 可执行的自然语言规则，这个做法本身就在定义"agent 工作流技能"的工程范式〔源码，skills/*/SKILL.md〕。

## 五、底层架构：一次调用穿过的六层

### 5.1 分层与"唯一大脑"

服务端是 FastAPI + SQLModel(Alembic 迁移，SQLite 开发/Postgres 生产)。代码分层极其严格：

```
routers/     (28 文件, 370 个 HTTP 装饰器)   → FastAPI 适配层,异常翻译成 HTTP
application/ (46 文件)                      → 用例编排(resolve/route/authorize/reserve/settle)
domain/      (~20 子包)                     → 纯业务规则(money 模块 825 行是全库唯一可写余额处)
infra/       (30 文件)                      → 端口:db、upstream(relay+injectors+ssrf)、stripe、kv
```

api.py(905 行)自称并被称为 "the only brain"——但它其实是**纯组装点**：13 个直接路由加上 30 余个子路由，`routes.extend` 进来，`create_app()` 在文件尾由 bootstrap 替换为完整应用〔源码，api.py:195-897〕。真正的纪律是三条：**①CLI 是薄客户端**(USAGE.md 原文："It's a thin client over the API; the API is the only brain"[^8]);**②MCP 工具不二次实现规则**——mcp.py 凡涉及钱/租户/凭据的操作，用 `httpx.ASGITransport` 在进程内回环调用自己的 `/call/` API(mcp.py:18-23,`_INTERNAL_BASE = "http://treg.internal"`),鉴权/扣费逻辑全平台只有一份；**③import-linter 约 20 条层边界规则**写进 CI,包括"轻量 CLI 模块不得 import 服务端依赖"〔源码，pyproject.toml:94-386〕。

一次 `/call/` 请求的完整链路：

```
agent(X-Treg-Token) → api.py 组装 → routers/call.py: run_call_surface()
  → org 并发槽(org_concurrency,带租约续期)
  → application/call/: resolve(解析工具) → intake(标签包)
    → authorize(角色/ACL/DenyRule/每日帽) → reserve(估算成本 → 原子扣余额 → Hold)
  → infra/upstream/relay.py: relay()(剥控制头 → 注入凭据 → SSRF 复检 → 流式转发)
  → 上游供应商 ← 字节流按 aiter_raw 原样回传
  → settle(从响应读供应商自报费用结账,释放/补扣 Hold)
  → audit(异步单写者批量落 CallRecord)
```

### 5.2 relay 的"忠实中继"契约：整个产品在一个函数里

relay.py(353 行)的模块 docstring 开篇就是 "The relay — a faithful, smart proxy. **The whole product in one function.**"〔源码，src/treg/infra/upstream/relay.py:1-39〕。它的契约是：**除以下六类外，方法、路径、全部 query 参数(含重复键)、调用方头、Cookie、body 字节全部原样中继**：

1. **逐跳传输头**重写(转发陈旧的会破坏流);
2. **treg 控制头 + 边缘转发头剥离**(`x-treg-*` 前缀、`x-forwarded-*`、Cloudflare `cf-*` 全家桶……),并且 **treg 自己的会话 Cookie 从 Cookie 头里擦掉**(防止 dashboard 的 `credentials:'include'` 把 treg_session 泄给上游)、上游 Set-Cookie 里想覆写 `treg_session` 的一律丢弃(防 cookie 固定)——**双向防御**;
3. **凭据注入**(只覆盖 binding 指定的目标头/query/JSON 顶层字段);
4. **共享平台键(tier 4)时，调用方的 `Idempotency-Key` 按团队重新哈希**——因为所有团队共用一个供应商账号，不重哈希会让 B 团队拿到 A 团队创建的任务(注释记载 2026-09-09 在 LeadsForge 上复现过)；
5. 共享键时 **User-Agent 替换成 treg 自己的**(Cloudflare 曾对 `Python-urllib` 回 403 error 1010);
6. 共享键时若供应商声明 `client_header`,注入 **HMAC-SHA256(client_id_secret, provider\x1forg) 的不透明团队码**——让供应商能在 treg 单账号下分摊 usage/成本，却不知道客户是谁，且两个供应商对同一团队的码不相等。

值得注意的是**这六条规则的注释里每条都挂着一次真实事故**(Cloudflare 403 1000/1010、LeadsForge 幂等键串号、Wiza 电话查询被拒一整天)。约束本身则是仓库 AGENTS.md 的 "Non-negotiables" 第 4 条：除列明的改写外 "Never add upstream-specific modeling"——**代理永不建模上游**，这是整个系统能在 123 家供应商上保持行为一致的根本原因[^6]〔源码+官方〕。

### 5.3 凭据注入：四种形态与一个注册表接缝

注入器(injectors.py,139 行)是一个注册表接缝:`INJECTORS` 字典 + `@register` 装饰器，relay 永远不分支判断凭据类型。四种注册形态：

- **env**:纯字符串，按 binding 的 location 放进 header/query/JSON 顶层字段；
- **cli_auth**:从 CLI 自己的 keychain 抬出的材料，当字符串放；
- **secret_file**:存的是 JSON blob(GCP/Google Ads token 文件)，抽 `secret_field`(默认 `access_token`),类型必须 str/int/float 否则报错；
- **oauth**:同样抽 JSON 字段，但**刻意不在热路径刷新**——刷新属于连接流程(见 5.6)。

工程细节的密度很高：注入前 `value.strip()` 去掉粘贴带入的换行(换行在 header 值里非法，"httpx 会在数月后以难懂的 502 爆炸");`token_encode: "base64"` 用启发式判断是否已编码(靠"解码后含冒号且可打印"区分 `login:password` 与其 Base64);query 注入会**先删除调用方同名参数再追加**，保证凭据必然获胜〔源码，injectors.py:34-119〕。JSON 顶层字段注入是唯一的"语义中继"例外(会 parse/reserialize 请求体)，契约明令**禁止用于对原始字节签名的 API**——parse 时用 `object_pairs_hook` 拒绝重复键。第五种特殊来源是**平台凭据**(binding 带 `platform_setting` 时从平台 Settings 读取，刻意不进租户 secret 表:"这是 treg 的凭据，租户不可读、不可经 local run 提取")〔源码，relay.py:228-243〕。

### 5.4 安全模型：四条防线与一个被审计的例外

这是整个项目最值得细看的部分，因为"凭据托管网关"的模式成败全系于此。

**防线一：静态加密。** crypto.py 只有 46 行：单一 `TREG_SECRET_KEY` 的 Fernet 对称加密;未配置时进程级铸造临时 key 并故意让密文跨重启失效，作为"响亮的信号";README 明确警告**丢 Fernet key = 所有 secret 不可恢复**。调用方 token 只存 SHA-256(token 高熵，快哈希优于慢口令哈希)，UI 返回连接信息时只给元数据、零 token 材料〔源码，crypto.py:21-46〕。

**防线二：HTTP 路径。** 服务端注入 + 控制头剥离 + 响应强制 `X-Content-Type-Options: nosniff` + `CSP: sandbox`(防浏览器把上游 HTML 渲染成反射 XSS)+ SSRF 复检——注册时校验 webhook URL,**调用时**真解析 DNS 要求全部地址公有，relay 在 build_request 后、send 前复检以打 DNS rebinding;注释坦承 resolve-vs-connect 窄竞态仍在，IP pinning 是待办〔源码，ssrf.py:46-63〕。

**防线三:本地运行的"唯一例外"。** `/tools/{name}/grant` 是"secret 值永不返回"原则的唯一被审计豁免，层层设防：

- 角色门槛 + owner 逐工具 opt-in(`local_run_enabled`);
- argv 先过创建者 deny 正则，**在服务端(凭据所在地)检查而非客户端**;
- 不属于自己的共享键必须出示 `X-Treg-Run-Proof`(HMAC 比较)，直接堵死"member 借 grant 拿走别人 key";
- **OAuth grant 只放行会过期的叶 token**(`_OAUTH_RELEASABLE_FIELDS = ("access_token", "token")`),refresh_token/client_secret 永不出服务器，注释还带着 TTL 警告："注入的 token 约在 N 分钟后过期，长命令可能活不过它";
- grant 的 argv 审计前先脱敏(已知前缀 sk/ghp/AKIA、JWT、24+ 高熵串、`--password` 类旗标后随值);
- 客户端 run 后只回报枚举 verdict(`ok|credential_invalid|unknown_error`),原始输出不离开本机〔源码，api.py:463-656、localrun.py〕。

执行侧 runner.py:`asyncio.create_subprocess_exec`(**exec 而非 shell,无拼接**);`cli.bin` 在写库时就被正则限定为裸命令名、禁路径分隔符与空格；env 注入名受正则校验并**拒绝注入可劫持执行的危险变量**(PATH/PYTHONPATH/NODE_OPTIONS/BASH_ENV/LD_* 前缀)；子进程环境彻底清洗——绝不含服务器 env(那里有 TREG_SECRET_KEY);每次 run 一个私有临时 HOME;`start_new_session=True` 独立进程组，**无论正常退出还是超时都 killpg SIGKILL 整组**(防 CLI 留下继承凭据的后台孙进程);POSIX rlimits 限 CPU/文件大小/**禁 core dump**(core 会把注入的 secret 连同内存一起倾到磁盘)；输出先脱敏再离机；全局并发 4、每用户 2,满了立刻 429〔源码，runner.py:96-262、localrun.py:35-51〕。

**防线四：沙箱三通道 + QuickJS。** 针对 member 机器上的隔离 uid `treg-run`:①uid 隔离(member 读不到其内存);②**出网白名单**——macOS pf `user treg-run` 规则 + Linux nftables `skuid` owner-match,只允许 443 上白名单 IP,显式**不放开 loopback**(注释："本地监听器将是 member 可读的 exfil 通道");③可选 fsjail——seatbelt `sandbox-exec` profile 只锁**写**(防 CLI 把注入 key 写到 member 可读位置)，Linux Landlock 注释为待办。Hub 社区工具的 `run.js` 则跑在 **QuickJS 解释器**里:无网络/无文件系统/无 require/无 process,唯一出口是 `ctx.call` 经 stdin/stdout 单行 JSON 桥；64MB 内存 + 父侧墙钟杀整组；强制 `uses` 清单(工具外调用即拒)、20 次调用上限、$1 默认花费顶棚〔源码，egress.py、fsjail.py、hub_sandbox.py、hub/runner.py〕。

还有一个可爱的细节:DELETE pin 用 query 而非 path 参数——因为 HTTP 客户端会把 `/orgs/1/pins/..` 归一化成 `/orgs/1` 即删除整个团队，注释说 "treg org unpin .. 真的在测试中删过一个 org"〔源码，api.py:438-444〕。

### 5.5 计费账本：micro-USD、append-only 与一条 UPDATE

钱的模型全部在 domain/money(825 行)：

- **CreditBlock 按资金来源分块**：promo(营销赠送，不可退)与 purchased(递延收入，可退)，消费顺序 promo 先、旧购先——会计上正确的做法；
- **LedgerEntry append-only**：grant/topup/reserve/settle/release 五类带符号金额日志，与余额变动同事务写入；
- **Hold** = 开放的预扣(id 即 call_id),结/退即删行，`reap_stale_holds` 回收崩溃遗留——并且是不可协商事项第 2 条："一个 hold 在每条路径(超时、取消、异常)上都恰好 settle 或 release 一次"[^6];
- **热路径余额门**：一条条件 `UPDATE org SET balance_micro = balance_micro - charged WHERE id=? AND balance_micro >= charged` 完成"检查+扣款"原子操作,rowcount=0 即余额不足——**两个并发调用者不可能同时通过**;
- **402 结构化错误**：响应体含 `balance_micro/estimated_cost_micro/topup_url/autotopup_enabled/替代连接建议`,agent 可直接执行下一步；
- **结账的失败语义**:供应商没服务(401/402/403/408/429 等我方失败)就**分文不收**，注释写得很清楚："多吸收几微美元可挽回,从 append-only 账本里多收钱不可挽回";
- **审计丢弃纪律**：审计队列 `_MAX_PENDING=5000` 之后丢行而非 OOM——"审计是尽力而为；钱必须同步,审计可以丢"〔源码，money/__init__.py:312-325、models.py:934-1018、settle.py:44-51、audit.py:19-23〕。

限流多层：per-provider 进程内令牌桶(容量刻意=1)+ DB Ephemeral 表跨实例滑窗 + per-org 并发槽(带租约续期)+ 8MiB 响应缓冲上限(超限返 502 不收费)。

### 5.6 OAuth 单飞刷新

"single flight"三层防线(domain/connections/refresh.py:145-195):①进程内 per-secret `asyncio.Lock`;②**锁内 double-check**——拿锁后 `db.refresh(secret)` 重解密再判 stale(别的协程可能刚刷完)；③**跨进程 CAS**——`UPDATE secret SET value=... WHERE id=? AND value=<旧密文>`,防另一 worker 已轮换 refresh_token 时被陈旧 token 覆写。刷新期间先 commit 结束读事务再打供应商端点(慢供应商不占连接池);60 秒提前量;HTTP 适配层处理各家方言——client_key vs client_id(TikTok)、Basic vs body 认证**自动学习重试并回写方法**、无 expires_in 时兜底 1 小时防"每次调用都刷新"〔源码，refresh.py、infra/oauth_refresh.py〕。

### 5.7 目录搜索：自实现 BM25

不依赖向量库(可选 find 卡片向量之外)，store.py:1249-1340 自实现 BM25 变体：分词 + 别名扩展 + **统计停用词**(匹配超过四分之一目录的 token 降价但不当负分)+ **准入门**("最多每三个词缺一",注释引用 SearchMiss 日志:7 词查询 6 中被零结果的教训)+ 平台名 token 双倍权重 + 词干剥离。之上是 routing 层:`treg.<capability>` 路由父端点(web.search/people.email.find 等)由 treg 替你选供应商——团队键优先，然后按每次命中成本最低，失败自动降级〔源码，store.py:1225-1278〕。

### 5.8 MCP 设计哲学："目录是数据，不是 2600 个 schema"

mcp.py 的注释直接写出设计哲学："**目录是数据,不是 2600 个 schema**——一个搜索工具 + 一个详情工具 + 一个调用工具"。把 4,100 个端点逐一暴露成 MCP 工具会淹没模型上下文；三件套模式让 agent "先搜后调"。配合读/写分离的 hint、目录面隐藏生成类平台的差异策略，这是对 MCP 生态"工具爆炸"问题的清醒回答〔源码，mcp.py:10-13〕。

## 六、2026-10-23 安全事件：凭据托管模式的软肋

任何关于 treg 的严肃分析都无法绕开 17 天前的这场事件。官方事件报告[^5]的关键事实(本人逐字核验)〔官方〕：

- **2026-10-23 01:30(UTC)**:确认生产 web 服务存在**未授权的服务端命令执行**;02:24-02:35 关闭易受攻击功能并轮换密钥;03:20-10:30 逐个供应商轮换自有凭据;08:50 封禁疑似攻击者账户；
- **攻击方式**:"a vulnerability in a server-side feature"(服务器端某功能的漏洞)——**具体技术细节未披露**，官方表示待补救与外部审计完成后发布完整事后分析；
- **影响(保守视为已泄露)**：用户自行添加的 API 密钥、通过 treg 连接的所有账户(Slack/LinkedIn/Facebook/Meta Ads)、全部 treg 访问令牌(已重置)、treg 自有凭据——但数据库日志显示攻击者未执行任何数据库查询;Instagram 因平台不允许代撤销需用户手动移除；
- **官方定性**：这是"针对平台公司、以窃取 AI 推理服务 API key 为目标的一系列攻击"的一部分；
- **应对**：禁用漏洞功能且"原形式不会恢复"、全部凭据轮换、令牌重置、赔偿覆盖整个 10 月的未授权使用、引入多支渗透测试团队。

对照源码看这件事格外有意味:runner.py 的 exec/脱敏/进程组击杀、fsjail、egress 白名单、QuickJS 沙箱——这些**为 CLI 运行设计的防线质量很高**，但事件入口是"server-side feature"的服务端命令执行，说明**防线设计与实际暴露面之间存在缺口**(具体哪个 feature、哪条路径，官方尚未披露，不宜猜测)。更重要的结论是结构性的：**treg 的价值主张是"凭据集中托管后 agent 更安全"，而集中托管同时制造了单点故障**——一旦服务端沦陷，所有下游凭据一起沦陷，且 BYOK 的第三方 key 平台无权代撤(只能请用户自己去各供应商后台轮换)。10-23 事件把这个模式的阿喀琉斯之踵完整演示了一遍。时间线上它恰好发生在 HN 热帖(10-06)之后 17 天——增长最陡的时段撞上信任最重的打击，后文的潜力评估必须把"信任重建"作为第一变量。

## 七、工程质量：事故驱动的注释文化

这个代码库最独特的资产不是某段代码，而是**注释文化**：几乎每个防御性设计旁边都挂着一次真实踩坑记录。随手摘几条(全部实读)〔源码〕：

- mcp_install.py 曾教用户 `bearer_token_env_var`,环境变量丢失后 Codex **静默丢弃全部 treg 工具**，"burnt their daily Codex allowance on a $0.13 job"——所以现在永不建议环境变量引用 token;
- conftest.py 给每个测试进程独立命名 sqlite 文件，因为第一次并行跑出现过 **1022 个错误**；并且强制清空所有 TREG_PLATFORM_KEY_*——注释说开发者 .env 里带真实充值的键，不清空套件会在他笔记本上真花钱；
- CI 的 gitleaks 检查曾被一次 commit 掉，导致 PR 卡了三周，2026-08-24 恢复为必需检查；
- e2e-server.sh **故意用真实厂商键**跑 e2e("真实的上游会用真实的错误体回答——这正是 e2e 的全部意义"),每次跑生成新 Fernet key(e2e_check.py 断言:成功调用不落证据列、真实失败保留截断的错误体、**真实平台键在所有证据列中零泄漏**)。

量化面：296 个测试文件、官方称 2,700+ 独立用例、双数据库(SQLite + Postgres 16)+ Redis 的 CI 矩阵、Playwright E2E、gitleaks 全历史扫描、六类目录校验 workflow、import-linter 层边界门禁、93 个 alembic 迁移带 baseline/expand-safety 专项测试〔源码，tests/、.github/workflows/ci.yml〕。frontend 是 Vue 3.5 + Vite + TypeScript(不是 React),wheel 构建带资产断言(不许混入 .mp4、必须引用编译产物)。甚至 UI 都有一份对标 Vercel design.md 的 143 行设计规范，其中反复强调 "Make visual promises truthful——看起来可点击的元素必须真的执行动作"〔源码，design.md〕。

**评估结论：工程质量在开源项目里属第一梯队，且远超其"四个月、小团队、AI 辅助"的出身**。另一个值得记录的架构细节:仓库 AGENTS.md 披露 treg.to 的真实生产拓扑在私有仓 treg-internal(公共仓只承载可移植行为,生产配置/事件证据/私有工具在内部仓，"描述 treg.to 只谈机制，绝不谈数字、日期、主机、实例数")[^6]〔源码+官方〕。

## 八、商业模式与竞争格局

### 8.1 "批发转零售"的本质

treg 的商业模式剥掉外壳是：**用自己的资金订阅/签约 Semrush、Crunchbase、X API 等数据能力，按次拆卖给 agent**。官方声明 0% 加价("we earn on volume pricing with vendors - not on you"),收入靠与供应商的量价差；BYOK 永远免费(唯一例外：走 treg 自家 X app 的连接按次计费,因为 X 向 app owner 收费)[^1]〔官方〕。llms.txt 甚至明确允许把 treg 嵌入第三方计费产品、按客户 ID 限额开票[^2]。

官方自办的基准测试(注意：自办基准需打折)称：292 人邮箱查找测试中 treg 准确率 90.4%,与 Clay(89.7%)、Freckle(90.1%)打平，但**每正确行成本 $0.0056 vs Clay 的 $0.0395(7 倍差距)，单次延迟 0.44s vs 11.2s**[^9]〔官方〕。

### 8.2 竞品对比

| 项目 | 聚合对象 | 模式 | 与 treg 的关键差异 |
|---|---|---|---|
| **OpenRouter** | LLM 模型 | 按上游 token 价转发 + 手续费 | treg 整套话语体系刻意对标它("0% markup" 就是攻击其 fee 模式) |
| **Composio**(30.9k star) | 应用动作(1000+ toolkits) | SDK 开源 + 托管 key | 面向"执行操作"(per-user OAuth),开发者 SDK;treg 面向"买数据与生成能力"，agent 直接用的消费层 |
| **Toolhouse** | 5000+ 服务 | $5/1000 请求 | 已上移到"AI worker"定位,与"按次零买数据"错位 |
| **Zapier MCP** | 应用目录 | 订阅制 | 自动化导向,无按次计费的数据 API 层 |
| **mcp-jungle / tarvis.io** | 自托管 MCP 目录 | 开源/一键自托管 | HN 评论区指出的"低配替代":社区说用它们可达同样目的——**这正是 treg 壁垒问题的直接证据**[^10]〔社区〕 |

**treg 的真实差异化**：它不是又一个"MCP 工具集成层",而是**付费数据 API 的批发转零售 + BYOK 凭据托管 + agent-first 原生分发**三件事的合体。竞品要么做动作集成(Composio/Zapier),要么做开发者基础设施，没人做“把订阅墙后的数据 API 拆成单次零售”这门生意。HN 评论者 florianpipes 的评价可作为需求侧佐证："通过单一订阅访问数据源的无状态 API 集合，是 GTM 自动化的一半"[^10]〔社区〕。

### 8.3 许可证的商业护栏

名义 Apache 2.0,GitHub 许可证字段实为 NOASSERTION——附加条款禁止把代码再分发为竞争性托管/注册服务(需书面许可)。这是"开源获客 + 托管变现"的标准双轨：代码开放换信任与贡献，托管服务收费，同时用条款挡住"拿代码开竞品"的路。

## 九、潜力与风险：三块薄冰上的锋利定位

### 9.1 潜力：四张真牌

1. **需求是真的**。"任务→任一供应商→结果"这个抽象在模型层(OpenRouter)已被充分验证，迁移到工具层是合理外推；四个月 7k star 与 HN 讨论证明开发者共鸣存在。"agent 需要零买数据能力而人类不需要"是 agent 经济带来的新需求结构。
2. **agent-first 分发是真创新**。五橱窗、llms.txt 行为契约、SKILL.md 金钱/证据/伦理纪律、TOON 输出格式、父进程识别——把"agent 是第一等用户"落到了每一处细节，这套方法论本身就有独立于 treg 的复用价值(值得每个做 agent 产品的人研究，包括它引导 star 的灰色部分)。
3. **工程执行是第一梯队**。四个月、小团队、AI 辅助，产出 11.2 万行带完整测试矩阵与层边界门禁的代码库，且注释即风险登记簿——这个迭代质量是它能在安全事件后存活的前提。
4. **BYOK + 开源自托管的对冲选项**。凭据阶梯让"只用它做团队凭据网关、完全 BYOK、甚至自托管整栈"成为合法用法，给不信任托管的用户留了台阶——这既是获客面也是信任对冲。

### 9.2 风险：三块薄冰 + 两个未知数

1. **供应商 ToS(最大悬顶)**。Semrush、X、LinkedIn 们的条款是否允许第三方持卡转售，HN 社区已点名("这些提供商的服务条款大多不允许此类用途")[^10]。模型聚合层没有这个问题——模型厂商的 API 本来就是按量售卖的；而 $139/月订阅模式与"拆开零卖"天然冲突。一旦头部供应商发律师函或做 API 指纹封禁，目录价值瞬间坍缩。**这是 OpenRouter 类比在最深处的破裂点:treg 聚合的"商品"在法律上不属于它**。
2. **凭据托管的信任(已被攻破一次)**。10-23 事件证明单点故障不是理论；且 BYOK 第三方 key 平台无权代撤，事故处置成本极高。渗透测试与事后分析的结果将直接决定企业客户敢不敢把真键放进来。
3. **壁垒结构性薄弱**。目录 YAML 可复制、代码 Apache 2.0 可自托管(mcp-jungle/tarvis 已是活例)、HMAC 团队码等机制也可抄。真正剩下的护城河只有：持卡转售的供应商关系与议价能力、per-call 计费与审计的运营复杂度、以及凭据托管的信任存量——第三项刚受重创。
4. **0% 抽成的收入结构**：收入完全依赖批发价差，会被供应商直营按量 API 与自托管竞争者双向挤压——支付行业"收单+转接"式的薄利生意，天然向持牌合规方集中。
5. **两个未知数**：无公开融资与法人信息(执行持续性与安全事件的赔偿能力存疑)；star 增长含 agent 引导成分(真实开发者采用率需要用调用数据而非 star 衡量，后者不公开)。

### 9.3 赛道终局判断

"工具聚合层"会存在——agent 经济需要这一层，就像电商需要支付网关、模型层需要 OpenRouter。但它的终局更像**收单/转接生意:薄利、重运营、强信任门槛、天然向合规方集中**。treg 抢占了叙事先手("OpenRouter for agent tools" 是 2026 年最锋利的 positioning 之一)，验证了需求，交付了第一梯队的工程——但它站在供应商 ToS、凭据托管信任、零壁垒代码三块薄冰上，且第一块薄冰的裂缝(BYOK 泄露)已经真实出现过一次。

## 十、结论与选型建议

**定位判断**：treg 是 2026 年"agent 工具经济"叙事下定位最锋利、工程最扎实的开源基础设施项目之一。它对"工具藏在订阅墙后"这个痛点的解法(批发转零售 + 服务端凭据注入 + agent 原生分发)是完整的，且源码质量经得起逐行审计。但它不是 OpenRouter 的结构等价物——OpenRouter 转售的商品(API token)在法律上属于可转售品，treg 转售的商品(订阅拆分)在法律上属于灰色品；这个差异决定两者的天花板不同。

**选型建议**(按场景)：

- **适合**：轻量、按次采购外部数据/生成能力的 agent 工作流(lead-gen、竞品分析、SEO 侦察、UGC 批量生产)；团队内共享厂商凭据而不想给每个成员发 key 的场景(纯 BYOK 模式，零计费风险)；个人开发者想给 coding agent 装一个"外部世界接口"。
- **谨慎**：生产关键路径(官方托管无公开 SLA;安全事件善后仍在进行)；企业敏感凭据(等渗透测试与事后分析报告落地再评估)；对成本稳定性敏感的批量负载(定价与路由随时调整)。
- **规避**：供应商 ToS 明确禁止转售的商业化产品内嵌场景(法务先审)；把高价值自有 key 放进共享团队的场景(Grant 例外面虽小但存在)。
- **折中**：自托管(dev-local.sh 本地一键起，Render Blueprint 生产化，SQLite 起步)——代码开放，运营复杂度(93 个迁移、双仓生产拓扑)自担。

**一句话总结**：treg 验证了"数据 API 零买"这个真需求，交付了教科书级的 agent-first 工程与一份可以逐行审计的安全设计；但它把整个商业模式押在"供应商允许转售、平台永不沦陷、代码零壁垒"三个假设上——第一个假设在法务上最脆弱，第二个假设 17 天前刚被现实击穿过一次。赛道大概率成立，treg 未必是终局赢家；无论它自己走多远，它的 llms.txt 分发方法论、SKILL.md 行为契约范式与 relay 的"忠实中继"架构，都值得进 2026 年 agent 基础设施的案例册。

[^1]: superdesigndev. "Treg (OpenRouter for Tools)" README. https://github.com/superdesigndev/treg
[^2]: treg.to. "llms.txt"(给 AI agent 的官方指令文档,含 star 引导、免费额度、金钱纪律等原文). https://treg.to/llms.txt
[^3]: Hacker News. "Show HN: SuperDesign.Dev"(2025-06-25,Jason 以 jzdesign1993 账号发布). https://news.ycombinator.com/item?id=44376003
[^4]: GitHub API. superdesigndev 组织与 treg 仓库/贡献者数据,2026-11-03 快照. https://api.github.com/repos/superdesigndev/treg
[^5]: treg.to Blog. "Security incident 2026-10-23"(官方事件报告,2026-10-25 发布). https://treg.to/blog/security-incident-2026-10-23
[^6]: superdesigndev. "treg/AGENTS.md"(仓库工程铁律:不可协商事项、treg-internal 双仓约定). https://github.com/superdesigndev/treg/blob/main/AGENTS.md
[^7]: superdesigndev. "treg/plugin/README.md"("分发包装,不是第二个产品"与五橱窗策略). https://github.com/superdesigndev/treg/tree/main/plugin
[^8]: superdesigndev. "treg/USAGE.md"(620 行官方使用文档). https://github.com/superdesigndev/treg/blob/main/USAGE.md
[^9]: treg.to Blog. "Work email finding bench"(自办基准:292 人邮箱查找,treg 90.4% vs Clay 89.7%,成本 7 倍差). https://treg.to/blog/work-email-finding-bench
[^10]: Hacker News. "Treg (OpenRouter for Tools)"讨论帖(2026-10-06,48 分/10 评论,含 ToS 质疑、mcp-jungle/tarvis 自托管替代等社区声音). https://news.ycombinator.com/item?id=49980790
