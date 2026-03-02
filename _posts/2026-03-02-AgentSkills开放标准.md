---
layout: post
toc: true
title: "深入理解 Agent Skills 开放标准"
categories: Agent
tags: [LLM, agent, agentic-ai, agi, ai]
author:
  - vortezwohl
  - 吴子豪
---
Anthropic 在 2025 年 10 月推出的Agent Skills，并于 12 月 18 日将其发布为跨平台可移植的开放标准，核心解决了大语言模型驱动的通用 Agent 向领域专用 Agent转化时的三大痛点：上下文窗口资源浪费、能力碎片化无法复用、领域知识难以灵活扩展。该标准以文件系统为基础、渐进式披露为核心设计原则，让开发者能通过简单的文件夹 / 文件组织，为 Agent 封装领域化的流程知识、可执行脚本和资源，同时实现了跨平台、跨框架的复用（已被 Claude 生态、OpenAI Codex、Gemini、Spring AI Alibaba 等支持）。

## Agent Skills 开放标准

Agent Skills 开放标准的本质是为 AI Agent 定义了一套 “能力封装与加载的通用语言”，它摒弃了定制化 Agent 的碎片化开发模式，用轻量、无依赖的文件系统结构，让 “通用 Agent + 可插拔 Skills” 成为大模型 Agent 落地的主流范式。

大模型 Agent 的能力落地始终存在一个核心冲突：**希望 Agent 拥有尽可能多的领域能力，但 Agent 的上下文窗口有限，且单次任务仅会用到少数能力。** 此前的三种解决方案均有明显缺陷$^{[1]}$：

- **全量加载**: 将所有领域知识写入系统提示, 资源浪费严重, 可扩展性差.

- **多 Agent 架构**: 每个 Agent 单独封装单一领域能力, 存在任务转接成本, 且本质仍是局部全量加载.

- **RAG 检索**: 按需加载知识但易导致流程性知识碎片化, 准确率上限仅 70% 到 80%.

Agent Skills 开放标准通过**渐进式披露**解决了这一矛盾：让 Agent 先记住 “有什么能力”（轻量元数据），需要时再加载 “如何使用能力”（核心流程），特殊场景再调取 “详细资源 / 代码”（按需加载），既保证了能力的丰富性，又最大化利用了上下文窗口。

此外, Skills 还具备以下核心特性$^{[2]}$:

- **轻量化与无依赖**: 基于纯文件系统，无需复杂的中间件 / 数据库，开发者仅需掌握 Markdown/YAML 即可开发，门槛极低；

- **组合性与可插拔**: 多个 Skill 可独立开发、按需加载，Agent 可叠加多个领域 Skill（如同时加载 PDF 处理 + BigQuery 分析），实现能力的组合扩展；

- **无界能力扩展**: 通过 “核心文件 + 附属资源” 的结构，Skill 的内容容量不再受限于 Agent 的上下文窗口，理论上可封装无限量的领域知识；

- **与工具链深度整合**: 支持与代码执行（Bash/Python）、MCP（Model Context Protocol）服务器、外部工具（如数据库 / 云服务）协同，Skill 可封装工具调用流程，让 Agent 无需重新学习工具使用方法。

### 开放标准的核心规范

该标准定义了无平台依赖的基础结构，所有兼容该标准的 Agent / 框架均需遵循，这是其跨平台可移植性的核心，核心规范包含两部分：

#### 标准化的目录结构

每个 Skill 是一个独立的文件夹，以技能名命名，核心为必需的SKILL.md，搭配可选的脚本、参考文档、资源文件夹，一个基础的 Skill 文件结构如下：

```
skill-name/
├── SKILL.md  # 必需：元数据+核心指令，标准入口
├── scripts/  # 可选：可执行脚本（Python/Bash/Node.js），提供确定性操作
├── references/  # 可选：参考文档（md/pdf），存放细分场景的详细说明
└── assets/  # 可选：资源文件（模板/配置/数据源），支撑技能执行
```

> 更具体的 Skill 目录与文件标准参见[深入理解 Skill 的标准文件结构](#深入理解-skill-的标准文件结构).

## 设计 Skill 的五大核心原则

Anthropic 官方将 “构建 Skill” 类比为 “为新员工准备入职指南”，核心是把领域的流程性知识封装为 Agent 可理解、可执行的结构化内容。设计 Skill 需遵循五大核心原则，同时按标准化步骤落地，确保 Skill 的可用性、可扩展性和兼容性。

1. **渐进式披露为核心**: 所有内容设计均需围绕"分层加载", 避免将所有知识写入单一文件, 应该让 Agent 仅加载当前任务所需内容.

2. **面向 Agent 视角**: `name` 和 `description` 需贴合 Agent 的意图匹配逻辑, 指令步骤需清晰可执行, 符合 Agent 的工具调用习惯.

3. **轻量化核心, 精细化拆分**: `SKILL.md` 主体仅保留通用核心流程, 细分场景/罕见操作拆分为附属文件, 减少上下文占用.

4. **代码与文档融合**: 将确定性或高成本的操作封装为可执行脚本, 脚本引入文档, 让 Agent 可直接执行而非生成代码.

5. **安全优先**：不包含未审计的外部网络调用、敏感数据操作，脚本需做边界校验，避免恶意使用导致的环境风险。

## 深入理解 Skill 的标准文件结构

单个 Skill 的完整目录结构如下$^{[3]}$: 

```
[skill-id]/  # 技能唯一标识目录（小写字母/数字/连字符，如pdf-form-processing）
├── SKILL.md          # 【必需】技能核心入口：元数据+通用逻辑，所有调用的起点
├── scripts/          # 【可选】确定性可执行脚本：解决Agent生成代码不稳定的问题
│   ├── extract_fields.py  # 示例：PDF表单字段提取（Python）
│   ├── convert_pdf.sh     # 示例：PDF格式转换（Bash）
│   └── requirements.txt   # 示例：脚本依赖清单（如pypdf>=4.0）
├── references/       # 【可选】细分场景参考文档：避免SKILL.md内容臃肿
│   ├── encrypted_pdf.md   # 示例：加密PDF处理特殊步骤
│   └── invoice_spec.pdf   # 示例：行业发票格式规范（静态文档）
├── assets/           # 【可选】静态资源：支撑技能执行的模板/配置/数据
│   ├── templates/    # 示例：PDF表单模板（invoice_template.pdf）
│   ├── configs/      # 示例：API密钥配置（pdf_api_config.json）
│   └── datasets/     # 示例：测试用PDF样本（test_invoice.pdf）
└── tests/            # 【非标准但最佳实践】技能测试用例：验证脚本/逻辑正确性
    └── test_extract.py    # 示例：验证extract_fields.py的测试代码
```


定义|类型|核心功能|适用场景
|:--:|:--:|--|--|
|`[skill-id]/`| 目录   | 技能的唯一命名空间，名称必须与SKILL.md中name字段完全一致，Agent 通过此名称匹配技能 | 所有场景（基础要求）|
|`SKILL.md`| 文件   | 技能的唯一入口，包含元数据（Agent 预加载）和核心逻辑（Agent 触发后加载）| 所有场景（必需）|
|`scripts/`| 目录 | 存放确定性、可复用的可执行脚本，Agent 可直接调用（无需重新生成代码）| 需要稳定执行的操作（如数据提取、格式转换）|
|`references/`| 目录 | 存放细分场景的详细文档，仅在 Agent 需要时加载（减少上下文占用）| 复杂场景的补充说明（如特殊规则、行业规范） |
|`assets/`| 目录 | 存放静态资源，作为脚本 / 文档的辅助材料| 需要模板、配置、测试数据的场景|
|`tests/`| 目录 | 验证技能正确性的测试用例（非标准但推荐）| 技能开发 / 迭代阶段的自测|

- **`SKILL.md` 结构细节$^{[3]}$**

    `SKILL.md` 是 Skill 的灵魂，也是 Agent 唯一的 “入口文件”，其结构严格分为前置 YAML 元数据和Markdown 主体内容两部分，缺一不可: 

    ```markdown
    ---
    # 第一部分：YAML前置元数据（必需，严格格式，Agent预加载）
    name: pdf-form-processing  # 必需：与目录名一致的唯一标识
    description: Extract text and fill form fields from PDF invoices, support encrypted PDF decryption.  # 必需：Agent匹配意图的核心依据
    compatibility: Python 3.8+, pypdf>=4.0, PyCryptodome  # 可选：环境/依赖要求
    license: MIT  # 可选：许可证类型（共享技能时用）
    version: 1.0.0  # 可选：技能版本
    tags: [pdf, invoice, form, extract]  # 可选：关键词，辅助Agent匹配
    metadata:  # 可选：自定义扩展元数据（行业/企业自定义）
    domain: finance
    author: dev@example.com
    ---

    # 第二部分：Markdown主体内容（核心逻辑，Agent触发后加载）
    ## 1. 技能概述
    This skill is used to process PDF invoices, including field extraction, form filling and encrypted PDF decryption.

    ## 2. 快速开始（核心流程）
    ### 2.1 Basic Field Extraction
    Run the script to extract form fields from PDF:
    `python scripts/extract_fields.py [pdf_file_path]`
    ### 2.2 Encrypted PDF Handling
    For encrypted PDFs, refer to the detailed steps in `references/encrypted_pdf.md`.

    ## 3. 资源引用
    - Template file: `assets/templates/invoice_template.pdf`
    - API config: `assets/configs/pdf_api_config.json`

    ## 4. 注意事项
    - Do not support scanned PDF (only editable PDF)
    - Encrypted PDF requires password input in the script parameter
    ```

- **`SKILL.md` 核心部分详解$^{[3]}$**

    - **前置 YAML 元数据** (Agent 预加载层)

        这部分是Agent**启动时唯一加载的内容**（仅占~100 token），是技能“被发现、被匹配”的关键，字段约束如下：

        |字段|必需/可选|约束规则|核心作用|
        |:--:|:--:|--|--|
        | `name` | 必需 | 1-64字符，仅小写字母/数字/连字符，与目录名完全一致 | Agent识别技能的唯一ID |
        | `description` | 必需 | 1-1024字符，用Agent可理解的语言描述功能/场景 | Agent匹配用户指令意图的核心依据（如用户说“填充PDF发票”，匹配该描述） |
        | `compatibility` | 可选 | 列出环境/依赖（如Python版本、库） | Agent执行前检查环境，避免执行失败 |
        | `license`/`version`/`tags` | 可选 | 无严格约束，按需定义 | 技能共享、版本管理、辅助匹配 |
        | `metadata` | 可选 | 自定义键值对 | 行业/企业扩展（如金融领域标记、作者信息） |

    - **Markdown 主体内容** (Agent触发后加载层)

        这部分是技能的核心逻辑，需遵循**渐进式披露**原则（控制在5k token内），核心模块包括：

        - **技能概述**：补充`description`的细节，说明技能边界（如“仅支持可编辑PDF，不支持扫描件”）；

        - **快速开始**：提供通用执行流程，包含脚本调用示例、基础步骤，是Agent执行的核心参考；

        - **资源引用**：明确指向`scripts/`/`references/`/`assets/`中的文件，告诉Agent“何时用、用哪个”；

        - **注意事项**：说明异常处理、边界条件（如加密PDF需要密码），降低执行错误率。

- **Skill 的调用逻辑$^{[1]}$** (渐进式披露)

    Agent对Skill文件的调用完全遵循“**先轻量、后详细，先通用、后细分**”的渐进式披露原则，完整调用流程如下：

    1. **预加载** (仅加载元数据)

        Agent启动时，遍历所有Skill目录，仅读取每个`SKILL.md`的**YAML元数据**（`name`/`description`为主），拼接成“技能列表”写入系统提示。

        - 此时Agent仅知道“有什么技能、能做什么”，不加载任何脚本/文档/资源；

        - 示例：Agent系统提示中会包含“pdf-form-processing：Extract text and fill form fields from PDF invoices...”。

    2. **触发** (加载 `SKILL.md` 主体)

        当用户指令匹配到某技能的`description`（如用户说“帮我提取PDF发票的表单字段”），Agent触发该技能，读取`SKILL.md`的**完整Markdown主体内容**，加载到上下文。

        - 此时Agent获取技能的通用流程，但仍不加载`scripts/`, `references/`, `assets/`中的文件；
        
        - 示例：Agent加载到“执行 python scripts/extract_fields.py [pdf_file_path]”这一指令，但未执行脚本。

    3. **按需调用** (加载脚本/文档/资源)

        Agent解析`SKILL.md`中的资源引用，根据当前任务需求，**仅加载所需文件**：

        1. **调用scripts/脚本**：Agent调用代码执行工具（如Python/Bash），直接执行脚本（如`python scripts/extract_fields.py invoice.pdf`），仅将**执行结果**（如字段列表）加载到上下文，不加载脚本本身内容；
        
        2. **调用references/文档**：Agent调用文件读取工具，读取指定文档（如`cat references/encrypted_pdf.md`），仅加载当前任务所需的文档片段（如加密PDF解密步骤）；
        
        3. **调用assets/资源**：Agent通过脚本或文档中的路径，访问静态资源（如加载`assets/templates/invoice_template.pdf`作为填充模板），资源本身不加载到上下文，仅在脚本执行时使用。

    4. **执行后释放**

        任务完成后，Agent释放临时加载的`SKILL.md`主体、脚本执行结果、参考文档内容，仅保留元数据在系统提示中，避免上下文溢出。

- **文件相互引用的规则$^{[3]}$** (核心可移植性保障)

    Skill内的文件引用必须遵循严格规则，确保跨平台可移植（无论 Agent 部署在 Windows/Linux/Mac，引用都有效）。

    - **核心规则**：仅使用相对路径

        所有文件引用必须以`SKILL.md`所在目录为根目录，使用**相对路径**，示例：

        |引用目标|正确写法（相对路径）|错误写法（绝对路径）|
        |----------|----------------------|----------------------|
        | 脚本文件 | `scripts/extract_fields.py` | `/home/user/skills/pdf/scripts/extract_fields.py` |
        | 参考文档 | `references/encrypted_pdf.md` | `C:\skills\pdf\references\encrypted_pdf.md` |
        | 资源文件 | `assets/templates/invoice_template.pdf` | `D:/assets/invoice_template.pdf` |

    - **`SKILL.md` 引用其他文件**:

        在Markdown主体中，明确说明“引用路径+使用方式”，示例：
        ```markdown
        ### 加密PDF处理步骤
        1. 执行解密脚本：`python scripts/decrypt_pdf.py [pdf_path] [password]`；
        2. 参考解密规则：阅读 `references/encrypted_pdf.md` 的第3章；
        3. 使用解密模板：加载 `assets/templates/decrypted_invoice.pdf`。
        ```

    - **`references` 中的参考文档引用脚本或资源**:

        参考文档中可直接标注相对路径，示例:

        ```markdown
        ### 高级解密示例
        使用 `scripts/advanced_decrypt.py` 脚本，配合 `assets/datasets/encrypted_test.pdf` 测试数据，可解密AES-256加密的PDF。
        ```

    - **引用最佳实践**

        - **路径风格统一**: 所有引用使用 Unix 风格路径分隔符（/），Agent 工具链会自动适配 Windows 的\；

        - **避免循环引用**: 禁止脚本 A 引用脚本 B，脚本 B 又引用脚本 A（会导致执行死循环）；

        - **路径注释**: 在脚本 / 文档中注释路径逻辑（如# 上级目录 → assets/templates），便于维护；

        - **版本适配**: 在 SKILL.md 中说明资源版本要求（如 “assets/templates 仅兼容 PDF 1.7 格式”）。

## 参考文献

[[1](https://claude.com/blog/equipping-agents-for-the-real-world-with-agent-skills)] Anthropic Research Team. Equipping agents for the real world with Agent Skills. *Claude Official Blog*, 2025.

[[2](https://agentskills.io/home)] Anthropic Research Team. Agent Skills Documentation. *AgentSkills.io*, 2025.

[[3](https://agentskills.io/specification)] Anthropic Research Team. Agent Skills Specification.  *AgentSkills.io*, 2025. 