ClaudeCode 是 Anthropic 公司推出的终端原生 AI 编程助手，基于 Claude 系列大模型打造，它不仅是一个代码补全工具，更是一个能够深度理解项目上下文、直接执行开发任务的 AI 开发伙伴；它具备完整的项目感知能力，能够跨文件分析代码结构、理解业务逻辑和依赖关系，支持自然语言交互来自动生成、修改、重构代码，运行测试套件并分析结果，执行 Git 操作（提交、分支、合并），诊断和修复 Bug，规划并实施从需求分析到部署上线的多步骤复杂任务；同时提供精细的权限控制系统，支持沙箱环境执行危险命令，分级权限审批和操作审计，确保开发安全；它无缝集成到终端、主流 IDE（VS Code、IntelliJ）和桌面应用中，支持通过 npm、Homebrew、Docker 等多种方式安装，能够处理日常开发（原型开发、代码审查、文档生成）、复杂工程（系统重构、性能优化、技术债务清理）以及运维部署（CI/CD 配置、环境搭建、故障排查）等全流程任务，成为开发者简化繁琐工作、提升效率的智能助手；

> 由于 Anthropic 政策限制, 其在中国境内使用受限，用户需要自行解决网络访问和 API 调用问题。本文

## 安装 Claude Code

安装 Claude Code 之前, 我们需要安装 [`NodeJS`](https://nodejs.org/en/blog/release/v24.14.0) 和 [`npm`](https://www.npmjs.com/) 工具.

1. **安装 `NodeJS` 和 `npm` 工具**

    [点此下载](https://nodejs.org/dist/v24.14.0/node-v24.14.0-x64.msi)

2. **使用 `npm` 安装 Claude Code**

    ```
    npm install -g @anthropic-ai/claude-code 
    ```

3. **验证安装**

    ```
    claude --version
    ```

    输出以下内容代表安装成功

    ```
    2.1.71 (Claude Code)
    ```

## 开始在中国大陆使用 Claude Code

在中国境内使用 Claude Code 受到若干政策限制约束, 不论是注册 `Anthropic` 账户 (需要通过谷歌登录), 还是购买 `Claude` 服务 (需要 Visa/Master/AMEX 等支付方式, 且服务价格昂贵), 很难顺利使用. 所以 DeepSeek 为国内的 Claude Code 用户提供了支持.

### Claude Code 接入 DeepSeek 服务

1. **登录 [DeepSeek 官网](https://www.deepseek.com/), 创建 [API KEY](https://platform.deepseek.com/api_keys) 并[付费](https://platform.deepseek.com/top_up)**

2. **拿到已创建好的 API KEY, 并保存**

    ```
    sk-a9*****************************f
    ```

3. **创建 Claude Code 配置文件**

    在用户目录下 `.claude` 路径创建一个 `settings.json` 文件, 具体路径通常是 `C:\Users\{用户名}\.claude\settings.json`

    文件内容如下:

    ```json
    {
        "env": {
            "ANTHROPIC_AUTH_TOKEN": "sk-a9*****************************f",
            "ANTHROPIC_BASE_URL": "https://api.deepseek.com/anthropic",
            "ANTHROPIC_MODEL": "deepseek-reasoner",
            "ANTHROPIC_DEFAULT_OPUS_MODEL": "deepseek-reasoner",
            "ANTHROPIC_DEFAULT_SONNET_MODEL": "deepseek-reasoner",
            "ANTHROPIC_DEFAULT_HAIKU_MODEL": "deepseek-chat",
            "CLAUDE_CODE_SUBAGENT_MODEL": "deepseek-chat",
            "CLAUDE_CODE_MAX_OUTPUT_TOKENS": "32000"
        },
        "permissions": {
            "allow": [],
            "deny": []
        },
        "alwaysThinkingEnabled": false
    }
    ```

4. **开始使用 Claude Code**

    ![alt text](/images/ClaudeCode快速开始/image.png)

## 参考资料

[[1](https://claude.com/product/claude-code)] Anthropic Research Team. Claude Code. Claude Official Site, 2025.