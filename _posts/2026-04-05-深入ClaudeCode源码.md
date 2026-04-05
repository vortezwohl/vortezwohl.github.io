---
layout: post
toc: true
title: "深入 Claude Code 源码：学会与你不信任的伙伴（LLM）共事"
categories: LLM
tags: [LLM, Code Agent, Claude Code, Harness Engineering]
author:
  - vortezwohl
  - 吴子豪
  - Codex
---

读完 Claude Code 源码，我最大的感受不是“它能调用很多工具”，而是：**它一直在防模型出错，防模型乱来，防环境反过来坑系统**。如果用一句更直白的话概括：**Claude Code 不是把模型直接放进终端里干活，而是先给模型套上一层厚重的“赛博护栏”，只允许模型在护栏里行动。** 而这层“赛博护栏”，就是我所理解的 harness...

它不是某一个单独文件，也不是某一个类，而是一整套运行规则。它做的事情很像一个经验丰富的技术主管：

- 模型说“我要调用这个工具”，它先检查你参数写对了没有；
- 模型说“我要执行这个命令”，它先判断这个命令危险不危险；
- 模型说“这个目录里的配置我来执行一下”，它先问这个工作区到底值不值得信任；
- 模型说“我已经做完了”，它还会追问：你是真的做完了，还是只是嘴上说做完了？

所以，Claude Code 真正强的地方，不只是“会干活”，而是“**会在不信任模型的前提下，约束模型安全地干活**”。

如果写在 Paper 中，我们说 agent runtime、policy enforcement、verification pipeline；换成更易懂的话，就是：**它认定了模型会犯错，所以系统从一开始就在“不信任模型”的前提上设计。**

## Claude Code 在解决什么问题？

### 模型经常会把工具参数写错

这是最基本、也最现实的问题。

模型并不是一个稳定的 API 调用器。它经常会把：

- 数字写成字符串；
- 布尔值写错；
- 数组格式写错；
- 本来不该传的字段也一起传进去。

源码里有这样一行注释：

```markdown
// Source: src/services/tools/toolExecution.ts:614

// Validate input types with zod (surprisingly, the model is not great at generating valid input)
```

这句注释的意思是：**别信模型生成的参数，必须先校验。**

也就是说，在 Claude Code 里，模型不是“直接执行工具的人”，模型更像是“提申请的人”，真正能不能执行，要过系统的严格检查。

### 模型会在复杂流程里把状态搞乱

比如一轮对话里，模型可能会：

- 先发一个工具调用；
- 执行到一半被中断；
- 或者模型 fallback 到另一个模型；
- 或者流式输出中途失败。

这时候如果系统处理不好，就会出现一种很糟糕的情况：

- 前面已经记了 “我要调用工具”；
- 后面却没有对应的工具结果；
- 对话历史变成“半截状态”；
- 下一轮继续跑的时候，整个上下文就乱了。

Claude Code 为这种情况提供了兜底，其源码里有这样一段逻辑：

```ts
// Source: src/query.ts:123-145

function* yieldMissingToolResultBlocks(
  assistantMessages: AssistantMessage[],
  errorMessage: string,
) {
  for (const assistantMessage of assistantMessages) {
    const toolUseBlocks = assistantMessage.message.content.filter(
      content => content.type === 'tool_use',
    ) as ToolUseBlock[]

    for (const toolUse of toolUseBlocks) {
      yield createUserMessage({
        content: [
          {
            type: 'tool_result',
            content: errorMessage,
            is_error: true,
            tool_use_id: toolUse.id,
          },
        ],
      })
    }
  }
}
```

这段代码在做的事情是：**如果某个工具已经发起了调用，但后面流程崩了，那也要补一个失败结果，不能让记录悬在半空。** 

这很重要，因为 Claude Code 很清楚：不是只有“成功执行”才重要，**失败时能不能把状态补完整**也同样重要。

### 工作区本身可能不安全

这是很多人第一次读 Claude Code 源码时最容易忽略的一点。

普通人的直觉是：

- 用户打开了一个项目；
- 工具开始工作；
- 读配置、执行 hook、加载扩展；
- 一切顺理成章。

但 Claude Code 不这么想。

它的思路是：**你打开的项目目录，本身就可能带着恶意配置。**

比如：

- 项目里的 `.claude/settings.json` 可能被人预埋了危险设置；
- hook 可能会执行某些命令；
- skills、agents、外部 include 文件也可能成为注入点。

所以 Claude Code 把“工作区信任”放在很前面，而且地位很高。

源码里有一段注释写得非常明确：

```markdown
// Source: src/interactiveHelpers.tsx:125-126

// Always show the trust dialog in interactive sessions, regardless of permission mode.
// The trust dialog is the workspace trust boundary
```

翻译一下就是：**无论你权限模式怎么调，在交互模式下，都要先过“这个工作区值不值得信任”这一关。** 这不是普通权限确认框，而是更靠前的一层边界判断。

### 不是“用户点了允许”就万事大吉

很多系统的权限模型很简单：

1. 模型想干一件事；
2. 弹窗；
3. 用户点允许；
4. 执行。

Claude Code 不是这么单薄。

它在问的是更复杂的问题：

- 这个操作是不是本来就该禁止？
- 这个操作是不是必须总是询问？
- 这个操作是不是虽然看起来正常，但其实碰到了敏感路径？
- 这个操作是不是在自动模式下也不能放？
- 这个决定是不是被某条规则覆盖掉？

它不是把“允许”当成唯一控制点，而是做成一个多层判断流程。

## 我理解的 Harness：不信任模型，所以对模型深度监管

如果一定要解释 harness，可以这样说：**harness 就是一层负责“接住模型输出、检查模型意图、限制模型行动、补偿模型失误”的系统外壳。** 它的核心不是提高模型自由度，恰恰相反，它在控制模型的权限边界。

更具体地，Claude Code 的 harness 主要在做三件事：

### 把模型输出当成“待审输入”

模型说：

- “我要调用 Bash”
- “我要编辑这个文件”
- “我要进入 plan mode”
- “我要调用某个 MCP 工具”

Claude Code 不会立刻执行，它会先做一连串检查：

- 参数结构对不对；
- 值本身合不合理；
- 这个工具在当前模式下是否允许；
- 是否碰到了敏感路径；
- 是否需要用户确认；
- 是否应该放进沙箱执行。

也就是说，模型输出进入系统后，先经过一层层校验，最后才会变成真正动作。

### 把工具执行当成“受管流程”

Claude Code 并不只是“把命令发出去然后等结果”。

它在意的是整个执行过程是否还保持一致。

比如：

- 哪些工具可以并发；
- 哪些工具必须串行；
- 其中一个 Bash 失败了，其他并发任务要不要一起停；
- 流式执行过程中如果模型 fallback，之前的半截执行记录怎么处理；
- 如果用户中途打断，怎么收尾才不会把对话历史搞坏。

这些都不属于“模型能力”，属于**运行时托管**。

### 把“做了”与“证明做了”分开

这是我觉得 Claude Code 非常成熟的一点。

它不是只追求“任务做完”，还追求“你得能证明自己做对了”。

在多 agent 协作里，这一点尤其明显。

源码里有这样几条提示：

```markdown
- workers self-verify before reporting done. This is the first layer of QA; a separate verification worker is the second layer.
- For verification: "Prove the code works, don't just confirm it exists"
- For verification: "Try edge cases and error paths"
```

翻译一下就是：

- 实现的人先自己验证；
- 但这还不够；
- 最好再有一个独立验证者来检查；
- 验证不是说“代码在那儿”，而是要证明“代码真的能跑、边界情况也成立”。

这就把 agent 系统从“会行动”拉到了“会自证”。

## Claude Code 是在“对模型零信任”的基础上构建的

### 先校验，再执行

这是最基本的一条原则。

源码里先用 schema 校验参数，再做额外校验：

```ts
// Source: src/services/tools/toolExecution.ts:614-621,682-685

const parsedInput = tool.inputSchema.safeParse(input)
if (!parsedInput.success) {
  ...
}

...

const isValidCall = await tool.validateInput?.(
  parsedInput.data,
  toolUseContext,
)
```

这段逻辑所做的事：

1. 先看格式对不对；
2. 再看内容本身合不合理；
3. 都通过了，才进入后面的权限和执行阶段。

所以模型不是在“直接使用工具”，而是在“提交一份工具调用申请”。

这也是 Claude Code 零信任原则最朴素的一层：

**先假设模型给出的东西不可信，再决定要不要把它变成真实动作。**

### 即使 schema 理论上会拦，系统还会额外再防一层

有个很典型的例子是 Bash 输入里的 `_simulatedSedEdit`。

源码里写着：

```markdown
// Source: src/services/tools/toolExecution.ts:756

// Defense-in-depth: strip _simulatedSedEdit from model-provided Bash input.
```

这句话背后的意思是：**即使按设计，模型本来就不该传这个字段，系统还是再手动剥掉一次。**

这是一种很明显的安全思路：

- 不把“理论上不会发生”当成安全保证；
- 哪怕 schema 会拦，也再补一道保险。

### 不相信单一模块的“放行”

Claude Code 很警惕一种情况：

某个 hook 说“可以”，于是系统就一路绿灯。

它没有这么做。

源码里有这样的逻辑：

```markdown
// Source: src/services/tools/toolHooks.ts:388-394

Hook approved tool use for ${tool.name}, but deny rule overrides
Hook approved tool use for ${tool.name}, but ask rule requires prompt
```

**就算 hook 说能过，也不能直接算数。还要继续看全局规则是不是反对。** 这意味着系统不是把权限交给某个单点，而是用多层规则相互制衡。

### 不相信“自动模式”一定安全

Claude Code 支持 auto mode，也就是某些场景下系统自动判断是否批准，不总是打扰用户。但它没有因此把所有风险都交给自动分类器。

源码里有这样一行注释：

```markdown
// Source: src/utils/permissions/permissions.ts:526-533

// Non-classifier-approvable safetyCheck decisions stay immune to ALL auto-approve paths
```

意思是：**有些安全检查，根本不允许自动放行。**

也就是说，自动模式不是万能通行证。  
碰到某些高风险情况，系统仍然坚持人工边界。

这非常符合“零信任”思想：

- 不信模型；
- 也不完全信自动分类器；
- 甚至连系统自己的自动放行路径，也要继续设上限。

## 工作区信任：Claude Code 的第一关

很多人会把“权限”和“信任”混成一件事。  
Claude Code 把它们分得很开。

### 什么叫权限

- 你现在能不能执行这个动作？
- 这个工具这次能不能跑？

### 什么叫信任

- 这个工作区本身是不是安全的？
- 这里面的配置、hook、include、env 到底能不能影响系统？

Claude Code 的做法是：**先判断工作区值不值得信，再谈你在这个工作区里能做什么。**

源码里这句注释很关键：

```markdown
// Source: src/interactiveHelpers.tsx:126, src/utils/hooks.ts:1992

// The trust dialog is the workspace trust boundary
```

还有一句注释也很关键：

```markdown
// Source: src/utils/hooks.ts:1992

// SECURITY: ALL hooks require workspace trust in interactive mode
```

这两句结合起来，意思是：

- trust 不是 UI 提示词；
- trust 是安全边界；
- 没过这条边界，hook 这种能力就不该执行。

即 Claude Code 把“目录是否可信”提升成了系统级问题。

## 配置来源不是平等的

这一点很容易被忽略，但其实非常关键。

Claude Code 很清楚：**不是所有配置来源都该拥有一样大的权力。**

特别是一些危险能力，例如：

- 跳过危险模式确认；
- 自动模式 opt-in；
- 一些带明显安全影响的行为。

源码里明确写了类似这样的注释：

```markdown
// Source: src/utils/settings/settings.ts:879-880,893-894

projectSettings is intentionally excluded —
a malicious project could otherwise auto-bypass the dialog (RCE risk)
```

翻译成白话：**项目自己的配置不允许替用户自动同意危险设置，因为恶意项目可以借这个机会直接把系统带歪（导致远程代码执行漏洞，RCE）。**

这是非常好的安全意识。因为如果项目配置也和用户本地配置一样有权，那任何人只要往仓库里塞一份配置文件，就等于替用户点了“允许”。Claude Code 明显不接受这种风险。把这段话说得更直接一些：**Claude Code 认为“仓库里的配置”也是不可信输入的一部分。** 这正是零信任原则在配置层的体现。

## 沙箱不是“可选增强”，而是硬边界的一部分

Claude Code 的 sandbox 设计，给我的感觉不是“有个安全功能”，而是“它在主动补环境层的漏洞”。源码里有几段非常典型的注释：

```markdown
// Source: src/utils/sandbox/sandbox-adapter.ts:230,247,257

// Always deny writes to settings.json files to prevent sandbox escape
// Block writes to .claude/skills in both original and current working directories.
// SECURITY: Git's is_git_directory() treats cwd as a bare repo ...
```

这些代码不是抽象意义上的“加安全性”，而是在处理非常具体的真实风险：

- 不让模型改 settings，因为一旦模型能写配置文件，很多限制都可能被反向篡改。

- 不让模型随便写 `.claude/skills`，因为 skill 本身就很有能力。如果一个系统允许模型自动改 skill，本质上就是让它改自己的能力边界。

- 防 git 相关的特殊逃逸面，这一点有很强的工程倾向。作者不是只防普通文件写入，而是注意到了：

  - git 对目录结构有自己的解释方式；
  - 某些伪造结构可能影响后续 unsandboxed git 行为；
  - 所以要专门补 scrub 逻辑。

这些设计不只是“做权限控制”，而是“预知真实系统会在哪些地方漏风”。

## Bash 部分的设计最具代表性

如果说 Claude Code 源码里最能体现“零信任”和“防御性工程”的地方，我认为是 Bash 模块。因为 shell 是最危险、最容易被模型破坏的地方。Claude Code 对 Bash 的态度不是：

- 简单做几个关键词黑名单；
- 看着差不多就执行。

它做的是更麻烦、但也更靠谱的路线：

- **先尽量把命令真正解析明白**

  源码里有一句核心注释：

  ```markdown
  // Source: src/tools/BashTool/bashPermissions.ts:1672-1674

  // tree-sitter produces either a clean SimpleCommand[] ... or 'too-complex'
  ```

  翻译一下：

  **要么把这个命令解析成清清楚楚的结构，要么就承认它太复杂，别自作聪明。**

  这非常重要。

  很多系统最危险的地方，不是“完全没检查”，而是“没看懂却以为自己看懂了”。

  Claude Code 在这里明显更保守：

  - 看不懂，就提高警惕；
  - 太复杂，就不要轻易自动放；
  - 不把模糊理解当成安全依据。

- **它防的是“解析差异”这种更隐蔽的问题**

  源码里专门处理 brace expansion、quoted brace、unicode whitespace、mid-word hash 等情况。

  比如有一条报错信息：

  ```plaintext
  'Command contains quoted brace character inside brace context (potential brace expansion obfuscation)'
  ```

  这不是普通用户会想到的问题，但安全工程里很典型：

  **你以为你看到的是一种意思，shell 真正执行时却是另一种意思。**

  也就是：

  - 系统的解析；
  - shell 自己的解析；

  两边如果不一致，就可能被利用。

  Claude Code 显然在努力把这类“你以为的命令”和“真正执行的命令”之间的差距缩小。

- **即使某些子段通过了，也还要回头检查原命令**

  源码里有一句很值得注意：

  ```markdown
  // Source: src/tools/BashTool/bashPermissions.ts:1984-1985

  // SECURITY FIX: When pipe segment processing returns 'allow', we must still validate the ORIGINAL command
  ```

  这句话背后的意思很清楚：

  **不能因为命令拆开看都没问题，就默认整体也没问题。**

  比如：

  - 前半段安全；
  - 后半段安全；
  - 但中间的重定向、拼接、展开方式可能有问题。

  Claude Code 在这里的态度依旧是：**拆开检查不够，还得回头看整体。**

## 为什么要确保“消息轨迹必须完整”

这是读 `query.ts` 时很容易被低估的一点。

Claude Code 不只是一个“调用工具然后打印结果”的程序，它同时在维护一条很严格的消息轨迹。

为什么？

因为后面所有东西都依赖这条轨迹：

- 下一轮模型输入要靠它；
- transcript 要靠它；
- 恢复会话要靠它；
- tool/result 对应关系要靠它；
- thinking block 的规则也要靠它。

源码里有两句注释特别能说明问题：

```markdown
// Source: src/query.ts:554

// Note: stop_reason === 'tool_use' is unreliable -- it's not always set correctly.
```

```markdown
// Source: src/query.ts:715

// that would cause "thinking blocks cannot be modified" API errors.
```

这说明作者很清楚：

- API 并不是永远可靠地给你理想信号；
- thinking block 也有自己的严格约束；
- 只要消息轨迹坏掉，后面一整套系统都可能连锁出错。

所以 Claude Code 宁可花很多力气修补轨迹，也不愿带着半坏状态继续往前跑。

## 结果验证很重要，Claude Code 不接受“差不多行了”

Claude Code 的整体风格不是：

- 能跑就行；
- 模型说完成了就算完成；
- 输出看着像 JSON 就当真。

它更像是在不断追问：

- 你真的按 schema 输出了吗？
- 你真的调用了该调用的工具吗？
- 你是真的验证过，还是只是复述了一遍代码？

### structured output 不只是“提醒一下”

源码里有这样的要求：

```markdown
// Source: src/utils/hooks/hookHelpers.ts:61

Use this tool to return your verification result. You MUST call this tool exactly once at the end of your response.
```

```markdown
// Source: src/utils/hooks/hookHelpers.ts:80

You MUST call the SyntheticOutput tool to complete this request. Call this tool now.
```

为了让这件事更具体一点，其实源码不是只写一句提示，它还真的把这个要求挂进了停止阶段的检查逻辑里：

```ts
// Source: src/QueryEngine.ts:327-332

const hasStructuredOutputTool = tools.some(t =>
  toolMatchesName(t, SYNTHETIC_OUTPUT_TOOL_NAME),
)
if (jsonSchema && hasStructuredOutputTool) {
  registerStructuredOutputEnforcement(setAppState, getSessionId())
}
```

这段代码的意思是：**只要这次任务要求结构化输出，Claude Code 就会主动注册一条“你必须真的调用输出工具”的约束，而不是只靠模型自觉。**

这不是“温柔提示”，而是明确规则：**你不能只是说“这是结果”，你必须通过指定工具提交结果。**

也就是说，Claude Code 更相信“机器可检查的结果”，而不是“模型自称的结果”。

### 多 agent 场景下，验证是独立的一层

源码里的态度非常明确：

- 实现者先自查；
- 自查不算最终结论；
- 再来一个验证者独立检查；
- 验证时要试边界情况和错误路径。

这其实是在把“写代码”和“证明确实写对了”拆成两个阶段。对 agent 系统来说，这一步非常重要。因为模型最擅长的一件事，就是把“看起来像做完了”伪装成“真的做完了”。

## Claude Code 最值得学习的地方

### 不要把模型视作可靠的执行者

要把它视作：

- 申请者；
- 提议者；
- 有能力但不稳定的组件。

这样整个系统设计才足够健壮。

### 不要把“用户允许一次”当成唯一安全边界

真正可靠的系统，需要多层边界：

- schema；
- validation；
- permission；
- trust；
- sandbox；
- verification。

一层失误，不至于全盘崩。

### 不要只关心成功路径

真正的工程质量，往往体现在失败路径：

- 中断后怎么办；
- fallback 后怎么办；
- 半截 tool_use 怎么补；
- 轨迹不完整怎么修；
- 并发中一个任务炸了怎么收尾。

Claude Code 在这方面下了很大功夫。

### 不要只让模型“做事”，还要逼它“自证”

这是 agent 系统很关键的升级点。不是让模型更会说，而是让模型交出**可以检查的结果**。

## 我的理解和感悟

读完 Claude Code 源码后，我对它最本质的理解是：**它不是在努力把模型训练成一个完全可靠的工程师，而是在搭一个足够结实的工程系统，在零信任的假设下，让一个完全不稳定的模型在受控边界内工作。**

前一种思路是：

- 相信模型会越来越聪明；
- 所以给它更多能力。

后一种思路是：

- 模型再聪明也一定会犯错；
- 所以系统必须先把边界立住。

Claude Code 走的是第二条路，而且这正是它源码最有价值的地方。它真正成熟的地方，不在于“功能很多”，而在于它始终在追问这些更难的问题：

- 模型错了怎么办；
- 模型越界怎么办；
- 配置不可信怎么办；
- 执行到一半崩了怎么办；
- 模型说做完了，但其实没做完，怎么办。

最后，如果一定要给 Claude Code 下一个最简洁的定义，我会说：**Claude Code 是一个把不可信的大模型关进工程护栏里的 agent 运行时，而那圈护栏被称为 “Harness” 。**
