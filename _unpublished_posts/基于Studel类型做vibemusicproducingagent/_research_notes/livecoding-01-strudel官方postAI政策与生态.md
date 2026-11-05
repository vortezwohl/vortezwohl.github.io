# 研究线A · Strudel 生态与 AI 政策 (2026-10 实测)

## 1. Strudel 官方 "post-ai humane policy" (决定性事实)
- 帖: https://old.reddit.com/r/strudel/comments/1xexepq/ (r/strudel, 2026-10-25, 98分/66评, 全量精读)
- 发布者: yaxu(TidalCycles/Strudel 创始人 Alex McLean), 政策全文: https://codeberg.org/uzu/strudel/src/branch/main/POLICY.md (issue #2095 收反馈)
- 政策要点(原文直引):
  - "contributors and other users of our projects should not have to engage with LLM-generated code and text as part of their community work."
  - "We recognise that LLM-based scraping and generation of (A)GPL code shared by ourselves and others is of questionable legality. For example, under many juristictions, LLM-generated work does not have a copyright holder and can therefore not be licensed under the (A)GPL."(官方论点: LLM 生成物无版权主体→无法按 (A)GPL 授权)
  - "Contributions including LLM-generated content will be declined."(PR 级拒绝)
  - "LLM-generated text (e.g. commit messages, issue comments, forum posts etc) will be flagged, and ultimately deleted."
  - "We strongly discourage others from forking/importing our projects into corporate AI platforms, including Microsoft Github."(反 GitHub/反公司化 AI 平台)
  - 仓库已迁 Codeberg: codeberg.org/uzu/strudel (自托管 forgejo CI)
  - ★关键豁免: "A contributor's personal use of LLMs and related technology is their own business, whether for research, analysis, bug discovery or other assistive technology, so long as the output is not included in any contributions to our projects."
- yaxu 追加澄清: "We just want to interact with human code while working on strudel."(约束对象=项目贡献, 不是用户怎么用)
- 社区反应: 压倒性支持([9分] "i stand with this ✊", [8分] "Amen!"); 少数异议被踩(garlicfiend -3: "AI assisting you in learning to play guitar vs playing the guitar for you... gobsmacked"); [Xenon_Chameleon]: "There is such a massive difference between people using LLMs for things like research and bug checking versus tech bros using it to vibe-code slop projects for clout and/or money."
- 评论中提到的 fork: "Dexter U.S. version"(有人把 strudel 移植到 windows 桌面, 不打算遵守政策 unfork——"Like I am going to unfork it")

## 2. 对"基于 Strudel 做 vibe producing agent"的直接约束(初步推论, 详见 08 报告)
- 允许: agent 为"用户个人使用"生成 Strudel patterns——政策明文豁免;
- 敌意区: fork Strudel 进商业 AI 产品("corporate AI platforms"被点名劝阻); 用 Strudel 源码做 AI 训练/再分发被官方认为法律上可疑;
- 设计含义: (a) agent 定位为"Strudel 语言的外部 IDE/agent"(类似 Cursor 之于任意语言), 不吞并引擎; (b) 或自研兼容 pattern DSL; (c) 引擎若要复用 strudel 代码需 AGPL 开源整体产品——商业闭源路线需自研引擎;
- 社区进入策略: r/strudel + uzu discord 是强反 vibe-slop 文化, 营销话术必须避开"AI 替你写歌", 强调"表达性乐器/人机协作"。

## 3. 生态新工具(2026-10~11, 时间窗证据)
- **Rustel** (r/livecoding 2026-11-03, 65分/23评): Rust 原生 Strudel 引擎, "runs Strudel patterns without needing a browser"; 终端 studio(live editing/piano roll/scope/spectrum/mixer, Vim 模式, auto-evaluate); CLI: `rustel play --watch yourScore.strudel`; 支持 samples/synths/effects/MIDI/OSC/Ableton Link/VST3/KabelSalat/stems 导出/session tapes("nothing you play is ever lost"); AGPL; rustel.cc + github.com/tzfm/rustel; "99% strudel feature complete"。→ CLI+watch 模式=天然 agent 可驱动的执行后端。评论区疑问集中在 sample 版权(作者逐个核实并文档化)。
- **Stave** (r/livecoding 2026-10-24, 8个月开发, beta): "keeps your Strudel code as the source of truth and adds the views a producer expects: a Timeline, a step grid and piano roll, a Mixer"; monaco 编辑器; p5/Hydra/GLSL 可视化经 .viz() 挂载; 代码↔视图双向同步("Edit on the Timeline... and Stave writes it back into your code"); stave.live, AGPL 开源(github.com/MrityunjayBhardwaj/Stave)。→ "代码为源+GUI 制作人视图"方向的直接先行者(无 agent 层)。
- r/strudel 子版已独立存在(r/TidalCycles 部分迁出/并存)。

## 4. LLM×live coding 既有实验存量
- r/TidalCycles 2025-01: "I made claude.ai convert my poem into TidalCycles code"——当时 Claude 对 TidalCycles 不熟("as of now claude doesn't seem to be familiar with TidalCyles"); 2年后(2026) 模型对 strudel/tidal 语法的熟悉度待实测(写入 08 报告的可行性项)。
