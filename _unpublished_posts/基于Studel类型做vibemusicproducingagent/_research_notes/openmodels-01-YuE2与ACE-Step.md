# 研究线B补充 · 开源音乐模型技术底座 (2026-11 实测 GitHub)

## YuE2 (multimodal-art-projection/YuE)
- 13.1k stars / 1.5k forks / 229 commits; 新闻更新至 2026-10(活跃)
- 架构: 符号规划+音频生成两阶段——输入歌词+风格提示 → 先生成**可编辑的旋律与和弦谱(ABC notation)** → 再渲染 48kHz 立体声全曲; AR-NAR 混合 Transformer + 流匹配
- 能力: 零样本翻唱(SheetSage2 音频转谱后按新风格重唱); **README 提及"智能体化的乐谱编辑"**(官方已向 agent 方向靠拢); WildSongBench 评测集; MERT2 音乐理解
- License: 代码/文档 Apache 2.0; 权重 CC BY-NC 4.0+附加许可(个人可商用生成内容; 公司商用需联系授权)
- ★架构启示: "符号层可编辑+渲染器"与 Strudel agent 的"改 pattern 代码→出音频"范式同构; ABC 谱面是天然的 agent 编辑接口

## ACE-Step (ace-step/ACE-Step)
- 5.0k stars / 656 forks; v1.5 (2026-01); 148 open issues(活跃)
- **Apache 2.0(权重也开放=商业可用)**; 4 分钟生成约 20s(A100), 快于 LLM 基线 15×; 19 语言
- 操作面: Text2Music / Retake / Repainting / Edit / Extend / Remix(flow-edit) / 歌词编辑
- LoRA 生态: Lyric2Vocal, Text2Samples, RapMachine; Roadmap: ControlNet 类 StemGen(单轨生成=为旋律配伴奏), Singing2Accompaniment
- ★定位: 本地部署/合规商用首选开源引擎; 与 agent 结合的"重绘/局部编辑"接口已内建

## 组合判断(技术可行性结论)
- "agent(生成/修改 Strudel pattern 或 ABC 谱) + 本地/云端开源模型渲染 + 用户本地持有资产"的完整技术链路 2026 年已全部存在且许可兼容:
  - 交互引擎: Strudel(AGPL, 浏览器) / Rustel(AGPL, CLI) / 自研
  - 渲染: ACE-Step(Apache)/YuE2(非商业需授权)/商用 API
  - 符号资产: strudel mini-notation / ABC / MusicXML / MIDI
