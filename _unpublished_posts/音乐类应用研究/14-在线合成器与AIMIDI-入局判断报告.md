# 在线合成器体验站 × AI 生成 MIDI · 入局判断报告

> **调研对象**: 独立开发者做"在线(Web)合成器体验网站"，并叠加"AI 生成 MIDI 乐谱"能力的三明治形态——免费合成器体验层 + AI MIDI 引擎付费层 + 云保存/分享留存层。
>
> **调研日期**: 2026-10-29（所有时效性数据以当日核验为准）
> **调研方法**: 继承本仓库 09 号报告(手机端模块合成器，综合 5.5/10)的既有事实存量；本轮新增 GitHub webaudio/midi-generation 主题实测、awesome-webaudio Apps 类目全量清单(40+ 条目)、Magenta.js 官方文档、SOUNDRAW/AIVA 定价页、Reddit 三连实测(r/synthesizers 版块搜索 / AI-MIDI 关键词搜索)，中英文双语检索。
> **置信度标注**: **【事实】**= 有可追溯来源；**【推断】**= 基于事实的合理推算；**【数据缺失】**= 公开渠道查不到。

---

## 0. 调研定约（KIQ）

| # | 关键情报问题 | 挂靠决策 |
|---|---|---|
| KIQ1 | Web 合成器体验站的赛道供给与变现先例？ | 纯体验站做不做 |
| KIQ2 | AI 生成 MIDI 的技术地基与竞品动态？ | AI 层可行性 |
| KIQ3 | 用户真实需求与付费意愿信号？ | 变现形态 |
| KIQ4 | 社区文化对付费的接受度？ | 获客与定位 |
| KIQ5 | 与 LovePageMaker(礼物页)主线的资源冲突？ | 优先级排序 |

**与 09 号报告的关系**: 09 号裁决的是"移动端 App 形态"(iPad 优先/AUv3)，本报告裁决"Web 体验站形态"——平台转移带来的成本结构与获客面变化是本轮新增变量。

---

## 1. 执行摘要（结论先行）

**总裁决: 纯"合成器体验站"(无 AI/无付费/无留存闭环)——不进；"体验站 × AI MIDI"杂交形态——黄灯偏绿(5.5-6/10)，仅作为第二产品序列，在 LovePageMaker 现金流验证之后再启动。**

一句话: **体验站是"别人的营销预算"，AI-MIDI 才是"你的付费理由"——免费合成器当钩子、AI 引擎当商品、DAW 导出当付费墙。**

五条核心结论:

1. **Web 合成器体验站没有付费先例**【事实】: awesome-webaudio Apps 类目 40+ 个在线产品(Pluto Synth/Online Sequencer/Drumhaus/synflow/Binary Synth/mdrone/NoiseCraft/GridSound/mod-synth.io...)全部免费或开源；教学向的 Ableton Learning Music/Synths 与 Chrome Music Lab 是巨头用品牌预算做的免费营销品。**"体验站"作为独立生意无收入模型——它是为卖 DAW/卖品牌服务的，你既没有下游产品也没有品牌预算。**
2. **AI-MIDI 的品类心智已形成且供给在加速**【事实】: r/SunoAI 出现 "midilab.ai is like SUNO for MIDI clips"(72分/35评)；r/Logic_Studio 出现独立开发者的 "AI MIDI generator (Free/BYOK)"(85分/84评)；r/ableton 音乐理论 MIDI 工具(152分)；r/musicprogramming "40篇论文的免费MIDI生成器"(51分)。痛点原话: **"struggled with drum programming"**——AI-MIDI 解决乐理/编曲技能缺失，与合成器的音色能力天然互补。
3. **浏览器端 AI 生成技术地基成熟且处于真空窗口**【事实】: Magenta.js(MusicRNN/MusicVAE/GrooVAE)支持 WebWorker 推理+ServiceWorker 离线缓存，19.8k★；主库 2026-01 归档但已分散到独立仓库继续维护；社区新引擎 SkyTNT/midi-model(Transformer 符号生成，385★)活跃。**"浏览器端 AI 音乐生成"话题在 Magenta 归档后处于真空期。**
4. **合成器社区反付费、崇开源**【事实+推断】: r/synthesizers 年度热帖 12 帖中 8 帖是硬件捡漏/遗物/收藏，软件帖仅 4 帖但全是免费开源发布且千赞(免费 Roland JP-8000 模拟器 2656 分/开源 Juno 1800 分/阿根廷开源合成器 1368 分)。**向该社区卖付费产品必被抵制；但"免费发布"的注意力红利实测巨大——获客靠免费/开源发布，变现靠 AI 引擎。**
5. **AI 音乐服务的价格带已被验证**【事实】: AIVA 免费(版权归平台)→€11/月(标准)→€33/月(Pro 版权归用户)；SOUNDRAW ¥41-125/月(stems 导出分层)。**"版权分层"是已验证的转化引擎**——照抄到 MIDI 形态: 免费生成版权归平台/付费解锁版权+导出。

---

## 2. 赛道结构(Web 端全景)

### 2.1 供给五象限

| 定位 | 代表 | 免费? | 对本产品的威胁 |
|---|---|---|---|
| 教学游乐场 | Ableton Learning Music/Synths、Chrome Music Lab | 全免费 | **高**——教学体验的完美参照物，品牌预算碾压 |
| 玩具沙盒 | Pluto Synth、Online Sequencer、Drumhaus、synflow(浏览器模块化)、Binary Synth、mdrone | 全免费 | 中——分走"随便玩玩"流量 |
| 专业 DAW | BandLab(1 亿注册用户)/Soundtrap | 免费+订阅 | 低——不同场景 |
| 复刻向 | SynthView、Moog/Roland 官方网页 Doodle | 免费 | 低——品牌纪念品 |
| 模块化 | NoiseCraft(1.2k★)、mod-synth.io、GridSound(1.9k★)、AudioNodes | 免费 | 中——最接近本形态的免费替代 |

**关键缺口**: 40+ 清单中**零付费产品**——既是警示(无先例)也是机会(做出付费能力即赛道内零竞争)。

### 2.2 技术地基(可行性)

- **Magenta.js**【事实】: MusicRNN(旋律/鼓)、MusicVAE(多轨/三重奏/和弦条件)、GrooVAE(鼓律人性化)、MidiMe(浏览器内少量数据微调)；checkpoint 托管 GCS；WebWorker 后台推理不阻塞 UI；PWA 离线方案已有先例(Piano Genie)；
- **维护状态**: magenta/magenta 主库 2026-01-06 归档只读，"transitioned to using individual repositories"——**生态未死，工程上仍可用**；
- **新引擎**: SkyTNT/midi-model(385★，事件 Transformer)、LLMidi(LLM 驱动 VST3)、EmoTune-AI 等持续涌现；
- **参照系**: signal(在线 MIDI 编辑器)2.4k★、Scribbletune(3.8k★)——Web 音乐工具开发者生态健康。

### 2.3 AI-MIDI 需求信号(Reddit 实测 2026-10-29)

| 帖 | 分/评 | 内容 |
|---|---|---|
| r/Logic_Studio | 85/84 | "musician and dev who struggled with drum programming, so I built a native macOS AI MIDI generator (Free/BYOK)" |
| r/SunoAI | 72/35 | "midilab.ai is like of like SUNO for MIDI clips, getting really great results" |
| r/ableton | 152/17 | 音乐理论 MIDI 分析+按个人风格生成和弦进行，completely free |
| r/musicprogramming | 51/45 | "Free MIDI generator based on 40 research papers on music theory" |
| r/microtonal | 45/17 | "DAW that has music theory baked into every inch" |

**解读**: "AI MIDI generator"是真实的高互动需求词簇；供给端全部是免费/Freemium 独立开发者；**无人把它与"直接可玩的合成器体验"组合——组合空白是本形态的立足点**。

### 2.4 社区基因(r/synthesizers 年度热帖实测)

- 硬件叙事 8/12(捡漏/遗物/收藏/Rack 装修)；
- 软件叙事 4/12 且全部免费开源发布: JP-8000 模拟器免费(2656分)/Ultramaster KR-106 开源(1800分)/Tolhuin 开源(1368分)；
- **启示**: 免费发布=千赞注意力红利(获客通道)；付费=社区禁区(变现必须绕开社区情感)。

---

## 3. 变现经济学

**价格锚(AI 音乐服务已验证)**【事实】:
- AIVA: 免费月 3 次下载(版权归 AIVA/需署名) → €11/月 标准 → €33/月 Pro(版权归用户+300次/月+WAV)；
- SOUNDRAW: ¥41.58/月起(mp3 无限) → ¥91.58/月(wav+stems) → 企业 API；
- 参照: 09 号报告的 miRack 终身销量折算($75-150万/6年)证明"合成器工具付费"是小众生意。

**本形态的变现设计**【推断】:
1. 免费层: 合成器全功能可玩 + AI 每日 N 次生成(带水印音频/无导出)——体验站层即获客层；
2. 付费层 $8-12/月: 无限 AI 生成 + MIDI/WAV/Ableton Set 导出 + 云保存 patch；
3. 版权分层(照抄 AIVA): 免费生成内容商用需授权，Pro 用户拥有完整版权；
4. 天花板参照: 与 miRack 同量级($12-25万/年)——**手工作坊生意量级，不是创业公司故事**。

**获客设计**【事实+推断】:
- r/synthesizers 免费发布(千赞模式实测存在)；
- SEO: "browser synth / online synth / free synth / AI midi generator"——付费竞品零占位；
- GEO: "browser ai midi"话题真空期(Magenta 归档后)；
- 每台合成器自带 "Learn this knob" 交互教学(对标 Ableton Learning Synths 的钩子体验)。

---

## 4. 产品形态雏形(三明治)

```
首屏 = 免费可玩合成器(体验层/SEO钩子)
  · 复古合成器 ×5 (Juno/808/Jupiter 风格, WebAudio 实现)
  · 每台内置 "Learn this knob" 交互教学
二层 = AI MIDI 引擎(付费层)
  · 风格/和弦/鼓点生成 → 拖进合成器直接弹奏
  · Magenta.js 本地推理(隐私+离线)
  · 导出: MIDI / 音频 / Ableton Set
三层 = 云保存 + patch 分享(留存层, 第二曲线)
```

**MVP 收敛**(若启动): 1 台合成器 + 1 个 AI 鼓点/和弦生成器 + 导出付费墙——约 6-8 周单人工程(比移动端版 18-24 个月轻一个量级，Web 免上架/免 AUv3 适配/免 DSP 移动端优化)。

---

## 5. 一票否决复查与优先级裁决

| # | 否决项 | 结果 |
|---|---|---|
| 1 | 付费容量 < 生存线×10 | **纯体验站命中**(无变现模型)；杂交形态未命中(有 AIVA/SOUNDRAW 价格带参照) |
| 2 | 单一平台依赖 >60% | 未命中(Web 无商店审核) |
| 5 | 单人不可持续 | 黄灯: MVP 可单人(6-8周)，完整三明治需 6-12 个月 DSP+AI 工程 |
| 7 | 法律风险 | 未命中(Magenta.js MIT 类开源/WebAudio 自研；注意 GPL 传染仅在选择 fork VCV 代码时触发——自研引擎规避) |

**与 LovePageMaker 主线的冲突裁决**【推断，依据 08/09 号报告时间窗】:
- 礼物页: 1-2 人 × 6 周可验证，2027-02-14 情人节死线明确，W1 落地页即可开跑；
- 合成器站: MVP 6-8 周 + 冷启动周期，无死线；
- **结论: 礼物页优先级 >> 合成器站。合成器站作为"礼物页现金流验证后的第二产品"，若礼物页 30 天实验失败，合成器站可作为备选启动项(MVP 收敛版)。**

---

## 6. 验证计划(若启动)

| 周 | 动作 | 量化目标 | 杀线 |
|---|---|---|---|
| W1-4 | MVP: 1 合成器 + AI 鼓点生成器 + 免费发布到 r/synthesizers | 发布帖 ≥500 赞；注册留存 ≥30% | 帖 <200 赞且无自然转发 → 杀 |
| W5-8 | 付费墙上线($8-12/月) | 免费→付费 ≥1% | <1% → 杀或改免费+捐赠模式 |
| W9+ | 扩合成器至 5 台 + patch 分享社区 | 次月留存 ≥40% | — |

**参照物**: miRack 六年 677 美区评分 / Endlesss 拿融资仍关停(反面) / Koala Sampler 独立口碑(正面)。

---

## 7. 来源索引与置信度

**【事实】高置信**: 09 号报告存量事实(VCV 无移动版/miRack/Drambo/市场规模/Endlesss 关停)；awesome-webaudio Apps 类目清单(40+ 条目实测)；GitHub webaudio 主题(NoiseCraft 1.2k/Strudel 3k/Glicol 3k/GridSound 1.9k/Signal 2.4k/Scribbletune 3.8k)；Magenta.js 官方文档(模型清单/WebWorker/离线方案)与主库 2026-01-06 归档状态；SkyTNT/midi-model 385★；SOUNDRAW/AIVA 官网定价页；Reddit 三连实测(r/synthesizers 年度热帖/AI-MIDI 关键词搜索，2026-10-29)。

**【推断】中置信**: Web 合成器站天花板与 miRack 同量级；三明治形态的转化率假设；免费发布千赞的可复制性。

**【数据缺失】**: Ableton Learning Synths 用户规模(无公开数据)；midilab.ai 收入；BandLab 音乐工具内付费转化率。

---

## 8. 附: 本轮调研过程记录

- GitHub topics/webaudio 与 topics/midi-generation 页面抓取；
- awesome-webaudio readme.md(master 分支，经 GitHub API 定位小写文件名)Apps 段全量提取；
- magenta/magenta 仓库归档状态核验 + magenta-js/music 官方文档；
- SOUNDRAW / AIVA 官网定价页直读；
- Reddit 登录态浏览器实测: r/synthesizers 版块搜索("browser OR web synth")、全站搜索("MIDI pack OR midi generator OR AI midi")——前者 25 结果、后者 22 结果，逐条读取标题/分数/版块；
- 直连失败说明: pluto.live / plutosynth.com / strudel 官网 / signal.vercel.app / awesome-webaudio 主域名 / Google 搜索在本网络不可达，已用镜像与替代源补齐，无证据缺口。
