# 04 · AIGC 科研绘图: 竞品动作与社区态度的精确边界

> 数据源: 帖12/25/26/28 + 各帖内 AI 相关评论交叉 + 搜索快照(AI 簇部分被限流, 用子版块与既有帖子补偿); 引用格式 [帖NN|用户名|分数]
> 结论概览: 科研社区对"AI 画图"的态度不是反对 AI, 而是反对**不可验证的语义**。禁区在期刊发表环节, 缓冲带在概念草图/图标/教育场景; 供给侧的真正空位是"AI 生成但可编辑可验证的矢量输出"。

## 一、社区态度光谱(带原话)

### 1.1 嘲讽/娱乐化(最大声量)
- Frontiers "巨大老鼠" AI 论文撤稿事件成为 meme 母题: "That rat is fucking hung. It's like Doug Dimmadome's hat." [帖25|thecrowphoenix|37]
- 精神分析式的失败机理描述(值得全文引用):
  > "There are letters but they don't mean things. It has aesthetic features my brain expects from a diagram of a signalling pathway, but arranged nonsensically. We got some transmembrane noodles and a rubber duck, and it's like the model kind of gets that certain things should be paired or that they eventually dimerize. But it's a hot mess" [帖25|MakeChinaLoseFace|5]
- "AI Generated Scientific Figures" 在 r/behindthebastards(52分) 是纯娱乐帖 — AI 科研图已经是公共笑料品类

### 1.2 道德抵制(组织化出现)
- "I will draw figures in illustrator for free for people just to not use AI, I'm begging you" [帖19|raeurae|1, 药理四年级博士]
- "Inkscape isn't ai slop" / "Then get skills" [帖26|3dprintingn00b|10/4]
- AI 图工具(Illustrae.co)在 r/labrats 推广性发帖 ratio 只有 0.36 [帖26] — 直接的社区排斥信号
- 系统化反对理由(整理自长评论): "People here aren't fans of AI use in the sciences... AI tends to hallucinate a lot and doesn't understand context and nuance around science, so you end up generating something that looks reasonable but is often inaccurate" [帖26|You_Stole_My_Hot_Dog|3]
- "那些 AI 应用哥们该跪拜真正的工具开发者": "Those ai app chuds should bow down and worship you"(对确定性代码工具的敬意) [帖44|Enigmatic_Baker|9]

### 1.3 实用主义采纳(沉默多数)
- "This is one problem that AI has solved for me. Build the basic analysis and then tell AI to create the figure. Still needs hand holding but much quicker" [帖14|noma887]
- 十年老手也倒戈: "I have about ten years of experience making graphs with matplotlib... it's still faster and easier to have Claude Code make my plots. I'd be EXTREMELY cautious about asking an LLM to do your stats without closely checking it, but 'plot this .csv' followed by telling it how to adjust it visually is something they do exceptionally well" [帖18|iggywing|5]
- "It's literally faster to ask an LLM to make a plot then tweak it as I want than to remember what bespoke syntax I need. Is it lazy? Yes. But ya boi has grants and papers to write." [帖18|1337HxC|2]
- 图标级使用: "I use Illustrator + ChatGPT for icons... good for basic icons" [帖19|ParticularBed7891|4]; "controversial take: i have a gpt subscription, it can generate icons that i then assemble in powerpoint" [帖19|Augustthesecond|0分被踩但坚持]; nano banana 实例(酵母 SCRAMBLE 示意图) [帖01|molecular_data]
- 学生写码刚需: "Python with chat gpt generally carried my doctoral thesis" [帖17|Krazoee]; "No need to learn R. Just ask Claude to write the code"(被赞 17 分的中间派: "Just make sure you actually read the code") [帖17|MakeLifeHardAgain/Spacebucketeer11]

### 1.4 质量守门人(学术信任结构)
- "Have a couple of students and postdocs trying to fudge their way through R with AI, and it's an unmitigated disaster. One postdoc actually showed up to a meeting with data misrepresented and axes mislabeled... learn R enough to understand how it works. Then use AI to help only when you're able to implement the resulting code and understand what it's doing." [帖20|gertalives|10]
- "I wouldn't trust them. Definitely wouldn't publish them since I couldn't verify how the data was handled." vs 反驳 "It gives me the script and I can see exactly how the data are handled" → 关键分歧点 = **输出是否可审计** [帖18|Zeno_the_Friend vs mtnsbeyondmtns]
- "AI does decent work, but it retains that distinct 'look' regardless of what you tell it to make" [帖19|FieryVodka69] — 风格同一性问题在 AI 上同样成立

## 二、期刊/资助合规边界(产品生死线)

- BioRender 官方话术(两处一致): "most journals and grant funders don't accept AI-generated figures for final submissions. So AI can be helpful for early concepting, but a compliant illustration tool is needed to publish... check whether AI tools train on your data (we don't)" [帖01/帖32|Avery-at-BioRender]
- 社区对合规的真实感知分层:
  - 发表 = 禁区: "It gets complicated with publication though unless you purchase the illustrations. For talks/posters it is probably fine." [帖05|FTP4L1VE|2]
  - 演示/教学/组会 = 宽松区: "I suppose if you were *just* using it for yourself it's not much of an issue, but don't use it for any reports, publications, or communications" [帖26|You_Stole_My_Hot_Dog]
  - 概念草图 = 鼓励区: "AI can be helpful for early concepting"(官方) 与 "I generate rough drafts... with AI" 多处
- 学术界自己的信号: 研究者主动区分 established vs uncertain [帖12|Muhlwa_Sholanke: "Most science explainers smooth over the line between established and uncertain. Glad this one labels it instead."] — **不确定性标注是可信度的核心**
- 用户的成熟怀疑: "I know AI-generated scientific content should always be verified against the literature, so I'm mainly interested in using it to create high-quality illustrations rather than as a source of scientific information" [帖28|OP] — 付费意愿存在, 但建立在"我知道要验证"的前提上

## 三、AIGC 竞品动作盘点(从帖子与快照汇总)

| 玩家 | 动作 | 社区反应 |
|---|---|---|
| BioRender | Generate Custom Figure(2026-03, 文本→3稿) / Restyle Icon(2026-07, 图标风格迁移: 水彩/彩铅/3D/线稿等8种) / Graphing & Analysis(2026-05-21) | 官方话术强调"不训练你的数据+合规"; 社区对其涨价与条款的怨气未消 [帖01/帖32/帖51] |
| PaperBanana | 开源文本→科研示意图(避免"mouse giant fantasy testis"式崩坏) | "layout rendered actually decent"; 缺矢量输出被吐槽 [帖01] |
| Anima(Claude Code skill) | 概念→交互式 SVG 科学图(等距几何+引用+确定性标注) | r/ClaudeAI 193分, "Most science explainers smooth over the line between established and uncertain. Glad this one labels it" [帖12] |
| Illustrae.co | 提示词生科研图, 进 r/labrats 推广 | ratio 0.36, 被视为 slop [帖26] |
| nano banana(Gemini) | 学术免费一年, 被用于实验室示意图 | 正面个案 [帖01/帖04] |
| figurelabs | 文本式生图(免拖图标) | 被指 hidden advertisement; "credits to upload my own files = hard nope" [帖01] |
| OpenTikZ | "Flaticon for academic TikZ"(CC0 图标+agent 编辑模板+PNG→TikZ+描述→TikZ) | r/LaTeX 47分, "Hand-writing TikZ is slow; AI generated diagrams are hard to edit and had to get vector quality" [帖40|OP 自述痛点] |
| DeTikZify / octree tikz-generator | 图→TikZ 代码 | 小众工具持续出现 [帖41] |
| 独立 vibe-code 图标生成器 | biographics.nitro.bio(开源许可+付费AI图标) / Laprablenia 自用版 | 个人可自建生态 [帖24/帖31/帖32] |
| (背景)Adobe Illustrator | 内置 AI 图像生成("gets dumb things wrong all the time, but can work as a starting image") | 当草图垫子用 [帖19|geneticats] |
| SciSpace | 学术 AI 全家桶(Agent/文献综述), 订阅+币值双轨 | **计费信任崩塌案例库**: r/academia 主帖 "Avoid SciSpace – Extremely Misleading, Overpriced, and Potentially a Scam"(32分/43评) [帖72]; "coins burn ridiculously fast... can't see coin pricing unless you're already a paid subscriber"; "a few were clear fabrications — journals, years, and authors that simply do not exist"(伪造引用); 退款陷阱("100% money-back guarantee" 但限 300 tokens 内); 官方 2026 年才上线 unlimited+Verified Report 补救 |
| Napkin AI | 通用文本→视觉(演示/营销向) | **科研社区自然声量≈零**: 搜索仅命中 LinkedIn 营销帖与泛工具清单, 无任何科学用户自发讨论 — 该品类在科研场景是空白而非强敌 |
| Mind the Graph | BioRender 低价模仿者 | 同样**自然声量≈零**(搜索全是噪音), 仅在 BioRender 讨论串被顺带提及 "similar to biorender but a little cheaper" [帖13] — 存在感问题比定价问题更致命 |
| splotch.ink / Jotform AI flowchart | 通用 AI 流程图 | 教师侧顺带推荐 [帖71] |
| 教育/教学侧 AI 需求 | 教师白板示意图: "accurately takes a prompt and can make a science diagram" | 社区建议用 ChatGPT+Python 而非图像生成("as those can contain inaccuracies") [帖71] — 教育场景同样以准确性优先 |

## 四、AI 输出的四大结构性缺口(全部有原话背书)

1. **非矢量**: "very annoying it doesn't output to svg/PDF/vector graphics. Tracing to convert to vector graphics from PaperBanana's PNG/JPG output is awful" [帖01|pastaandpizza]; "The biggest issue I have is the AI images aren't vector files" [帖19|ParticularBed7891]
2. **语义不可验证**: "optimized to produce sophisticated-looking images rather than clear scientific communication" [帖19|OP]; "aesthetic features... arranged nonsensically" [帖25]
3. **不可编辑**: "AI generated diagrams are hard to edit" [帖40|OP]; 用户理想 = AI 出草图 → 自己可控精修
4. **风格泯然**: "it retains that distinct 'look' regardless of what you tell it to make" [帖19|FieryVodka69]; 与 BioRender 同质化焦虑同构

## 四点五、学术 AI 工具的"信任崩塌"完整案例: SciSpace(帖72, 2025-11 → 2026-10 全程)

这是本研究最有完整弧线的负面样本, 对任何"学术+AI+订阅"产品都是必读反面教材:

1. **产品定位与现实的落差**: "SciSpace markets itself as an academic tool, yet it hallucinates more [than ChatGPT]" — 学术标签抬高了验证标准
2. **伪造引用是存在性危机**: "A bunch of them were completely unrelated, Some were nonexistent, And a few were clear fabrications — journals, years, and authors that simply do not exist" — 对科研工具, 引用造假 = 直接死刑判决
3. **订阅+币值双轨制被解剖**: "You pay for a subscription, and then it still uses a 'coin system.' The coins burn ridiculously fast... priced higher than the subscription itself. Worst of all, you can't see the coin pricing unless you're already a paid subscriber"
4. **退款暗黑模式**: "You can cancel anytime. All plans have a 100% money-back guarantee for 24 hours. This is bullshit! A refund is only available if you consume just 300 tokens" + "24 hrs to catch their payment or no refund"
5. **官方补救的节奏问题**: 投诉集中于 2025-11 → 2026-01, 官方(SCiSpaceSupport)2026-06 起才在该帖逐条回复, 2026-10 上线 unlimited Agent + Verified Report — **补救周期≈11 个月, 期间每一轮搜索"SciSpace"的用户都会先看到处刑帖**
6. **社区验证方法论**(帖54 的回答与帖72 互相印证): "start with a tiny dataset whose group means and sample counts you've calculated by hand. check those against the plotted values... readable code can still implement the wrong calculation"

## 五、可执行的定位启示(供 05 报告引用)

1. **"AI 在环路里, 不在出版链上"**: 生成→可编辑矢量→人手终审, 把 AI 放在 draft 阶段而非 submission 阶段, 与官方话术和社区道德同时兼容
2. **可验证性即卖点**: 语义校验(图标↔实体映射表)、引用溯源、established/uncertain 标注——Anima 的被赞点就是可复制的路线
3. **图标库+AI 的正确组合**: 不是"AI 画一切", 而是 AI 补长尾图标(植物物种/物理仪器缺口, [帖19|Augustthesecond]), 主体仍是可编辑矢量资产
4. **教育与替身场景先行**: 学生报告/组会/课程 = 低风险高频场景(帖26 OP 的真实需求就是"教育用途的简单示意"), 从这里建立习惯再向出版级爬升
