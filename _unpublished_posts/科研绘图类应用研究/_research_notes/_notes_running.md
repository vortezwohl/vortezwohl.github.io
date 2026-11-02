# 跑动笔记(研究中间洞察, 供最终报告引用)

> 状态: 截至 2026-11-02, 已通读 帖01-帖22。本文件为增量洞察池, 最终报告写成后可归档。
> 引用规范: [帖NN|用户名|分数] 均为原话直引, 存档于同目录 帖NN_*.md

## A. BioRender 生态(帖01-10)

### A1. 定价愤怒是第一主题(全时代最稳定信号)
- $35-42/月个人版被反复称为 ridiculous / kidney-money / insane [帖01|CTLeafez|5; 帖13|East_Tumbleweed2447]
- 实验室订阅续费时 PI 抱怨成本是常见场景 [帖01|WildPieee 开帖原文]
- 免费版+截图 是普遍灰色工作流: "I know a lot of people use the free version and screenshot the results" [帖01|JoshFungi|12], 西语用户描述开飞行模式/虚拟卡/盗版等规避手段 [帖01|VaultzzU]
- 学生/自费 vs 机构付费的态度分裂: "I love it as long as my department pays for it" [帖04|deleted|23] vs "expensive to pay for yourself" [帖04|labratinthelibrary]
- 隐藏定价: "talk to their chat support folks, there are hidden pricing tiers for students and postdocs" [帖07|anomalousdiffraction|8]; 学生版 PO/经费采购流程反而更贵更麻烦 [帖07|cornholio702|11]
- 试用转付费陷阱: 忘关试用被扣 $225 且"必须提前一天取消否则完蛋" [帖07|masoca1]; 零登录零导出仍被扣 7 个月 $273, 只退最近一月 [帖06|lakekat] → 计费策略是口碑雷区
- 生效中的价格冲击: Prism 2026-10 涨价 127%, 大学开始推荐 BioRender 代替 [帖18|OP]

### A2. 审美同质化/风格识别 = 生态反噬
- "I can identify their common icons very easily and they just look so boring... like Excel default colors" [帖01|SynbiosVyse|2]
- "paper with figures made from it are so recognizable and drab" [帖01|FireG99]; "modern version of clipart, I'm starting to see the same exact biorender objects in every paper" [帖07|limpbizkit6|13]
- "it's the program that has caused everyone's figures to look the same in the past ten years" [帖06|boarshead72|18]; Canva 类比 [帖06|Bach_Chemist]
- "too cliche due to being overused and it's so easy to tell" [帖02|OP]; "It fills a niche but the images are now over used. I use it, but damn i don't like seeing it anymore" [帖04|TheTopNacho|13]
- 反面证据: BioRender "significantly increased the legibility and visual aesthetics of publication drawings" [帖02|MysteriousMacrophage|127, 全帖最高赞]
- Style 可识别性双向刃: "you can see if something was done with biorender by the style" [帖05|FTP4L1VE]; TikZ 同样被风格识别 [帖22|PersonalityIll9476: "I'd recognize that style anywhere"]

### A3. 授权/条款是持续流血点(法律焦虑)
- 引用要求争议: 2024 前要求 caption 内引用 URL 被骂 "walking advertisement"/"predatory" [帖05|OP]; 2024 放宽为 caption/references/acknowledgements 任一 [帖05|emath99 引官方帮助页]
- DPI 门槛: 免费版导出分辨率低到期刊(300-600dpi)不可用, watermark 移除是付费功能 → "免费版实际不可用于发表" [帖05|toastedbread47|22]
- 发布许可条款: "需要他们特别许可才能发表付费账户创建的图" → 用户直接流失 "Stopped using their services... Now I am happy with free Inkscape" [帖04|MrBacterioPhage|17]
- ToS 细节恐惧: 仅第一/共同第一作者可发表含 BioRender 内容的图 [帖02|born_to_pipette 引 ToS 原文]
- Open Access 兼容性: BioRender 许可被指 "nightmare and infectious and a trap", 与 CC 许可冲突, 分享者对下游使用负责 [帖19|DELScientist|29]
- 公司滥用: 公司用 basic plan 做营销物料, 员工焦虑是否举报 [帖03 全帖] → B2B 合规市场存在

### A4. 免费生态位图谱(用户实际推荐清单)
- NIH BioArt (bioart.niaid.nih.gov): 出现频率最高的免费图标库 [帖01/04/05/19/13]
- bioicons.com (可加 draw.io 扩展) [帖01/04/05/19]; smart.servier.com / Servier Medical Art [帖01/04/08/13]; phylopic.org [帖01]; labicons.net [帖01]
- SBGN + Newt editor(分子机制图示标准) [帖08|bergsonilhemgh] — 小众但刚需
- 图标引用负担: "7 icons = 7 additional citations. Are people actually doing this?" [帖19|1337HxC] — 免费库的隐性合规成本
- 覆盖缺口: 植物物种只有 6 种且 3 种是核桃花生 [帖19|Augustthesecond]; 物理科学几乎空白 [帖07|nthlmkmnrg|24 "wish they had something like this for the physical sciences"]
- "Someone needs to just sit down and generate a suite of like 300 similarly-styled icons and publish it" [帖19|pacific_plywood] — 用户亲口说出产品机会

### A5. PowerPoint 是隐形霸主(不搜索不知道)
- 127 分高赞: "I draw them in PowerPoint lol" [帖02|MysteriousMacrophage]; "PowerPoint is the new photoshop!" [帖02|lea949]
- "A lot of PIs just make figures in powerpoint lol" [帖08|stray-lyght]; 合作社用旧工具绑架: "still use ppt for the sake of (ahem older) collaborators" [帖19|to_fit_truths]
- 甚至 Excel 画图 [帖02|Scr33ble] + 注册表改 600DPI 导出的黑魔法 [帖13|ginngeer]
- r/okbuddyphd meme: "PowerPoint figures have weak aura" (456分) [帖25 待读]

### A6. 矢量/导出痛点(结构性技术缺口)
- BioRender 付费版导出仍非真矢量: "even the paid version of BioRender only allows for exporting rasterised images"; PDF 导出也非矢量 [帖07|xRyuuzetsu; 帖07|born_to_pipette]
- AI 生成图同样卡在位图: "very annoying it doesn't output to svg/PDF/vector graphics. Tracing... is awful" [帖01|pastaandpizza on PaperBanana]; "AI images aren't vector files so it can be a pain to remove backgrounds" [帖19|ParticularBed7891]
- 用户补救工作流: BioRender→高清PDF→Illustrator image trace→矢量 [帖07|limpbizkit6] — 痛苦且低质
- matplotlib/MATLAB 草图 → SVG → Inkscape 精修 是编程党标准管线 [帖19|Incandescent_Banana|11]

### A7. AI 分歧在科学社区极其尖锐
- 反对方: "AI figure generators 输出是 LinkedIn image slop... optimized for sophisticated-looking images rather than clear scientific communication" [帖19|OP 核心洞察]; "I'd rather give biorender the cite than Anthropic" [帖05|matertows]; "I will draw figures in illustrator for free for people just to not use AI, I'm begging you" [帖19|raeurae]; 审美层面 "about as immediately recognizable as AI artwork" [帖01|mcgarad]
- 支持方: "This is one problem that AI has solved for me. Build the basic analysis and then tell AI to create the figure. Still needs hand holding but much quicker" [帖14|noma887]; Illustrator+ChatGPT icons 组合 [帖19|ParticularBed7891]; nano banana(Gemini) 画酵母 SCRAMBLE 示意图实例 [帖01|molecular_data]
- BioRender 官方的话术(重要竞争情报): "most journals and grant funders don't accept AI-generated figures for final submissions. So AI can be helpful for early concepting, but a compliant illustration tool is needed to publish... check whether AI tools train on your data" [帖01|Avery-at-BioRender] — 官方把「合规+不训练你的数据」当护城河话术
- 社区对官方话术的逆反: "Generative AI will eventual kill their business model" [帖05|RedditBResearch]; "There's zero reason to use biorender anymore... use AI to make the image in vector format" [帖05|ProteinEngineer]
- 克隆呼声: "Someone do this with biorender" (ChemDraw 克隆帖下 51 分高赞) [帖16|chimera4373]

### A8. 开源克隆潮已开始(供给侧信号)
- betterbiorender.com [帖03|victorz96]; americaIsFuk 自建 12k 开源图标库, 判断 "proprietary icons/clip art 才是 BioRender 真正的护城河", 提出社区驱动非AI图标贡献 [帖05|americaIsFuk]
- DrawPlotia(PhDumb 自推), 多个自荐帖 [帖05]; "I hope someone Opensources BioRender like what happened to Adobe this week" [帖05|Informal_Dark_3870] — 指向 Affinity 被免费化事件
- ChemDraw 已被 vibe-code 克隆潮淹没: reshiki.com, bondcraft.net, chempirical.com, enlacecraft, ErezDraw, ChemDraft, Reactio Draw(AGPL, 支持CDX/CDXML兼容+Word集成) [帖15/16]
- "ChemDraw became too big to care about a user's needs, and now it's easier to make your own version" [帖16|resonanse_cascade|32]
- 克隆潮的边界: "It really takes a lot of time to get basic features working, so I don't think vibecode will fully replace paid software" [帖16|OP 回复]; "I want reliable software with predictable functions, not to wait for Claude to code it from scratch every time" [帖16|Grand-Tea3167 反方 → OP 回击]
- 社区识破软广的眼睛很尖: figurelabs 被指 hidden advertisement [帖01|bzbub2|10], "credits to upload my own files, hard nope" [帖01|1337HxC]; NoahAI/Lucis/Nanobanana 一句话评论疑似水军 [帖19|Specialist_Ride_8072; 帖04|Adept_Yogurtcloset_3 等]
- OP 本身可疑: 帖14 OP(Motor_Cash6011) 在每条回复下追问"用什么工具...我可以分享链接/视频" = 典型 stealth marketing [帖14 多处]

### A9. 主流工作流画像(多帖交叉验证)
- 编程党: R/ggplot 或 Python/matplotlib+seaborn 出图(+统计), 矢量导出; Inkscape/Affinity/Illustrator 组版精修; 图标来自 BioRender/免费库
- GUI 党: Prism(数据+统计+图) / Origin; PowerPoint 组版; BioRender 画机制图
- 混合管线是常态: "matplotlib → SVG → Inkscape" / "R → PDF → Affinity" / "BioRender 图 + Illustrator 精修"
- 记忆点: "ggplot in R... I recognize ggplot2 in practically every paper in my field" [帖13|noodles0311] — 又一个风格同质化证据
- MATLAB 许可断供风险: "Good luck recreating any of your work after you leave academia and do not have the license" [帖13|DerBanzai]; 离开学界后 Prism 同样断供 → 长期可复现性焦虑是 R 迁移的核心动机 [帖17|deleted|123: "I knew when I left my grad lab that I wouldn't have access to Graphpad... with R I could continue to work on revisions"]

### A10. 学习曲线与知识传授缺口
- "做图 skill issue" 争论 [帖14]; 有人 PhD 期间专门修科研图形设计课并称 "one of the best things I did" [帖14|MisterBreeze]
- YouTube/课程需求: R Programming 101 publication-quality ggplot 系列 [帖13|amazngsgr]; "Which courses or tutorials are genuinely worth the time" 是 OP 的问题清单 [帖13]
- 高频具体技巧在评论区口口相传: PPT 改注册表出 600dpi, ggprism 主题, ggsignif 显著性标记, patchwork 拼版, ggh4x forcepanelsize, colorbrewer/cividis 色盲安全色

## B. Prism/数据绘图定价与迁移(帖17/18/20)

- Prism 年费 ~$990-142-260/区段 + 2026-10 涨 127%; 学生半价需 PI 信头纸 [帖20|RedBeans-n-Ricely]; 试用叠加 bug(3设备×3邮箱) [帖17|ScaryDuck2]
- 盗版完全常态化且被高赞: "ARRRRRR" 207分 [帖20|Azylim]; "My whole PhD was run on cracked versions" [帖20|unbalancedcentrifuge]; Taobao $2-4 [帖17|Commercial-Loss-5117; 帖20|SeasonComprehensive7]; Prism 8 crack 在 ResearchGate 传播 [帖17|Substantial_Gene_15 49分]
- 但社区主流建议仍是学 R(225分 top): ggprism 包让 ggplot 输出 Prism 风格 — "Prism 的外观"本身是被渴望的审美资产 [帖20|aiwendil22|109; 帖17|azbkthompson|27]
- 反方真实存在且强烈: "R and Python are so ridiculously hard to use compared to Prism they shouldn't ever be compared... I have been dabbling for literal years in R and still couldn't get exactly the graphics I want" [帖20|TheTopNacho|31]; "Saying 'just learn R' is blatantly insulting" — 快速小图场景 GUI 不可替代 [帖18|YYM7: 12个样本柱状图, GraphPad 粘贴比 jupyter 快]
- 统计正确性焦虑(双刃): Prism 被喷让博士不会统计 "990USD/year license... wrong analysis because it's so easy to click two buttons" [帖20|Enzymology2491]; 但 AI 生成 R 代码同样被资深科学家打回 "data misrepresented and axes mislabeled... learn R enough to understand how it works. Then use AI" [帖20|gertalives|10]
- 文件格式锁定: "We share prism files to update graphs... your 49 year old boss doesn't care and just wants the prism file" [帖18|Boneraventura|1, 真实感最强的一条]; 公司 "slave to prism" [帖18|ZachF8119]
- BioRender 2026-05-21 上线 Graphing & Analysis(吃 Prism 地盘): CSV/XLS/Prism 文件拖入, 统计检验推荐+解释, 异常值一键剔除, Style Match, 实时协作, SOC 2; add-on $20/4mo 学生 [帖18|Avery-at-BioRender 多条] → BioRender 正从插图横向扩张到数据图
- 独立替代品死亡样本: figurezen.com 发布于 2026-07 推荐帖, 2026-11-02 已 404 [帖20|TrainingAdvance1551→AdCommercial3569 复查] — 4个月寿命
- 活着的独立挑战者: LabPlot(最像 Prism 界面), JASP/jamovi(统计), SimplePub(浏览器SVG无账号, 但无误差线/统计), Graphitix(开源 .graph 格式, 自曝 rough edges), Excel 内嵌免费 Prism 替代(r/pharmacology 自推: WB定量/qPCR ΔΔCt/IC50/ELISA 拟合)
- OriginPro 无家可归: findareddit 专门问"有没有讨论 OriginPro 的 subreddit" — 社区真空 [快照|OriginPro]; 需求散落在 Chempros/PhD/材料
- Igor Pro(r/matlab 8评): 与 Claude AI 结合的尝试出现

## C. 通用痛点(帖11/13/14/19/21/22)

- 时间黑洞: "days if not weeks making the figures" [帖14|g33ksc13nt1st]; "hundreds of hours go into every set of figures for a major paper" [帖14|limpbizkit6]; "congrats, you're now at the halfway point"(数据齐=论文一半) [帖14|IHTFPhD|9]
- 图=论文的门面与偏见: "If they don't care how the figures look like... what else didn't they care about? I typically put those in the bin" [帖14|g33ksc13nt1st] — 审稿人真的以图取人
- 机构级需求信号: "suggesting they hire a full time figure maker... enough interest to keep them fully subscribed" [帖14|Rhawk187]
- 可复现性/一致性: "create matlab scripts so I could regenerate them quickly... if I needed to relabel anything, change colours/fonts, increase resolution" [帖11|GoodMerlinpeen|10]; "How does everyone make figures and tables look the exact same?!" [帖14|Meizas]; 模板自存是民间解法 [帖18|TheTopNacho]
- 无障碍约束: 色盲安全色/黑白打印/纹理替代 [帖14|Efficient_Rhubarb_43; 帖21|2kgweight|27]; cividis/colorbrewer 口口相传 [帖21|Bohrealis]
- 海报文化冲突: "Why are scientific posters required to be ugly?" [帖21]; "attractive posters also attract people" [帖21|Donut_Earth]; better posters 资源链 [帖21|alexa-488]
- TikZ 风格识别与学习曲线: gold standard + time sink [帖22]; 生态工具: q.uiver.app(画图→TikZ), tikzit, yEd, GraphViz, typst+cetz(新派), Python→TikZ transpiler(84分, 6年前) [帖22]
- 字体/排版: "same type face, stroke size, alignment issues" 是高影响论文的统一特征 [帖14|limpbizkit6]

## C2. 批次3补充洞察(帖23-52)

### 许可规避的民间标准操作(重要行为数据)
- 「发表前买 1 个月许可」= 社区公开传授的合规做法: "just buy a licence for 1 month before publishing... about 40USD for the month" [帖24|gene100001]; "My understanding is yes (and it's what a lot of people do)" [帖30|deleted] — BioRender 的按月定价事实上催生了"发布月付一次"模式
- 砍价有效: 写信说明博士生贫困被降到 $15/月 [帖24|ArkanaeL|11]
- 免费版高清导出 hack: "Blow the image up as large as you can on your screen, zooming in, then screenshot"(103分置顶) [帖30|2024account]; 多账号轮试用 [帖30|Shh04]; "If I would be allowed to use lab money to pay I would, but fuck using my own personal money to pay for work stuff" [帖30|omgu8mynewt|14 — 个人垫付费的道德怨恨]
- 描摹规避: "If you really like a biorender icon just trace over it in your graphics software... only takes like one minute"(知识产权灰色建议被公开给出) [帖24|Spavlia|7]

### BioRender 图标的科学准确性风险(新发现的主题)
- "students insist on using biorender, they often also have the wrong cell types or there is something wrong in their diagrams, which at best looks a bit silly in a lab meeting and at worst makes them look bad at a conference" [帖30|Chidoribraindev|9]
- 实例: 蜱虫幼虫图缺失 → 用户涂掉若虫的后腿冒充幼虫 → 学术会议上被当众指出 [帖30|CrisperWhispers|7] — 通用图标诱导科学性错误, 这是"图标库依赖"的深层代价

### 免费/开源资源全景表(帖42 的完整清单, 最权威的一手汇总)
- 图标库: NIH BioArt / BioDraws / SciDraw / Servier SMART / Biosketch / Iconify / Health Icons / Bioicons / OpenClipart / PhyloPic
- 工具: Inkscape / Penpot(协作) / SVG-Edit / Affinity Designer(已并入 Canva)
- 该帖 OP 明确在寻找: 许可坑(期刊发表/商业课程/经费deck) — 许可咨询是真实需求 [帖42]
- "There is still no good website that provide most of the life sciences illustrations"(BioDraws 作者自评) [帖23|Steamed_Momo] — 供给侧自认不完整

### Inkscape 的 UX 批评(免费冠军的软肋)
- "Inkscape suffers from programmer UI syndrome" [帖24|theScrapBook]
- 学习成本被反复承认但被"所有权"价值对冲: "You'll never get caught by institutional licencing, the skill is with you for as long as Inkscape is viable...I should donate to them" [帖24|deleted|16]

### 独立开发者动向(供给侧实时观察)
- biographics.nitro.bio: 开源许可+付费 AI 图标生成, 在多个帖子里自推 [帖24/31/32]
- DrawPlotia/OpenBrender(PhDumb): 聚合 PD/CC0 图标源+SVG编辑+PDF/PNG/SVG 导出+基础统计, Claude Code vibe-code 而成, 双子版块连发 [帖33/34/51]
- OpenTikZ: "Flaticon for academic TikZ", CC0, 四模式(单图标/改模板/PNG→TikZ/描述→TikZ) [帖40] — 其 OP 自述痛点极为精准: "Hand-writing TikZ is slow; AI generated diagrams are hard to edit and had to get vector quality"
- QuickStats(WebR 浏览器内 R): 89分但 ratio 0.84 — "Uggghhhh another vibe coded slop"(41分置顶) + "You can tell it's vibe coded as the app uses the same default colours that Claude uses"(**Claude 默认审美已成为可识别指纹**) [帖57]; 差异化质问 "What is the added benefit over JASP and Jamovi?"(17分); "publication-ready"宣称被直接挑战
- 分域专用包的巨大好评: chemdiagrams(反应能级图) 390分 ratio 1.0 — "Hated making these by hand. This is sweet" [帖44] — 垂直图型自动化是深水区金矿

### 期刊把图重排的现实
- "the journal remade it to fit their style guide anyway. I should have just submitted the original pencil sketch" [帖32|Big_Knife_SK|9] — 精修投入可能被期刊重排版部分抵消
- dpi 执行不一致: "the journal had a strict 600 dpi minimum... they just went ahead and published the low-res ones" [帖39|jmhimara]

### 教学缺口的原话级确认
- "every important tool I've only learned about through tribal knowledge... It's so dumb that th[ere is no formal teaching]" [帖47|howtotailslide, 末年博士]
- "Prism has infected my entire field... I hate those stupid stars"(显著性星号文化) [帖47|therealityofthings]
- "you can always spot an Excel graph. And the analysis accompanying the Excel graph is almost always inadequate" [帖56|rangom1]
- 经典书目在评论中自发流传: Tufte / Wilke(Fundamentals of Data Visualization) / Cairo(Truthful Art) / storytellingwithdata [帖39/56]

### 图形摘要(GA)的功能性证据
- 有机合成领域 GA 高价值: "it's kinda fun... quick to figure out what transformations they are doing" / "the difference between 'exactly what I needed' and 'this is way outside my scope'" [帖27]
- 物理/分析化学 GA 质量差: "often look like MS clipart with some equations and a picture of a distribution floating around 🥲" [帖27|7ieben_]
- 分布之痛: RSS 不带 GA 链接 [帖27]
- GA 差异化动机: "Since the rampant use of biorender, I feel like the scientific illustration became boring and template-based" [帖51|OP]; BioRender 对策= Restyle Icon(8种风格迁移) [帖51|Avery]
- 忙碌医生场景: "Have to make a graphical abstract... Am a busy doctor and don't have much interest in learning graphic design" → PPT→PDF矢量/BioRender模板/外包(Fiverr) [帖48/31]

### 工程制图侧(R/CFD/ECE)
- 工程: Visio/PowerPoint/draw.io/yEd/Bluebeam/CAD导出 [帖35]; "companies don't care much for beauty in internal documents"(工业界 vs 学术界审美分层) [帖35|jakkemaster]
- CFD: ParaView/Tecplot/Fieldview + IPE + Inkscape(+TeX插字) [帖38]
- 3D 示意: Blender 是统一答案但学习曲线陡 [帖37/45]

### Origin 生态(帖43)
- Origin→Illustrator 导出精修管线 [帖43|Ok-Ease5589]
- Origin vs Python 阵营对立: "Origin is only good if you can't program basic plotting" vs "OriginPro can significantly accelerate your workflow... I recommend my students to learn both" [帖43]
- 学术诚信边界争论: 峰位偏移可视化被指 "intellectual dishonesty"(后自认过激) — 可视化伦理是敏感区 [帖43]

### "publication quality" 的定义(帖39, 概念锚)
- 主流理解=印刷复现性而非分析价值: "A lo-res bitmap is not 'publication quality' no matter how perfect it is as a tool for quantifying... A hi res piece that's a terrible display (dynamite plots) can be 'publication quality'" [帖39|efrique|8]

### 物理基础图重画浪费(机会级引语)
- "Imagine all the things academics could accomplish during the time wasted drawing the same diagrams that have been drawn a million times already... We need some sort of stock bank for this shit. *See Standard Motivation D.1A for context*" [帖50|chuckie219|1]; OP: "At least for figures a public bank would be epic"

## D. 待验证假设(供后续批次与报告)
1. 「期刊合规」是 AI 绘图工具的阿喀琉斯之踵还是机会?(BioRender 话术 vs 政策实际) — 需 AI cluster 搜索恢复后验证
2. 图形摘要(graphical abstract)是情绪消费品类(2410分/1687分 meme 帖), 存在「一键像样的 GA」需求 [帖08 待读/帖26/27/28]
3. 插图师自由职业市场 vs 工具替代: r/scienceillustration/medicalillustration 供给侧怎么看待 AI — 待读
4. PowerPoint 生态有没有被工具化的空间?(学界事实标准但无科研向插件生态)
5. 「Prism 外观」+「统计正确性」双需求能否被一个免费 GUI 工具承接?(LabPlot/JASP 都只占一头)
