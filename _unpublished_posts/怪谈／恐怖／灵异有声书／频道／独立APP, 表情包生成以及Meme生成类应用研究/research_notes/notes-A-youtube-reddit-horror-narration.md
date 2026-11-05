# A · YouTube/Reddit 恐怖口播生态调研笔记

> 调研日期：2026-11-05
> 调研人：竞品研究 agent（为"恐怖/怪谈口播频道 + 垂直阅读 APP"立项做需求与竞争验证）
> 方法：WebSearch 多轮发现（约 25 次查询）+ WebFetch/webReader 直读原文页面。**覆盖度诚实声明**：本轮工具链受网络限制，直读全文成功的页面共 4 个（Wikipedia 的 [R/nosleep](https://en.wikipedia.org/wiki/R/nosleep)、[The NoSleep Podcast](https://en.wikipedia.org/wiki/The_NoSleep_Podcast)、[MrBallen](https://en.wikipedia.org/wiki/MrBallen) 三个词条全文 + webReader 成功抓取的 [old.reddit.com/r/nosleep 首页](https://old.reddit.com/r/nosleep/)，首页当日热帖如 "I never knew what my wife's job was…" 17.1k 分/690 评论为实读）；其余 Reddit 帖、YouTube 页、Graphtreon 页的直读均被超时/限流阻断，相关数字与用户引语经由搜索摘要转述并附原始 URL，**凡未能逐字核验原文的均已标注"待核实"或"经转述"**。数字时效以 2025 下半年～2026 年为主。
> 与同目录其他笔记的边界：本笔记只覆盖"英文 YouTube/Reddit 恐怖口播生态"，不覆盖中文平台、表情包应用（见 notes-B）、小语种市场（见 notes-C）。

---

## 1. 频道格局表

### 1.1 头部与腰部频道（订阅数为 2025–2026 区间口径，来源附后）

| 频道 | 订阅数（约） | 视频/总播放 | 成立 | 内容形式 | 更新频率 | 近三年态势 |
|---|---|---|---|---|---|---|
| MrBallen（大盘标杆，非纯恐怖） | 11.4M（[HypeAuditor](https://hypeauditor.com/youtube/UCtPrkXdtCM5DACLufB9jbsA)） | 943 视频；月均 40M 播放（[Wikipedia](https://en.wikipedia.org/wiki/MrBallen)） | 2020 | 无脸真人出镜（黑背景+军装+手部比划），纯口播讲故事 | 周更播客+高频 YT | 2020 年从 0 起步，5 年到 11M+，全生态最快 |
| Mr. Nightmare | 7.16M（[频道页经搜索转述](https://www.youtube.com/@MrNightmareTV)；[Mr. Nightmare Wiki 称 2025 年 6.8M/14 亿播放](https://mrs-nightmare.fandom.com/)） | ~610 视频 / 16–17 亿播放 | 2014 | 无脸：库存素材/氛围画面+口播，"真实恐怖经历"合集 | 官方招牌写"每周 2 更"但实际近两年仅月更 1–2 条（[r/MrNightmare 讨论转述](https://www.reddit.com/r/MrNightmare/)） | 稳健缓增：2024 中 6.5M → 2025 末 7M+ |
| Corpse Husband | ~7.55M（[Amazing YouTubers Wiki 转述](https://www.youtube.com/channel/UCFNTq9XKHDNy_1-2lL0kqCg)） | ~103 视频 | 2015（恐怖内容 2018-2020 鼎盛） | 无脸+极低音色，真实恐怖故事 | **实质停更**（2021 起转音乐，因健康问题退网，[r/CorpseHusband](https://www.reddit.com/r/CorpseHusband/comments/11i76qv/what_happened_to_corpse)） | 恐怖口播线已死，订阅数靠音乐期存量 |
| Scary Interesting | 2.05M / 3.94 亿播放 / 314 视频（[Social Blade](https://socialblade.com/youtube/handle/scaryinteresting)） | 单条均播 ~125 万，极高播/投比 | ~2020–2021（**待核实**） | 纪录片式旁白：真实灾难/失踪/生存故事 | 周更级 | 5 年到 2M，是"后 MrBallen 时代"最快头部之一；**被社区质疑用 AI 声音**（见 4.3） |
| Lazy Masquerade | 1.91M（[vidIQ 转述](https://www.youtube.com/c/LazyMasquerade/videos)） | ~384–426 视频 / 4.26 亿播放 | 2015 | 英音+暗色氛围动画，_listener 投稿 | 月更级 | 平稳 |
| MrCreepyPasta | 1.72M / 3.79 亿播放 / 3,340 视频（[Social Blade 转述](https://socialblade.com/)） | 量大管饱 | 2011 | creepypasta 念读鼻祖之一，无脸 | 高频 | 平台老牌，增长平缓（月增≈0，月播 265 万） |
| Let's Read! | ~1.3–1.5M（[ChannelsLike](https://www.channelslike.com/@lazymasquerade)、[youtubers.me](https://us.youtubers.me/lets-read/youtuber-stats)）**口径不一，待核实** | 数百条+播客化（单期近 3 小时） | ~2014 | 真实恐怖/露营/网络怪奇，播客双轨 | 周更（播客日历稳定） | 平稳偏增，WatchMojo 榜 #5 |
| The Dark Somnium | 884K（[vidIQ 转述](https://www.youtube.com/@Darksomnium)），旧数据 450K | ~1,000 视频 / 1.92 亿播放 | ~2018 | 原创配乐+音效+多配音，nosleep 念读 | 周更 | **约 3 年翻倍**，播客加入 Bloody FM 网络 |
| Darkness Prevails / Unexplained Encounters | 884–909K / 2.26 亿播放 / 1,499 视频（[vidIQ 转述](https://www.youtube.com/@darknessprevails)） | 日更级 | ~2015 | 真实经历投稿，矩阵化运营 | **接近日更** | 矩阵扩张（见 1.2） |
| Being Scared | **数据冲突待核实**：一份列表仅 ~1.97K 订阅（[YouTube 列表转述](https://www.youtube.com/c/BeingScared)），与其历史体量（数十万级、"雨声讲鬼故事"格式开创者）明显不符 | 单条 33–42K 播放 | ~2016 | 恐怖故事+雨声/环境音（"Scary Stories Told In The Rain"，有 6 小时合集、"Press Play If You Can't Sleep"系列） | 月更级 | 本体平缓；**创始人把频道社区做成了 APP（见 1.3），商业意义远大于频道本身** |
| The Dark Swamp: Horror Stories | 订阅数**待核实**（检索未获确切值） | 播客端 950+ 集 | ~2021 | **AI 声音**：Darkness Prevails 本人声线克隆（[Feedspot 播客列表注明](https://podcasts.feedspot.com/horror_story_podcasts/)），近每日更新 | **每日** | Darkness Prevails 矩阵的 AI 化新线（同矩阵还有 Just Creepy ~198K、Southern Cannibal ~235K） |
| Dr. Creepen / Dr. Creepen's Dungeon | 主频道 ~36.5K（[频道页转述](https://www.youtube.com/channel/UCcZ_-5180OBED8NBkZgkRmQ)）**与"几大恐怖频道"常被并列提及的印象冲突，待核实**（疑只统计了其一子频道） | ~1.4K 视频 | ~2016 | 多投稿源念读 | 周一/三/五 | 长尾稳定 |
| Chilling Scares | **未获数据，待核实**（检索无结果，可能体量小或已改名） | — | — | — | — | — |
| Nick Crowley / Nexpo / Wendigoon（相邻大盘，供参照） | 百万级 | — | — | 恐怖+网络谜团+深度解读（策划密度更高） | — | r/mrballen 社区最常推荐的同好迁移目标（[r/mrballen 转述](https://www.reddit.com/r/mrballen/)） |

### 1.2 关键格局判断

- **事实**：头部梯队（MrBallen/Mr. Nightmare/Corpse）合计占走品类的绝大多数注意力；Corpse 离场后其"低音炮真实故事"需求悬空。
- **事实**：恐怖口播是 YouTube 公认增长最快的故事类目之一。行业指南称该类 12 个月订阅留存 70–85%（平台均值 40–55%）（[FluxNote 转述](https://fluxnote.io/blog/horror-stories-youtube-channel-guide-2026-start-and-monetize)，**经转述待核实**）。
- **事实**：2025–2026 增速最快的模式是"**矩阵 + AI 声音 + 日更**"（Darkness Prevails 家族的 The Dark Swamp 线），以及"**真实故事纪录片化**"（Scary Interesting，被疑 AI 配音）。两者都成立于 2020 前后、3–5 年做到百万订阅。
- **推断**：纯人肉念读的老牌频道（MrCreepyPasta、Lazy Masquerade）进入平台期；增长动能已经转移到"选题材题+工业化制作"一侧，而新入场者的声音是否真人正在变成次要变量（见第 4 节的社区撕裂）。

### 1.3 垂直标杆：Being Scared → Chilling APP（对"独立 APP"路线的直接证据）

- Being Scared 创始人 Dane Petrali 于 2021 年推出恐怖垂直流媒体 **Chilling**（影视+音频恐怖故事+社区），宣称"由频道评论区/社区讨论点燃想法，**把观众转化为 13,000+ 月付费用户**"（[Republic 融资页](https://republic.com/chilling)，融资页为自报口径）。
- 2022 年在 Republic 众筹，估值 **$10M**（[KingsCrowd 评估](https://kingscrowd.com/chilling-on-republic-2022)）；获 [Bloody Disgusting 报道](https://bloody-disgusting.com/sponsored/3680056/scareds-chilling-app-changes-game-lifestyle-horror)（2021-09，注意此为赞助内容）。
- **推断**：这是"频道→APP"路径被完整走通的唯一公开样本；13K 付费/百万级订阅 ≈ 1–2% 付费转化量级。

---

## 2. MrBallen 现象拆解

### 2.1 规模数据（全部有来源）

- **起点**：2020 年，前海军 SEAL（2014 年阿富汗手榴弹袭击重伤、2017 年医疗退役，紫心勋章），TikTok 首条（Dyatlov 事件）**当天 500 万播放**（[Wikipedia](https://en.wikipedia.org/wiki/MrBallen)）。
- **YouTube**：11.4M 订阅、943 视频、月均 40M 播放（[Wikipedia](https://en.wikipedia.org/wiki/MrBallen)、[HypeAuditor](https://hypeauditor.com/youtube/UCtPrkXdtCM5DACLufB9jbsA)）。
- **播客**：《MrBallen Podcast: Strange, Dark & Mysterious Stories》2022-02 上线；2022-09 音频版权独家授权 Amazon Music（Wondery 体系三年）；**2025-10-27 转签 SiriusXM 多年独家协议，Variety 报道合约价值"高达约 $70M"（按对赌听众指标支付）**（[SiriusXM 官方新闻稿](https://investor.siriusxm.com/news-events/press-releases/detail/2220/siriusxm-inks-exclusive-deal-with-star-youtube-storyteller)、[Variety](https://variety.com/2025/biz/news/mrballen-podcast-john-allen-siriusxm-1236564999)、[Hollywood Reporter](https://www.hollywoodreporter.com/business/digital/mrballen-siriusxm-podcast-deal-1236410445)）。SiriusXM 新闻稿称其播客"下载量超 1,000 万、在 Apple/Spotify 榜单持续居首"。
- **收入**：**Forbes 2026 Top Creators 榜第 21 位，报告年收入 $24M**（[Wikipedia 引 Forbes](https://en.wikipedia.org/wiki/MrBallen)）。
- **出版/衍生**：图像小说《MrBallen Presents: Strange, Dark & Mysterious Stories》2024-10 由 Ten Speed Graphic 出版（[podnews](https://podnews.net/press-release/mr-ballen-book)）；Ballen Studios（妹妹 Evan Allen——普利策得主——2023-12 起任负责人）运营多档衍生播客（Medical Mysteries 等）+ 2026 巡演 + 每年十月"Balloween"营销节点（[Foundry 案例研究](https://www.builtbyfoundry.io/blog/mrballen-11m-subscribers-2-podcast-deals-graphic-novel-every-october)，**经转述**）。

### 2.2 格式为什么有效（有出处的拆解）

1. **开头即给"结局钩子"**：先抛结果/悬念结论（哪怕听似荒诞），让听众确信"听完必有回报"（[MrBallen Podcast 自述集 "Science Behind the Storytelling" 文字稿](https://podcasts.happyscribe.com/mrballen-podcast/science-behind-the-storytelling)）。
2. **扣住 twist 不放**：故意延宕关键反转，用"讲叙者显然知道去哪"的确定性换信任（[Summify 拆解](https://summify.io/discover/mrballen-how-to-tell-stories-better-than-99-of-people-do-thi-VxeHnV)，**经转述**）。
3. **选题铁律**：只要"线性情节+强结尾"的真实故事；细节只为铺垫反转服务。
4. **受害者视角+原始恐惧**：幽闭、失踪、意外，普适恐惧而不是小众题材（[r/mrballen 社区讨论](https://www.reddit.com/r/mrballen/)）。
5. **零画面纯口播**：无脸形式把制作成本压到"讲故事能力"本身，且天然适配播客二次分发。
6. **真人信用背书**：军人重伤经历构成"为什么我讲真实故事"的强人设——这是 AI 声音频道无法复制的部分（推断）。
7. **选题工业化**：Hollywood Reporter 称其靠家庭成员（含两位普利策记者）协助选稿（[THR](https://www.hollywoodreporter.com/business/digital/mrballen-siriusxm-podcast-deal-1236410445)，**经转述**）。

### 2.3 跨平台路径（时序）

TikTok 引爆（2020）→ YouTube 长视频（2020–2021，史上最快破百万订阅梯队）→ 播客（2022，Amazon/Wondery）→ 图书（2024）→ 流媒体独家大deal（2025 SiriusXM ~$70M）→ 巡演/基金会/衍生矩阵（2023–2026）。**推断**：这条"短引流→长留存→播客伴随→IP 出版→订阅锁定"的顺序已是品类标准打法模板。

---

## 3. Reddit 写作生态与授权惯例

### 3.1 社区规模

| 社区 | 规模 | 备注 |
|---|---|---|
| r/nosleep | **18M+ 成员**，全站前 50 大 subreddit（[Wikipedia](https://en.wikipedia.org/wiki/R/nosleep)）；2015 约 4M（Guardian 经转述）、2020 约 14M（[Tubefilter](https://www.tubefilter.com/2020/02/26/horror-subreddit-r-nosleep-goes-dark-to-protest-youtubers-stealing-their-stories)）、2024 约 17M（CrimeReads 经转述） | 2010-03-24 创建；首帖文化"everything is true, even if it's not"（[官方 immersion 规则](https://www.reddit.com/r/nosleep/wiki/index/immersion)：*"on r/nosleep, everything is true even if it's not"*） |
| r/TwoSentenceHorror | ~1.87M（[Redpulse 2026-08 转述](https://redpulse.com/)，**待核实**） | 2014-03 创建；催生 CW 电视剧 Two Sentence Horror Stories |
| r/creepypasta | ~915K（[Redpulse 2026-10 转述](https://redpulse.com/)，**待核实**） | 2010-03 创建 |
| r/ShortCreepyStories | **待核实**（未检索到可靠值；社区普遍估数十万级） | — |
| r/creepy（参照） | ~14.8M | 图片向 |

### 3.2 版权与旁白授权规则（事实，官方 wiki 口径）

- 作者保留完整版权：在 r/nosleep 发帖**不**等于授权旁白；只有作者本人可发自己的故事，其他用途需联系版主（[r/nosleep IP 规则 wiki](https://www.reddit.com/r/nosleep/wiki/index/intellectualproperties)：*"you may only post stories you wrote yourself; other cases require moderator contact"*）。
- 版主专门建立了三个周边社区处理旁白版权教育/举报：r/NosleepWritersGuild、**r/SleeplessWatchdogs**（监督 YouTube 旁白侵权）、r/YTNarratorsGuild（[Wikipedia](https://en.wikipedia.org/wiki/R/nosleep)）。
- **2020-02 "Writer's Blackout"**：r/nosleep 关灯（转私有）约一周，抗议 YouTuber 偷故事不署名不付费（[Tubefilter 2020-02-26](https://www.tubefilter.com/2020/02/26/horror-subreddit-r-nosleep-goes-dark-to-protest-youtubers-stealing-their-stories)、[The Verge 2020-02-27](https://www.theverge.com/2020/2/27/21156447/reddit-subreddit-nosleep-horror-story-protest-youtubers-stealing-stories)、[NoSleep Podcast 官方博客（Vice 采访整理）](https://www.thenosleeppodcast.com/blog/the-r-nosleep-blackout-vice-interviews)）。
- **行业惯例**：Mr. Nightmare、Let's Read! 等"合规旁白"做法是**先私信作者取得授权，通常免费（换曝光）而非付费**；付费个案需私下协商（[FluxNote 指南](https://fluxnote.io/guides/scary-story-videos-guide)、[faceless.my](https://faceless.my/niches/faceless-horror-channel)，均为第三方转述；社区一手讨论见 [r/NoSleepOOC "What YouTubers respect the Authors"](https://www.reddit.com/r/NoSleepOOC/comments/fagsz8/what_youtubers_respect_the_authors_of_rnosleep)、[被无授权旁白的作者自述](https://www.reddit.com/r/NoSleepOOC/comments/9eglwf/found_a_youtube_narrator_that_didnt_ask)）。例外是 The NoSleep Podcast：官方播客**向作者付费/支付报酬**（"one of the few podcasts in its genre that pays its creative contributors"，[Wikipedia](https://en.wikipedia.org/wiki/The_NoSleep_Podcast)）。
- **对"AI 生成故事收集民俗素材"的直接含义（推断）**：若内容为 AI 生成则不存在"Reddit 作者授权"问题，但会被社区按 AI 内容对待（见第 4 节）；若改编 Reddit 故事，授权是合规硬门槛，侵权会被 SleeplessWatchdogs 公开点名（[大规模盗用点名帖 2025](https://www.reddit.com/r/SleeplessWatchdogs/comments/1mxvooa/massive_amount_of_nosleep_stolen_on_giant)）。

### 3.3 r/nosleep 的商业化外溢（需求侧证据）

- Wikipedia 记载的改编链：Penpal（2011 帖→2012 小说）→《The Spire in the Woods》版权售 Amblin（2016）→ Ryan Reynolds 挂名制片一部改编（2018）→ Netflix 购入《My Wife and I Bought a Ranch》（2020）→ Amazon 购入《The Left Right Game》（Tessa Thompson 制片）→ 3000 Pictures 购入婆婆故事（2022，Jessica Knoll 改编）→ 选集剧《Tales From the Void》（2024）→ 小说《We Used to Live Here》（Marcus Kliewer）→ 电影《The Third Parent》（2026）。来源：[Wikipedia](https://en.wikipedia.org/wiki/R/nosleep)。
- **推断**：r/nosleep 是恐怖短篇的全球最大"题材上游"，其内容源源不断向 YouTube 旁白、播客、影视供货——用户计划"收集宗教/民俗素材交给 AI 生成故事"本质是在这个成熟供应链之外另建一条平行供给。

---

## 4. AI 内容禁令与社区情绪（原声引用+链接）

### 4.1 r/nosleep 对 AI 故事的处理（事实边界先说清）

- **r/nosleep 官方没有一份被检索直接命中的"AI 禁令公告"原文**（多轮检索未能定位公告原帖，**此点标注待核实**）。可确证的事实链是：
  1. 现行规则"只能发自己写的故事"本身已封死纯 AI 代笔的合规空间（[IP wiki](https://www.reddit.com/r/nosleep/wiki/index/intellectualproperties)）；
  2. ChatGPT 上线当月（2022-12），r/NoSleepOOC（官方场外版）出现社区质询帖：*"What is the community (and in particular moderation) going to do about AI-generated stories?"*，楼主明示倾向"AI-generated stories should be banned"（[原帖](https://www.reddit.com/r/NoSleepOOC/comments/za9thj/what_is_the_community_and_in_particular)，**经搜索转述，引语为帖子标题+楼主立场概述**）；
  3. 2025–2026 年出现多位写作者被永久封禁的抱怨帖，其中有人声称被误判为 AI：*"I am a prolific Nosleep writer and I got permanently banned"*（[r/horrorwriters 原帖](https://www.reddit.com/r/horrorwriters/comments/1vav6ht/i_am_a_prolific_nosleep_writer_and_i_got)，**经转述**）；
  4. 康奈尔新闻 2025-10 以 Reddit 版主视角报道 AI 内容成为"三重威胁"，并引用写作社区绝对禁令措辞 *"you are not allowed to use AI in this subreddit"*（[Cornell News](https://news.cornell.edu/stories/2025/10/ai-generated-content-triple-threat-reddit-moderators)，**经转述**）。
- **事实**：The NoSleep Podcast 发布正式 **No-Gen-AI 政策页**，承诺不使用生成式 AI（[thenosleeppodcast.com/about/no-gen-ai-policy](https://www.thenosleeppodcast.com/about/no-gen-ai-policy)，页面直读失败，政策存在性由检索结果确认，**原文引语待核实**）。

### 4.2 社区情绪：反对方原声（引语经搜索结果转述，链接为原帖）

- *"This is ruining YouTube. I can't watch this AI slop anymore. It's just endless narrating, saying the same shit over and over."*（"这正在毁掉 YouTube。AI 垃圾我看不下去了，永远是无尽的旁白、车轱辘话来回说。"）——[r/youtube "Anyone else hate how most of the videos on YouTube are AI"](https://www.reddit.com/r/youtube/comments/1rebb3g/anyone_else_hate_how_most_of_the_videos_on)
- *"Disappointing to see classic online horror narrators being replaced by AI voice nonsense. The algorithm buries the human channels."*（"看着经典恐怖旁白被 AI 声音 nonsense 取代很失望，算法还把真人频道埋掉。"）——[r/horror "Epidemic of AI voices in horror narration"](https://www.reddit.com/r/horror/comments/1uubcv6/dk_if_this_is_the_subreddit_for_it_but_there_is)
- *"A lot of people find AI generated voices annoying or grating."*（"很多人觉得 AI 声音烦人或刺耳。"）——[r/SleeplessWatchdogs "Question about AI narration"](https://www.reddit.com/r/SleeplessWatchdogs/comments/1m1dnqu/hi_i_have_a_question_about_ai_narration_am_i)
- r/NewTubers 创作者咨询"恐怖频道该请真人还是用 AI"，回复普遍称 AI 旁白"don't seem genuine / cheap"（"显得不真诚/廉价"），观众会直接跳过（[原帖](https://www.reddit.com/r/NewTubers/comments/1f7iej8/should_i_hire_a_voice_actor_or_use_ai_for_new)，**经转述**）。
- r/creepypasta 求推荐"非 AI 配音"旁白频道的帖子本身就是需求证据（["Good narration channels that aren't voiced by AI"](https://www.reddit.com/r/creepypasta/comments/1iwx29g/good_narration_channels_that_arent_voiced_by_ai)）。

### 4.3 中立/接受方原声

- *"I don't mind either type of voice over."*（"两种配音我都不介意。"）——[r/HFY "Original Stories / AI Voice or Not"](https://www.reddit.com/r/HFY/comments/1k1rc7f/original_stories_ai_voice_or_not)（**经转述**）
- r/aitubers 讨论"观众为什么讨厌 AI 插图"时同样指出"plenty of viewers don't mind it"（不少观众无所谓）（[原帖](https://www.reddit.com/r/aitubers/comments/1nsc6d6/why_do_people_hate_ai_generated_illustrations_for)，**经转述**）。
- r/NoSleepOOC 有专门讨论"nosleep 听众到底介不介意 AI 朗读"的帖子（["Do nosleep listeners prefer AI for reddit horror, or real"](https://www.reddit.com/r/NoSleepOOC/comments/13sryez/do_nosleep_listeners_prefer_ai_for_reddit_horror)，**两方观点并存，原文未逐字核验**）。
- **被质疑/疑似 AI 化的头部频道**：Scary Interesting 被公开讨论是否 AI 生成配音（[r/youtube "Scary Interesting AI generated?"](https://www.reddit.com/r/youtube/comments/1fzhet7/scary_interesting_ai_generated)，**经转述**）；The Dark Swamp 则是**官方口径的 AI 克隆声线**（Feedspot 注明，见 1.1），其日更 950+ 集的产能只有 AI 才能支撑（推断）。
- **平台政策拐点（事实）**：YouTube 于 **2025-07-15** 把"repetitious content"政策改名为"**inauthentic content**"并明确指向量产/AI 垃圾内容（[YouTube 官方政策页](https://support.google.com/youtube/answer/1311392?hl=en)、[Fliki 解读](https://fliki.ai/blog/youtube-monetization-policy-2025)）。恐怖旁白板块随后进入清算期：r/youtube 帖子标题即 *"The horror narration space is taking an absolute beating"*（恐怖旁白板块正被暴打），称连真人频道也因"inauthentic content"被下架变现（[原帖](https://www.reddit.com/r/youtube/comments/1rxzrtq/the_horror_narration_space_is_taking_an_absolute)，**经转述**）；老牌频道 Chilling Tales for Dark Nights（2012 年开台）**2026-03-19 被取消变现、2026-03-24 申诉恢复**，2026 年内两次被打击（[YouTube 官方社区帖](https://www.youtube.com/post/UgkxnjnNfmPCmM2MJ5qeUWvrScwj4b_s5OnZ)、[其 Facebook 恢复公告](https://www.facebook.com/chillingtalesfordarknights/posts/-update-weve-been-re-monetized-as-of-today-march-24-2026-our-youtube-monetizatio/1278731454236599)）。

---

## 5. 用户需求信号与抱怨（原声引用+链接+中文翻译）

> 每条引语经搜索结果转述获得，未能逐字核验原帖全文的已在句末标注。链接均为原帖。

### 5.1 睡前/助眠用途（最强、最一致的需求信号）

1. *"I like falling asleep to horror podcasts but so many have… [calm narration]"* — 原帖核心表述："恐怖播客通常用一种平静的叙述方式，能把我催眠"（"often narrated in a calm sort of way that lulls me to sleep"）。[r/podcasts](https://www.reddit.com/r/podcasts/comments/viq7wy/i_like_falling_asleep_to_horror_podcasts_but_so)
2. *"I've found listening to horror stories helps me sleep."*（"我发现听恐怖故事帮我入睡。"）[r/podcasts](https://www.reddit.com/r/podcasts/comments/r8sjdj/any_good_podcasts_where_someone_is_telling_scary)
3. *"I listen to horror stories every night in bed. Am I alone?"*（"我每晚在床上听恐怖故事，只有我这样吗？"）——高赞回复归因为"叙述者声音抚慰"，像 ASMR。[r/TooAfraidToAsk](https://www.reddit.com/r/TooAfraidToAsk/comments/9opq1d/i_listen_to_horror_stories_every_night_in_bed_am)
4. *"I can't be the only one who listens to creepy stories to fall asleep"*（"不会只有我靠怪谈入睡吧"）[r/Paranormal](https://www.reddit.com/r/Paranormal/comments/1iqjnkx/i_cant_be_the_only_one_who_listens_to_creepy)
5. 求频道推荐帖标题本身就是需求："Favourite horror story YouTube channels to fall asleep to?"（"你们睡前最爱听的恐怖故事频道？"），回复中有人提到夜奶喂娃时循环听。[r/horror](https://www.reddit.com/r/horror/comments/1gn4h3n/favourite_horror_story_youtube_channels_to_fall)
6. 一位听众自述睡醒循环听"spooky stories, morbid facts and internet drama... it puts me in such a peaceful mood"（"怪谈、猎奇事实和网络瓜……让我心情特别安宁"）（同上 r/horror 帖内回复，**经转述**）。
7. 产品侧印证：Being Scared 的"雨声+鬼故事"系列直接把标题写成"Press Play If You Can't Sleep"（睡不着就点开），并出 6 小时合集（[频道检索转述](https://www.youtube.com/c/BeingScared)）。

### 5.2 追更/陪伴行为

8. SiriusXM 收购公告的官方描述：*"an audience that shows up for every new episode"*（"每集必到的听众"），播客下载超 1,000 万。[SiriusXM 新闻稿](https://investor.siriusxm.com/news-events/press-releases/detail/2220/siriusxm-inks-exclusive-deal-with-star-youtube-storyteller)
9. r/MrNightmare 社区对偶像"月更 1–2 条"的持续讨论本身就是追更焦虑的证据：粉丝计算着 "Terrifying Take" 栏目"隔一周甚至一个月才更"（[r/MrNightmare 讨论，经转述](https://www.reddit.com/r/MrNightmare/)）。
10. Let's Read! 单期播客时长近 3 小时（如 EP222 "22 True Scary Horror Stories"），长合集是稳定格式而非偶然（[Wikitubia/频道页转述](https://youtube.fandom.com/wiki/Lets_Read!)）。

### 5.3 对 AI 声音的态度（正反并陈，详见 4.2/4.3，此处不重复引语）

11. 反感方补充：AI 旁白"车轱辘话"（saying the same shit over and over）被点名最多的频道类型就是恐怖故事汇编（[r/youtube 1rebb3g](https://www.reddit.com/r/youtube/comments/1rebb3g/anyone_else_hate_how_most_of_the_videos_on)）。
12. 接受方补充：r/HFY 读者"两种配音都不介意"（[原帖](https://www.reddit.com/r/HFY/comments/1k1rc7f/original_stories_ai_voice_or_not)）。
13. **推断**：态度分布与"用途"强相关——睡前/背景音用户对声音真实度最不敏感（要的是节奏和氛围），主动追更的核心粉丝最反感 AI（消费的是"人"）。

### 5.4 对广告的抱怨

14. r/MrNightmare 专帖讨论"Nightmare 视频的变现/广告"，35 条回复，核心不满是合集视频里的**不可跳过广告**，被形容为"heavily criticized, with LOADS of dislikes"（"被喷惨，点踩极多"）（[原帖](https://www.reddit.com/r/MrNightmare/comments/1kdrac2/thoughts_on_the_monetization_on_mr_nightmares)，**经转述**）。
15. 行业文按用户视角描述痛点：恐怖氛围"经常被突然插入的响亮不可跳过广告毁掉"（"psychological tension is often ruined when a loud, unskippable commercial suddenly appears"）[horrorworldglobal 频道盘点](https://horrorworldglobal.com/top-10-scary-horror-youtube-channels-websites)（**经转述，信度一般**）。
16. r/PetPeeves 关于 YouTube 广告频率的泛讨论中也有"专门找无广告版本的 Mr Nightmare 视频"的用户行为（[原帖](https://www.reddit.com/r/PetPeeves/comments/1uaatzi/the_frequency_of_youtube_ads_now)，**经转述**）——**推断**：无广告长合集是付费/APP 端最直接的付费点。

---

## 6. 变现数据

### 6.1 YouTube 广告分成（RPM/CPM）与去变现风险

- **恐怖口播类 RPM 经验值约 $3–$7**（口径：多个 Reddit 自述+Facebook 玩家小组基准的交叉）。具体自述：Whisper Stories 月播 ~30 万 × $5 RPM ≈ **$1,500/月**（[r/NewTubers "My Thoughts About the Horror Stories Niche"](https://www.reddit.com/r/NewTubers/comments/1g731d5/my_thoughts_about_the_horror_stories_niche_on)，**经转述**）；Facebook 玩家小组基准"月播 3M ≈ $5–10K/月、RPM $5–7、月更 8–10 条"（经搜索转述，**待核实**）；FluxNote 称头部恐怖频道月入 $10K–50K+（[指南](https://fluxnote.io/blog/horror-stories-youtube-channel-guide-2026-start-and-monetize)，第三方估计）。
- CPM 具体咨询帖：["[QUESTION] How much CPM does a Horror narration channel get?"](https://www.reddit.com/r/PartneredYoutube/comments/ywl5xo/question_how_much_cpm_does_a_horror_narration)（**经转述**）。
- YouTube 官方**没有发布过恐怖/真实犯罪类的 RPM 基准**（[FluxNote 指南](https://fluxnote.io/guides/best-niches-for-story-youtube-channels)明确指出这一点）。
- **去变现是结构性风险（事实链）**：2025-07-15 "inauthentic content"政策生效（[官方](https://support.google.com/youtube/answer/1311392?hl=en)）→ 2026-03 Chilling Tales for Dark Nights 两度被取消变现（见 4.3）→ 恐怖旁白板块"taking an absolute beating"（[r/youtube](https://www.reddit.com/r/youtube/comments/1rxzrtq/the_horror_narration_space_is_taking_an_absolute)）；另有创作者社区抱怨恐怖题材被激进降推荐（[Morbidly Beautiful "YouTube's War on Horror"](https://morbidlybeautiful.com/youtube-war-horror)，**经转述，页面直读失败**）。
- **推断**：红/暗黑题材广告主竞价天然低（limited ads），品类天花板靠"量×会员/订阅"而非广告单价。

### 6.2 Patreon/会员公开数据（Graphtreon 口径）

| 主体 | 公开数据 | 来源 |
|---|---|---|
| Mr. Nightmare | Patreon 558 名会员、约 **$48/月**（Graphtreon 缓存 $44/月） | [Graphtreon](https://graphtreon.com/creator/mrnightmare)、[Patreon 页](https://www.patreon.com/mrnightmare)（**Graphtreon 页面直读失败，经搜索转述**） |
| Lazy Masquerade | 有 Patreon（$1/月档+免费档），**未公开会员数/收入，Graphtreon 未收录到** | [patreon.com/lazymasquerade](https://www.patreon.com/lazymasquerade) |
| MrBallen | **无公开 Patreon 收入口径**；其会员化走 SiriusXM Podcasts+（提前 7 天+免广告）与 Apple 订阅 | [SiriusXM blog](https://www.siriusxm.com/blog/tag/mrballen-podcast) |
| The NoSleep Podcast | 走自有订阅（Season Pass → Sleepless Universe），**不走 Patreon**；早期付费实验期收入约 **$3,762/月**（2020 Supercast 案例文，经转述）；Wikipedia 称其 2015 年下载 60 万、现约 **200 万月听众** | [Wikipedia](https://en.wikipedia.org/wiki/The_NoSleep_Podcast)、[thenosleeppodcast.com](https://www.thenosleeppodcast.com/sleepless-universe) |

- **推断**：恐怖口播频道的 Patreon 普遍弱（头部也只有几百人档位），钱更多在"播客订阅+平台 deal+IP 授权"；粉丝愿为"免广告长合集/抢先听"付费，而不愿为"按月支持"付费。

### 6.3 频道出售/收入自述

- r/NewTubers 上恐怖频道收入自述（见 6.1 引帖）与一条协作帖"刚过变现门槛的恐怖故事频道找合作"（[r/NewTubers](https://www.reddit.com/r/NewTubers/comments/1ooezaq/collaboration_opportunity_monetized_horror_story)，**经转述**）；未找到恐怖频道整频道出售的公开成交价案例，**标注待核实**。
- 参照锚点：Scary Interesting 第三方估计月入 $33K–$109K（[OutlierKit 转述](https://outlierkit.com/)，估算口径不明，**强烈存疑待核实**）。

---

## 7. 播客生态

| 播客 | 事实数据 | 商业模式 |
|---|---|---|
| The NoSleep Podcast | 2011-06-13 上线（r/nosleep 衍生），David Cummings 创立；周更；**2026 年已到第 24 季**；约 **200 万月听众**（Wikipedia 口径）；2015 年 60 万下载；嘉宾含 Mike Flanagan、Elijah Wood | **Season Pass 按季付费**（S3 前引入）→ 自有订阅平台 Sleepless Universe（免广告全长度+抢先+花絮；免费档听删减版）；**向作者和配音付酬**；Spotify 不支持其季票（[Wikipedia](https://en.wikipedia.org/wiki/The_NoSleep_Podcast)） |
| Lore | 2015 上线，Aaron Mahnke；**5 亿+ 下载**（[aaronmahnke.com/about](https://www.aaronmahnke.com/about)；2023-01 时"近 5 亿"，[Podnews/Libsyn 公告](https://podnews.net/)）；Amazon Prime 改编剧 2 季；The World of Lore 图书三部曲；衍生 Grim & Mild 工作室（Unobscured/Cabinet of Curiosities 等）；2023-01 与 Libsyn AdvertiseCast 达成广告代理（经转述） | 广告赞助 + 图书/影视 IP + 工作化多节目矩阵 |
| MrBallen Podcast | 2022-02 上线；Amazon Music 独家（2022-09）→ **SiriusXM 多年独家（2025-10，据 Variety 价值最高约 $70M）**；SiriusXM 口径 10M+ 下载、双平台榜单头部；其 YouTube 播客副频道 ~477K 订阅/159 视频（[HypeAuditor 转述](https://hypeauditor.com/)） | 广告（SiriusXM 独家代理）+ 订阅（Podcasts+：免广告、提前 7 天）+ YouTube 流量反哺 |
| The Dark Somnium（播客侧） | Spotify 4.9 分、~11.7K 评分；Bloody FM 网络成员（[Apple/Spotify 页转述](https://podcasts.apple.com/us/podcast/the-dark-somnium/id1546000046)） | 网络广告分成 |
| The Dark Swamp（播客侧） | 950+ 集、Podbean/Spotify/Apple 多平台、AI 克隆声线日更（[Feedspot](https://podcasts.feedspot.com/horror_story_podcasts/)） | 广告+矩阵导流 |

- **推断**：恐怖类播客是"长尾内容最适合的变现容器"（伴随性场景=通勤+睡前），订阅制在该品类已被 NoSleep（Season Pass 十几年）和 MrBallen（Podcasts+）双向验证。

---

## 8. 机会与空白点

> 标注：【事实】=有来源支撑；【推断】=基于事实的逻辑延伸，需验证。

1. 【事实】需求真实且巨大：r/nosleep 18M+ 成员、MrBallen 5 年 11M 订阅+$24M 年收入、Lore 5 亿下载、NoSleep Podcast 14 年长跑 24 季——"恐怖叙事"是被多平台反复验证的常青品类。
2. 【事实】最强的使用场景是**睡前/背景音**（大量一手帖），且用户明确抱怨广告打断氛围、主动寻找"无广告长合集"。【推断】"无广告 + 长合集 + 定时关闭 + 声音连续性"是 APP 端相对 YouTube 的结构性差异点；Chilling APP（13K 月付费、$10M 估值）证明该路径存在，但量级离"大生意"有距离。
3. 【事实】供给端正处在"AI 大清洗"窗口：YouTube 2025-07 inauthentic content 政策 + 2026-03 老牌频道两度去变现 + 社区对 AI slop 的强烈反感。【推断】纯 AI 声音口播的 YouTube 路线合规与口碑风险都处于历史最高位；若用 AI 生成故事，**真人配音（或至少真人声线克隆授权）+ 显著的人工策划密度**是差异化生存线；"收集民俗/宗教素材→AI 生成故事"若走 YouTube，选题工业化（MrBallen 式线性情节+强结尾筛选）比"产量"重要。
4. 【事实】版权上游有硬约束：Reddit 故事必须逐篇获得作者授权，r/nosleep 有专职监督社区和黑历史（2020 熄灯事件）。【推断】用户计划的"民俗/宗教素材 AI 生成"恰好绕开该版权链，但内容质量与"真实感"（nosleep 核心魅力是第一人称真实感）是必须重建的资产；宗教/民俗差异化题材在英文大盘里供给稀薄（Lore 偏历史民俗、大频道偏"真实经历"），存在题材空位。【待验证】民俗题材在英语恐怖听众中的付费意愿。
5. 【事实】变现结构：广告 RPM $3–7 且去变现风险高；Patreon 弱；真正的钱在**订阅（播客季票/Podcasts+）、平台 deal、IP 授权**。【推断】单频道天花板约"月播 300 万≈$1.5–2 万广告"量级，需要多平台矩阵（YouTube+播客+付费订阅）才能接近 MrBallen 结构；中文市场（B 站/播客）不在本笔记范围，需另行调研。
6. 【事实】头部老频道更新放缓（Mr. Nightmare 月更、Corpse 离场）。【推断】"稳定周更的真人声线+民俗怪谈选题"存在接住存量需求的机会窗口。

---

## 9. 证据索引（全部 URL）

**官方/平台**
1. https://support.google.com/youtube/answer/1311392?hl=en （YouTube 变现政策，2025-07-15 inauthentic content）
2. https://investor.siriusxm.com/news-events/press-releases/detail/2220/siriusxm-inks-exclusive-deal-with-star-youtube-storyteller
3. https://www.siriusxm.com/blog/tag/mrballen-podcast
4. https://www.thenosleeppodcast.com/about/no-gen-ai-policy
5. https://www.thenosleeppodcast.com/sleepless-universe
6. https://www.thenosleeppodcast.com/blog/the-r-nosleep-blackout-vice-interviews
7. https://ballenstudios.com

**Wikipedia（全文实读）**
8. https://en.wikipedia.org/wiki/R/nosleep
9. https://en.wikipedia.org/wiki/The_NoSleep_Podcast
10. https://en.wikipedia.org/wiki/MrBallen

**Reddit（原帖链接；除 r/nosleep 首页外均为经搜索转述，未逐字核验全文）**
11. https://old.reddit.com/r/nosleep/ （webReader 实读）
12. https://www.reddit.com/r/nosleep/wiki/index
13. https://www.reddit.com/r/nosleep/wiki/index/intellectualproperties
14. https://www.reddit.com/r/nosleep/wiki/index/immersion
15. https://www.reddit.com/r/NoSleepOOC/comments/za9thj/what_is_the_community_and_in_particular
16. https://www.reddit.com/r/horrorwriters/comments/1vav6ht/i_am_a_prolific_nosleep_writer_and_i_got
17. https://www.reddit.com/r/NoSleepOOC/comments/g4qmbp/free_content_now_copyrighted
18. https://www.reddit.com/r/NoSleepOOC/comments/3zegmu/permission_to_use_stories_for_narration_work
19. https://www.reddit.com/r/NoSleepOOC/comments/fagsz8/what_youtubers_respect_the_authors_of_rnosleep
20. https://www.reddit.com/r/NoSleepOOC/comments/9eglwf/found_a_youtube_narrator_that_didnt_ask
21. https://www.reddit.com/r/SleeplessWatchdogs/comments/1mxvooa/massive_amount_of_nosleep_stolen_on_giant
22. https://www.reddit.com/r/SleeplessWatchdogs/comments/1t9mn2p/where_does_everyone_stand_on_youtubes_new_nosleep
23. https://www.reddit.com/r/SleeplessWatchdogs/comments/1m1dnqu/hi_i_have_a_question_about_ai_narration_am_i
24. https://www.reddit.com/r/youtube/comments/1rebb3g/anyone_else_hate_how_most_of_the_videos_on
25. https://www.reddit.com/r/horror/comments/1uubcv6/dk_if_this_is_the_subreddit_for_it_but_there_is
26. https://www.reddit.com/r/youtube/comments/1rxzrtq/the_horror_narration_space_is_taking_an_absolute
27. https://www.reddit.com/r/youtube/comments/1fzhet7/scary_interesting_ai_generated
28. https://www.reddit.com/r/creepypasta/comments/1iwx29g/good_narration_channels_that_arent_voiced_by_ai
29. https://www.reddit.com/r/NewTubers/comments/1f7iej8/should_i_hire_a_voice_actor_or_use_ai_for_new
30. https://www.reddit.com/r/HFY/comments/1k1rc7f/original_stories_ai_voice_or_not
31. https://www.reddit.com/r/aitubers/comments/1nsc6d6/why_do_people_hate_ai_generated_illustrations_for
32. https://www.reddit.com/r/NoSleepOOC/comments/13sryez/do_nosleep_listeners_prefer_ai_for_reddit_horror
33. https://www.reddit.com/r/podcasts/comments/viq7wy/i_like_falling_asleep_to_horror_podcasts_but_so
34. https://www.reddit.com/r/podcasts/comments/r8sjdj/any_good_podcasts_where_someone_is_telling_scary
35. https://www.reddit.com/r/TooAfraidToAsk/comments/9opq1d/i_listen_to_horror_stories_every_night_in_bed_am
36. https://www.reddit.com/r/Paranormal/comments/1iqjnkx/i_cant_be_the_only_one_who_listens_to_creepy
37. https://www.reddit.com/r/horror/comments/1gn4h3n/favourite_horror_story_youtube_channels_to_fall
38. https://www.reddit.com/r/MrNightmare/comments/1kdrac2/thoughts_on_the_monetization_on_mr_nightmares
39. https://www.reddit.com/r/PetPeeves/comments/1uaatzi/the_frequency_of_youtube_ads_now
40. https://www.reddit.com/r/NewTubers/comments/1g731d5/my_thoughts_about_the_horror_stories_niche_on
41. https://www.reddit.com/r/NewTubers/comments/1hj9rmy/is_500_a_month_realistic
42. https://www.reddit.com/r/NewTubers/comments/1ooezaq/collaboration_opportunity_monetized_horror_story
43. https://www.reddit.com/r/NewTubers/comments/121kcy6/does_horror_stories_narration_channels_get
44. https://www.reddit.com/r/PartneredYoutube/comments/ywl5xo/question_how_much_cpm_does_a_horror_narration
45. https://www.reddit.com/r/PartneredYoutube/comments/1lvc5gc/
46. https://www.reddit.com/r/CorpseHusband/comments/11i76qv/what_happened_to_corpse

**媒体/行业**
47. https://variety.com/2025/biz/news/mrballen-podcast-john-allen-siriusxm-1236564999
48. https://www.hollywoodreporter.com/business/digital/mrballen-siriusxm-podcast-deal-1236410445
49. https://www.builtbyfoundry.io/blog/mrballen-11m-subscribers-2-podcast-deals-graphic-novel-every-october
50. https://podnews.net/press-release/mr-ballen-book
51. https://www.tubefilter.com/2020/02/26/horror-subreddit-r-nosleep-goes-dark-to-protest-youtubers-stealing-their-stories
52. https://www.theverge.com/2020/2/27/21156447/reddit-subreddit-nosleep-horror-story-protest-youtubers-stealing-stories
53. https://news.cornell.edu/stories/2025/10/ai-generated-content-triple-threat-reddit-moderators
54. https://morbidlybeautiful.com/youtube-war-horror
55. https://fliki.ai/blog/youtube-monetization-policy-2025
56. https://antinodeaudio.com/work/youtube-ai-monetization-policy-2025-eng
57. https://air.io/en/monetization/youtube-monetization-policy-changes-2026-a-complete-dated-timeline
58. https://fluxnote.io/blog/horror-stories-youtube-channel-guide-2026-start-and-monetize
59. https://fluxnote.io/guides/best-niches-for-story-youtube-channels
60. https://fluxnote.io/guides/scary-story-videos-guide
61. https://faceless.my/niches/faceless-horror-channel
62. https://podcasts.feedspot.com/horror_story_podcasts/
63. https://videos.feedspot.com/horror_youtube_channels
64. https://aaronmahnke.com/about
65. https://www.lorepodcast.com/about
66. https://en.wikipedia.org/wiki/Lore_(podcast) （未直读成功，经搜索转述）
67. https://blog.dropbox.com/topics/work-culture/podcast-creator-aaron-mahnke-interview
68. https://www.grimandmild.com/lore
69. https://horrorworldglobal.com/top-10-scary-horror-youtube-channels-websites

**数据/统计/案例**
70. https://hypeauditor.com/youtube/UCtPrkXdtCM5DACLufB9jbsA
71. https://socialblade.com/youtube/handle/scaryinteresting
72. https://us.youtubers.me/lets-read/youtuber-stats
73. https://www.channelslike.com/@lazymasquerade
74. https://graphtreon.com/creator/mrnightmare
75. https://www.patreon.com/mrnightmare
76. https://www.patreon.com/lazymasquerade
77. https://graphtreon.com/top-patreon-creators/podcasts
78. https://republic.com/chilling
79. https://kingscrowd.com/chilling-on-republic-2022
80. https://bloody-disgusting.com/sponsored/3680056/scareds-chilling-app-changes-game-lifestyle-horror
81. https://www.thehypemagazine.com/2022/10/17/chilling-app-offers-horror-fans-the-ultimate-content-experience
82. https://chillingapp.com/playlist/dane-petrali
83. https://podcasts.happyscribe.com/mrballen-podcast/science-behind-the-storytelling
84. https://summify.io/discover/mrballen-how-to-tell-stories-better-than-99-of-people-do-thi-VxeHnV
85. https://www.thestorytellinglabpodcast.com/items/the-secret-to-building-tension-in-mrballen's-suspenseful-stories
86. https://www.youtube.com/post/UgkxnjnNfmPCmM2MJ5qeUWvrScwj4b_s5OnZ
87. https://www.facebook.com/chillingtalesfordarknights/posts/-update-weve-been-re-monetized-as-of-today-march-24-2026-our-youtube-monetizatio/1278731454236599
88. https://youtube.fandom.com/wiki/Lets_Read!
89. https://www.youtube.com/channel/UCcZ_-5180OBED8NBkZgkRmQ
90. https://www.youtube.com/@Darksomnium
91. https://podcasts.apple.com/us/podcast/the-dark-somnium/id1546000046
92. https://twu-ir.tdl.org/bitstreams/94b1bd0f-a2cd-4457-b02a-6d06f96a6bfe/download （nosleep 学术研究）
93. https://journals.uni-lj.si/elope/article/download/7849/8200

**待核实清单（诚实汇总）**：r/nosleep 官方 AI 禁令公告原帖与生效时间；Chilling Scares 频道数据；Dr. Creepen 系列各子频道订阅拆分；The Dark Swamp 订阅数；Scary Interesting 成立年份与收入估计；Being Scared 当前订阅数（检索结果自相矛盾）；r/ShortCreepyStories 成员数；NoSleep Podcast 当前 Patron/订阅收入（Sleepless Universe 定价）；所有"经转述"引语的逐字原文；Whisper Stories 与 Facebook 小组基准的原始出处。
