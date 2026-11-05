# D · AI 内容管线、SaaS 工具与平台政策调研笔记

> - **调研日期**：2026-11-05
> - **方法**：WebSearch 发现线索 → 官方定价页/政策原文页/Reddit 帖子用 WebFetch 与 web_reader 逐一核实；价格数字以官方页面为准，无法直读的页面经搜索摘要交叉印证并标注"待核实"。
> - **实际阅读页数**：13 个工具定价页 / 政策原文页 / 官方博客 / 讨论帖（ElevenLabs 定价、OpenAI API 定价、YouTube AI 披露官方博客、KDP 内容指南、SCP 授权指南、Crayo 定价、Revid 定价、Epidemic Sound 定价、SEJ 政策报道、YouTube YPP 政策页（部分）、Google Play AI 政策页（部分）、r/nosleep 规则页、2 个 Reddit 帖子正文）。
> - **覆盖度声明**：本机网络对 `support.google.com`、`reddit.com`、`old.reddit.com`、`trustpilot.com` 直连超时或 403，Reddit 评论树与 Trustpilot 星级只能通过搜索摘要与 web_reader 截断内容取证，相关引用标注"经搜索摘要转述"或"待核实"；所有价格以访问日为准，可能变动。
> - **本篇文件状态**：仅本文件为本次新增，未改动仓库其他文件。

---

## 0. 一句话结论（TL;DR）

**技术管线完全可行且便宜（自建方案单集约 $1–6），但"恐怖口播"恰是 2025-07-15 YouTube "inauthentic content" 政策与后续批量除名行动的首要打击区**；用 AI 配音的恐怖叙述频道正在被大规模取消变现，社区已出现专职追踪此事的版块（r/SleeplessWatchdogs）。最大风险不是成本也不是版权，而是**平台政策执行的不确定性**；最大机会点是"作者授权/素材合规"与"政策风险监测"两个无人服务的环节。

---

## 1. TTS 配音工具对比与成本

### 1.1 ElevenLabs（官方定价页核实，2026-11-05）

| 档位 | 月费（月付） | 月度积分 | 备注 |
|---|---|---|---|
| Free | $0 | 10,000 | **无商用授权** |
| Starter | $1 首月，之后 $6 | 30,000 | **含 Commercial License**、含 Dubbing Studio |
| Creator | $11 首月，之后 $22（年付 $18.33） | 121,000 | 含商用、含 Dubbing |
| Pro | $99 | 600,000 | 含商用、含 Dubbing |
| Scale | $299 | 1,800,000（3 席位） | 含商用 |
| Business | $990 | 6,000,000（10 席位） | 含商用 |
| Enterprise | 定制 | 定制 | 含托管配音制作 |

**积分换算（官方口径）**：
- Multilingual v2 / Eleven v3：1 积分 = 1 字符（约 1,000 字符 ≈ 1 分钟语音）。
- Flash v2.5 / Turbo v2.5：**0.5 积分/字符**（同等积分语音时长翻倍）。
- API 等效价：v2 ≈ $0.08/千字符；Flash/Turbo ≈ $0.04/千字符。
- **Dubbing（多语言配音）按分钟计积分**：自动配音带水印 2,000 积分/分钟、去水印 3,000 积分/分钟；Dubbing Studio 去水印 10,000 积分/分钟。→ 一集 20 分钟故事去水印自动配音 = 60,000 积分 ≈ Pro 档月积分的 1/10，折合约 **$9.9–11/集**（事实，由定价页数字直接折算）。
- 官方有 **Creepy Voices 音色库**（elevenlabs.io/voice-library/creepy-voices），社区最常用的恐怖叙述音色是 **Adam**（低沉、压迫感，"scary reddit voice" 教程默认音色）。
- 免费档音色**不可商用**；商用从 Starter（$5–6/月）起即可，门槛极低。

**每小时成品语音成本（按 60,000 字符/小时折算）**：

| 方案 | 折算每小时成本 |
|---|---|
| Creator 档 + Multilingual v2 | ≈ $11/小时（121k 积分 ≈ 2 小时语音） |
| Pro 档 + Multilingual v2 | ≈ $9.9/小时 |
| Pro 档 + Flash v2.5 | ≈ $5/小时 |
| API 按量 v2 / Flash | $4.8 / $2.4 每小时（$0.08 与 $0.04 每千字符） |

> 注意：以上是"套餐积分折算"，套餐是包月买断制，积分不结转（Crayo 等同类产品同理）；产出不足时每小时成本会更高。

### 1.2 OpenAI TTS（官方 API 定价页核实）

- **tts-1**：$15 / 100 万字符 → 一集 20 分钟（约 18–20k 字符）≈ **$0.27–0.30**；每小时语音 ≈ **$0.9–1.5**。
- **tts-1-hd**：$30 / 100 万字符 → 约 2 倍价。
- **gpt-4o-mini-tts**：按 token 计费（文本输入 $0.60/1M，音频输出 $12/1M token），支持用自然语言指令控制语气（"whispered, tense"），适合恐怖氛围微调；折算每分钟约 $0.02–0.05（**估算，待核实**，token 与音频时长换算比率官方未在定价页直接给出）。
- 音色：经典 6 音色（alloy/echo/fable/onyx/nova/shimmer）+ gpt-4o-mini-tts 专属音色（Ash/Ballad/Coral/Sage/Verse/Marin/Cedar 等 11–13 个）。无专属恐怖音色，靠指令调整。
- 商用：OpenAI 服务条款允许输出物商用，无需额外授权费（事实：定价页无限制条款；通用条款允许拥有输出物——细节以最新 ToS 为准）。

### 1.3 Fish Audio（定价页经搜索摘要核实，官方链接在手，数字待复核）

- Free：8,000 积分/月（约 7 分钟），仅个人使用。
- Plus：$15/月（年付约 $11/月），约 200 分钟高品质语音。
- Teams：$100/月起；Max：$749/月（2,500 万积分）。
- API 按量：TTS ≈ $15/1M 字符；Drama 3 模型促销价 $10/1M 字符（后涨至 $50）。
- 来源：[fish.audio/plan](https://fish.audio/plan)、[官方定价文档](https://docs.fish.audio/developer-guide/models-pricing/pricing-and-rate-limits)。
- 亮点：中文语音克隆社区口碑较好；开源版 Fish Speech 可自部署。

### 1.4 开源方案（GitHub/对比文章核实）

| 模型 | 许可证 | 特点 |
|---|---|---|
| **F5-TTS** | MIT | 零样本克隆强，生成开头偶有瑕疵 |
| **GPT-SoVITS** | MIT | 1 分钟数据即可微调，中/日语质量突出 |
| **StyleTTS 2** | MIT | 英文自然度基准领先，韵律拟人 |
| Kokoro | Apache-2.0 | 轻量快速 |
| XTTS-v2 | Coqui 公共许可（限制条款） | 零样本克隆好但许可有约束 |

- 边际成本 ≈ 0（自有 GPU 或按租用 GPU 折算 $0.1–0.3/集）。
- 社区对比讨论：[r/LocalLLaMA 对比帖](https://www.reddit.com/r/LocalLLaMA/comments/1g36cm6/)：F5-TTS 开头有 artifacts；GPT-SoVITS 胜在微调；中文怪谈首选 GPT-SoVITS。

### 1.5 Reddit 用户真实评价（恐怖叙事场景）

1. **r/ElevenLabs《Which voices for horror narrations?》**（已读正文）：恐怖频道主求"calm and clear voice"推荐。链接：https://www.reddit.com/r/ElevenLabs/comments/1m5if11/
2. **r/ElevenLabs《Voice for creepy story》**（经搜索摘要转述）：推荐 **Callum**——"a very good voice for creepy story narration… default settings sound pretty good"（中文：Callum 非常适合恐怖故事旁白，默认设置就已经相当好）。链接：https://www.reddit.com/r/ElevenLabs/comments/180mk1r/
3. **职业配音演员（r/ElevenLabs《ElevenLabs for horror voice》）**（经搜索摘要转述）：自述近一年几乎所有客户都改用其声音的 AI 克隆版在 ElevenLabs 上出活——"almost all clients have shifted to using their AI voice duplicates"（几乎所有客户都转用了我的 AI 声音克隆）。链接：https://www.reddit.com/r/ElevenLabs/comments/1on3wog/
4. **TikTok 恐怖故事音色讨论**（经搜索摘要转述）：做 creepy Reddit story 视频的用户普遍用 Adam 音色配低音调参数。链接：https://www.reddit.com/r/ElevenLabs/comments/1nyt5lq/
5. **质量波动抱怨（r/ElevenLabs《Deteriorating quality?》）**：付费用户抱怨近期音质下降——长篇口播需留出重生成返工余量。链接：https://www.reddit.com/r/ElevenLabs/comments/1ukpyty/
6. **听感分化（r/singularity）**（经搜索摘要转述）：评价两极——"literally indistinguishable"（几乎无法分辨）vs 均匀词距导致"sound robotic"（听起来机械）。链接：https://www.reddit.com/r/singularity/comments/10nkwq0/

**小结**：恐怖旁白是 ElevenLabs 社区验证过的高频用例（有现成音色库与教程），单小时 $5–11 的套餐成本对 60 集/月的量（≈20 小时语音）是 $100–300/月；追求极限成本走 API（tts-1 约 $1/小时语音）或开源自建（≈0）。

---

## 2. 剧本生成工具

### 2.1 通用大模型 vs 专用写作工具

| 工具 | 定价（待核实项已标） | 恐怖题材适配 |
|---|---|---|
| 通用 LLM（GPT/Claude/GLM 系） | API 按量，单集脚本 $0.01–1.5 | 无内容审查硬门槛差异大；长故事逻辑一致性问题需靠大纲+分章+校对流程解决 |
| **NovelAI** | Tablet $10 / Scroll $15 / Opus $25（官方页直连超时，**待核实**） | 卖点是**无审查长文生成**（Opus 档文本生成不限量），社区用于暗黑题材 |
| **Sudowrite** | Hobby $19 / Pro $29–35 / Max $59–129（**待核实**） | Muse 模型官方宣称"handles mature and explicit fiction without refusals: romance, erotica, horror, dark themes"（[官方博客](https://sudowrite.com/blog/sudowrite-vs-dreamgen-nsfw)） |

### 2.2 真实用户证据（逻辑一致性与"AI 味"）

1. **r/WritingWithAI《sudowrite keeps sanitizing my dark content》**（经搜索摘要转述）：暗黑题材作者抱怨 Sudowrite"softens"其内容，与无审查宣传不符——专用工具的"无审查"并非绝对。链接：https://www.reddit.com/r/WritingWithAI/comments/1u7kwj5/
2. **RoyalRoad 论坛**（经搜索摘要转述）：写手报告 AI 生成故事存在 plot holes 与前后不一致。链接：https://www.royalroad.com/forums/thread/143343
3. **Sudowrite Trustpilot**（经搜索摘要转述）："works well, reasonably priced... but the writing is pretty horrible"（工具好用价格合理……但写出来的东西相当糟）。链接：https://www.trustpilot.com/review/www.sudowrite.com
4. **AI 味检测在恐怖社区是硬规则**：Creepypasta Wiki（Miraheze）站规原文："Stories strongly suspected to be AI-generated (it's easier to tell than you think)" will be summary deleted（"被强烈怀疑是 AI 生成的故事——这比你想的容易辨认——可直接删除"）。链接：https://creepypasta.miraheze.org/wiki/Creepypasta_Wiki:Site_Rules ；Fandom 版 Quality Standards 原文："Do not post AI-generated stories or stories with segments that are AI-generated"。链接：https://creepypasta.fandom.com/wiki/Creepypasta_Wiki:Quality_Standards
5. **DrCreepen 事件（r/creepypasta）**（经搜索摘要转述）：YouTube 最大 creepypasta 朗读频道之一被指控谎称人写、实际用 AI 生成故事，社区反弹并被 wiki 拒收——"AI 味"被发现即社死，且是说谎+侵权双重指控。链接：https://www.reddit.com/r/creepypasta/comments/1pr62oy/
6. **学术佐证**：Cornell News（2025-10）报道 AI 内容已成 Reddit 版主"triple threat"（识别、举报、治理三重负担）。链接：https://news.cornell.edu/stories/2025/10/ai-generated-content-triple-threat-reddit-moderators

**结论**：写恐怖故事的瓶颈不是"生成"，而是**长篇逻辑一致性**与**反 AI 味**。恐怖社区对 AI 味高度敏感且会主动用规则武器化（直接删帖/封人）。若走"AI 生成→人工深度改写+事实/逻辑校对"流程，通用 LLM + 自建校对清单比专用工具更省；专用工具（NovelAI/Sudowrite）的价值主要在无审查与角色一致性记忆，但对怪谈口播不是刚需。

---

## 3. 无脸（faceless）视频自动化 SaaS 格局与口碑

### 3.1 定价与功能对比（官方页核实为准，其余标注）

| 工具 | 档位与价格 | 产能限制 | 来源 |
|---|---|---|---|
| **Crayo.ai**（官方页核实） | Hobby $19/月（250 视频 credits、30 分钟 AI 配音）；Clipper $39/月（1000 credits、120 分钟配音）；Pro $79/月（3000 credits、180 分钟配音） | 1 credit = 1 条视频；积分不结转；**不退款** | https://crayo.ai/pricing |
| **AutoShorts.ai**（搜索摘要，**待核实**） | Starter $19/月（约每周 3 条）；Daily $39/月（1 条/天）；Hardcore $69/月（2 条/天） | 每档仅 1 个系列；自动发 TikTok | https://autoshorts.ai |
| **Revid.ai**（官方页核实） | Hobby $39/月；Growth $99/月（2000 credits ≈ 25–50 条 30 秒视频）；Ultra $199/月（12,000 credits） | 40–80 credits/条 30 秒视频；70+ 语言配音；支持 API/MCP/CLI | https://revid.ai/pricing |
| **Pictory**（搜索摘要，**待核实**） | Starter $29 / Professional $59 / Teams $199（月付）；年付约 8 折 | 按视频分钟/credits 计 | https://pictory.ai/pricing |
| **InVideo AI**（搜索摘要，**待核实**） | Plus 约 $28–35/月（50 AI 分钟）；Max 约 $48–60/月（200 AI 分钟） | 按生成分钟计 | https://invideo.io/pricing |
| **Vsub**（第三方汇总，**待核实**） | Starter $29（5,000 credits ≈ 29 分钟视频）；Pro $49（≈60 分钟）；Premium $99（≈81 分钟）；Premium+ $199；终身授权 $735–$2,985 | 视频上限 2 分钟起 | https://vsub.io/pricing |

**关键事实**：这批工具全部为**短视频（<5 分钟）设计**。以"月产 60 集×20 分钟=1,200 分钟配音"衡量：Crayo 最高档仅 180 配音分钟、Vsub Premium 仅 81 视频分钟——**没有一个 faceless SaaS 按长篇音频叙事的产能设计**（事实，由各定价页数字直接对比）。

### 3.2 用户抱怨与口碑

1. **Crayo（Trustpilot，经搜索摘要转述）**：付费后客服无响应、退款难；Trustpilot 评级数字因反爬 403 无法直读（**待核实**）。链接：https://www.trustpilot.com/review/crayo.ai
2. **Crayo 官方 FAQ（核实）**：明确"不提供退款"。
3. **r/FakeGuru《Is youtube automation courses a scam?》**（经搜索摘要转述）：大量"$10k blueprint"卖课套路讨论。链接：https://www.reddit.com/r/FakeGuru/comments/1ddtdbo/
4. **Substack 调查文**（经搜索摘要转述）：faceless automation 卖课被指"predatory scam"——$1,497 课程配伪造收入截图与假评价（"91% 五星但 87% 评论者行为可疑"）。链接：https://substack.com/home/post/p-156560541
5. **r/Entrepreneur《Is YouTube Automation a real thing?》**（经搜索摘要转述）：业务模式本身可成立，但周边生态被卖课和虚假收入证明污染；真实成本（外包脚本/配音/剪辑）被系统性淡化。链接：https://www.reddit.com/r/Entrepreneur/comments/pb9o4s/
6. **创作者共识（X/PaddyG96）**（经搜索摘要转述）：automation 可以赚钱但不推荐新手；"通常卖课的那个人就是赚课程钱的"。链接：https://x.com/PaddyG96/status/1731364246770872606

**"卖课乱象"结论（事实+推断）**：事实是收费课程普遍 $500–1,500、收入证明普遍造假或幸存者偏差、YouTube 政策收紧后此类课程风险激增；推断是 2025-07 后新入场者按课程教的"模板化批量"打法，恰是政策重点打击对象，课程内容已系统性过时。

---

## 4. 平台政策风险（逐条，附官方链接）

### 4a. YouTube 2025-07-15 "inauthentic content" 更新

- **官方政策页**：https://support.google.com/youtube/answer/1311392 （本机直连超时，经 web_reader 部分核实：页面标题为 YPP 政策条款，含频道变现政策约束）。
- **要点**（经 [Search Engine Journal 报道](https://www.searchenginejournal.com/youtube-targets-mass-produced-content-in-monetization-update/550337) 核实，引用官方表述）：
  - 原"repetitious content"改名 **"inauthentic content"**，明确把**mass-produced（批量生产）与 repetitious（重复）**内容排除出变现资格；
  - YouTube 官方口径："a minor update to YouTube's longstanding YPP policies… clarifying what we mean by inauthentic and repetitive content"（对 YPP 长期政策的"小更新"，澄清何为不真实与重复内容）；
  - 违规后果：从 YPP 除名或暂停。
- **不是对"无脸/AI"一刀切**：原创、有人工改造价值的内容仍可变现（多个分析一致）。

**创作者社区反应（恐怖口播是重灾区）**：

1. **r/NewTubers《The Truth About 15 July Update》**（已读部分正文）："There's been loads of misinformation being spread on this sub and many other YT subs about what's happening on July 15"（社区流传大量关于 7-15 更新的错误信息）——恐慌与辟谣并存。链接：https://www.reddit.com/r/NewTubers/comments/1lvz23r/
2. **r/youtube《The horror narration space is taking an absolute being/beat》**（经搜索摘要转述）：三年恐怖口播博主自述**用真人声音+自己音效设计**仍被以 inauthentic content 取消变现。链接：https://www.reddit.com/r/youtube/comments/1rxzrtq/
3. **r/ElevenLabs《YouTube is demonetizing channels using ElevenLabs audio》**（经搜索摘要转述）：创作者报告自己与"hundreds of thousands of other channels"（数十万频道）疑因 ElevenLabs 音频被取消变现。链接：https://www.reddit.com/r/ElevenLabs/comments/1suntcc/
4. **r/NewTubers《This Demonetization Wave is Scary》**（经搜索摘要转述）：除名潮"hitting horror narration channels daily"（每天都在打击恐怖口播频道）。链接：https://www.reddit.com/r/NewTubers/comments/1souc12/
5. **r/SmallYoutubers**（经搜索摘要转述）：自己配音但用 AI 画面也被取消变现。链接：https://www.reddit.com/r/SmallYoutubers/comments/1s1hhqr/
6. **r/SleeplessWatchdogs**：为追踪本次打击专设的版块，YouTube 对 nosleep 类频道的新一轮打击有误伤真人叙述者的记录。链接：https://www.reddit.com/r/SleeplessWatchdogs/comments/1t9mn2p/

**风险评估**：恐怖口播+AI 配音 = 政策执行的最高危组合（事实：上述帖子集体指向）；纯 AI 声音+批量模板化=高概率失去变现；"AI 配音但有人工改造、原创故事、高制作密度"仍是灰色地带，有真人叙述者被误伤的案例，也有大量 AI 频道存活的案例——执行具有不可预测性。

### 4b. YouTube AI 内容披露要求（官方博客原文核实）

- **官方公告**：[How we're helping creators disclose altered or synthetic content](https://blog.youtube/news-and-events/disclosing-ai-generated-content/)（2024-03-18）；帮助文档：https://support.google.com/youtube/answer/14328491
- **必须披露**（原文要点）：内容"a viewer could easily mistake for a real person, place, scene, or event"（观众可能误以为真实的人/地点/场景/事件）且由 altered/synthetic 手段制作——包括"a person's voice to narrate a video"（**用 AI 生成人声旁白**即属此类）。
- **豁免**："clearly unrealistic, animated, includes special effects, or has used generative AI for production assistance"（明显不真实/动画/特效/AI 仅辅助制作，如脚本创意、字幕、调色）。
- **标签位置**：一般内容在展开描述区；健康/新闻/选举/财经等敏感话题在视频本体加显著标签。
- **不披露的后果**：官方原文"look at enforcement measures for creators who consistently choose not to disclose"，且"YouTube may add a label even when a creator hasn't disclosed it"（平台可代打标签）。
- **对恐怖怪谈口播的含义**：AI 配音的"像真人叙述"若配写实画面需打披露标签；**纯音频+超自然故事通常"明显不真实"可豁免**，但若用写实场景 B-roll 则需披露（推断，基于豁免条款文字）。

### 4c. Reddit 各恐怖版块对 AI 内容的禁令

- **r/nosleep 规则页（web_reader 核实原文）**：
  - "No A.I.-generated stories, no plagiarism, no fanfiction"（禁 AI 生成故事、禁抄袭、禁同人）；
  - **叙述者授权规则原文**："Please attempt to secure permission from the authors before posting, narrating, or sharing a story elsewhere."（在他处发布、朗读或分享故事前，请先设法取得作者许可）。链接：https://www.reddit.com/r/nosleep/wiki/rules
  - → 含义：**把 r/nosleep 故事拿去做 AI 口播 = 双重违规**（内容上禁 AI + 朗读需作者许可），且故事版权归作者个人（非 CC）。
- **r/NoSleepOOC 治理讨论**：版主与社区多次讨论 AI 禁令与"AI slop"泛滥。链接：https://www.reddit.com/r/NoSleepOOC/comments/za9thj/ 、https://www.reddit.com/r/NoSleepOOC/comments/1sahct8/
- **真人叙述者 vs AI 频道的冲突（经搜索摘要转述）**：《There is a massive rise in stolen AI read stories》——人类叙述者抱怨 AI 频道盗用故事"four videos a day on four different channels"（四条频道一天四条）无法竞争。链接：https://www.reddit.com/r/NoSleepOOC/comments/191449r/
- **Creepypasta Wiki**：AI 生成内容全站禁发（见第 2 节证据）。

**含义**：Reddit 生态内（nosleep/creepypasta）**不存在合法的"抓素材→AI 生产"路径**；合法路径只剩"民俗/公有领域素材"与"自写原创"。这也反证第 5 节素材选择的正确方向。

### 4d. Google Play 对 AI 生成内容 App 的政策

- **官方政策页**：[AI-Generated Content policy](https://support.google.com/googleplay/android-developer/answer/14094294)（web_reader 核实部分原文）："AI 生成内容政策适用于通过文本、语音和图像提示输入的任意组合生成的任何内容"；开发者必须在 Play Console 的 **App Content（应用内容）部分声明应用是否含 AI 生成内容**；带生成式 AI 功能的应用必须内置安全护栏（内容审核、举报/标记机制）。Developer Policy Center：https://play.google.com/about/developer-content-policy/
- **新增"AI Asset Declaration"**：上架时还需声明商店素材（图标/截图/宣传图）是否 AI 生成（[Reddit 讨论帖](https://www.reddit.com/r/androiddev/)，经搜索摘要转述，**原帖链接待核实**）。
- **对独立 APP 的含义（推断）**：**内容型 App（怪谈有声书库）若不向用户提供 AI 生成功能，主体是"内容合规"而非"genAI 功能合规"**——重点是恐怖/灵异内容的分级（年龄分级问卷）、版权来源可追溯；若 App 内置"AI 生成怪谈"功能才触发完整 AI 政策（审核+举报机制+声明）。商店素材 AI 生成声明对所有开发者适用。

### 4e. Amazon KDP 对 AI 书籍的披露政策（官方原文核实）

- **官方指南**：[KDP Content Guidelines](https://kdp.amazon.com/help/topic/G200672390)
- **原文要点**：
  - "We require you to inform us of AI-generated content (text, images, or translations) when you publish a new book…"（出版新书或改版重发时必须向 Amazon 披露 AI 生成内容——文本/图像/翻译）；
  - AI 生成定义包含"even if you applied substantial edits afterwards"（即使事后大幅人工编辑仍算 AI 生成，**必须披露**）；
  - "You are not required to disclose AI-assisted content"（AI 仅辅助——自查、润色、纠错——**无需披露**）；
  - 责任条款："You are responsible for verifying that all AI-generated and/or AI-assisted content adheres to all content guidelines."
- **数量限制**：每日每账号最多 3 本新书（2023-09 起，[The Guardian 报道](https://www.theguardian.com/books/2023/sep/20/amazon-restricts-authors-from-self-publishing-more-than-three-books-a-day-after-ai-concerns)；[Authors Guild 说明](https://authorsguild.org/news/amazon-adds-to-kdp-generative-ai-policy-caps-daily-self-publishing-uploads)）。
- **披露只对 Amazon，不展示给买家**（广泛报道，**待核实**）。
- **含义**：电子书线可行且披露成本低（对平台不对读者）；但"AI 生成"标签义务会封死"假装人写"的路；KDP 对内容质量投诉的打击（账号封禁）是真实风险。

---

## 5. 版权与素材合规

### 5.1 Creepypasta 经典角色 IP：Slender Man 案例

- 2009 年 Something Awful 论坛用户 **Eric Knudsen（Victor Surge）** 创作 Slender Man，并**以其真名注册了相关图像与文字版权**（[Slender Man Wiki](https://theslenderman.fandom.com/wiki/Victor_Surge)；[NBC News](https://www.nbcnews.com/storyline/slender-man-stabbing/slender-man-creator-speaks-stabbing-i-am-deeply-saddened-n122781)）。
- 角色物权属**多方争议**：Marble Hornets（Troy Wagner/Joseph DeLage）发展了流行形象设定，Derrick Wirtz 贡献了标志性图像；法学分析见 CYN Smith《Beware the Slender Man: Intellectual Property and Internet Memes》（[Univ. of Florida Law Review, 2018, PDF](https://scholarship.law.ufl.edu/cgi/viewcontent.cgi?article=1408&context=flr)）。
- 已有诉讼先例：Phame Factory 起诉 Sony 的 2018 电影《Slender Man》（[TheWrap](https://www.thewrap.com/sony-faces-slender-man-lawsuit-sending-cease-desist-letters)）。
- **结论**：Slender Man 是"集体创作+个人版权注册+诉讼先例"三重雷区，商用改编有真实法律风险；**用"瘦长鬼影"这个名字与形象做商业内容不可取，做"风格类似但原创"的角色安全**（推断）。

### 5.2 SCP 的 CC-BY-SA 3.0 约束（官方授权指南原文核实）

- 来源：[SCP Wiki Licensing Guide](https://scp-wiki.wikidot.com/licensing-guide)
- 派生作品**必须**：① 署名（作品名+作者 screen name+原文链接）；② **同样以 CC-BY-SA 3.0 发布**——"Attribution alone is not enough"；游戏可用 GPLv3 替代。
- **可商用**："You can sell the remixes you make based on the SCP Foundation"，但买方同样可依 ShareAlike 自由复制。
- 官方不能豁免："The staff will not and cannot 'suspend' the Creative Commons license for any reason whatsoever."
- **重大例外：SCP-173 旧图**（基于石井孝之 Untitled 2004）不在 CC 下，官方原文警告"Do not sell anything related to SCP-173's former image"，商用其形象"may result in serious legal action"。SCP-173 文本本身仍是 CC。
- **含义（事实+推断）**：SCP 恐怖有声内容**合法可做**（含商用），成本是署名规范+衍生作品开放同等许可；如果 APP 闭源收费但不"销售 SCP 内容本身"而以 SCP 为素材做音频，属派生作品，音频也需 CC-BY-SA 3.0 标注（推断，建议法务复核）。

### 5.3 r/nosleep 故事授权惯例

- 规则原文（已核实）："Please attempt to secure permission from the authors before posting, narrating, or sharing a story elsewhere."——叙述需**先获作者许可**；故事版权归作者。
- 社区现实：大量 AI 口播频道不获许可直接朗读（第 4c 节证据），已有集中举报社区（r/SleeplessWatchdogs）。
- **含义**：走 nosleep 素材 = 需要逐作者私信授权的脏活，无工具化方案（见第 7 节机会点）。

### 5.4 民间传说/民俗素材的版权状态

- **事实（法务通识，建议律师复核）**：民俗故事本体（无可识别作者）属公有领域；格林童话原版（1812–1857）全球公有领域，可自由改编。
- **仍受保护的层**：现代**译本**（如 Jack Zipes、Maria Tatar 译本有独立版权）、**衍生作品**（迪士尼《白雪公主》电影、现代小说重述）、注释/插画/汇编。
- **实操结论**：以 Project Gutenberg 等处的**原始版本**为底本改编安全；避免照搬现代译本与影视化设定；宗教题材（圣经故事原型等）文本本体多为公有领域，但**特定现代译本（如新国际版 NIV）有版权**，引用需用公版译本（推断+通识）。
- **民俗内容平台风险提示（推断）**：民俗素材版权无风险≠平台政策无风险；"宗教+恐怖"内容可能触发广告主友好度限制与地区敏感性，需在题材层面留缓冲。

---

## 6. 单集成本模型（20 分钟恐怖口播）

### 6.1 假设表（全部显式）

| 假设项 | 取值 | 依据 |
|---|---|---|
| 单集时长 | 20 分钟成品音频 | 任务设定 |
| 语速换算 | 900–1,000 字符/分钟（英语） | ElevenLabs 官方口径（1,000 字符 ≈ 1 分钟） |
| 单集字符量 | ≈ 18,000–20,000 字符（约 3,000–3,400 词） | 由语速换算 |
| 脚本工作流 | 大纲→分章生成→人工/规则校对，含 2–3 轮重生成，进出合计 3–8 万 token | 常规 LLM 用量估算（推断） |
| LLM 单价 | 轻量模型（GPT-4o-mini/GLM Flash 类）$0.3–0.6/1M 输出 token；Claude Sonnet 类 $3–15/1M 输出 token | 各官方定价页区间（通用行情，单值待核实） |
| TTS 用量 | 一遍成音 + 10% 重录余量 = 20k–22k 字符/集 | 社区反馈重生成常见（1ukpyty 帖） |
| 背景音乐 | Epidemic Sound 订阅制无限下载；或 CC0 公版 | 官方页面（价格数字未渲染，**待核实**，历史区间 $10–18/月） |
| 封面图 | AI 生成（自有工具）或无版权图库 | — |
| 剪辑 | 路线 A=全自动（FFmpeg/模板，人力≈0）；路线 B=外包（$15–50/集） | 市场行情（推断） |

### 6.2 单集成本表（美元）

| 环节 | 极简自建线 | 标准付费线 | 全托管外包线 |
|---|---|---|---|
| 脚本 LLM | 0.01–0.05（轻量 API） | 0.3–1.5（Sonnet 级+人工审） | 含在文案外包 $10–30 |
| TTS 配音 | ≈0（开源自部署+GPU 电费 0.1–0.3） | 1.5–3.3（ElevenLabs Flash–v2 套餐折算）或 0.3（OpenAI tts-1 API） | $30–80（真人/精品 AI 工作室） |
| 背景音乐 | 0（CC0） | 0.2–0.3（订阅摊薄） | 1–5（单轨授权） |
| 封面图 | 0–0.2 | 0.3–1（AI 生图 API） | 5–15（设计外包） |
| 剪辑合成 | ≈0（自动化管线，一次性开发） | 0.5–2（半自动+人工抽检） | 15–50（外包） |
| **单集合计** | **≈ $0.1–0.6** | **≈ $2.8–8.6** | **≈ $61–180** |

### 6.3 月产 60 集（20 分钟/集）月成本区间

| 方案 | 构成 | 月成本 |
|---|---|---|
| **极简自建** | 开源 TTS（GPT-SoVITS/F5-TTS）+ 轻量 LLM API + CC0 音乐 + FFmpeg 自动剪辑 | **$30–120/月**（含 GPU/电费/LLM API） |
| **标准付费（推荐基线）** | ElevenLabs Pro $99（v2 档可出 33 集，Flash 档 66 集 → 或 Scale $299 出 100 集）+ LLM $20–60 + Epidemic $10–18 + 封面 API $10 + 半自动剪辑人力 | **$150–420/月** |
| **高质感半外包** | Scale 档 TTS + Sonnet 级脚本+人工校对工时 + 音乐订阅 + 封面外包 + 剪辑外包（部分集） | **$700–2,000/月** |
| **全外包（课程/代理模式）** | 文案+配音+剪辑全部外包 | **$3,700–10,800/月** |

> 敏感性提示：单集 TTS 成本在"自建≈0 → ElevenLabs v2≈$3.3 → Dubbing 多语言版≈$10–37/集"间摆动；多语言矩阵（每语言一条产品线）成本约等于单语言×语言数。

---

## 7. 管线 SaaS 机会点（事实 vs 推断标注）

### 7.1 已确认的事实（可支撑机会判断）

1. **长篇音频叙事产能缺口**：所有 faceless SaaS 按短视频设计（Crayo 顶配 180 配音分钟/月、Vsub Premium ≈81 视频分钟/月、AutoShorts 顶配 60 条/月短视频），20 分钟×60 集=1,200 分钟/月的口播产能无一家能覆盖（事实，定价页对比）。
2. **授权清权（rights-clearance）无工具**：r/nosleep 朗读需逐作者授权，现状是人工私信；搜索未发现任何"作者授权撮合/清权记录"工具（事实：规则存在+工具缺失）。
3. **政策风险监测无商业工具**：YouTube inauthentic content 打击已催生自组织社区（r/SleeplessWatchdogs）追踪误伤案例，说明需求真实存在；搜索未发现商业化"频道政策风险预警"产品（事实：社区存在+产品缺失）。
4. **多语言配音成本高企**：ElevenLabs Dubbing 去水印 $10–37/集（20 分钟折算），是单语言 TTS 的 3–10 倍（事实，定价折算）；无恐怖题材专优的多语言配音产品。
5. **AI 味/一致性校对无专用工具**：恐怖社区以"人肉鉴定"删帖（Miraheze 站规原文"it's easier to tell than you think"），DrCreepen 事件表明"AI 味翻车"是真实商业风险；未发现面向创作者的事前检测/改写工具（事实：社区以人工执行）。
6. **现有工具贵且体验差**：Crayo 不退款+客服失联投诉（Trustpilot）、Sudowrite"工具好内容糟"、ElevenLabs 质量波动帖——头部产品均有公开不满（事实，第 1–3 节证据）。

### 7.2 推断（机会假设，需验证）

- **【推断】垂直一体管线产品**（民俗素材库→结构化怪谈生成→一致性/版权/政策三重校对→长篇配音→多平台分发+风险监测）无在位者；但"恐怖怪谈"垂类市场天花板存疑，更适合作为长篇音频内容工厂的通用能力先做，恐怖做首发场景。
- **【推断】最可防御的切入点是"清权+合规"层**：作者授权记录、素材来源链、平台政策执行案例库——这类资产靠数据积累形成壁垒，比拼生成能力（红海）更难被大模型迭代抹平。
- **【推断】"民俗怪谈素材库"本身可成为产品**：公版民俗故事的结构化语料（按文化圈/母题/恐怖类型/版权状态标注）是生成管线的上游稀缺资产，同时服务自用与对外售卖。
- **【推断】频道矩阵管理 SaaS**（多频道统一配音资产、披露标签合规、除名预警）需求从 SleeplessWatchdogs 社区自组织可证，但付费意愿未验证（**待验证**）。

---

## 8. 证据索引

**官方政策与定价页（本次实际读取）**

1. ElevenLabs 定价页：https://elevenlabs.io/pricing
2. OpenAI API 定价页：https://developers.openai.com/api/docs/pricing （TTS 指南：https://developers.openai.com/api/docs/guides/text-to-speech）
3. YouTube YPP 政策页（inauthentic content）：https://support.google.com/youtube/answer/1311392
4. YouTube AI 披露官方博客：https://blog.youtube/news-and-events/disclosing-ai-generated-content/
5. YouTube AI 披露帮助文档：https://support.google.com/youtube/answer/14328491
6. Google Play AI 生成内容政策：https://support.google.com/googleplay/android-developer/answer/14094294
7. Google Play Developer Policy Center：https://play.google.com/about/developer-content-policy/
8. Amazon KDP 内容指南（AI 披露）：https://kdp.amazon.com/help/topic/G200672390
9. SCP 授权指南：https://scp-wiki.wikidot.com/licensing-guide
10. r/nosleep 规则页：https://www.reddit.com/r/nosleep/wiki/rules
11. Creepypasta Wiki Quality Standards：https://creepypasta.fandom.com/wiki/Creepypasta_Wiki:Quality_Standards ；Miraheze 站规：https://creepypasta.miraheze.org/wiki/Creepypasta_Wiki:Site_Rules
12. Crayo 定价页：https://crayo.ai/pricing
13. Revid 定价页：https://revid.ai/pricing
14. Epidemic Sound 定价页：https://www.epidemicsound.com/pricing/ （价格数字未渲染，计划名已核实）
15. SEJ 政策报道：https://www.searchenginejournal.com/youtube-targets-mass-produced-content-in-monetization-update/550337
16. Fish Audio 定价：https://fish.audio/plan ；https://docs.fish.audio/developer-guide/models-pricing/pricing-and-rate-limits
17. ElevenLabs Creepy 音色库：https://elevenlabs.io/voice-library/creepy-voices

**Reddit/社区用户发言（含链接；标注方式：[已读]=web_reader 读到正文，[转述]=经搜索摘要转述）**

1. [已读] r/ElevenLabs 恐怖叙述音色求推荐帖：https://www.reddit.com/r/ElevenLabs/comments/1m5if11/
2. [转述] Callum 音色好评：https://www.reddit.com/r/ElevenLabs/comments/180mk1r/
3. [转述] 职业配音演员客户转用 AI 克隆：https://www.reddit.com/r/ElevenLabs/comments/1on3wog/
4. [转述] "数十万频道疑因 ElevenLabs 音频被取消变现"：https://www.reddit.com/r/ElevenLabs/comments/1suntcc/
5. [已读部分] 恐怖口播博主被除名（真人声音仍中枪）：https://www.reddit.com/r/youtube/comments/1rxzrtq/
6. [转述] 除名潮"hitting horror narration channels daily"：https://www.reddit.com/r/NewTubers/comments/1souc12/
7. [已读部分] r/NewTubers 辟谣帖（7-15 更新）：https://www.reddit.com/r/NewTubers/comments/1lvz23r/
8. [转述] AI 画面+真人声音被除名：https://www.reddit.com/r/SmallYoutubers/comments/1s1hhqr/
9. [转述] 叙述者控诉 AI 频道盗故事"一天四条"：https://www.reddit.com/r/NoSleepOOC/comments/191449r/
10. [转述] 配音演员因 AI 频道失业：https://www.reddit.com/r/NoSleepOOC/comments/12lul40/
11. [转述] nosleep 版主 AI 禁令讨论：https://www.reddit.com/r/NoSleepOOC/comments/za9thj/
12. [转述] Sudowrite"净化"暗黑内容投诉：https://www.reddit.com/r/WritingWithAI/comments/1u7kwj5/
13. [转述] DrCreepen AI 欺骗指控：https://www.reddit.com/r/creepypasta/comments/1pr62oy/
14. [转述] "几乎无法分辨 vs 机械感"听感分化：https://www.reddit.com/r/singularity/comments/10nkwq0/
15. [转述] faceless 卖课调查（$1,497 课程+假评价）：https://substack.com/home/post/p-156560541 ；r/FakeGuru：https://www.reddit.com/r/FakeGuru/comments/1ddtdbo/ ；r/Entrepreneur：https://www.reddit.com/r/Entrepreneur/comments/pb9o4s/
16. [转述] Trustpilot Crayo 投诉（客服失联/退款难）：https://www.trustpilot.com/review/crayo.ai
17. [转述] r/SleeplessWatchdogs 打击追踪：https://www.reddit.com/r/SleeplessWatchdogs/comments/1t9mn2p/
18. [转述] 开源 TTS 对比：https://www.reddit.com/r/LocalLLaMA/comments/1g36cm6/
19. [转述] ElevenLabs 质量波动：https://www.reddit.com/r/ElevenLabs/comments/1ukpyty/

**版权与法务**

20. Slender Man 版权法学分析（PDF）：https://scholarship.law.ufl.edu/cgi/viewcontent.cgi?article=1408&context=flr
21. Sony Slender Man 诉讼：https://www.thewrap.com/sony-faces-slender-man-lawsuit-sending-cease-desist-letters
22. KDP 限流报道：https://www.theguardian.com/books/2023/sep/20/amazon-restricts-authors-from-self-publishing-more-than-three-books-a-day-after-ai-concerns ；https://authorsguild.org/news/amazon-adds-to-kdp-generative-ai-policy-caps-daily-self-publishing-uploads
23. Cornell News（AI 内容治理）：https://news.cornell.edu/stories/2025/10/ai-generated-content-triple-threat-reddit-moderators

**待核实清单**：AutoShorts/Pictory/InVideo/Vsub 精确价格；Fish Audio 当前档位数字；Epidemic Sound 具体月费；NovelAI/Sudowrite 现价；Trustpilot 各家星级；gpt-4o-mini-tts 分钟折算；KDP 披露"不向买家展示"表述；Android 商店素材 AI 声明原帖链接。
