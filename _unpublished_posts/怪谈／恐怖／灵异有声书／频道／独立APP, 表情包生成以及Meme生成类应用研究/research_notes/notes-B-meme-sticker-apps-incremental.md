# B · 表情包/Meme 生成应用增量调研笔记

> 调研日期：2026-11-05
> 方法：iTunes Search/Lookup API（美区+中国区商店结构化数据）、App Store 用户评论 RSS（5 个应用 × 50 条）、微信表情开放平台官方协议原文、媒体报道与行业报告、Reddit 开发者帖全文、Imgflip API 官方文档。所有数字附来源链接；无法核实的明确标注"待核实"。
>
> **与已有研究的边界声明**：同仓库 `_unpublished_posts/头像图生成, 表情包生成以及Meme生成类应用研究/`（2026-11-02，Reddit 社区视角）已覆盖：①关系头像是最大机会；②表情包工具链断层在"生成式+一键导入"；③meme 最拥挤且 AI slop 污名最重；④Supermeme ~$5k MRR 证明钱在 B 端营销；⑤NSFW 是最高危红线。**本笔记不重复以上论证**，只补量化数据、中国微信生态、AI 原生新玩家、Meme API/B2B、恐怖题材交叉五个增量面，并在第 6 节明确标注哪些结论沿用、哪些被新证据修正。

---

## 1. 应用商店量化数据

### 1.1 头部应用现状（美区 App Store，iTunes API 实测，2026-11-05）

| 应用 | 开发商 | 评分 | 评分数 | 最近更新 | 备注 |
|---|---|---|---|---|---|
| Mematic – The Meme Maker | Trilliarden Entertainment GmbH | 4.74 | 163,228 | 2026-09-17 | 自称"最流行 meme 制作器，10M+ 下载"（自报口径，[mematic.com/about 转述](https://mematic.com/about)） |
| Sticker.ly – WASticker | NAVER Z | 4.77 | 150,510 | 2026-11-02 | 活跃更新 |
| GIPHY: The GIF Search Engine | Giphy, Inc. | 4.77 | 729,983 | 2026-10-30 | 现属 Shutterstock（见 1.4） |
| GIF Keyboard (Tenor) | Tenor | 4.66 | 1,433,081 | **2024-09-30** | **停更约两年**：iOS 端独立 GIF 键盘实质衰退的硬证据 |
| Sticker Maker Studio | Tamara Vardanyan | 4.72 | 80,171 | 2026-10-07 | WhatsApp 贴纸工具 |
| Top Sticker Maker – WaSticker | Aplicativos Legais | 4.73 | 98,831 | 2026-08-25 | 同类 |
| WhatSticker | WHATSTICKER APPS SRL | 4.58 | 76,545 | 2026-10-21 | 同类 |
| Piñata Farms Meme Generator | Pinata Farms | 4.78 | 49,537 | 2026-11-04 | 总评分高但近期评论区被差评淹没（见 1.5） |
| Meme Generator Plus ZomboDroid | ZOMBODROID d.o.o. | 4.91 | **仅 100** | 2026-11-03 | Android 巨头的 iOS 重制版（2022-12 上架），iOS 上规模极小 |

数据来源：iTunes Search/Lookup API（`itunes.apple.com/search|lookup`，country=us），2026-11-05 实测。

### 1.2 Android 侧关键事实：ZomboDroid（注意：常被误写为"ZomboDuck"）

- ZomboDroid 的原版 **Meme Generator Free**（2011-09 上架）累计 **30M+ 下载、4.7 分**，后被 Google Play 政策封禁下架；开发商自述全系列应用累计下载 30M+。来源：[开发者自述帖 r/androiddev](https://www.reddit.com/r/androiddev/comments/fagk5j/)、[zombodroid.com](https://www.zombodroid.com)。
- 一份第三方分析称其 meme 应用月收入 **$60K+**（口径与核实方式不明，**待核实**）。
- **含义（推断）**：Android 端 meme 工具的量级远大于 iOS，但政策封禁风险（模板版权、历史违规）是被验证过的真实风险。

### 1.3 Sticker.ly 与头部工具的收入/下载估计

- Sticker.ly 月下载约 **280 万**（双端）、MRR 约 **$110K+**；Sensor Tower 近月估计 $50k–$100k/月。来源为第三方增长分析（growwithplutus.com，经搜索转述，**原文页未能直接核验，待核实**）；Sensor Tower 追踪页：[app.sensortower.com/overview/com.snowcorp.stickerly.android](https://app.sensortower.com/overview/com.snowcorp.stickerly.android)。
- Sticker.ly Android 端评分 4.3、评论数约 **236 万**（Google Play 页面数据，经搜索转述）；iPhone 美区 4.77/15 万评分。NAVER Z 未单独披露 Sticker.ly 的 MAU（其披露口径聚焦 ZEPETO）——**MAU 无公开数字，待核实**。
- 头部市场为**巴西、印度、墨西哥**（WhatsApp 重度市场）。
- **趋势判断（基于硬证据的推断）**：
  - 跌：独立 GIF 键盘（Tenor GIF Keyboard 停更两年）；GIPHY 平台价值大幅缩水（$400M → $53M，见 1.4）；中国区老一代表情工具停更（见 1.6）。
  - 涨：AI 功能与 WhatsApp 贴纸工具（Sticker.ly/WhatSticker/Sticker Maker Studio 均保持 2026 年内高频更新）；行业报告给出上行预期（Meme Template Generator App Market：2025 年 $1.5B → 2034 年 $4.76B，CAGR 13.7%，[Growth Market Reports](https://growthmarketreports.com/report/meme-template-generator-app-market)，报告口径宽泛，仅作方向参考）。

### 1.4 平台级事件：GIPHY 三度易主

- Meta 2020 年以约 **$400M** 收购 GIPHY → 英国 CMA 反垄断强制剥离 → **2023 年 5 月以 $53M 现金卖给 Shutterstock**（亏损约 $2.6 亿）。来源：搜索多来源交叉（CMA/媒体报道转述）。**修正点**：市面上流传的"Meta 2025 年把 GIPHY 卖给 PicsArt"查无实据，勿引用。
- **含义**：通用 GIF/贴纸内容平台的独立估值逻辑已被证伪——平台钱在"分发集成"（键盘/聊天集成）而非"独立 App"。

### 1.5 Piñata Farms：AI meme 玩家的失败样本（近期评论 RSS 全文阅读）

Piñata Farms 是 2026 年仍在活跃更新的"AI meme"头部应用（49,537 评分），但其最近用户评论几乎清一色 1-2 星，主题高度集中：

> "Every prompt fails the AI check. Even the example they give fails to work… Total fail not one success."（Chet4d，1 星——AI 生成成功率极低）
>
> "I recently looked on my purchases… there's a charge for $39.99 for a year subscription. I never signed up for it."（Jbizzle95，1 星——未经同意扣费）
>
> "When you upload a video they will never put it into the system for use unless you pay the 'get it faster' fee. Two videos have been in line for over a year."（PatrickBags，1 星——付费加速队列）
>
> "It's not free after I paid the subscription and now have to buy coins. lol scam."（Eside4lifex2，1 星——订阅+金币双重付费）

来源：App Store 评论 RSS（`itunes.apple.com/us/rss/customerreviews/id=1461925825/sortby=mostrecent/json`）。
**含义**：AI meme 工具的"AI slop 污名"在上一份研究里是社区舆论层面的，这里补上了**商业执行层面**的实证：生成质量不稳定 + 激进订阅变现 = 评分与口碑双杀。这是 AI meme 赛道的警示基线。

### 1.6 中国区 App Store 表情包工具（iTunes API 实测，country=cn）

| 应用 | 评分 | 评分数 | 最近更新 | 备注 |
|---|---|---|---|---|
| GIF制作-gif动图制作器 | 4.74 | 174,074 | 2026-09-09 | 中国区评分数最多的表情/GIF 工具 |
| 斗图表情-斗图gif表情包制作 | 4.82 | 151,341 | **2020-08-14** | 老一代头部，已停更约六年 |
| ImgPlay - gif制作器&表情包 | ImgBase | 4.83 | 84,504 | 2026-11-04 仍在更新，中国区少数活跃头部 |
| 蛋啵 | 厦门弘天创氏 | 4.90 | 39,913 | 2025-10 更新 |
| 斗图神器 | ZephyrMoor | 4.58 | 13,669 | 2026-08 更新 |
| 闪萌表情 | Beijing iShion | 4.70 | 4,399 | 2025-04 后停更（闪萌曾是微信/GIPHY 合作的表情库） |

**观察（推断）**：中国区表情工具在 App Store 存在"历史评分存量很大但更新停滞"的现象（斗图表情/闪萌）；Mematic 类欧美 meme 工具在中国区几乎没有存在感（搜索"Mematic"返回的均为小应用）——两个市场几乎是平行的产品世界，与上一份研究"中国生态需单独验证"的判断一致。

---

## 2. 中国微信表情生态与表情 IP 经济

> 上一份研究明确标注"微信表情生态未经验证"。本节用官方协议原文 + 一个 2025-2026 的真实创作者账本 + 人民日报报道做直接验证。

### 2.1 微信表情开放平台变现规则（官方协议原文核实）

来源：[《微信表情开放平台赞赏功能使用协议》](https://sticker.weixin.qq.com/cgi-bin/mmemoticon-bin/readtemplate?t=deal/reward)（2026-11-05 抓取原文）。

- **赞赏（打赏）**：
  - 第 1.3 条："赞赏所得：指微信用户赠与款项的金额**扣除相应渠道技术服务成本后**的本功能用户所得。"
  - 第 6.3 条："赞赏所得计算公式：赞赏所得=微信用户赠与金额-渠道技术服务成本。"
  - **关键修正点**：常见说法"打赏全额归创作者"不准确——协议原文是"扣除渠道技术服务成本"，且第 3.1.3 条规定"腾讯有权调整成本比例、决定暂时或永久免收"，**抽成比例并非固定公开数字**。
  - 结算：第 3.1.2/6.5 条，定期结算至创作者绑定的微信支付账户；结算周期腾讯可调整并公告。
  - 税费：第 6.4 条由创作者自理。
- **付费表情**：平台技术服务费为"（未计税收入 – 有争议收入）× **10%**"，创作者拿约 90%，每月 15 号结算上月收益；个人结算超 500 元会被代征增值税（1%）及附加，另有个税三档累进（口径来自下文收入账本文章对平台规则的转述，比例与官方教程页一致，官方教程页正文抓取为空，**10% 比例标"高置信待核"**）。
- 审核：创作者账本称"审核很严格，远超小程序"，分批次反馈问题。

### 2.2 创作者真实收入账本（2025-2026，一手数据）

来源：猎者出海对博客园作者"数字游民9527"账本的拆解，[liezhe.com/weixin-biaoqingbao-zhangben](https://www.liezhe.com/weixin-biaoqingbao-zhangben)（2026-11-05 全文阅读）。

- 2025 年 4 月用两天制作 **6 组表情**上架（2 组付费，每组 10 微信豆），截至 2026-01：**累计发送约 36 万次，总收入仅 125 元**。
- 其中"柴犬阿红"单组下载 13,849 次、发送 361,127 次。
- 收入构成：付费下载约 30 个真实用户（平台抽成 10%）；**打赏"9 成以上来自朋友"**——陌生用户付费几乎为零。
- 作者结论（原话）：

> "没事儿设计两组玩儿可以，权当爱好。如果奔着功利心来的，可以放弃了，投产太低，既需要水平，也需要运气。"——数字游民9527（博客园，经猎者出海转述）

- **核心机制结论（事实）**：微信表情是"用的人多、付钱的人少"的生意，**发送量≠收入**；只有两个变现渠道（赞赏+付费表情），量级在打赏率万分之五到千分之一（CBNData 转述口径，待核实）。

### 2.3 历史头部案例（媒体报道口径）

来源：松果财经《小小表情包，十亿大生意》，[woshipm.com/it/4585316.html](https://www.woshipm.com/it/4585316.html)（2026-11-05 全文阅读；文章发布于 2021-05，数据为历史口径）。

- **乖巧宝宝**（钟超能）：两年下载 **1.5 亿次**，超 11 万人打赏，作者累计收到打赏约 **50 万元**——至今仍是"微信表情个人创作者收入天花板"的标志性案例。
- **萌力星球**（野萌君/汪蛋等）：截至 2020-10，表情全球累计下载 **19 亿+**、累计发送 **800 亿+**；与大英博物馆、光大银行、罗技等联名；变现靠 IP 授权而非表情本身。
- **十二栋文化**（长草颜团子/制冷少女等 300+ 形象）：微信上线近百套表情（3 套收费 1 元），**表情是流量入口，营收在 IP 开发与授权**。
- **蚊子动漫**（蘑菇头）：盈利主要在 IP 授权/品牌合作，但**盗版猖獗一度令其遭遇经营危机**。
- 央视网 2017 年报道"90 后小伙制作表情包两年赚 50 万"（乖巧宝宝同款叙事，见搜索转述，[news.cctv.com](http://news.cctv.com/2017/12/07/ARTIxcTFbZ77Q0RpZUCMOeSW171207.shtml)）。
- 淘宝表情包店铺：单价 1-3 元、月销几百到几千，"轻松月入 3000 元"（媒体转述口径，待核实）。

### 2.4 奶龙：表情 IP → 国民级 IP 的变现天花板（2026 最新）

来源：人民日报 2026-01-15 第 10 版《一只"奶龙"背后的创新链》，[paper.people.com.cn](http://paper.people.com.cn/rmrb/pc/content/202601/15/content_30132851.html)（2026-11-05 全文阅读）；[LEC 授权展报道](https://www.chinalicensingexpo.com/media_center/detail?id=17508)。

- 起源：深圳第七印象，2019 年设计形象，**2020 年初抖音短视频走红**（表情包/短视频双起源）。
- 商业数据：**全网粉丝超 3800 万、流量破千亿；合作品牌 200+（腾讯、美团、伊利、比亚迪、周大福等）；衍生品 2000+ SKU；销往 57 个国家和地区；带动全产业链产值过亿元**。
- 模式：**"内容+授权"双轮驱动**——表情/短视频只是流量入口，钱在授权（玩具、文具、食品、主题乐园）。
- 风险面：海外侵权"维权链条长、成本高"（人民日报原文）。
- **与上一份研究的衔接**：上一份研究建议"自创 IP 角色做主题包运营，学习奶龙"——本节验证了奶龙模式的完整路径与量级上限：**表情 IP 的钱不在表情内，在授权/周边**；个人创作者（2.2）与 IP 公司（2.4）之间是两个完全不同的生意。

### 2.5 抖音 / B站 / 输入法侧生态

- **抖音表情开放平台**（[effect.douyin.com/emoji](https://effect.douyin.com/emoji)）：官方上传入口，流程=上传→审核（约 1 个工作日，要求原创、美观、辨识度高）→上线抖音表情商城；同一平台体系（像塑 AR 特效开放平台）对创作者有现金流量权益。IP 限定表情商城有"假面骑士 55 周年限定表情包"等先例（抖音站内视频佐证）。
- **B 站**："大会员专属表情"体系（会员特权型，非创作者自由售卖）；UGC 表情包主要作为二创内容流转，无官方分成通道（口径为搜索转述，**待核实**）。
- **输入法渠道**：[百度输入法开放平台](https://shurufa.baidu.com/open)明确提供 **IP 合作模式**（接入方提供 IP 授权与素材，协同制作皮肤/表情上架表情中心，用于品宣而非直接分成）+ 积分商城入驻；[搜狗输入法皮肤表情开放平台](https://open.shouji.sogou.com)面向设计师原创内容，承诺"亿级流量曝光"。**含义（推断）**：输入法渠道是"品宣换曝光"不是"卖表情赚钱"的渠道。
- **抖音表情包视频变现**：知乎案例《做抖音表情包，10 天收益 8000+》——变现来自**表情包视频内容的平台广告分成**，而非表情本身付费（[zhuanlan.zhihu.com/p/514441811](https://zhuanlan.zhihu.com/p/514441811)，文章较旧，量级**待核实**）。

---

## 3. AI 原生新玩家（2025-2026）

### 3.1 平台原生功能：Meta AI 贴纸

- 2023-09-27 Meta Connect 发布"AI stickers"（Emu 模型驱动），嵌入 WhatsApp/Messenger/Instagram 聊天，输入 prompt 直接生成贴纸（[Meta Newsroom](https://about.fb.com/news/2023/09/introducing-ai-powered-assistants-characters-and-creative-tools)）。
- Meta AI 助手 2025 年声称达 10 亿月活（2024 年底 5 亿，翻倍口径为 Meta 公开宣传转述，**宣传口径待核实**）。
- **含义（推断）**：平台原生 AI 贴纸"免费+零跳转"，独立 AI 贴纸 App 在"随手生成单张贴纸"场景没有生存空间；独立 App 必须赢在**包管理/跨平台导入/角色一致性**这些原生功能不做的事。

### 3.2 Telegram：AI 贴纸以第三方 bot 生态为主

- Telegram 没有原生 AI 贴纸生成；生态由第三方 bot 与工具构成（图像生成 bot + 贴纸格式转换 bot，如 Stickerify 类；见 [Pixelbin 2025-09 综述](https://www.pixelbin.io/blog/how-to-create-stickers-using-ai)）。
- 一手证据：r/iosapps 开发者帖（2024-10-15，[reddit 1fzp2dc](https://www.reddit.com/r/iosapps/comments/1fzp2dc/stickerai_is_live_on_app_store_generate_stickers/)，2026-11-05 全文阅读）——StickerAI 开发者与用户原声：

> "Apple review is the biggest hurdle for AI sticker apps: we were rejected twice for missing content moderation/reporting."（开发者 u/Alternative-Cow-6814——**AI 贴纸 App 被苹果拒审两次，被要求补内容审核+举报功能后才过审**）
>
> "the killer feature for me is the one-tap add to WhatsApp pack. Making custom packs used to need Photopea + background removal + webp export."（u/tg_sticker_addict——**一键导入贴纸包是杀手功能，替代了抠图/转格式全流程**，直接佐证上一份研究"生成式+一键导入"断层论）
>
> "10 free stickers is too few, then it's paywall."（u/meme_lord_99）
>
> "the stickers have the classic AI look, glossy and same-face. Not sure people will keep using them after the novelty wears off."（u/sarah_k.pixel——**AI 同脸审美疲劳**）

- **含义**：①苹果对 AI 生成内容 App 有明确的审核要求（审核/举报 UI 是上架前置成本）；②用户对"AI 味"贴纸的新鲜感存续性存疑，角色一致性/个人照片衍生是对冲手段（与上一份研究"真人衍生包"结论互证）。

### 3.3 商店里的 AI 贴纸/meme 新品（具体产品 + 评分证据）

| 产品 | 上架时间 | 现状（iTunes API 实测 2026-11-05） | 判断 |
|---|---|---|---|
| AI Sticker Maker – Funny Memes | 2021-05 | 4.62★ / 24,600 评分，2025-11 仍更新 | AI 贴纸类 iOS 存量头部 |
| sticker maker – holos | **2025-10-05** | 4.88★ / 仅 51 评分 | 2025 新品，起步口碑好、规模尚小 |
| Bake some stickers with AI | 2024-12 | **2.5★ / 仅 4 评分** | AI 生成体验差即死掉的样本 |

- Android 侧：Sticker Maker（marsvard，[Google Play](https://play.google.com/store/apps/details?id=com.marsvard.stickermakerforwhatsapp)）4.7★ / **234 万评论**，是 WhatsApp 贴纸工具的 Android 巨头（口径为搜索转述，**待核实**）。
- **失败对照**：Piñata Farms（1.5 节）——"AI meme"概念存量头部，商业化口碑崩坏。
- **趋势小结（事实+推断）**：2024-2026 没有出现"AI 贴纸爆款独立 App"——现有证据（holos 51 评分、Bake 4 评分、StickerAI 依赖 Reddit 冷启动）显示该品类在商店内仍是小盘子，流量入口在平台原生功能和 bot，不在商店搜索。

### 3.4 Product Hunt 侧

- memes 话题下的活跃产品：Supermeme.ai（AI meme 营销，110+ 语言）、WhatToMeme（AI 生成 meme 创意）（[producthunt.com/topics/memes](https://www.producthunt.com/topics/memes)，经搜索转述）。
- **排除项（零信任核实）**：2025-08 Product Hunt 周榜的"StickerX"实为 macOS 桌面便签应用（web_reader 抓取产品页核实），与 AI 贴纸无关，勿误引。
- **含义**：PH 上没有跑出高热度 AI 贴纸消费者产品；有热度的都在 B 端（meme 营销方向），与第 4 节互相印证。

---

## 4. Meme API / B2B

### 4.1 Imgflip API（官方文档全文核实，[imgflip.com/api](https://imgflip.com/api)）

- **免费层**：`get_memes`（热门模板列表）+ `caption_image`（生成带 Imgflip 水印的 meme）。
- **Premium：$9.99/月起**，含：
  - `caption_gif`（GIF meme）：每月前 50 次免费，之后 **$0.02/次**；
  - `search_memes`（1M+ 模板库搜索）：每月前 200 次免费，之后 **$0.005/次**；
  - 去水印：每月前 100 次，之后 **$0.01/次**；
  - `automeme`：前 50 次免费，之后 $0.02/次；
  - **AI meme 端点**：支持接入 OpenAI（GPT）生成文案或 Imgflip 经典模型；前 50 次/月免费，之后 $0.02/次；
  - 私有模板上传。
- **含义**：Imgflip 把"AI 文案+模板渲染"做成了按量计费的 API，起步价低于 Supermeme——**做 B 端 meme 服务时，Imgflip 是现成的基础设施而不是竞品壁垒**（事实，定价以官方页为准）。

### 4.2 Supermeme 之后的 B2B 格局（2025-2026）

- **Supermeme.ai 现状**：免费层 10 credits/月；付费 **$19.99/月起**（Solo Startup，年付口径），含品牌套件、去水印、**API 访问**、110+ 语言（[supermeme.ai/pricing](https://supermeme.ai/pricing)，经搜索转述）。上一份研究的"$5k MRR"结论**沿用**，本份补充其定价与 API 化现状。
- **替代/新玩家**（[G2 替代榜](https://www.g2.com/products/supermeme-ai/alternatives)、[MemeFast 2025-2026 对比文](https://memefast.app/blog/best-ai-meme-generators)，经搜索转述）：
  - **MemeFast**：$2.99/月起（该对比文出自其自家博客，**利益相关，定价待核实**）；
  - **Predis.ai**（社媒内容+ meme）、**Simplified**、Canva（通用设计工具内嵌 meme 模板）；
  - Opus Clip（视频切片方向，被 OMR 列为替代）。
- **格局判断（推断）**：B2B meme 营销没有出现第二个 Supermeme 量级的纯玩家，竞争主要来自**通用 AI 内容工具的功能内嵌**（Canva/Predis/Simplified）。上一份研究"钱在 B 端营销"的结论**沿用量级、修正格局**：纯 meme 工具的 B 端窗口在收窄，差异化要靠"品牌一致性+批量多语言+API"的组合而非单独的 meme 生成器。

---

## 5. 恐怖题材 × 表情包/Meme 交叉机会

> 背景衔接：用户同时运营怪谈/恐怖/灵异有声书频道（见同目录研究主题）。本节评估表情包/meme 作为**频道粉丝运营物料**的交叉机会。

### 5.1 恐怖 meme 社区规模

- **r/creepy**（Reddit 恐怖主社区）：当前公开口径 **62.7 万周访客 / 约 4900 周贡献**（Reddit 于 2025-09 移除订阅数、改用周访客/周贡献指标，[PCMag 报道口径](https://www.pcmag.com/news/reddit-swaps-subreddit-member-counts-for-active-users)——旧口径曾为 100 万+ 会员）。
- 恐怖 meme 更分散地生活在 r/cursedimages、r/TIHI 等泛梗社区（具体成员数 Reddit 已不再公开，**规模待核实**）。
- **判断（推断）**：恐怖 meme 是"常青小众"——没有独立的大版块，但渗透进所有大 meme 社区；万圣节是天然流量脉冲。

### 5.2 恐怖 IP 贴纸的热度证据

- **伊藤润二官方 LINE 贴纸**（"You'll Meet a Horrible Fate"，2017-04-20 上架）：定价 **240 日元（约 $2.20）/100 LINE 币**；后续有富江、双一等多个官方套组（[Anime News Network](https://www.animenewsnetwork.com/interest/2017-04-23/horrify-your-friends-with-junji-ito-line-stickers/.115044)）——**恐怖 IP 付费贴纸是被验证过的商品形态**。
- 第三方恐怖贴纸经济：Amazon/Etsy/TikTok Shop 上伊藤润二相关（富江/漩涡）贴纸包 50-120 张装、$5-13，明确按"Halloween stickers"营销（搜索多店铺佐证）；WhatsApp/Telegram 有第三方恐怖贴纸包（如 HORROR MANIAC: JUNJI ITO by Acid Pop）。
- **万圣节季节性**：LINE 等平台每年 10 月有万圣节贴纸促销档期；第三方恐怖贴纸商品普遍以万圣节为营销节点（以上游店铺营销口径佐证，具体销量数据**待核实**）。
- **注意（事实）**：恐怖贴纸的合法变现前提是 IP 授权——伊藤润二贴纸全部为官方发行；个人自创"怪谈角色贴纸包"才能规避授权风险（与 07 风险研究结论一致）。

### 5.3 恐怖频道用贴纸/表情包做粉丝运营的先例

- **Creepypasta 62 件周边贴纸包**（Amazon 在售）：恐怖网络文学 IP 的贴纸周边已成标准品类。
- **SrPelo**（Spooky Month 系列，YouTube 恐怖/怪诞动画创作者）：通过 Creator Ink 出官方周边（含贴纸类商品）——**"怪谈系列内容 → 季节性周边/贴纸"的频道变现闭环已有先例**。
- **Morbid**（真实罪案/恐怖播客）：以社区梗与口头禅做周边（Printful 案例库收录）——**"频道梗 → 表情包/周边"路径成立**。
- 拆箱/晒贴纸内容（#StickerJunkie 等）表明贴纸是**可拍、可分享**的互动物料。
- **交叉机会评估（推断，基于以上先例）**：
  1. **低成本高确定性**：把频道怪谈角色的关键形象做成"角色情绪贴纸包"（8-16 张），作为频道粉丝群（Telegram/Discord/微信）的免费运营物料 → 直接复用上一份研究"生成式+一键导入"产品能力；
  2. **季节性脉冲**：万圣节/中元节做主题包投放（伊藤润二与万圣节贴纸商品化已验证需求节点）；
  3. **边界**：贴纸是**频道的运营与品牌物料**，不是独立收入线（个人微信表情账本证明直接付费极弱；IP 授权才是大头，但那需要角色先成为 IP——奶龙路径，重内容运营）。

---

## 6. 增量机会清单与结论

### 6.1 与已有研究的衔接（沿用 / 修正 / 补充）

| 已有研究结论 | 本份增量证据 | 处置 |
|---|---|---|
| 关系头像是最大机会（★★★★★） | 未涉及，无新证据 | **沿用** |
| 表情包工具断层在"生成式+一键导入" | StickerAI 用户原话"一键导入 WhatsApp 包是杀手功能，替代 Photopea+抠图+webp 全流程" | **沿用并强化**（一手评论佐证） |
| meme 最拥挤、AI slop 污名最重 | Piñata Farms 差评墙：AI 生成失败率极高 + 激进订阅扣费 → 商业口碑双杀 | **沿用并补充商业实证** |
| Supermeme ~$5k MRR，钱在 B 端 | Supermeme $19.99/月起且已 API 化；新玩家弱，竞争来自 Canva/Predis 内嵌 | **沿用量级，修正格局**（纯 meme B 端窗口收窄） |
| NSFW 是最高危红线 | 新增：苹果对 AI 贴纸 App 强制要求内容审核+举报 UI（StickerAI 两次拒审） | **沿用并补充上架合规成本** |
| "微信表情生态未经验证" | 官方协议原文 + 2025-2026 一手账本 + 人民日报奶龙报道 | **已验证**（见 6.2-E1/E2） |

### 6.2 增量机会清单（每条标注 事实(F) / 推断(I)）

- **E1（F→I）：微信表情直接付费≈0，不是收入线。** 官方协议证实打赏需扣渠道成本且比例由腾讯单方可调；一手账本 36 万次发送仅换 125 元、打赏 9 成来自朋友。→ 推断：面向 C 端的"微信表情赚钱"叙事不成立；表情的价值是**IP 冷启动素材**。
- **E2（F）：表情 IP 的钱在授权，天花板是奶龙。** 3800 万粉丝、200+ 品牌、2000+ SKU、57 国、产值过亿（人民日报）；历史对照组：乖巧宝宝打赏 50 万（个人天花板）、萌力星球 19 亿下载后转 IP 授权。→ 推断：若做表情，必须"内容角色化"并预留授权商品化的路径设计。
- **E3（F）：独立 GIF/贴纸平台估值逻辑已证伪。** GIPHY $400M→$53M（Shutterstock 接手）；Tenor GIF Keyboard 停更两年。→ 推断：不要做"平台"，要做"工具+渠道集成"。
- **E4（F）：AI 贴纸独立 App 尚无爆款，商店内是小盘子。** holos（51 评分）、Bake（4 评分）、AI Sticker Maker（24,600 评分已是存量头部）；PH 无高热度消费级产品。→ 推断：赛道未被商店流量验证，冷启动依赖社区（Reddit/bot 生态），与上一份研究的社区冷启动打法一致。
- **E5（F→I）：一键导入（WhatsApp/Telegram/Discord/iMessage 矩阵）仍是最大差异化，且苹果合规（审核+举报 UI）是 AI 类目上架前置成本。**（StickerAI 帖双证）
- **E6（F）：B 端 meme 基础设施已商品化。** Imgflip API $9.99/月 + AI meme 端点按量计费；Supermeme $19.99/月含 API。→ 推断：做 B 端营销工具时自研渲染无壁垒，价值点在"品牌一致性+多语言批量+分发"，不在生成本身。
- **E7（I）：恐怖 × 表情包交叉是"运营物料"机会而非"收入"机会。** 恐怖 IP 贴纸商品化已验证（伊藤润二 240 日元、Amazon/Etsy 万圣节贴纸 $5-13）；频道先例（SrPelo/Morbid/Creepypasta）均为周边/社区互动用途。→ 建议：为怪谈频道产出**自创角色**贴纸包（规避 IP 授权风险），万圣节/中元节节点投放，作为粉丝群增长与品牌记忆物料。
- **E8（F）：Android 是量级主战场但政策风险被验证。** ZomboDroid 30M+ 下载原版被 Google Play 封禁（模板版权/政策），iOS 重制版仅 100 评分。→ 推断：双端发布时 Android 侧的模板版权合规要前置设计（自创模板/UGC 声明）。
- **E9（I）：头部工具的变现模式给定价锚点。** Mematic 买断转订阅引发差评反弹（评论佐证）；Sticker.ly 免费内容被 paywall+成人广告反噬（"Predatory Ad practices"评论）；Piñata 订阅+金币双轨被骂 scam。→ 推断：温和定价（一次性/低订阅）+ 无成人广告是口碑差异化的低成本手段。
- **E10（I）：中国区 App Store 表情工具"存量评分大、更新停滞"（斗图表情 2020 停更仍 15 万评分）。**→ 推断：中国区工具层有空位，但分发主战场在微信/抖音生态内，独立 App 不是首选形态。

### 6.3 剩余不确定与后续动作

- Sticker.ly 的 MAU、月下载 280 万与 $110K MRR 为第三方口径（**待核实**），如需精确值需 Sensor Tower/AppMagic 付费数据；
- Imgflip/微信 10% 之外的具体分成细节（增值税代征等）以创作者后台实际结算为准；
- 万圣节恐怖贴纸的具体销量数据未获得（仅验证商品存在与营销节点）；
- 后续如继续：①用点点数据/七麦补中国安卓侧数据；②实测 Imgflip API 免费层验证渲染质量；③对 SrPelo/Creator Ink 拆解"内容→贴纸周边"的具体转化数据。

---

## 7. 证据索引

### 7.1 官方/一手页面（2026-11-05 实际阅读全文或结构化数据）

1. 微信表情开放平台《赞赏功能使用协议》原文：https://sticker.weixin.qq.com/cgi-bin/mmemoticon-bin/readtemplate?t=deal/reward
2. 数字游民9527 表情包收入账本拆解（猎者出海）：https://www.liezhe.com/weixin-biaoqingbao-zhangben
3. 人民日报《一只"奶龙"背后的创新链》（2026-01-15 第 10 版）：http://paper.people.com.cn/rmrb/pc/content/202601/15/content_30132851.html
4. 松果财经《小小表情包，十亿大生意》（人人都是产品经理）：https://www.woshipm.com/it/4585316.html
5. Imgflip API 官方文档（定价/限流原文）：https://imgflip.com/api
6. Reddit StickerAI 开发者帖及评论全文：https://www.reddit.com/r/iosapps/comments/1fzp2dc/stickerai_is_live_on_app_store_generate_stickers/
7. iTunes Search/Lookup API 实测（美区+中国区，2026-11-05）：Mematic id491076730 / Sticker.ly id1458740001 / GIPHY id974748812 / GIF Keyboard id917932200 / Sticker Maker Studio id1443326857 / Top Sticker id938243056 / WhatSticker id1147094379 / Piñata Farms id1461925825 / Meme Generator Plus ZomboDroid id6443486587 / AI Sticker Maker id1553251597 / holos id6748560506 / Bake some stickers id6639608927；中国区：斗图表情 id1155901497 / GIF制作 id1301961579 / ImgPlay id989843523 / 蛋啵 id1594998254 / 斗图神器 id1538907432 / 闪萌 id1362975891 等
8. App Store 评论 RSS（5 应用 × 50 条）：`itunes.apple.com/us/rss/customerreviews/id={491076730,1458740001,974748812,1443326857,1461925825}/sortby=mostrecent/json`

### 7.2 行业报告/新闻（搜索交叉，部分口径待核实）

9. Meme Template Generator App Market（$1.5B→$4.76B, CAGR 13.7%）：https://growthmarketreports.com/report/meme-template-generator-app-market
10. AI-Generated Animated Sticker Market（Dataintelo）：https://dataintelo.com/report/ai-generated-animated-sticker-market
11. ZomboDroid 封禁自述（r/androiddev）：https://www.reddit.com/r/androiddev/comments/fagk5j/ ；官网 https://www.zombodroid.com
12. GIPHY→Shutterstock 交易（2023-05，$53M，CMA 强制剥离口径，多来源搜索交叉）
13. Meta AI 贴纸发布（2023-09-27）：https://about.fb.com/news/2023/09/introducing-ai-powered-assistants-characters-and-creative-tools
14. 伊藤润二 LINE 贴纸（ANN，2017-04）：https://www.animenewsnetwork.com/interest/2017-04-23/horrify-your-friends-with-junji-ito-line-stickers/.115044
15. Reddit 指标改革（2025-09 移除订阅数，PCMag 口径）：https://www.pcmag.com/news/reddit-swaps-subreddit-member-counts-for-active-users
16. 抖音表情开放平台：https://effect.douyin.com/emoji ；百度输入法开放平台：https://shurufa.baidu.com/open ；搜狗开放平台：https://open.shouji.sogou.com
17. Supermeme 定价页（搜索转述）：https://supermeme.ai/pricing ；替代品对比：https://www.g2.com/products/supermeme-ai/alternatives 、https://memefast.app/blog/best-ai-meme-generators
18. Sticker.ly Sensor Tower 追踪页：https://app.sensortower.com/overview/com.snowcorp.stickerly.android ；MRR 口径 growwithplutus.com（待核实）
19. 央视网《90 后小伙制作表情包两年赚得 50 万》（2017）：http://news.cctv.com/2017/12/07/ARTIxcTFbZ77Q0RpZUCMOeSW171207.shtml
20. 奶龙 LEC 授权展报道：https://www.chinalicensingexpo.com/media_center/detail?id=17508
21. 知乎《做抖音表情包，10 天收益 8000+》：https://zhuanlan.zhihu.com/p/514441811

### 7.3 用户评论/创作者发言引用清单（原文 + 中文翻译）

| # | 谁 | 原文（节选） | 中文翻译 | 来源 |
|---|---|---|---|---|
| 1 | SPSteve（Mematic，1星） | "I paid for the app once, and after deleting and re-downloading, it's subscription only? No thanks" | "我以前付钱买断过，删了重装后就只剩订阅制了？不谢。" | App Store 评论 RSS（Mematic） |
| 2 | Rottmnt_Donnie_Fan（Mematic，5星） | "Practically 95% of the entire app is free to use… No ads at all." | "整个应用几乎 95% 的功能都能免费用……而且完全没广告。" | 同上 |
| 3 | vivekvarmagunturi（Sticker.ly，1星） | "Dark patterns, app is good but content is behind pay walls and adult ads." | "暗黑模式，应用本身不错，但内容都锁在付费墙后面，还有成人广告。" | App Store 评论 RSS（Sticker.ly） |
| 4 | Fazegonza（Sticker.ly，1星，西语） | "Iba a pagar 7 días gratis y me cobró 21$ es una estafa" | "本来说是 7 天免费试用，结果被扣了 21 美元，是骗局。" | 同上 |
| 5 | 匿名（GIPHY，1星） | "tried to make 3 gifs and it took two hours because the app kept freezing and force closing" | "想做 3 张 GIF 花了两小时，应用一直卡死闪退。" | App Store 评论 RSS（GIPHY） |
| 6 | 匿名（Sticker Maker Studio，1星） | "insanely annoying gambling/misleading game ads that you cant instantly click out of" | "极度烦人的博彩/误导性游戏广告，还不能马上点掉。" | App Store 评论 RSS（Sticker Maker Studio） |
| 7 | 匿名（Sticker Maker Studio） | "If you have both WhatsApp and WA Business installed, the app forces the sticker pack onto WA Business" | "如果同时装了 WhatsApp 和 WA Business，应用会强行把贴纸包装到 WA Business 里。" | 同上 |
| 8 | Jbizzle95（Piñata Farms，1星） | "there's a charge for $39.99 for a year subscription. I never signed up for it" | "我的购买记录里有一笔 $39.99 的年订阅扣费，我根本没订阅过。" | App Store 评论 RSS（Piñata Farms） |
| 9 | Chet4d（Piñata Farms，1星） | "Every prompt fails the AI check… Total fail not one success." | "每个提示词都过不了 AI 检查……全军覆没，一次都没成功。" | 同上 |
| 10 | PatrickBags（Piñata Farms，1星） | "Two videos have been in line for over a year" | "两个视频在队列里排了一年多。" | 同上 |
| 11 | u/Alternative-Cow-6814（StickerAI 开发者） | "we got rejected twice before approval; they asked us to add moderation and reporting features" | "过审前被拒了两次；苹果要求我们补内容审核和举报功能。" | Reddit r/iosapps 1fzp2dc |
| 12 | u/tg_sticker_addict | "the killer feature for me is the one-tap add to WhatsApp pack. Making custom packs used to need Photopea + background removal + webp export" | "对我来说杀手级功能是一键把贴纸加进 WhatsApp 贴纸包。以前做自定义包得用 Photopea+抠图+webp 导出。" | 同上 |
| 13 | u/sarah_k.pixel | "the stickers have the classic AI look, glossy and same-face… novelty wears off" | "贴纸有一股经典 AI 味，油光光的还千篇一律同张脸……新鲜感会消退。" | 同上 |
| 14 | u/meme_lord_99 | "10 free stickers is too few, then it's paywall." | "免费只有 10 张太少了，之后就是付费墙。" | 同上 |
| 15 | 数字游民9527（微信表情创作者） | "没事儿设计两组玩儿可以，权当爱好。如果奔着功利心来的，可以放弃了，投产太低" | （原文即中文）做两组玩玩可以，当爱好；想赚钱就放弃，投入产出太低。 | 博客园，经猎者出海转述 |

（完）
