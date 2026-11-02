# 03 · Meme 生成类：需求与竞品研究

> 对应 idea：③基于特定主题生成 meme 或漫画。

## 1. 工具格局：模板+贴字仍是王道，AI 只在特定场景有增量

### 1.1 事实基线：meme 制作的主流工具与用户习惯（r/AskReddit、r/memes）
- 主流工具：**imgflip**（模板站，含 tier maker）、**Mematic**（移动端，"免费版够用"）、**GIMP/Photoshop**（硬核用户）、**Canva/Kapwing**（营销向）、danktank.app 等。
- Mematic 的商业模式（免费带水印）养出了整个 "made with mematic" 反讽文化（r/ComedyNecrophilia "mematic is the worst fucking app ive ever used" 1558 分）——**水印既是获客手段也是社区嘲讽对象**。
- imgflip 推出 AI 生成后遭老用户抵制（r/antiai "Et tu, Imgflip?" 83 分）："有人用 uBlock 把 AI 板块藏掉"；最有杀伤力的一条评论（23 分）："**prompt 一个 meme 比在模板框里填字还慢**"。

### 1.2 "AI meme" 的三场舆论战（决定产品叙事怎么写）
1. **懒惰之争**（r/aiwars "Memes generated using meme templates is peak laziness" 66 分/124 评）：反 AI 方认为贴模板也是劳动；AI 方反讽"贴字的先把水印去了"、"模板限制了表达"。→ **AI meme 在"梗圈守门人"处口碑差，但在普通用户/营销人群处无感甚至偏好**（"模板抓不住小众场景" 8 分）。
2. **slop 之争**：r/lovethissmug "I love 'desloppified' AI-generated memes"（10885 分）与 r/desloppification 社区（"We are losing the ancient texts" 980 分）——AI 图像大量冒充经典梗图格式，出现"人类重绘 AI 图"的反向需求（有人当场接单重绘）。→ **"不 slop"本身是可营销的定位**（r/headshots "2026 年的重点是别被抓住用 AI slop——诚实营销有效" 17 分同源逻辑）。
3. **真实身份之争**（r/comics "Here's How I Create Comics Using AI!" 1670 分）：画师标题党钓鱼，实为 GIMP 手工，最高赞"Actual Intelligence"（1132 分）。重要副产品：**真人画师被冤枉成 AI 已常态化**——对产品的启示是"AI 标识/水印"既是合规礼貌也是社区生存策略。

## 2. 变现实证：Supermeme.ai 的三点一线（本方向最重要案例）
- r/SideProject 原帖：**Supermeme.ai 运营 3 年、MRR ≈ $5,000、3 人团队（全员有本职工作）**。客户画像是**社媒营销人员**（把 meme 用于广告与社媒运营），不是玩梗的普通用户。
- 团队实验过的失败项：GPT Store 分发（"tried it, couldn't get much from it"）。
- 用户反馈两条：输出"可用"（别家 "always quite cringe"）；**"希望有移动端，快速生成发到群聊"**。
- 同类：另有独立开发者发帖 "AI Meme Generator makes $5000/month"（14 评），同类营收规模反复出现 → **B2B 营销 meme 是 AI meme 唯一被反复验证的付费场景**。

## 3. 漫画/四格：character consistency + 笑话结构是两个真门槛
- r/generativeAI "How I make AI comics with a character who stays the same"（社区方法论帖）：金句"**画画现在是简单的部分。难的是：值得追的角色、不漂移的形象、四格内成立的笑话**"。方法论：六行角色卡 + "锁定外观"一句话（发型/眼睛/体型/服装/鞋/配饰）每格原样粘贴 + 全程单一画风 + 四句结构（setup/build/turn/payoff）。
- 细节漂移处理：耳环/肩带放过，**瞳色与签名特征=硬失败必须重摇**；"文字+参考图>单独任一"；每个角色保留一张"最佳基准图"。
- 工具阵亡案例：**Dashtoon Studio 2026-08 关停**（转型 AI 视频短剧）→ 一体化 AI 漫画工具商业化艰难；创作者转用 SD+LoRA+Krita 自组管线，或 Qwen 图生图保一致性。
- 付费摩擦案例：某工具"要求手机号才能付款，不敢给"。

## 4. 用户痛点清单
1. **模板场景 AI 无优势**：贴字 30 秒 vs prompt 3 分钟。→ 不要做"AI 模板 meme 生成器"。
2. **slop 观感**：AI 图易被识别、被嘲讽；构图完成度与梗图语义（谁看谁、字压哪里）AI 都不擅长。
3. **冷启动地狱**：MemeFast（iOS，AI meme 生成器）发布帖被怼 "More AI slop with a subscription 😭"、"paid actor vibes"（创始人疑似自演被识破）。→ 独立 meme App 在 Reddit 冷启动=送人头。
4. **角色一致性**是漫画场景核心痛点（方法论文帖的存在本身就是证据）。
5. **角色互换视频 meme**（r/StableDiffusion）：TikTok 上爆火的"同一模板换角色"视频没有好工具，Wan 2.x 是最好的开源方案但仍需分镜处理，Higgsfield/Viggle 效果一般。
6. **分发即归宿**：meme 的终点是发出去——工具若不打通"生成→直接发群/发帖"，就是半成品（Supermeme 用户明确要"快速生成发到群聊"）。

## 5. 机会点小结（与 idea③ 对齐）
1. **降级预期**：独立"AI meme 生成器"是三个 idea 中天花板最低、舆情最差的方向；建议作为头像/表情包产品的**附属功能**（用已有形象生成梗图=形象资产复用），而不是独立产品。
2. **可行切口 A——形象驱动四格漫画**：把"你的头像/表情包形象"放进四格/条漫，解决"角色一致+笑话模板"，卖给社交媒体个人用户（轻）与营销人员（付费，Supermeme 已验证付费意愿在营销侧）。
3. **可行切口 B——中文圈主题 meme**：奶龙式角色 meme 在海外（菲律宾/欧美）自发生长（r/AskChina 考据帖、r/rednote 求图帖），"主题 meme + 表情包"可共用一套角色资产；但同样必须自创 IP 规避版权。
4. **可行切口 C——营销批量图**：若走 B 端，学 Supermeme：模板质量>生成自由度，卖"可用性"不卖"AI"。
