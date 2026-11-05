# 01 · 竞品档案 · 生成式音乐 SaaS（提示词整曲生成）

> 数据时点: 2026-11-05。定价来自官网直读（Suno）与多源交叉核实（其余）。置信度标注同总览。

## 1. Suno（品类心智霸主，2026-09 经历塌方）

### 1.1 现状【事实】(suno.com/pricing 直读)

| 层级 | 价格 | credits | 下载上限 | 关键差异 |
|---|---|---|---|---|
| Free | $0 | 50/日 | — | 仅 v6-mini，上传≤8min，共享队列 |
| Pro | $10/月（年付$8） | 2,500/月 | **20 首/月** | 商用权、2 种 stem 分离、优先队列、自定义模型 |
| Premier | $30/月（年付$24） | 10,000/月 | **60 首/月** | **Suno Studio 2.0**（MIDI/效果器/自动化/自建插件 API）、3 种 stem、30min 上传、personas |

- 模型线: v6 / v6-wild / v6-mini；credits 不滚存。
- **结构性矛盾**: 创作量（2500-10000 credits）与下载配额（20/60 首）严重不对价——平台故意把"你的作品"扣在云端。

### 1.2 2026-09 塌方事件（r/SunoAI 一手评论，见 _research_notes/reddit-01/04）

- 背景: UMG/Sony/Warner 2024 起诉 → Warner 2025-11 和解 → Suno 承诺只用授权数据训练新模型 → 2026-09 v6 上线并**移除 v4.5/v4.5+/v5/v5.5 全部旧模型**。
- v6 差评原话（top-sorted 高赞直引）:
  - [61分] "v6 is atrocious. Musically lifeless and sounds the most synthetic. Despite sounding better fidelity wise, the feel of everything sounds fake."
  - [12分] "audio fades out randomly, and volume changes. Lyrics turn into static or sound like the singer has walked away from the mic."
  - [11分] 能力清单级批评: "certain drum patterns, guitar styles, vocal styles, song structures, and entire genre behaviors that 4.5 could produce are either heavily resisted or flattened."
- 用户称 v6 为 "WarnerSlop"、"copyright cleaned and watermarking model"；出现 chargeback 行动与华纳 CEO 内部邮件泄露风波。
- **付费意愿反证**: [12分] "id pay 100 bucks a month if it meant I kept getting to create with 4.5."——塌方是政策性的，不是需求性的。

### 1.3 iOS 端现状【事实】(iTunes API)

- 4.88/5，389,464 评分；近期差评主题: "bait and switch"、"failed generations to scam you out of credits"（失败也烧积分）、"hard to work with and download. Don't pay for pro."

### 1.4 对本项目的含义【推断】

- Suno 自身在向"producing"转型（Song Editor→Stems→Sliders→Studio 2.0 Chat Bar），证明方向正确；但它被锁死在（a）授权数据的版权清洗直链（模型越"干净"越平庸）与（b）云端锁定商业模式里——**这两个约束 Strudel-agent 路线天然不存在**。

## 2. Udio（已死——平台锁定风险教科书）

【事实，多源核实 + r/SunoAI 旁证】

- 2025-10-29 与 UMG 和解并宣布共建 2026 新平台；**同日禁用所有下载**（音频/视频/stems）；舆论哗然后开约 48h 下载窗口再永久关闭；截至 2026-10 作品仍锁死，仅限个人非商用，用户只能外录翻录。
- Warner 随后同类和解；老服务关停，新平台以"UMG 授权目录+艺人声音/风格+分成"重启。
- r/SunoAI 用户: "When Udio fell, Suno spiked up"——**用户记住的不是产品多好，而是"平台可以单方面没收你的作品"**。

## 3. Riffusion → ProducerAI → Google

【事实，搜索核实】

- 开源频谱插值项目 → 转型消费级应用 ProducerAI → **2026-02 被 Google 收购**（PitchBook 列名 "Google Flow Music"）；三方统计 App 收入 ~$123.5K/月（2026 末）。
- 含义: 独立生成式音乐玩家的终点是被巨头吸收或死。"独立 SaaS"叙事在这个品类已经不成立。

## 4. 其余竞品速览（2026-11 价格带）

| 产品 | 定位 | 价格 | 备注 |
|---|---|---|---|
| Eleven Music (ElevenLabs) | voice+music 一体 | 信用点制（$1 首月 starter 起，含 10k credits/月） | 退订潮中被用户点名的去处之一 |
| Stable Audio 2.5/3.0 | 授权洁净+自托管 | 免费 50cr/月；Solo $12→Studio $199/月；API≈$0.20/首 | "版权干净"作为卖点 |
| Sonauto | 成本屠夫 | $11/月≈200首（~$0.06/首）→ Scale $330 | 单曲经济学最强 |
| Boomy | 大众玩具 | 免费层最宽松；$3-14/月解锁商用 | 早期 slop 重灾区 |
| Mureka（昆仑万维） | Suno 替代 | $8-30/月；API 经 platform.mureka.ai | 退订潮高频推荐："sounds so different than Suno" |
| MiniMax Music | 中国开源/闭源双线 | API ~$0.15/首；M Plan $264-1560/年 | **2026-08-20 起付费 API 停止对新用户开放、免费 endpoint 下线**（收缩信号）；Music 2.0 发布帖在 r/SunoAI 获 529 分 |
| SOUNDRAW / Beatoven / Mubert / AIVA | 背景音乐 B2B | $3-33/月区间 | 版权分层（免费版权归平台/Pro 归用户）是该细分已验证的转化引擎（14号报告已有，本轮复核） |

## 5. 品类级结论

1. **头部通吃结束**: Suno 独大、Udio 死、Riffusion 被收购——2024 年"百团大战"格局被版权危机+巨头收购清场。
2. **竞争轴转移**: 从"生成质量"转向 版权洁净度（Stable Audio）/成本（Sonauto）/生态锁定（Suno Studio 插件平台）/voice 一体化（Eleven）。
3. **开放侧承接**: 每一次闭源塌方，开源模型搜索量与迁移帖就出现尖峰——**闭源 SaaS 在为开源生态做用户教育**。【帖01 高赞评论群】
4. **"下载/所有权"是新的护城河维度**: 谁能让用户真正带走资产（音频+可编辑源），谁就接住这波被唤醒的所有权意识。【推断】
