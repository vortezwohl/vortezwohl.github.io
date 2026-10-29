# 帖09 · dataanalysis: 12年WhatsApp婚姻消息分析（159评，抓到97条全量）

- 来源: r/dataanalysis | 1633分 | 链接: https://old.reddit.com/r/dataanalysis/comments/1ppt5cz/
- 楼主: u/baxi87, 工具 Mimoto（iOS App, 本地处理, 支持WhatsApp+iMessage）, 长期愿景: 用聊天记录构建数字化身

## 正文要点
- 贡献平衡计分（消息长度/笑声/道歉/提问/图片视频/响应速度/发起对话）
- 启发式标签系统（轻量规则而非NLP模型）
- 全本地隐私优先；情绪分析因效果差而搁置；正在试验端侧小LLM

## 全部评论核心（97条）
#1 [119] 普通人怎么入门做这种dashboard → #2 [84|楼主] 从基础计数慢慢进化, 花了多年
#3 [15] 指标设计者偏袒自己? She could be sending 100% bangers vs you 100% filler; Has TW seen this?
#4 [12|楼主] 妻子看过并反馈改进了指标
#5 [4] 用AI工具(antigravity)手动导出数据建本地dashboard super easy; 开源项目jimwurst
#6 [49] I did the same thing for my soon to be cheating ex wife....it was VERY different(出轨侦查暗黑用法)
#7-9 [30/29/24] Minority Report式笑话: 用数据分析预测出轨
#10 [36] HR data nerd想用到职场 → #11 [55] As an employee, please no.(被监控恐惧)
#12 [25] I have downloaded the app. Now what I don't have a wife. Can u help(单身自嘲)
#13 [15] 对话曲线=长期关系的隐喻(蜜月期/同居下降/生子上升)
#15-16 [12/30] how has the relationship improved? → Ended up getting married and had kids
#17 [19] Rating this relationship as 81% is wild 💀(给关系打分的荒诞感)
#21 [8] vs #22 [15] 统计口径矛盾: laughed 2722(单个emoji) vs 4688(启发式模式含lol/ha ha)——口径透明度
#23 [7] Hawthorn效应: 知道被测量后行为会改变
#26-27 道歉次数多的调侃
#28/#37/#62/#86/#48 Android/Telegram/Web版需求(至少5问)
#32 [3] 数据解析bug(地区问题)——导出解析是工程难点
#40-43 [2-4] 响应时长51分钟质疑→空乘工作; 楼主: 所有统计都是关系语境化的, 没有标准行为
#55 [2] 怎么从WhatsApp提取12年原始数据?(高频问题)
#58 [2] call log数据找不到
#65 [3] WhatsApp可以导出聊天, 上传导出文件
#71 [2] What does your wife think about this?(知情同意)
#81 [2] I risk sharing passwords, credit card numbers in chat(隐私顾虑, 本地处理刚需)
#89 [2] 8条语音消息12年 thats insane(统计的趣味性)
#91 [2] You love her more than she does(按指标评判关系的危险倾向)

## 分析要点
1. 数据获取=第一道墙: 大量用户不知道/做不到导出; 导出体验即产品体验
2. 本地处理=刚需(#81敏感信息顾虑); 云端分析是信任负债
3. 量化关系的三重争议: 指标偏袒(#3)、打分荒诞(#17)、知情同意(#71)——产品设计要有"双人模式/共同解读"定位而非单方评分
4. 暗黑外溢: 出轨侦查(49赞)/职场监控(55赞被拒)——产品的社会风险面
5. Android缺口明确(5+次询问)——Web端可通吃
6. 口径透明(2722 vs 4688)必须内建解释, 否则反噬信任
7. 情感价值天花板: "曲线是关系的隐喻"+“最后结婚了”——礼物化叙事(我们的故事)比工具化叙事(分析)更动人
8. 供给: Mimoto(iOS)/jimwurst(开源)/streamlit分析器——赛道在萌芽但无一打通"礼物化"包装
