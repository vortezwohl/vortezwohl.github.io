# 帖02 · I made a "Spotify Wrapped" style website for my girlfriend

- 来源: r/relationshipgoals | 120分 | 18评 | 抓到17条
- 链接: https://old.reddit.com/r/relationshipgoals/comments/1qv13m8/
- 楼主: u/SirSkimmy (站点: https://nd24.is-a.dev/RelationshipWrapped/)

## 正文
Not sure if this is the right place to post this but I wanted to share it. My girlfriend and I just reached our one year anniversary and I really wanted to use my programming skills to make her something special. I had the idea to make a "Spotify Wrapped" style website to look back at our first year.

## 全部评论
#1 [22] This is sick dude! How did you pull things like top word, top emoji, number of texts?
#2 [15|楼主] iPhone unencrypted backup -> SQLite -> python script to analyze the data
#3 [2] Thanks for the idea, I'll poke around for something like that as well, our anniversary is in June but I'd like something similar for Valentine's
#4 [3|楼主] linked article: richinfante.com reverse-engineering-the-ios-backup
#5 [8|楼主] 站点链接
#7 [6|楼主] Thinking about making these for people or building out a program or template for people to use. Lmk if anyone would be interested
#8 [2] Definitely would be interested! This is so cool
#9 [1] Definitely interested.
#10 [1] Interested but technologically inept...
#11 [2] I honestly want to know how did you get all these stats😭😭😭 specially the ones which included emojis and photos per day
#12 [3|楼主] backup + code analysis, article is old so had to dig myself
#13 [3|楼主] Instagram data: request all data online -> big json; rest manual
#14 [1] Being a data engineer myself, this makes me both excited as well as hits the feelz
#17 [1] 100/10

## 分析要点
1. Wrapped 式回顾=聊天统计+表情统计+照片统计的组合, 用户复刻意愿强烈(情人节/纪念日双场景)
2. 楼主公开表达产品化意图(做给 people/模板), 2人响应——供给端在萌芽
3. "Interested but technologically inept"= 目标客群自画像
4. 实现成本: iOS备份逆向+Instagram导出+手工——SaaS 化空间巨大
