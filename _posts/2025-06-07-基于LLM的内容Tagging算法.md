---
layout: post
toc: true
title: "基于 LLM 的内容自动化 Tagging 实践"
categories: NLP
tags: [NLP, LLM]
author:
  - vortezwohl
  - 吴子豪
---

在推荐算法和搜索引擎优化等应用场景中，为内容添加标签（Tagging）是一项重要的基础工作。传统的标签分配方式通常依赖人工标注或基于关键词匹配的简单算法，前者成本高、效率低，后者则难以理解内容的语义信息。随着大语言模型（LLM）的发展，利用其强大的语义理解能力进行内容标签自动分配成为可能。但是, 当前使用 LLM 解决自动 Tagging 问题时，存在生成标签与实际标签难以精确对应的局限性。例如在标签格式匹配层面，LLM 可能因对空格、大小写等细节处理不当导致无法精准匹配原标签 ID，如将 “Secret relationship” 标签生成为 “secretrelationship”，因缺失空格且未保留首字母大写，与系统中预设的标签格式存在差异；在符号或格式规范的遵循上，LLM 也可能未能严格按照原标签的既定格式生成，像 “Rebirth/Reborn” 标签可能被简化为 “Rebirth”，忽略了原标签中 “/” 符号的保留，这类细节偏差会直接影响标签匹配的准确性，导致自动 Tagging 结果与实际需求出现偏差, 进而导致自动 Tagging 服务在生产环境下的可用性风险。

## 基于 LLM 的内容 Tagging 实践

为解决自动 Tagging 问题, 我设计了一种基于 LLM 的内容 Tagging 系统, 该系统通过构建专业提示词引导 LLM 从预定义标签池中选择合适标签，结合字符串匹配算法实现模糊标签名到标签 ID 的映射，为内容 Tagging 提供了自动化解决方案.

### 系统设计

- **标签池管理模块:** 负责从远程服务器获取可用标签列表.

    通过指定语言, 从内容服务器获取不同的标签列表:

    ```python
    TAG_POOLS = {'繁中': None, '英语': 'en', '西语': 'sp', '葡语': 'pt', '法语': 'fr',
        '俄语': 'ru', '印尼语': 'id', '泰语': 'th', '韩语': 'ko'}
        
    ...

    def get_tag_pool(lang: Lang) -> list:
        tag_pool_id = None
        match lang:
            case Lang.zh_cn:
                tag_pool_id = TAG_POOLS['英语']
            case Lang.en_us:
                tag_pool_id = TAG_POOLS['英语']
            case _:
                ...  # 目前只支持英语
        tag_pool = requests.get(f'https://authorserver-{tag_pool_id}.xxx.com/api/ProductionComplain/GetTagAllList?'
                                'key=e5b274b4945').json().get('data').get('list')
        tag_pool = [x for x in tag_pool if x.get('tag')]
        return tag_pool
    ```

- **标签匹配模块:** 将生成的模糊标签映射为系统内的标签ID, 确保标签匹配的高准确率.

    > **假设:** 模糊标签和其对应的真实标签之间的公共子序列要长于其与其他真实标签的公共子序列.

    通过文本相似度算法将生成的模糊标签映射到系统内的标签 ID, 这里采用的文本相似度算法为**最大公共子序列算法 (LCS)** (基于动态规划实现):

    ```python
    def longest_common_substring(s_0: str, s_1: str, ignore_case: bool = False) -> str:
        if ignore_case:
            s_0 = s_0.lower()
            s_1 = s_1.lower()
        m = len(s_0)
        n = len(s_1)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        max_length = 0
        end_index = 0
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s_0[i - 1] == s_1[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                    if dp[i][j] > max_length:
                        max_length = dp[i][j]
                        end_index = i - 1
                else:
                    dp[i][j] = 0
        if max_length == 0:
            return ''
        else:
            return s_0[end_index - max_length + 1: end_index + 1]

    def get_tag_id(tag: list | str, lang: Lang) -> list | int:
        tag_pool_map = get_tag_pool(lang)
        tag_pool = [x.get('tag', '') for x in tag_pool_map]

        def __tag_match(_tag: str) -> str:
            nonlocal tag_pool
            distances = list()
            for t in tag_pool:
                _tup = t, - len(longest_common_substring(_tag, t, ignore_case=True))
                distances.append(_tup)
            return sorted(distances, key=lambda x: x[1], reverse=False)[0][0]

        def __tag_to_id(_tag: str) -> int:
            for t in tag_pool_map:
                if _tag in t.get('tag', ''):
                    return t.get('id', 0)
            return 0

        return [__tag_to_id(__tag_match(x)) for x in tag] if isinstance(tag, list) else __tag_to_id(__tag_match(tag))
    ```

- **标签生成模块:** 通过构建提示词, 调用LLM生成标签.

    > **假设:** 一篇小说的开头和结尾可以决定其内容特征, 基于该假设压缩提示词长度, 加快生成速度.

    通过构建提示词, 引导 LLM 生成匹配的模糊标签, 将生成的模糊标签转为列表, 再通过文本相似度算法将生成的标签映射为 ID:

    ```python
    class Prompt(BasePrompt):
        def __init__(self, novel: str, tag_pool: list | str, tags_expected: int):
            super().__init__()
            self._prompt = {
                '小说原文': f'{novel[:1024]}...{novel[-1024:]}',
                '任务与目标': '你的目标是在认真阅读[小说原文]的基础上, 为[小说原文]打上标签.',
                '背景知识-标签池': tag_pool,
                '打标签指南': f'你只能从[标签池]中谨慎地选择{tags_expected}个已知的且与[小说原文]**最相关**的标签.',
                '需要的标签数量': tags_expected,
                '输出格式': 'list',
                '输出样例': [f'tag_{x + 1}' for x in range(tags_expected)]
            }
    ```

    ```python
    class TagGen(BaseGen):
        def __init__(self, lm_name: str = language_model_name, temperature: float = .2,
                    top_p: float = .1, random_seed: int = seed):
            super().__init__(
                lm_name=lm_name,
                temperature=temperature,
                top_p=top_p,
                random_seed=random_seed
            )

        def gen(self, novel: str, lang: Lang, tags_expected: int = 5) -> list[str]:
            prompt_str = Prompt(novel=novel, tag_pool=[x.get('tag') for x in get_tag_pool(lang)],
                                tags_expected=tags_expected).prompt
            tags = lm_invoke(msgs=[
                {'role': 'user', 'content': prompt_str},
                {'role': 'system', 'content': '请遵循任务要求为小说打上标签'}
            ], max_tokens=8192, lm_name=self._language_model_name,
                temperature=self._temperature, top_p=self._top_p,
                random_seed=self._random_seed)
            tags = (tags[tags.find('[') + 1: tags.rfind(']')]
                    .replace(' ', '')
                    .replace("'", '')
                    .replace('"', '')
                    .split(','))
            tag_ids = get_tag_id(tags, lang)
            return tag_ids
    ```

### 实验

- 标签池(以英文为例): 

    通过 API 请求得到库中的所有标签:

    ```json
    {"status":true,"code":"1","message":"Receive Successfully","data":{"list":[{"id":304322,"tag":"R18+"},{"id":306322,"tag":"Legend"},{"id":307322,"tag":"Crime"},{"id":308322,"tag":"Family"},{"id":310322,"tag":"Thriller"},{"id":315322,"tag":"Myth"},{"id":316322,"tag":"Mystery"},{"id":318322,"tag":"Adolescence"},{"id":320322,"tag":"Suspense"},{"id":322322,"tag":"Humor"},{"id":326322,"tag":"Modern"},{"id":328322,"tag":"Fantasy"},{"id":337322,"tag":"Gold digging"},{"id":340322,"tag":"Betrayal"},{"id":341322,"tag":"Forced love"},{"id":346322,"tag":"First love"},{"id":355322,"tag":"Revenge"},{"id":359322,"tag":"Pregnancy"},{"id":371322,"tag":"Divorce"},{"id":373322,"tag":"Secret relationship"},{"id":379322,"tag":"Love triangle"},{"id":380322,"tag":"Flash marriage"},{"id":381322,"tag":"Teacher and student"},{"id":391322,"tag":"Online dating"},{"id":395322,"tag":"Love at first sight"},{"id":404322,"tag":"Curse"},{"id":405322,"tag":"Sexual slave"},{"id":407322,"tag":"Cute Baby"},{"id":408322,"tag":"Celebrities"},{"id":417322,"tag":"Vampire"},{"id":418322,"tag":"Fairy"},{"id":419322,"tag":"CEO"},{"id":426322,"tag":"Doctor"},{"id":432322,"tag":"Twins"},{"id":433322,"tag":"Lawyer"},{"id":434322,"tag":"Lolita"},{"id":439322,"tag":"Police"},{"id":440322,"tag":"Bodyguard"},{"id":445322,"tag":"Multiple identities"},{"id":448322,"tag":"Mafia"},{"id":449322,"tag":"Nurse"},{"id":450322,"tag":"Playboy"},{"id":453322,"tag":"Housekeeper"},{"id":467322,"tag":"Scheming"},{"id":473322,"tag":"Attractive"},{"id":483324,"tag":"Contract marriage "},{"id":483325,"tag":"One-night stand"},{"id":483328,"tag":"High school"},{"id":483329,"tag":"Friends to love "},{"id":483331,"tag":"Rebirth/Reborn"},{"id":483333,"tag":"Time traveling"},{"id":483334,"tag":"Alpha"},{"id":483335,"tag":"Prince"},{"id":483336,"tag":"Royalty "},{"id":483338,"tag":"Badboy"},{"id":483339,"tag":"Badgirl"},{"id":483340,"tag":"Soldier "},{"id":483341,"tag":"Secretary "},{"id":483342,"tag":"Neighbor "},{"id":483346,"tag":"Killer"},{"id":483347,"tag":"Witch/Wizard"},{"id":483348,"tag":"Duke"},{"id":483350,"tag":"Sweet"},{"id":483351,"tag":"BE"},{"id":483352,"tag":"Drama"},{"id":483354,"tag":"GXG"},{"id":483355,"tag":"BXB"},{"id":483357,"tag":"Age gap"},{"id":483358,"tag":"Bully"},{"id":483359,"tag":"Office romance"},{"id":483360,"tag":"Twist"},{"id":483361,"tag":"Magical"},{"id":483364,"tag":"Lust/Erotica"},{"id":483365,"tag":"Arrogant/Dominant"},{"id":483367,"tag":"Noble"},{"id":483369,"tag":"Knight"},{"id":483370,"tag":"Mediaeval"},{"id":483372,"tag":"Romance"},{"id":483373,"tag":"Billionaires"},{"id":483375,"tag":"Workplace"},{"id":483376,"tag":"Kickass Heroine"},{"id":483442,"tag":"The King of Soldiers"},{"id":483443,"tag":"Runaway bride"},{"id":483444,"tag":"Mafia"},{"id":483445,"tag":"Hidden identities"},{"id":483446,"tag":"Rebirth"},{"id":483447,"tag":"Werewolf"},{"id":483448,"tag":"Substitute wife"},{"id":483449,"tag":"Flash Marriage"},{"id":483450,"tag":"Arranged marriage"},{"id":483451,"tag":"Transactional love"},{"id":483452,"tag":"Forbidden love"},{"id":483453,"tag":"Age gap"},{"id":483454,"tag":"Comeback"},{"id":483455,"tag":"Second chance"},{"id":483456,"tag":"Queen"},{"id":483457,"tag":"Unique occupation"},{"id":483767,"tag":"Revenge"},{"id":483768,"tag":"Female-centered"},{"id":483769,"tag":"Substitute bride"},{"id":483770,"tag":"CEO"},{"id":483771,"tag":"Divorce"},{"id":483772,"tag":"Pregnancy"},{"id":483773,"tag":"One-night stand"},{"id":483774,"tag":"Teen romance"},{"id":483775,"tag":"Personal growth"},{"id":483776,"tag":"Ex-wife"},{"id":483777,"tag":"Vampire"},{"id":483778,"tag":"Badboy"}]}}
    ```

- 测试程序

    ```python
    if __name__ == '__main__':
    novel = '''
    这种反差令人作呕。他勉强维持着公寓的房租，又开始给方便面，象征着菲亚特的自主和为凯文的无能和生活方式提供资金。“未来之家”的账户金额微乎其微。她牺牲了凯文的利益，换来了自己和凯文的经济安宁。
    搬回老家让他感到谦卑，但也让他净化了心灵。他买的每一件二手家具，他亲自挑选的每一罐油漆，都是为了从索菲亚那令人窒息的野心伪装中，重新找回他的独立和身份。
    分手几周后，伊森依然感到心痛，他参加了一场行业颁奖晚会。他因领导的一个项目获得了提名，而这个项目在 Innovatcch 成立之前甚至还没有成立。他差点就没去。
    然后他看到了他们。索菲亚和凯文·万斯一起来了。不像是首席执行官和下属。他们看起来像一对情侣。她的手充满占有欲地搭在他的胳膊上。凯文穿着一身昂贵的西装，打扮得光鲜亮丽，对着镜头微笑。
    后来，在一段尴尬的舞台表演中，索菲亚和凯文为了公关活动参加了一个愚蠢的“团队建设”游戏。她一边笑着，一边配合，抬头看着凯文，脸上带着一种伊森曾经以为只有他才会有的表情。这个女人总是声称，在他们长期的婚约面前，公开示爱是“不专业的”。他看着这一切，心里涌起一股冰冷的疏离感。痛苦依然存在，但这种痛苦被一种残酷的理解所掩盖。这就是现在的她。
    索菲亚看见他穿过舞厅。一瞬间，她的笑容僵住了。愧疚？后悔？凯文注意到她的目光，顺着她的目光看向伊森，这笑容瞬间消失了。凯文笑了笑，然后凑近索菲亚耳边低语了几句，又让她笑了起来，声音有点大。这是故意的、幼稚的挑衅。
    伊森毫无反应。他平静地抿了指控，这遭到了强大人物对其生活和能力的傲慢评判，揭示了他们根深蒂固的看法。)
    他没主动去找他们。如果索菲亚主动找他，她会怎么说？评判他的单身？评判他缺少伴侣？他几乎能听出她那傲慢的语气，她觉得他现在的生活似乎低人一等，因为他不在她身边，支撑着她的雄心壮志。
    他心里苦涩地讽刺着：“好吧，至少她终于公开和伴侣在一起了。”他把注意力集中在自己的酒上，冰块碰撞的叮当声，在喧闹的房间里，只听得清脆而冰冷。凯文试图再次吸引他的目光，无论是幸灾乐祸还是挑衅他，都徒劳无功。
        '''
        tags = TagGen().gen(novel, Lang.zh_cn)
        print(tags)
    ```

- 测试结果

    ```python
    [DEBUG] 2025-06-07 13:09:22,929 novgen.actor : Tag generated: ['Betrayal', 'Lovetriangle', 'Drama', 'Workplace', 'Transactionallove'], Tag id: [340322, 379322, 483352, 483375, 483451], Original Content: "
    [340322, 379322, 483352, 483375, 483451]
    ```

    生成的标签为: `['Betrayal', 'Lovetriangle', 'Drama', 'Workplace', 'Transactionallove']`

    对应的标签ID为: `[340322, 379322, 483352, 483375, 483451]`
