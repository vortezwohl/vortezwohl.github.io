---
layout: post
toc: true
title: "Waffo 与 Pancake.waffo 的区别：PSP 支付通道与 MoR 记录商户完整研究"
categories: Finance
tags: [Waffo, Pancake, Payment, Merchant of Record, MoR, PSP, 独立开发者, AI变现]
author:
  - 吴子豪
  - vortezwohl
  - ZCode
excerpt: >-
  "Waffo 和 Pancake 是不是两家支付服务商？"——不是。Pancake（Waffo Pancake，入口 pancake.waffo.com）是 Waffo 旗下的 Merchant of Record（MoR）产品线：Waffo 主平台是 PSP 支付通道，一次接入 173 个市场 480+ 种支付方式，但你是法律上的卖方，税务与拒付自己扛；Pancake 则让 Waffo 当发票上的卖方，向 47 个税务机关季度申报税务、提供海外法律主体、自动开 9 种语言发票、端到端处理拒付争议，责任按合同转移给 Waffo。本文基于 2026-10-21 的官方页面、官方定价页、第三方实测对比与 new-api 集成 issue 给出完整对照：两种模式的本质差异、费率与隐性成本（国内卡 2.9%+$0.50 起 / 全球卡 3.9%+$0.50 / 拒付 $25/笔 / 提现 1% 最低 $10）、中国开发者视角的微信与支付宝限制、new-api 卖 AI 额度的现成通道与 USD 定价坑，并按零信任原则明确区分已验证与未验证事实。
---

> **用途与边界**：本文是 2026-10-21 时点对 Waffo 与 Waffo Pancake 的调研结论，费率、覆盖范围、开通政策均可能继续变动，决策前务必以 [waffo.com](https://waffo.com) 当期页面与签约合同为准。文中区分"已验证"与"未验证"两类事实（见第七节）。调研环境无法直接解析 `pancake.waffo.com` 域名（DNS 失败），关于 Pancake 的事实均来自 waffo.com 官方页面与第三方评测的交叉印证。

## 〇、TL;DR：一句话结论

**Waffo 和 Pancake 不是两家公司，而是同一家公司的两种模式：**

- **Waffo**（waffo.com）＝平台本体：一家总部在香港的全球支付与变现基础设施公司，一次 API 接入覆盖 173 个市场的 480+ 种支付方式。它是 **PSP（Payment Service Provider，支付服务提供商）**：钱从你这里收，**你**是法律意义上的卖方（merchant of record），税务申报、拒付争议、合规责任都在你身上；
- **Pancake**（Waffo Pancake，入口子域 pancake.waffo.com）＝Waffo 旗下的 **Merchant of Record（MoR，记录商户）产品线**：Waffo 变成发票上的卖方，合同约定每笔交易的税务、合规、拒付责任止于 Waffo。面向不想在海外注册公司的个人开发者与 AI 创业公司。

类比：**Waffo ≈ Stripe（通道），Pancake ≈ Paddle / Lemon Squeezy（MoR）**。选哪个的本质问题只有一个：你愿不愿意自己当"法律上的卖家"。

为什么容易误以为是两家：Pancake 有独立子域名、独立 X 账号（@WaffoPancake）、独立文档站（waffo.mintlify.app）；new-api 等集成方在支付渠道列表里也把 "Waffo Pancake" 列为独立渠道名；且搜索引擎搜 "pancake.waffo" 经常返回的全是煎饼食谱（还会混入加密项目 PancakeSwap），官方信息反而沉底。但官网 [Merchant of Record 页面](https://waffo.com/en/merchant-of-record) 明确写着 "Waffo Pancake acts as the seller of record for every transaction"——它是 Waffo 的服务名，用同一个 Waffo 账号开通。

---

## 一、Waffo 主平台：PSP 模式

### 1.1 公司背景

- 主体：Waffo Hong Kong Limited，总部香港；定位"为 AI 时代打造的支付 API 公司"；
- 2026-02 官宣累计融资 3,000 万美元，其中 A 轮超过 1,500 万美元（官网首页公告条）；
- 官网合作方墙：New API、Lovart、OiiOii、DANA、GCash、TNG Digital、HSBC、Illuminate Financial 等——值得注意的是 **new-api（AI 网关项目）在列**，这与"AI 公司卖 credits/tokens"的第一目标行业定位直接相关。

### 1.2 产品矩阵

| 产品线 | 内容 |
|---|---|
| Payments | Checkout（托管收银台）、Payment Links、统一支付 API；173 市场 / 480+ 支付方式 |
| Revenue | 订阅与账单管理（Billing）、失败支付挽回（Recover）、反欺诈（Fraud Protection） |
| Money Management | 全球提现（Payouts）、汇率换汇（FX） |
| Pancake | MoR 服务（本文第二节） |
| Skills | 面向 AI Agent 的支付技能包，官方明确支持由 Agent 驱动接入 |

### 1.3 PSP 模式意味着什么

官方对 PSP 模式的定义很直白："A PSP only moves the money — shipping, tax, and chargebacks stay on you."（PSP 只负责资金流转，交付、税务和拒付仍由你承担。）落到实操上：

- 销售涉及的增值税 / 销售税 / 消费税**由你自行申报缴纳**（或另购税务服务）；
- 拒付（chargeback）争议**由你自己应诉**，通道方只收手续费；
- 消费者保护、发票合规由你负责；
- 作为交换，PSP 模式费率更低、支付方式覆盖更广、到账与账户控制权更大。

---

## 二、Waffo Pancake：MoR 模式

MoR（Merchant of Record，记录商户）＝交易单据上的法律卖方。官方表述："Waffo is the seller on the invoice — the liability for every transaction stops at us by contract."（Waffo 是发票上的卖方，合同约定下每笔交易的责任由我们承担。）

Pancake 替你承担的七件事（官方 MoR 页面）：

1. **销售税 / VAT / GST**：结账时自动计算，向 47 个税务机关季度申报缴纳；
2. **法律主体**：Waffo 持有海外经营所需的公司实体与税务登记，你不需要在境外注册公司；
3. **全球收款**：Visa / Mastercard / Apple Pay / Google Pay 等卡与钱包；
4. **五种货币定价**：USD / EUR / GBP / JPY / HKD；
5. **买家可用 135 种货币付款**：按真实汇率结算，不加汇率 Markup；
6. **合规发票**：自动生成 9 种语言的发票；
7. **欺诈防护与拒付争议**：争议由 Waffo 组装证据应诉到底。

目标用户与口号："you build the product, Waffo runs the payments"——明确面向个人开发者与 AI 初创公司，**个人与非法人主体可以申请**（第三方评测口径）。代价：生产环境开通需要过 Waffo 的资格审核；且 Pancake 当前支持的支付方式是主平台的**子集**（官方口径："接入 Pancake 当前支持的支付方式，并随区域覆盖扩大拓展业务"）。

---

## 三、核心区别对照表

| 维度 | Waffo 主平台（PSP） | Waffo Pancake（MoR） |
|---|---|---|
| 法律卖方 | **你** | **Waffo** |
| 税务申报 | 自己处理 | Waffo 向 47 个税务机关季度申报 |
| 海外主体需求 | 需要 | 不需要 |
| 拒付争议 | 自己应诉（$25/笔手续费） | Waffo 端到端处理 |
| 发票 | 自己开 | Waffo 自动开（9 种语言） |
| 支付方式覆盖 | 480+ / 173 市场 | 子集，随区域扩大 |
| 定价货币 | 按支付方式而定 | 5 种（USD/EUR/GBP/JPY/HKD），买家可付 135 种 |
| 开通门槛 | 即接即用 | 生产开通需资格审核 |
| 适合谁 | 有海外主体、量级大、要全覆盖 | 个人开发者、AI 独立开发、无海外主体 |
| 定位类比 | Stripe | Paddle / Lemon Squeezy |

---

## 四、费率与真实成本

### 4.1 官方定价页（waffo.com/en/pricing）

| 项目 | 价格 |
|---|---|
| 国内卡（domestic） | 2.9% + $0.50 起 / 成功交易 |
| 全球 / 跨境卡 | 3.9% + $0.50 / 成功交易 |
| 开通费 / 月费 | 无 |
| 拒付费 | $25/笔（叠加在交易费之上，仅成功支付被争议时收） |

### 4.2 第三方实测补充（anyway.sh 对比评测）

- 标准卡 / 数字钱包：3.9% + $0.50（与同类 MoR 产品 Anyway 持平）；
- 微信支付：3.9%、无固定费（仅限一次性合格订单）；
- 提现：**1%，最低 $10**——提 $500 实际费率 2%，提 $2,000 才降到 1%；
- 退款：原交易费不退，**另收 $1/笔**退款请求；
- ⚠️ 提现与退款费率来自第三方评测，未在官方定价页逐字核实。

### 4.3 怎么理解"MoR 更贵"

MoR 费率比纯通道高几个点，但价格里包了税务合规成本。自建等价能力（跨境税务申报服务 + 海外主体维护 + 拒付应对）远高于这点差价；对体量小的个人开发者，MoR 几乎总是更划算的选择。真正要警惕的是提现最低消费（$10）和退款附加费这类**隐性成本**，按自己的客单价和退款率算一遍真实成本再下结论。

---

## 五、中国开发者视角

结合 anyway.sh 实测与 new-api 社区信息，中国开发者最关心的几件事：

1. **买家侧支付宝尚不可用**：官方材料里的 Alipay 是**提现**通道（商户收款到支付宝），买家侧支付宝"尚未开放"——看到支付宝 logo 别误以为国内用户能买；
2. **微信支付限制**：仅限中国大陆与香港的一次性订单（USD/CNY），**不支持订阅**，且买家侧有累计月度限额——回头客可能在小额订单上就失去支付资格；
3. **订阅**：文档化的订阅续扣流程是卡 / USD，不要假设微信等其它方式能续扣；
4. **提现**：银行转账与支付宝均可（Finance 页），开户时需确认收款主体类型与币种；审核不保证通过（实名验证、产品审核、收款人姓名一致均适用）；
5. **new-api 官方集成**：Waffo Pancake 已作为 [new-api](https://github.com/QuantumNous/new-api)（AI 网关）的官方支付通道上线，卖 AI tokens / credits 可直接挂。但 [issue #7107](https://github.com/QuantumNous/new-api/issues/7107)（2026-08-30 开启）指出：checkout 币种在 `service/waffo_pancake.go` 中**硬编码为 USD**，非 USD 定价的商品直接"拉起支付失败"，钱包充值与订阅支付均受影响；截至本文时点该 issue 仍为 open 状态，无人工维护者回复。**结论：挂 new-api + Waffo Pancake，商品请直接用 USD 定价。**

---

## 六、怎么选

- **有海外公司主体、量级大、要全支付方式覆盖** → Waffo 主平台（PSP）：费率更低、覆盖 480+ 方式，自己扛税务与拒付；
- **个人开发者 / AI 独立开发 / 没有海外主体 / 不想碰跨境税务** → Pancake（MoR）：贵几个点，买断"法律卖方"这个角色；
- **在 new-api 上卖 AI 额度** → Pancake 是现成的官方通道，但注意第五节的 USD 定价限制与微信/支付宝的买家侧限制；
- **无论选哪个**：先用真实买家设备、目标市场、目标币种做一笔测试支付再放量（anyway.sh 的建议，认同）；把提现最低消费、退款附加费计入定价模型。

---

## 七、零信任声明：已验证 vs 未验证

**✅ 已从官方页面核实：**

- Pancake 是 Waffo 的 MoR 产品、同一 Waffo 账号开通；"PSP 只搬钱 vs Waffo 当发票卖方"的官方对比表述；
- MoR 七项服务及数字：47 个税务机关季度申报、5 种定价货币、买家 135 种付款货币、9 种语言发票；
- 官方定价：国内卡 2.9% + $0.50 起、全球卡 3.9% + $0.50、无开通费/月费、拒付费 $25/笔；
- 公司定位、173 市场 / 480+ 支付方式、2026-02 累计融资 $3,000 万（A 轮超 $1,500 万）；
- new-api 官方集成存在；issue #7107 的 USD 硬编码问题与 open 状态（2026-08-30 开启）。

**⚠️ 未验证 / 来自第三方：**

- `pancake.waffo.com` 子域名本身：调研环境 DNS 无法解析，未能直接访问核验；
- 提现 1%（最低 $10）、退款附加 $1/笔、微信支付 3.9% 无固定费——来自 anyway.sh 第三方评测；
- 微信支付买家侧累计月度限额的具体数值；
- 个人 / 非法人主体可申请、资格审核的具体标准与通过率；
- 国内卡 2.9% 档位的适用条件（收单侧主体要求等）；
- Waffo 的前身业务与更早期历史（本次未深挖）。

---

## 参考来源

- [Waffo 官网（英文）](https://waffo.com/en)
- [Waffo 官网（中文）](https://waffo.com/zh)
- [Waffo Pancake — Merchant of Record + Payout（官方）](https://waffo.com/en/merchant-of-record)
- [Waffo Pricing（官方定价页）](https://waffo.com/en/pricing)
- [Waffo Pancake vs Anyway（第三方实测对比）](https://anyway.sh/blog/anyway-vs-waffo-pancake)
- [new-api issue #7107：Waffo Pancake 结账币种硬编码 USD](https://github.com/QuantumNous/new-api/issues/7107)
- [Waffo Pancake 官方 X 账号](https://x.com/WaffoPancake)
- [Waffo Orders & Payments 文档（镜像站）](https://waffo.mintlify.app/features/orders-payments)
- [MkSaaS 的 Waffo 集成文档](https://mksaas.com/zh/docs/payment/waffo)
