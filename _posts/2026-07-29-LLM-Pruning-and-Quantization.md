---
layout: post
toc: true
title: "深入模型剪枝与量化：LLM 原理、工程实践与代码实现"
categories: AI
tags: [AI, LLM, Quantization, Pruning, Model Compression, Inference]
author:
  - vortezwohl
  - 吴子豪
excerpt: "模型剪枝和量化都属于模型压缩，但回答的是两类不同的问题：剪枝要决定哪些权重、通道、注意力头或层可以删除；量化要决定如何以更少比特近似原有数值。大语言模型的真实部署不能只比较 checkpoint 从多少 GB 变成多少 GB，还要同时理解权重显存、KV Cache、预填充与逐 token 解码、目标硬件内核、并发、首 token 延迟、长上下文质量和安全能力。本文从矩阵乘法与浮点数表示出发，系统梳理非结构化、N:M 和结构化剪枝，PTQ、QAT、权重/激活/KV Cache 量化，并解释 GPTQ、AWQ、SmoothQuant、SparseGPT、Wanda、LLM-Pruner 的关键直觉。文中给出 PyTorch、Transformers + bitsandbytes 和 llama.cpp/GGUF 的操作代码，最后准备 55 个可继续追问的高频面试问答，帮助新手建立从原理到一线工程决策的完整框架。"
---

> 本文按截至 **2026 年 7 月**可公开核验的论文与开源项目文档整理。论文结果依赖模型、校准集、位宽、硬件和运行时；不要把某篇论文中的压缩率或速度比直接当作线上承诺。部署前必须在目标模型、目标序列长度、目标硬件和业务集上复测。

## 先建立地图

神经网络推理里最常见的操作是矩阵乘法：

$$Y=XW+b$$

其中 $W$ 是权重。模型压缩有三条主要路线：

| 方法 | 改变什么 | 主要收益 | 常见误区 |
|---|---|---|---|
| 剪枝 | 删除权重、通道、头、层 | 少算或少访存 | 零值不自动带来 GPU 加速 |
| 量化 | 每个数使用更少比特 | 少占显存/内存，可能更快 | 4 bit 不必然四倍快 |
| 蒸馏 | 训练更小学生模型 | 得到真正小的稠密模型 | 需要数据和训练成本 |

**一句话：剪枝减少“要算多少元素”，量化减少“每个元素占多少比特”。** 两者可组合，但不能把压缩率当成端到端加速比。

真正要先回答五个问题：显存瓶颈是权重还是 KV Cache？目标是吞吐、TTFT（首 token 延迟）还是逐 token 延迟？运行时有目标低比特/稀疏 kernel 吗？质量按 PPL、领域正确率、JSON 合法率、长上下文还是安全集判断？是否有微调恢复预算？

## 一、LLM 为什么压缩难：先看成本在哪里

decoder-only Transformer 推理分为两段：

- **Prefill**：一次处理整个 prompt，算量集中，常受大矩阵乘影响。
- **Decode**：每次生成一个 token，持续读取全部权重和增长中的 KV Cache，常受内存带宽、cache 和调度影响。

权重裸存储的粗略估算：

$$M_{weight}\approx N_{param}\times b/8$$

7B 模型仅权重，FP16 约 14 GB，4 bit 约 3.5 GB。真实峰值还包括 scale/zero-point 元数据、workspace、运行时缓存和 KV Cache。

KV Cache 量级可简化为：

$$M_{KV}\approx2\times L\times T\times H_{kv}\times d_h\times bytes$$

其中 $L$ 是层数、$T$ 是缓存 token、$H_{kv}$ 是 KV 头数、$d_h$ 是头维度。长上下文和大并发时，KV Cache 往往超过权重，故只做 weight-only 量化并不能解决所有 OOM 或 decode 慢问题。

## 二、量化：用有限格子近似连续浮点数

仿射量化把浮点 $x$ 转为整数 $q$：

$$q=clip(round(x/s)+z,q_{min},q_{max})$$

反量化为：

$$\hat{x}=s(q-z)$$

$s$ 是 scale，$z$ 是 zero-point。对称量化一般令 $z=0$，常用于近零均值的权重；非对称量化更适合偏移的激活。误差由舍入与裁剪组成：异常值会拉大范围，使普通值只能落在很粗的格子中。

### 位宽、粒度与对象

- **FP16/BF16**：常规推理基线；BF16 指数范围更大。
- **INT8**：经常可稳定用于权重或激活路径。
- **INT4/NF4**：LLM 权重常用；NF4 是为近似正态权重设计的非均匀码本。
- **FP8**：依赖较新硬件和专用内核。
- **KV8/KV4**：对长上下文/高并发尤为关键。

粒度从粗到细为 per-tensor、per-layer、per-channel、per-group、per-token。粒度更细通常更准，但 scale 元数据和内核开销增加；W4 中常见 group size 32/64/128 正是折中。

### PTQ、QAT、QLoRA

- **PTQ**：训练后量化。成本低，LLM 部署的默认起点。
- **QAT**：训练时插入 fake quant，模型适应 round/clip 误差，通常更准但昂贵。
- **QLoRA**：冻结量化基座、训练 LoRA adapter 的低显存微调方法；不等价于“无损部署量化”[^8]。

## 三、LLM 量化方法：不要只背名字

### RTN

Round-to-Nearest 是最朴素的逐组四舍五入 baseline。它必须测：若 W8A16 RTN 已达标，就不必引入复杂算法；若 W4 崩溃，通常是异常值和粒度问题。

### GPTQ

GPTQ 的目标不是让量化后权重数字最接近原权重，而是在校准激活 $X$ 上最小化层输出误差：

$$\min_{\hat W}\|XW-X\hat W\|_2^2$$

它以近似二阶信息处理列块，并在量化部分权重后补偿未量化权重，因此能在一次后训练流程中完成低比特权重量化[^1]。典型定位：**PTQ、weight-only、W4A16**。是否加速由权重 pack 格式和兼容 kernel 决定。

### AWQ

AWQ 利用激活统计识别少量显著通道，通过等价缩放保护关键权重通道，再做低比特量化[^2]。它仍是 PTQ；“activation-aware”不等于必然把激活也量为 INT4，常见部署仍为 W4A16。

### SmoothQuant

激活异常值使 W8A8 困难。SmoothQuant 使用 $XW=(X/s)(sW)$ 的等价变换，把激活动态范围的一部分迁移到权重侧，使 W8A8 更可行[^3]。它解释了“权重量化通常容易，激活量化却常失败”的原因。

| 表示 | 价值 | 风险 |
|---|---|---|
| W4A16 | 权重显存显著下降，质量常稳 | decode 仍可能受带宽/反量化影响 |
| W8A8 | 可利用低精度 GEMM | 激活 outlier 与校准敏感 |
| W4A4 | 极低带宽 | 精度和硬件支持压力大 |
| KV8/KV4 | 长上下文、高并发省显存 | 注意力误差需要专项评测 |

## 四、剪枝：删掉什么才能真的更快

### 非结构化剪枝

按最小 $|w|$ 删除单个权重。优点是容易达到高稀疏率；缺点是稠密 GPU GEMM 通常仍计算完整矩阵，零不自动省时间。SparseGPT 使用近似二阶信息做一次性高稀疏 LLM 剪枝[^4]；Wanda 使用权重幅值和激活统计的组合重要性，避免昂贵重构[^5]。

### N:M 半结构化剪枝

每连续 $M$ 个元素保留 $N$ 个，例如 2:4。规则布局能匹配部分 NVIDIA 稀疏 kernel。不能说“2:4 必有两倍加速”：GPU 架构、shape、kernel、量化格式与端到端瓶颈都会影响结果。

### 结构化剪枝

删除完整通道、attention head、MLP neuron 或 Transformer block，得到更小的稠密矩阵，最容易在通用硬件上真实加速。代价是需要处理残差、投影、GQA/MQA、配置和 checkpoint 依赖；LLM-Pruner 是结构化 LLM 剪枝代表[^6]。

| 目标 | 优先选择 |
|---|---|
| 论文 sparsity/压缩率 | 非结构化 SparseGPT/Wanda |
| 特定稀疏硬件 | N:M |
| 通用部署的真实延迟 | 结构化剪枝或更小稠密模型 |
| 没有微调预算 | 先做成熟 PTQ 量化 |

## 五、剪枝与量化组合的正确顺序

默认建议：**先结构化剪枝和恢复微调，再做 PTQ 量化，最后导出目标运行时格式。** 剪枝改变 shape 和数值分布，先量化再大幅剪枝会让 scale、校准和 pack 失配。非结构化稀疏与 int4 可以联合，但只有运行时支持“该稀疏布局 + 该量化格式”的 fused kernel 时才可能获得真实收益。

## 六、可操作代码

### 1. 最小对称量化

```python
import torch


def symmetric_quantize(tensor: torch.Tensor, bits: int = 8) -> tuple[torch.Tensor, torch.Tensor]:
    """对浮点张量执行逐张量对称量化。

    Args:
        tensor: 待量化的浮点张量。
        bits: 有符号整数位宽，取值为 2 到 8。

    Returns:
        整数量化值和反量化所需的 scale。

    Raises:
        ValueError: 位宽不合法时抛出。
    """
    if not 2 <= bits <= 8:
        raise ValueError("bits 必须在 2 到 8 之间")
    qmax = (1 << (bits - 1)) - 1
    scale = tensor.abs().max().clamp_min(torch.finfo(tensor.dtype).eps) / qmax
    quantized = torch.clamp(torch.round(tensor / scale), -qmax, qmax).to(torch.int8)
    return quantized, scale


weight = torch.randn(4096, 4096)
qweight, scale = symmetric_quantize(weight)
reconstructed = qweight.float() * scale
print(((weight - reconstructed).norm() / weight.norm()).item())
```

这仅演示数值映射，没有 group-wise scale、pack、fused GEMM 和 GPU kernel；不可用它的耗时代表部署性能。

### 2. 最小剪枝对比

```python
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

layer = nn.Linear(4096, 11008, bias=False)
prune.l1_unstructured(layer, name="weight", amount=0.30)
print("非结构化零比例", (layer.weight == 0).float().mean().item())
prune.remove(layer, "weight")

# 真正缩小 shape 的结构化“物理裁剪”示例。
scores = layer.weight.detach().norm(p=2, dim=1)
keep = scores.topk(int(scores.numel() * 0.8)).indices.sort().values
smaller = nn.Linear(4096, keep.numel(), bias=False)
smaller.weight.data.copy_(layer.weight.data[keep])
print(smaller.weight.shape)
```

完整 LLM 不能只替换一个层：必须同步下游投影、残差维度、模型 config 与权重映射。生产结构化剪枝应使用专门工具并做恢复微调。

### 3. Transformers + bitsandbytes 加载 4 bit

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

model_id = "meta-llama/Llama-3.1-8B-Instruct"
config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    quantization_config=config,
)
inputs = tokenizer("用一句话解释量化。", return_tensors="pt").to(model.device)
ids = model.generate(**inputs, max_new_tokens=64, do_sample=False)
print(tokenizer.decode(ids[0], skip_special_tokens=True))
```

安装：`pip install -U torch transformers accelerate bitsandbytes`。模型权限、CUDA、GPU 算力和版本兼容性以官方文档为准[^9]；微调通常配合 PEFT/QLoRA，而非直接全量更新 packed 4 bit 权重。

### 4. 端到端生成粗测

```python
import time
import torch

inputs = tokenizer("解释 KV Cache。", return_tensors="pt").to(model.device)
torch.cuda.synchronize()
started = time.perf_counter()
result = model.generate(**inputs, max_new_tokens=128, do_sample=False)
torch.cuda.synchronize()
elapsed = time.perf_counter() - started
new_tokens = result.shape[-1] - inputs["input_ids"].shape[-1]
print(f"{new_tokens / elapsed:.2f} tokens/s")
```

生产压测还要固定 warmup、prompt/输出长度、batch、并发与采样参数，并分别记录 TTFT、ITL、P50/P95、显存峰值和错误率。

### 5. llama.cpp / GGUF 路线

```bash
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release
python convert_hf_to_gguf.py /path/to/hf-model --outtype f16 --outfile model-f16.gguf
./build/bin/llama-quantize model-f16.gguf model-q4_k_m.gguf Q4_K_M
./build/bin/llama-cli -m model-q4_k_m.gguf -p "解释模型量化：" -n 128
./build/bin/llama-bench -m model-q4_k_m.gguf
```

GGUF/`Q4_K_M` 是 llama.cpp 生态格式和量化类型，不等同于 GPTQ/AWQ；同为 4 bit 不能直接横比质量和速度[^10]。

## 七、一线工程的可复现流程

1. **定验收协议**：固定基座 hash、硬件/驱动、运行时版本、prompt 集、长度、并发和解码参数；写清质量与延迟红线。
2. **做 profile**：区分权重、KV Cache、workspace；分别测短/长 prompt 和低/高并发下的 prefill/decode。
3. **选最简单方案**：显存不够先评估成熟 W4A16；长上下文同时看 KV Cache；只有硬件支持和收益明确才做 W8A8/FP8/稀疏。
4. **校准集要代表线上**：覆盖语言、长度、代码、表格和领域符号；不大但要合规、可复现，避免把敏感线上文本外传。
5. **质量门禁不能只看 PPL**：加入业务任务、JSON/schema 合法率、工具调用、长上下文、人工盲评和安全集。
6. **发布可回滚**：artifact 记录算法/位宽/group/校准集/导出命令/报告；灰度监控 OOM、TTFT、ITL、吞吐、拒答和结构化输出失败率。

## 八、常用框架

| 工具 | 用途 | 重要边界 |
|---|---|---|
| PyTorch pruning / torchao | 原型、通用压缩实验 | 原型 API 不等于 LLM 部署栈 |
| Transformers + bitsandbytes | 快速 8/4 bit、QLoRA | 看版本、硬件和真实 kernel |
| AutoGPTQ / GPTQModel | GPTQ 生态 | pack/backend 必须兼容 |
| AutoAWQ / llm-awq | AWQ 生态 | 核对维护状态和运行时 |
| llama.cpp / GGUF | CPU、Apple、边缘 | GGUF 不与 GPTQ/AWQ混用 |
| vLLM + llm-compressor | 服务化与压缩 | 按支持矩阵选版本[^11] |
| NVIDIA ModelOpt / TensorRT-LLM | NVIDIA 高性能部署 | 强依赖硬件/版本[^12] |
| SparseGPT / Wanda / LLM-Pruner | LLM 剪枝研究 | sparse 文件不等于加速 |
| lm-eval-harness + 业务集 | 质量评估 | 必补长上下文和安全 |

## 九、55 个高频面试题

### 基础与数学

**1. 什么是量化？** 用有限整数/低精度格式近似浮点数，降低存储、带宽与可能的计算成本；必须说清对象、位宽、粒度和 kernel。

**2. 什么是剪枝？** 删除不重要的参数或结构，分非结构化、N:M 和结构化。

**3. 二者差异？** 剪枝少元素，量化少比特；前者看重要性与依赖，后者看数值误差与实现。

**4. 为什么 int4 不一定更快？** 无专用 kernel 时反量化/格式转换可能抵消收益，瓶颈还可能是 KV Cache、CPU 或网络。

**5. scale 与 zero-point？** scale 定义整数格宽，zero-point 对齐实数零；权重常对称，激活可非对称。

**6. 对称/非对称如何选？** 近零均值权重常对称；偏移激活可用非对称，仍取决于 kernel。

**7. 为什么细粒度更准？** 每个组的动态范围更贴合分布；代价是 metadata 和内核复杂度。

**8. 量化误差来源？** 舍入与裁剪；异常值会放大普通值分辨率损失。

**9. 为什么激活量化更难？** 激活随输入和 token 变化，并有跨通道 outlier；权重是固定可离线处理的。

**10. PTQ 与 QAT？** PTQ 训练后、成本低；QAT 训练中模拟量化、可能更准但昂贵。

**11. QLoRA 是什么？** 量化冻结基座上训练 LoRA adapter 的低显存微调，不是通用量化算法。

### LLM 量化

**12. RTN 的意义？** 最朴素 baseline，用于判断是否真的需要复杂 PTQ。

**13. GPTQ 优化什么？** 校准激活下的层输出误差，而非权重逐元素误差[^1]。

**14. GPTQ 为什么要校准集？** 输出误差依赖输入激活统计，分布错配会让量化失效。

**15. GPTQ 的 act-order？** 按激活显著性优先处理列的启发式，收益依赖实现与配置。

**16. AWQ 的直觉？** 保护少数激活显著通道，缩放后再量化以降低关键误差[^2]。

**17. SmoothQuant 的直觉？** 将激活异常值的部分尺度迁移到权重侧，改善 W8A8[^3]。

**18. NF4 为什么常见？** 非均匀 4 bit 码本贴近正态权重，常用于 QLoRA。

**19. W4A16 与 W8A8 怎么选？** 显存优先先 W4A16；硬件支持且需更高吞吐时评估 W8A8，并投入校准。

**20. 什么是 KV Cache 量化？** 压缩历史 K/V，长上下文和高并发收益大，但须评估注意力质量。

**21. 为什么 metadata 影响压缩率？** group scale、zero-point、padding、对齐都占空间。

**22. outlier 怎么处理？** 更细粒度、平滑、混合精度或保留异常通道。

### 剪枝

**23. 非结构化剪枝为何不加速？** 稠密 GEMM 不会跳过零；需要稀疏格式和内核。

**24. 结构化为什么易加速？** shape 真正变小，普通稠密 GEMM 也会少算。

**25. N:M 是什么？** 每 M 个值保留 N 个非零，适配特定硬件稀疏路径。

**26. magnitude pruning 的问题？** 小权重也可能对应大激活，不能代表输出重要性。

**27. SparseGPT 特点？** 一次性、近似二阶、可做高稀疏 LLM 剪枝[^4]。

**28. Wanda 特点？** 权重和激活共同评分，避免昂贵重构[^5]。

**29. LLM-Pruner 特点？** 面向结构化 LLM 剪枝及结构依赖[^6]。

**30. 删 attention head 一定安全吗？** 不安全，层和任务敏感度不同，GQA/MQA 还有额外映射依赖。

**31. 剪层还是剪 MLP？** 剪层更激进；剪 MLP 更局部，二者都要回归评测。

**32. 为什么剪后微调？** 剩余权重需重新分担功能，恢复容量突变带来的误差。

**33. 剪枝率等于加速比吗？** 不等于；加速由 kernel、访存、batch 和其他瓶颈决定。

### 工程与部署

**34. 7B 的 FP16/4 bit 如何估？** 裸权重约 14 GB/3.5 GB，再加 KV Cache、workspace 与 metadata。

**35. 长上下文为什么只量权重不够？** KV Cache 随长度和并发增长，可能主导显存。

**36. TTFT 和 tokens/s？** TTFT 看排队/prefill，tokens/s 多看 decode；都要单独报告。

**37. 校准集如何建？** 从目标分布分层抽样，覆盖语言、长度、格式和领域，重在代表性和合规。

**38. 为什么不能只评 PPL？** 不覆盖工具调用、结构化输出、安全、长上下文和业务正确率。

**39. artifact 记录什么？** 基座 hash、算法版本、bits/group、校准描述、导出命令、质量与性能报告。

**40. 灰度监控什么？** OOM、TTFT、ITL、吞吐、错误率、JSON 合法率、任务指标与拒答行为。

**41. 量化后质量下降如何定位？** 固定解码与数据，按任务/长度切片失败样本，检查校准、outlier、group 和 runtime。

**42. 模型变小不变快如何定位？** 分开 profile prefill/decode，检查 kernel、反量化、KV Cache、tokenizer、调度和网络。

**43. 两个 int4 为什么不能直接比较？** 算法、码本、group、校准、pack 与 kernel 都可能不同。

**44. AWQ、GPTQ、GGUF 如何选？** 先看目标运行时支持；格式兼容优先于算法名。

### 设计追问

**45. 先剪还是先量？** 默认先结构化剪枝/恢复，再 PTQ；任何联合方案需重新校准和评测。

**46. 稀疏与量化可叠加吗？** 可以，但必须有同时支持两种布局的 kernel，否则未必有端到端收益。

**47. 什么时候换更小原生模型？** 高压缩恢复成本高、无稀疏 kernel 或延迟仍不达标时，小稠密模型常更稳。

**48. “4 bit 损失多少精度”怎么答？** 没有通用百分比；给出模型、任务、校准、运行时和实测协议。

**49. 为什么混合精度有用？** 层/通道敏感度不同，保留少数敏感部分高精度可小成本保质量。

**50. 量化后能 LoRA merge 吗？** 通常高精度 merge 后再量化，或用 adapter 路径；任意修改 packed 权重不安全。

**51. 量化模型能全量微调吗？** 通常不用 packed 低比特权重直接全量更新，常用 QLoRA/PEFT 或保留 master weights。

**52. fake quant 与 STE？** fake quant 模拟 round/clip；STE 用近似梯度绕过不可导 round，能训练但需验证稳定性。

**53. 量化会影响安全对齐吗？** 会，需把越狱、拒答和误拒率放进回归门禁。

**54. 最成熟的压缩决策原则？** 先 profile 找主瓶颈，再选择硬件真实支持的表示，最后以质量和端到端指标验收。

**55. 如何一句话总结自己的实践？** “我把压缩视为受硬件和业务约束的端到端优化，不把位宽或零比例当结果，而用可复现校准、质量门禁、TTFT/ITL/吞吐和回滚机制证明结果。”

## 参考文献

[^1]: Frantar, E. et al. *GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers*. ICLR 2023. https://arxiv.org/abs/2210.17323
[^2]: Lin, J. et al. *AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration*. MLSys 2024. https://arxiv.org/abs/2306.00978
[^3]: Xiao, G. et al. *SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models*. ICML 2023. https://arxiv.org/abs/2211.10438
[^4]: Frantar, E.; Alistarh, D. *SparseGPT: Massive Language Models Can Be Accurately Pruned in One-Shot*. ICML 2023. https://arxiv.org/abs/2301.00774
[^5]: Sun, M. et al. *A Simple and Effective Pruning Approach for Large Language Models (Wanda)*. ICLR 2024. https://arxiv.org/abs/2306.11695
[^6]: Ma, X. et al. *LLM-Pruner: On the Structural Pruning of Large Language Models*. NeurIPS 2023. https://arxiv.org/abs/2305.11627
[^7]: Dettmers, T. et al. *LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale*. NeurIPS 2022. https://arxiv.org/abs/2208.07339
[^8]: Dettmers, T. et al. *QLoRA: Efficient Finetuning of Quantized LLMs*. NeurIPS 2023. https://arxiv.org/abs/2305.14314
[^9]: Hugging Face. *Transformers Quantization with bitsandbytes*. https://huggingface.co/docs/transformers/quantization/bitsandbytes
[^10]: ggml-org. *llama.cpp README and GGUF tools*. https://github.com/ggml-org/llama.cpp
[^11]: vLLM Project. *llm-compressor README*. https://github.com/vllm-project/llm-compressor
[^12]: NVIDIA. *TensorRT Model Optimizer README*. https://github.com/NVIDIA/TensorRT-Model-Optimizer
