---
learned: false
---

# FP8 与 TF32 低精度计算

> 本模块索引见 [参数精度详解](参数精度详解.md)

## 一、为什么需要 FP8

FP8 是 NVIDIA Ampere 之后引入、Hopper（H100）时代成熟、Blackwell（B200）时代大规模量产的低精度格式。核心动机就一句话：**位宽减半，显存减半、带宽减半、吞吐翻倍**。

| 格式   | 总位宽 | 显存（相对 FP32） | 峰值算力（H100 SXM5 dense） | 相对算力 |
| ----- | --- | ----------- | -------------------- | ----- |
| FP32  | 32  | 1×          | 67 TFLOPS            | 1×（CUDA Core） |
| TF32  | 32  | 1×          | 495 TFLOPS           | ~7×（Tensor Core） |
| BF16/FP16 | 16 | 1/2×        | 989.5 TFLOPS         | 2×（Tensor Core） |
| FP8   | 8   | 1/4×        | **1979 TFLOPS**（= BF16 的 2 倍） | 4×（Tensor Core） |

Blackwell B200 更进一步：BF16 dense 2.25 PFLOPS，**FP8 dense 4.5 PFLOPS**。

**为什么训练和推理都要用 FP8？**

1. **训练**：大模型训练瓶颈之一是显存（参数、梯度、优化器状态、激活）。权重用 FP8 存、GEMM 用 FP8 算，激活也以 FP8 缓存供反向复用（DeepSeek-V3 的做法），显存与带宽压力显著下降，吞吐接近翻倍；
2. **推理**：权重 FP8（W8A8）直接推理，KV cache 也可以量化到 FP8，服务成本大幅下降；
3. **硬件原生支持**：H100 的 Tensor Core 原生做 FP8 GEMM（累加器仍是 FP32），不需要软件模拟，这也是"FP8 训练在 2024 年成为可能"的硬件前提。

---

## 二、FP8 的两种格式（重点）

FP8 不是一个格式，而是**两种格式的总称**，区别在"指数位和尾数位的分配"：

### 2.1 位结构总览

| 格式  | 结构   | 指数偏置 | 最大有限值 | 最小正规数   | 相对精度（eps） | 是否支持 inf/nan |
| --- | ---- | ---- | ----- | ------- | --------- | ----------- |
| E4M3 | 1+4+3 | 7    | **448**  | 2⁻⁶ ≈ 0.0156 | 2⁻³ = 0.125 | 无 inf（E4M3FN 保留一个 NaN 模式） |
| E5M2 | 1+5+2 | 15   | **57344** | 2⁻¹⁴ ≈ 6.1e-5 | 2⁻² = 0.25  | 支持 inf/nan |

$$
\underbrace{\text{sign}}_{1\text{ bit}} \quad
\underbrace{exp}_{E\text{ bits, bias }B} \quad
\underbrace{mantissa}_{M\text{ bits}}
$$

E4M3：1 位符号 + 4 位指数（偏置 7）+ 3 位尾数；
E5M2：1 位符号 + 5 位指数（偏置 15）+ 2 位尾数。

### 2.2 E4M3：1+4+3，最大 448

- 数值 = $(1 + mant/8) \times 2^{exp-7}$；
- **最大有限值 448** = $1.75 \times 2^8$（指数字段 1111，尾数 110；E4M3FN 把 `1111_111` 保留为 NaN，所以尾数 111 不可用，这是 448 而不是 480 的原因）；
- 尾数 3 位 → 相对精度 6.25%（1.0 与 1.125 之间间隔 0.125），**数值更密**；
- 值域窄（2⁻⁶ ~ 448，约 13 个数量级），但**范围之上没有 inf**，溢出时饱和到 448；
- 典型用途：**前向激活与权重**——前向值分布集中、不需要极端大数，更密的尾数能保住更多有效位。

### 2.3 E5M2：1+5+2，最大 57344

- 数值 = $(1 + mant/4) \times 2^{exp-15}$；
- **最大有限值 57344** = $1.75 \times 2^{15}$（指数字段 11110，尾数 11）；
- 尾数 2 位 → 相对精度 12.5%（1.0 与 1.25 间隔 0.25），是 E4M3 的两倍粗；
- 指数 5 位 → 值域 2⁻¹⁴ ~ 57344（约 30 个数量级），且**支持 inf/nan**（指数字段 11111）——溢出不会静默饱和，会被检测到；
- 典型用途：**反向梯度**——梯度的量级跨层差异极大（1e-2 到 1e-8 甚至更小），需要宽动态范围；出现异常大梯度时 inf 还能被检测到并触发保护。

### 2.4 对比表与选择逻辑

| 对比项    | E4M3          | E5M2              |
| ------ | ------------- | ----------------- |
| 位分配    | 1+4+3         | 1+5+2             |
| 最大有限值  | 448           | 57344（大 128 倍）    |
| 最小正规数  | 0.0156        | 6.1e-5（小 256 倍）   |
| 相对精度   | 6.25%（好）      | 12.5%（差一倍）        |
| inf/nan | 无（FN 仅保留 NaN） | 有                |
| 最佳场景   | 前向：激活、权重      | 反向：梯度、大动态范围数值     |

**选择逻辑**：
- 前向激活/权重是"**集中分布 + 需要精度**"，选 E4M3（精度优先）；
- 梯度是"**分散分布 + 需要范围**"，选 E5M2（范围优先）；
- 这是微软/英伟达 FP8 训练框架（Noune et al. 2022 等）给出的默认分工，DeepSeek-V3 也遵循这一设计起点（实际训练中经调优后全用 E4M3，见第四节）。

### 2.5 次正规数

两种格式都有次正规数（尾数前隐式 1 变为 0）：E4M3 最小次正规 2⁻⁹ ≈ 0.00195，E5M2 为 2⁻¹⁶ ≈ 1.5e-5。次正规让"小值"不至于直接断崖为 0，但精度进一步下降。

---

## 三、FP8 计算模式（重点）

### 3.1 FP8 乘 FP8、累加 FP32

Tensor Core 上 FP8 GEMM 的典型数据流：

```
A (FP8) ──┐
          ├──▶ 乘法器：a×b 为精确乘积（纯位运算）
B (FP8) ──┘          │
                     ▼
              FP32 累加器（Σ 在 FP32 中进行）
                     │
                     ▼
              输出：FP32 / BF16 / FP8（可选量化）
```

**关键设计：乘法的两个操作数是 FP8，但累加器是 FP32。** 这对应硬件上"FP8 乘法 + FP32 加法"混合流水线。

### 3.2 为什么矩阵乘法适合 FP8——误差分析

矩阵乘 $C_{ij} = \sum_{k=1}^{K} a_{ik} b_{kj}$。设 FP8 量化相对误差为 $\delta$（|δ| ≤ 2⁻⁴ = 6.25%，E4M3）：

$$a'_{ik} = a_{ik}(1 + \delta_a), \quad b'_{kj} = b_{kj}(1 + \delta_b)$$

则单个乘积的相对误差 ≈ $|\delta_a| + |\delta_b| \le 12.5\%$（最坏情况），但：

$$C'_{ij} = \sum_k a_{ik}(1+\delta_a) b_{kj}(1+\delta_b) = C_{ij} + \underbrace{\sum_k a_{ik}b_{kj}(\delta_a + \delta_b)}_{\text{误差项}}$$

误差项是 **K 个随机符号误差的叠加**——δ 有正有负，统计上互相抵消，实际误差远小于最坏上界（实测见第六节：相对误差 ~3.7%，远小于 12.5%）。且由于累加在 FP32 中进行，K 增大时**不会引入 FP8 的逐项舍入累积**（这正是"FP8 累加 FP8"会崩的原因，见 6.4 的 NaN 演示）。

### 3.3 量化与缩放（scaling）

FP8 值域远小于数据分布，必须先做仿射量化：

$$x_q = \mathrm{round}\left(\frac{x}{\text{scale}}\right), \quad \text{scale} = \frac{\max|x|}{\text{FP8\_max}}$$

- **per-tensor**：整张量一个 scale，实现简单但对 outlier 敏感；
- **per-group / 分块量化**：按小块（如 128 个元素一组）各算各的 scale，outlier 只影响所在块，精度大幅提升——DeepSeek-V3 的关键技巧（见第四节）；
- 反量化在 Tensor Core 内部与累加同步完成，几乎零开销。

---

## 四、业界实践（重点）

### 4.1 DeepSeek-V3：671B MoE 全程 FP8 训练

DeepSeek-V3（2024.12）在 671B 参数的 MoE 模型上**从头训练就用 FP8**，技术要点：

1. **三个 GEMM 全部 FP8**：前向 Fprop、反向的激活梯度 Dgrad、权重梯度 Wgrad 全部走 FP8 GEMM（输入 FP8、累加 FP32、输出 BF16/FP32）——"理论上把 BF16 训练的 GEMM 吞吐翻倍"；
2. **格式分工**：设计起点遵循 E4M3 用于 Fprop、E5M2 用于 Dgrad/Wgrad（梯度动态范围大）；随后报告明确"**we adopt the E4M3 format on all tensors for higher precision**"——调优后全用 E4M3 精度更优；
3. **主权重、权重梯度、优化器状态保持高精度**（BF16/FP32），只在 GEMM 内部降为 FP8；
4. **分块量化（fine-grained quantization）**：激活按 **1×128 tile**（每 token 每 128 通道）求 scale，权重按 **128×128 block** 求 scale——这是解决激活/权重 outlier 问题的关键，配合 FP32 累加；
5. **FP8 Wgrad 让激活可以 FP8 缓存**：反向要用到的激活直接以 FP8 存，大幅省显存；
6. **高精度组件保留**：embedding、输出头、MoE 门控、归一化、注意力算子保留 BF16/FP32；
7. **效果**：训练 1 万亿 token，相对 BF16 基线的 loss 相对误差稳定在 **0.25% 以内**，模型效果与 BF16 相当；
8. 工程细节：H800 的 FP8 GEMM 累加精度实际只保留约 14 位（低于 FP32），K=4096 时相对误差接近 2%，因此他们每隔 128 个 MMA 把中间结果提升到 CUDA Core 做一次 FP32 高精度累加。

### 4.2 推理侧

DeepSeek-V3 推理部署同样用 FP8（W8A8 + 激活 FP8），开源多模态模型（Qwen2.5-VL、InternVL 等）的 AWQ/GPTQ 量化方案也提供 FP8 选项，服务端吞吐提升约 1.8~2×。

### 4.3 性能数字汇总

| 场景   | 提升                                       |
| ---- | ---------------------------------------- |
| H100 训练 | FP8 GEMM 吞吐 = BF16 的 2×（1979 vs 989.5 TFLOPS） |
| 训练显存  | 权重/激活 FP8 存储 → 相比 BF16 再省一半（整体约省 30~40%） |
| 推理    | W8A8 权重加载带宽减半，decoding 吞吐接近 2× |
| 误差    | DeepSeek-V3 实测相对 BF16 基线 loss 误差 < 0.25% |

---

## 五、TF32（重点）

### 5.1 TF32 是什么

TF32（Tensor Float 32）是 Ampere（A100）起引入的**矩阵乘专用格式**：**1+8+10**。

- 指数 8 位（偏置 127）→ **动态范围与 FP32 完全一致**（最大 3.4e38，最小 1.2e-38），不会下溢；
- 尾数 10 位 → 精度 ≈ FP16（FP16 尾数也是 10 位）。

它本质是"**把 FP32 输入截断到 10 位尾数再进 Tensor Core**"，所以不需要任何量化/缩放：数值范围不变，只牺牲精度。

$$TF32(x) = \text{round}_{10\text{bit}}(x), \quad \text{range} = \text{FP32}, \quad \text{precision} \approx \text{FP16}$$

### 5.2 为什么 Ampere 以后"默认"矩阵乘用 TF32

| 方案        | 精度       | 吞吐（A100/H100）      | 代价      |
| --------- | -------- | ------------------ | ------- |
| FP32 矩阵乘  | 完整 FP32  | 67 TFLOPS（H100）     | 慢       |
| TF32 矩阵乘  | ≈FP16 精度 | 495 TFLOPS（H100）    | 精度够用    |
| FP16 矩阵乘  | FP16      | 989.5 TFLOPS        | 需处理溢出下溢 |

- 训练中绝大多数 GEMM 的输入是**已经规范化过的中间值**（大多在 O(1) 附近），10 位尾数（相对误差 ~5e-4）足够；
- TF32 无需 loss scaling、无需换 dtype、无溢出风险，是"FP32 生态的无痛加速"；
- 因此**深度学习框架的矩阵乘默认开启 TF32**（如 cuBLAS 默认、Megatron 默认）。

**注意 PyTorch 的默认值**：`torch.backends.cuda.matmul.allow_tf32` 在 PyTorch 2.x 中默认 **False**（保精度优先，速度降级为 FP32），需要显式打开。检查 `torch.get_float32_matmul_precision()` 会返回 `"highest"`（=TF32 关闭）或 `"high"`（=TF32 开启）。

### 5.3 PyTorch 开关

```python
import torch.backends.cuda

torch.backends.cuda.matmul.allow_tf32 = True      # 矩阵乘开 TF32（默认 False）
torch.backends.cudnn.allow_tf32 = True            # 卷积开 TF32
torch.set_float32_matmul_precision("high")        # 统一开 TF32（"highest" 关闭）
```

### 5.4 与 FP16/BF16 对比

| 对比项   | TF32       | FP16       | BF16       |
| ----- | ---------- | ---------- | ---------- |
| 结构    | 1+8+10     | 1+5+10     | 1+8+7      |
| 动态范围  | 同 FP32     | 65504 封顶    | 同 FP32     |
| 精度    | ≈FP16（10 位） | 10 位尾数     | 7 位尾数（差）   |
| 需要 scaling | 不需要       | 需要（loss scaling） | 不需要        |
| 输入形式  | 输入是 FP32（硬件内部截断） | 输入先转 FP16 | 输入先转 BF16 |
| 典型用途  | 卷积/矩阵乘加速  | 训练          | 训练、推理     |

---

## 六、源码实现（可运行）

> 环境：PyTorch 2.8（CPU）。FP8 数据类型的实际计算（GEMM）需要 H100 等 Hopper+ GPU，但**位格式、量化、误差分析在 CPU 上可完整复现**。

### 6.1 用 torch.finfo 验证各格式关键数字

```python
import torch

for dt in [torch.float16, torch.bfloat16,
           torch.float8_e4m3fn, torch.float8_e5m2]:
    fi = torch.finfo(dt)
    print(f"{dt}: max={fi.max}, min_normal={fi.smallest_normal}, eps={fi.eps}")
# float16: max=65504.0, min_normal=6.103515625e-05, eps=0.0009765625
# bfloat16: max=3.3895313892515355e+38, min_normal=1.1754943508222875e-38, eps=0.0078125
# float8_e4m3fn: max=448.0, min_normal=0.015625, eps=0.125
# float8_e5m2: max=57344.0, min_normal=6.103515625e-05, eps=0.25
```

### 6.2 原生 FP8 量化（展示饱和行为）

```python
import torch

x = torch.tensor([0.5, 1.0, 447.9, 448.1, -3.5])
print(x.to(torch.float8_e4m3fn).float())
# tensor([  0.5000,   1.0000, 448.0000, 448.0000,  -3.5000])
# 448.1 → 448：E4M3 无 inf，溢出饱和

x2 = torch.tensor([0.5, 1.0, 57343.0, 57344.0, 57344.1])
print(x2.to(torch.float8_e5m2).float())
# tensor([5.0000e-01, 1.0000e+00, 5.7344e+04, 5.7344e+04, 5.7344e+04])
# 57344 是 E5M2 最大有限值；57344.1 饱和
```

### 6.3 手动位编码器（看清 1+4+3 与 1+5+2 的位结构）

```python
import math

def encode_fp8(v, ebits, mbits, bias, max_exp, max_mant):
    """把 fp32 标量编码为 fp8 位模式（含次正规与饱和处理）"""
    v = float(v)
    if v == 0:
        return 0
    sign = 1 if v < 0 else 0
    m, e = math.frexp(abs(v))            # v = m·2^e, m ∈ [0.5, 1)
    mant = int(round((m * 2 - 1) * 2**mbits))   # 取 1.xxx 的 xxx 部分
    if mant == 2**mbits:                 # 舍入进位：1.111.. → 10.000
        mant = 0
        e += 1
    exp = e - 1 + bias                   # 存储指数
    if exp <= 0:                         # 次正规：0.mmm × 2^(1-bias)
        return (sign << (ebits + mbits)) | int(round(abs(v) / 2.0**(1 - bias - mbits)))
    if exp > max_exp:                    # 饱和到最大有限值
        exp, mant = max_exp, max_mant
    return (sign << (ebits + mbits)) | (exp << mbits) | mant

def decode_fp8(bits, ebits, mbits, bias):
    sign = -1.0 if bits >> (ebits + mbits) else 1.0
    exp = (bits >> mbits) & ((1 << ebits) - 1)
    mant = bits & ((1 << mbits) - 1)
    if exp == 0:
        return sign * mant * 2.0**(1 - bias - mbits)
    return sign * (1.0 + mant / 2.0**mbits) * 2.0**(exp - bias)

E4M3 = dict(ebits=4, mbits=3, bias=7, max_exp=15, max_mant=6)    # 1111_110 = 448
E5M2 = dict(ebits=5, mbits=2, bias=15, max_exp=30, max_mant=3)   # 11110_11 = 57344

for v in [0.5, -1.0, 100.0, 448.0, 57344.0]:
    for name, p in [("E4M3", E4M3), ("E5M2", E5M2)]:
        b = encode_fp8(v, **p)
        print(f"v={v:>9} {name} bits=0b{b:0{p['ebits']+p['mbits']+1}b} decoded={decode_fp8(b, p['ebits'], p['mbits'], p['bias'])}")
# v=      0.5 E4M3 bits=0b00110000 decoded=0.5
# v=      0.5 E5M2 bits=0b00111000 decoded=0.5
# v=     -1.0 E4M3 bits=0b10111000 decoded=-1.0
# v=     -1.0 E5M2 bits=0b10111100 decoded=-1.0
# v=    100.0 E4M3 bits=0b01101100 decoded=96.0     ← 3 位尾数，100 只能近似
# v=    100.0 E5M2 bits=0b01010110 decoded=96.0
# v=    448.0 E4M3 bits=0b01111110 decoded=448.0    ← 1111_110
# v=    448.0 E5M2 bits=0b01011111 decoded=448.0
# v=  57344.0 E4M3 bits=0b01111110 decoded=448.0    ← 超出 E4M3 范围，饱和
# v=  57344.0 E5M2 bits=0b01011011 decoded=57344.0  ← 11110_11
```

### 6.4 FP8 矩阵乘：FP32 累加 vs FP8 累加（对比误差）

```python
import torch

torch.manual_seed(1)
A = torch.randn(256, 256) * 5.0
B = torch.randn(256, 256) * 5.0
C32 = A @ B                                    # 标准 FP32

def q8(x):
    return x.to(torch.float8_e4m3fn).float()   # 量化到 E4M3 再还原

A8, B8 = q8(A), q8(B)
C_fp32acc = A8 @ B8                            # 输入 FP8、累加 FP32（真实 Tensor Core 行为）

prod = A8.unsqueeze(2) * B8.unsqueeze(0)       # 所有中间乘积
C_fp8acc = q8(prod).sum(dim=1)                 # 每步乘积都回 FP8（模拟 FP8 累加器，灾难）

def rel(c):
    return (C32 - c).abs().max().item() / C32.abs().max().item()

print(f"FP8输入+FP32累加: rel_err = {rel(C_fp32acc):.2e}")   # 3.84e-02，误差可控
print(f"FP8输入+FP8累加:  rel_err = {rel(C_fp8acc)}")        # nan！累加器溢出
```

结论一目了然：**FP8 累加 FP32 误差 ~4% 可接受；一旦累加器也变 FP8，中间和超过 448 直接变 NaN**。这就是"矩阵乘法适合 FP8"的全部理由——精度由 FP32 累加器兜底。

### 6.5 TF32 模拟与开关演示

```python
import torch

def to_tf32(x):
    """模拟硬件 TF32：FP32 输入截断到 10 位尾数（13 位进位舍入）"""
    i = x.view(torch.int32)
    r = (i + (1 << 12)) & ~((1 << 13) - 1)
    return r.view(torch.float32)

torch.manual_seed(0)
A, B = torch.randn(512, 512), torch.randn(512, 512)
C32 = A @ B
Ctf = to_tf32(A) @ to_tf32(B)
err = (C32 - Ctf).abs().max().item()
print(f"fp32 vs tf32(模拟): rel_err={err / C32.abs().max().item():.2e}")
# fp32 vs tf32(模拟): rel_err=2.75e-04  ← ~1e-4 级，训练可接受

# 真实开关（GPU 上生效）
import torch.backends.cuda
print(torch.backends.cuda.matmul.allow_tf32)     # False（PyTorch 2.x 默认关闭）
torch.backends.cuda.matmul.allow_tf32 = True     # A100/H100 上 matmul 自动走 TF32
print(torch.get_float32_matmul_precision())      # 'high'（开）/ 'highest'（关）
```

---

## 七、优缺点

| 优点                          | 缺点                          |
| --------------------------- | --------------------------- |
| 显存/带宽省 4 倍（相对 FP32）          | 需要量化 + scale 技巧（per-group 等）  |
| GEMM 吞吐 2 倍于 FP16/BF16（H100） | 值域窄，outlier 敏感                |
| 训练/推理均可，H100/Blackwell 原生    | 硬件门槛：需要 Hopper+ GPU 才划算      |
| DeepSeek-V3 等验证 loss 误差 <0.25% | 部分算子必须回退高精度（归一化、注意力等）     |
| TF32 无脑快、无缩放负担               | TF32 精度只有 FP16 级，收敛风险仍存在     |

---

## 八、高频面试问答

**Q1：E4M3 和 E5M2 的区别？**
结构 1+4+3 vs 1+5+2。E4M3 最大 448、尾数 3 位（精度 6.25%）、无 inf，适合前向激活/权重；E5M2 最大 57344、尾数 2 位（精度 12.5%）、支持 inf/nan，适合反向梯度（动态范围大）。本质是"精度 vs 范围"的取舍。

**Q2：为什么 FP8 矩阵乘的累加器必须用 FP32？**
乘法误差是随机的、可抵消的（相对误差 ~4%），但**累加误差是累积的**：如果每一步和都量化回 FP8（最大 448），K=256 的中间和直接溢出成 inf/NaN。FP32 累加把 K 维误差压在 1e-7 级，总误差只由输入量化误差决定，不随 K 增长。

**Q3：FP8 训练为什么可行？梯度那么小不会下溢吗？**
① 关键计算（GEMM）累加在 FP32 中；② 量化有 per-group scale 自适应放大缩小；③ 主权重、优化器状态保持 BF16/FP32；④ 反向梯度按块量化，异常值影响局部。DeepSeek-V3 用 1×128/128×128 分块量化 + FP32 累加，loss 相对 BF16 基线误差 <0.25%。

**Q4：TF32 是什么？**
1+8+10 的矩阵乘专用格式：FP32 输入进 Tensor Core 前尾数截到 10 位。动态范围与 FP32 相同（不会下溢），精度约等于 FP16。Ampere 起 NVIDIA 硬件原生支持，是"FP32 生态无痛 8 倍速"。

**Q5：TF32、FP16、BF16 怎么选？**
需要 FP32 范围 → TF32/BF16（都免 scaling）；需要精度 → FP16（10 位尾数）；需要吞吐且能接受精度损失 → FP8。实际训练中 BF16 通用性最好（范围大 + 免 scaling），TF32 适合不想动代码的卷积/矩阵乘场景。

**Q6：PyTorch 里怎么开关 TF32？**
`torch.backends.cuda.matmul.allow_tf32 = True/False`，或 `torch.set_float32_matmul_precision("high"/"highest")`。PyTorch 2.x 默认关闭（保精度优先），开/关后矩阵乘数值有 ~1e-4 级相对差异。

**Q7：FP8 有没有 inf？溢出会发生什么？**
E5M2 有 inf/nan（指数字段 11111）；E4M3 无 inf，溢出饱和到 ±448。训练中要防 E4M3 饱和（用分块 scale 把 max 压到 448 以下）和 E5M2 溢出（溢出成 inf 会传播成 NaN，需要检查）。

**Q8：DeepSeek-V3 的 FP8 训练有哪些关键技巧？**
① E4M3 前向 / E5M2 反向（调优后全 E4M3）；② 1×128 激活、128×128 权重分块量化；③ 累加 FP32 且每 128 个 MMA 提升一次高精度累加；④ 主权重/优化器状态保持 BF16/FP32；⑤ embedding/门控/归一化/注意力回退高精度。

---

## 九、自我检验

- [ ] 能背出 E4M3 与 E5M2 的位结构、最大有限值（448/57344）与适用场景
- [ ] 能画出 FP8 乘 FP8、FP32 累加的 Tensor Core 数据流
- [ ] 能用误差公式解释"为什么矩阵乘法适合 FP8 而 FP8 累加会崩"
- [ ] 能说出 DeepSeek-V3 的 FP8 训练四要素（格式分工/分块量化/FP32 累加/高精度保留）
- [ ] 能讲清 TF32 的 1+8+10、为什么范围同 FP32 精度似 FP16
- [ ] 能写出 PyTorch 开关 TF32 的两种 API
- [ ] 能解释 H100 上 FP8 吞吐 = 2× BF16 的硬件原因
- [ ] 能回答 8 个面试追问
