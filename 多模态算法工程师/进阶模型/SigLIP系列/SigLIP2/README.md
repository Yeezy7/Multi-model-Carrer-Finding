# SigLIP 2 源码

> SigLIP 2: Multilingual Vision-Language Encoders with Improved Semantic Understanding, Localization, and Dense Features
> 论文: [arXiv:2502.14786](https://arxiv.org/abs/2502.14786) · 官方 Colab: [SigLIP2_demo.ipynb](https://colab.research.google.com/github/google-research/big_vision/blob/main/big_vision/configs/proj/image_text/SigLIP2_demo.ipynb) · Hugging Face 权重: [google/siglip2](https://huggingface.co/collections/google/siglip2-67b5dcef38c175486e240107)

## 目录结构

```
SigLIP2/
├── README.md                        # 本说明文件
├── LICENSE                          # Apache-2.0 许可证
├── big_vision/                      # Google 官方 JAX/Flax 实现（big_vision 仓库）
│   ├── models/
│   │   ├── vit.py                   # ViT 图像编码器（SigLIP2 基础）
│   │   └── proj/image_text/
│   │       ├── two_towers.py        # 双塔模型（SigLIP2 核心结构）
│   │       ├── text_transformer.py  # 文本编码器
│   │       ├── naflex_vit.py        # NaFlex 变体 ViT（支持原生宽高比/多分辨率）
│   │       └── utils.py             # 辅助工具
│   ├── configs/proj/image_text/
│   │   ├── README_siglip2.md        # 官方说明 + 全部模型权重下载链接
│   │   ├── SigLIP2_demo.ipynb       # 官方 demo 笔记本
│   │   └── common.py                # 公共配置
│   └── trainers/proj/image_text/
│       └── siglip.py                # 官方训练脚本（sigmoid 对比损失）
└── transformers/siglip2/            # Hugging Face PyTorch 实现（transformers 仓库）
    ├── modeling_siglip2.py          # Siglip2Model / Siglip2VisionModel / Siglip2TextModel 等
    ├── configuration_siglip2.py     # Siglip2Config / Siglip2VisionConfig / Siglip2TextConfig
    ├── image_processing_siglip2.py  # 图像处理器（含多分辨率、原生宽高比处理）
    ├── image_processing_pil_siglip2.py
    ├── tokenization_siglip2.py      # Gemma tokenizer（256k 词表）
    ├── processing_siglip2.py        # 处理器组合
    ├── convert_siglip2_to_hf.py     # big_vision 权重 → HF 权重转换脚本
    ├── modular_siglip2.py           # 模块化源文件（自动生成 modeling/configuration 的模板）
    └── __init__.py
```

## 两种实现的对比

| | big_vision（JAX） | transformers（PyTorch） |
|---|---|---|
| 框架 | JAX / Flax | PyTorch |
| 定位 | 官方训练 + 推理 | 推理 / 微调 |
| 权重格式 | .npz | safetensors |
| 模型类 | `Model`（two_towers） | `Siglip2Model` |
| 用途 | 学习训练细节、损失函数 | 直接加载 `google/siglip2-*` 权重使用 |

## 快速使用（PyTorch）

```python
from transformers import AutoProcessor, AutoModel
from PIL import Image

processor = AutoProcessor.from_pretrained("google/siglip2-base-patch16-224")
model = AutoModel.from_pretrained("google/siglip2-base-patch16-224")

image = Image.open("example.jpg")
inputs = processor(images=image, text=["a photo of a cat", "a photo of a dog"],
                   padding="max_length", return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits_per_image  # 图像-文本相似度
```

## 模型权重下载

官方 `.npz` 权重（JAX 版）见 `big_vision/configs/proj/image_text/README_siglip2.md`，共有 4 个规模：

- **ViT-B** (86M)：B/32, B/16
- **ViT-L** (303M)：L/16
- **So400m** (400M)：So400m/14, So400m/16
- **g** (1B)：g-opt/16

PyTorch 版权重直接通过 Hugging Face 加载：`google/siglip2-base-patch16-224`、`google/siglip2-so400m-patch14-384`、`google/siglip2-giant-opt-patch16-384` 等。

## 源码来源

- `big_vision/` ← [github.com/google-research/big_vision](https://github.com/google-research/big_vision)（main 分支，Apache-2.0）
- `transformers/siglip2/` ← [github.com/huggingface/transformers](https://github.com/huggingface/transformers)（main 分支，Apache-2.0）

仅保留了 SigLIP2 相关文件，完整仓库可访问上述链接获取。
