# 复杂文档多模态理解项目设计

## 1. 项目目标

将 Qwen3-VL-8B-Instruct 通过一次 Multimodal SFT 微调为一个更擅长复杂文档理解的最终模型，使其能够处理：

- 文档视觉问答（DocVQA）
- 图表问答（ChartQA）
- 信息图表问答（InfographicVQA）
- 合同与财报的结构化信息抽取（KIE）
- 合同与财报领域问答和数值推理

项目的目标不是单独刷高 DocVQA ANLS，而是验证一个统一的 Document VLM 在通用文档、图表和领域文档任务上的综合能力。

## 2. 模型形态

最终只训练、评测并部署一个模型：

```text
Qwen3-VL-8B-Instruct
        |
        | Multimodal SFT / QLoRA
        v
Final Document VLM
```

消融实验中的不同配置只作为研究对照，不作为独立部署模型。

## 3. 数据闭环

### 3.1 训练数据

将下列数据的训练划分统一转换为多模态指令数据后混合训练：

- DocVQA train：文档图片问答
- ChartQA train：图表理解与数值问答
- InfographicVQA train：复杂版面、文字、图表混合理解
- CUAD：公开合同 PDF 及合同条款标注
- FinQA：财务报告问答、支撑事实和数值推理程序
- FinTabNet：真实财务报告页面/表格及其结构标注
- 自建仿真数据：中文合同、资产负债表、利润表、现金流量表及困难版式样本

第一版目标为 300-500 份领域文档和约 2,500-4,000 条领域指令样本；通用数据按可用训练划分接入，不强制与领域数据按任务比例配平。

### 3.2 测试数据

所有公开数据只使用训练划分进行 SFT；各数据集原有测试划分完全隔离，仅用于最终评测：

- DocVQA test：通用文档问答
- ChartQA test：图表问答
- InfographicVQA test：复杂信息图表问答
- 自建合同/财报 test：领域 KIE、领域 QA 和数值推理
- Hard test subset：从测试数据中按小字体、低清晰度、扫描件、密集表格、倾斜和压缩等条件筛选，不参与训练

同一原始合同、财报或报告页面不得跨 train、validation、test 划分。领域数据按文档级别切分，而不是按页面或问答条目随机切分。

## 4. 统一样本格式

每条样本至少包含：

```json
{
  "id": "contract_0001_q03",
  "image": "images/contract/contract_0001_page_01.png",
  "conversations": [
    {
      "from": "human",
      "value": "<image>\n请提取合同金额和付款方式，使用 JSON 输出。"
    },
    {
      "from": "gpt",
      "value": "{\"合同金额\":\"人民币 580000 元\",\"付款方式\":\"合同签订后支付 30%，验收后支付 60%，质保期满支付 10%\"}"
    }
  ],
  "metadata": {
    "domain": "contract",
    "source_type": "public_real",
    "task": "structured_extraction",
    "source_dataset": "cuad",
    "document_id": "...",
    "license": "..."
  }
}
```

`task` 只用于追踪、过滤和评测，不用于强制采样配比。结构化抽取样本要求答案为合法 JSON；普通问答和数值推理样本保留自然语言答案，并在元数据中记录可选的支撑事实或 reasoning program。

## 5. 评测闭环

### 5.1 主结果

| 测试集 | 能力 | 指标 |
| --- | --- | --- |
| DocVQA test | 通用文档问答 | ANLS |
| ChartQA test | 图表理解 | Accuracy |
| InfographicVQA test | 复杂版面理解 | Accuracy / ANLS |
| 自建合同/财报 test | 结构化字段抽取 | Precision / Recall / F1 |
| 自建合同/财报 test | JSON 输出质量 | JSON validity、字段级 F1 |
| 自建财报 test | 数值推理 | 执行准确率、数值误差 |
| Hard test subset | 困难文档鲁棒性 | ANLS / F1 |

DocVQA ANLS 只能说明通用文档问答能力变化，不能替代合同/财报 KIE 的字段级评价；严格 JSON 输出也不直接用 ANLS 作为唯一指标。

### 5.2 消融实验

围绕最终方案中的策略做对照，例如：

- Base Qwen3-VL-8B-Instruct
- + Multimodal SFT
- + 困难样本增强
- + 高分辨率或 Dynamic Resolution 策略
- Full：最终配置

每个配置使用相同测试集和相同评测脚本。消融结果用于解释性能来源，最终只选择 Full 配置作为部署模型。

## 6. 数据处理产物

```text
doc_sft/
├── data/
│   ├── train.jsonl
│   ├── val.jsonl
│   ├── test_manifest.json
│   └── license_manifest.json
├── images/
│   ├── public_real/
│   └── synthetic/
├── scripts/
│   ├── download_real_data.py
│   ├── convert_public_data.py
│   ├── generate_synthetic_docs.py
│   ├── build_sft_jsonl.py
│   └── validate_dataset.py
└── README.md
```

原始公开文件、派生图片、标注转换结果和最终 SFT JSONL 分开保存。每个真实文件记录来源 URL、下载时间、原始数据集版本、许可证和哈希值；不默认重新分发底层合同 PDF。

## 7. 验证要求

- 校验所有图片路径存在且可解码；
- 校验每条样本恰好包含一个图像占位符并符合现有训练代码格式；
- 校验 JSONL 可逐行解析，结构化答案为合法 JSON；
- 校验 train/validation/test 不存在同一文档泄漏；
- 校验公开测试集没有进入训练 JSONL；
- 用固定的 base/final 推理脚本生成各测试集结果，并保存配置、版本和指标明细；
- 仿真样本由固定随机种子生成，保证可复现。

## 8. 范围边界

本阶段不把 DocVQA 测试数据转入训练，不将多个消融配置包装成多个最终模型，也不把 ANLS 作为所有任务的统一指标。第一版重点是完成一个可复现的训练-测试-消融闭环。
