import torch
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM, AutoProcessor

model_id = "Qwen/Qwen3-VL-8B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto")

"""LoraConfig + get_peft_model"""
lora_config = LoraConfig(
    r=32,                      # 秩
    lora_alpha=32,             # 缩放因子（实际 α/r = 2）
    target_modules=[           # 注入目标（含视觉塔与 LLM 的线性层）
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_dropout=0.05,         # Dropout
    bias="none",               # 偏置项不参与 LoRA
    task_type=TaskType.CAUSAL_LM,  # 任务类型
    modules_to_save=["mlp2"],  # 保存 LoRA 权重的模块
)

peft_model = get_peft_model(model, lora_config)     # 将 LoRA 注入模型
peft_model.print_trainable_parameters()  # 打印可训练参数
# trainable params: ~11,700,000 || all params: 7,600,000,000 || trainable%: 0.1538


"""
QLoRA 加载
QLoRA 只需把基座用 `BitsAndBytesConfig` 以 4-bit 加载
"""
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,                      # 4bit 量化
    bnb_4bit_quant_type="nf4",              # NF4 量化
    bnb_4bit_compute_dtype=torch.bfloat16,  # 计算精度
    bnb_4bit_use_double_quant=True,         # 双量化
)
model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map="auto")


"""训练与保存"""
from transformers import TrainingArguments, Trainer

args = TrainingArguments(
    output_dir="./qwen3vl-lora",     # 输出目录
    per_device_train_batch_size=1,   # 每设备训练批次大小
    gradient_accumulation_steps=8,   # 梯度累积步数
    learning_rate=2e-4,              # 学习率
    bf16=True,                       # bf16 精度
    logging_steps=10,                # 日志记录步数
    save_strategy="steps",           # 保存策略
    save_steps=200,                  # 保存步数
)
trainer = Trainer(model=peft_model, args=args, train_dataset=dataset)  # 训练器
trainer.train()  # 开始训练

# 只保存 LoRA 分支（几十 MB），不导出基座
peft_model.save_pretrained("./lora_weights")


"""合并权重与加载"""
from peft import PeftModel

# 方式一：不合并，直接加载基座 + LoRA分支
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16)
model = PeftModel.from_pretrained(model, "./lora_weights", torch_dtype=torch.bfloat16)

# 方式二：合并回基座权重（推理零开销，导出单一模型文件）
merged = model.merge_and_unload()  # 合并 LoRA 权重到基座
merged.save_pretrained("./qwen3vl-lora-merged")  # 保存合并后的模型