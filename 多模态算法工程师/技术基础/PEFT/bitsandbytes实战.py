import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,                      # 4-bit 加载基座
    bnb_4bit_quant_type="nf4",              # NF4 量化（可选 fp4）
    bnb_4bit_compute_dtype=torch.bfloat16,  # 计算精度：反量化后 bf16
    bnb_4bit_use_double_quant=True,         # 双重量化（省 ~0.37bit/权重）
)
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen-3-VL-Instruct", 
    quantization_config=bnb_config, 
    device_map="auto"                       # 自动切分到 GPU
)

# 基座已 4-bit 冻结，再用 peft 加 LoRA 即可训练
from peft import LoraConfig, get_peft_model, TaskType

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    task_type=TaskType.CAUSAL_LM,  # 任务类型
    bias="none",                   # 不训练 bias
)
peft_model = get_peft_model(model, lora_config)  # 注入 LoRA
peft_model.print_trainable_parameters()  # 打印可训练参数