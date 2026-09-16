from peft import LoraConfig, get_peft_model, PeftModel, TaskType
from LoRA import TinyLM


config  = LoraConfig(
    r=8,
    lora_alpha=16,
    bias="none",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    task_type=TaskType.FEATURE_EXTRACTION,                      # 不训练 bias
)
peft_model = get_peft_model(TinyLM(), config)    # 自动冻结基座 + 注入
peft_model.print_trainable_parameters()  # 打印可训练参数
# trainable params: 4,352 || all params: 19,104 || trainable%: 22.7806

# 训练完保存（只保存 LoRA 分支，几十 KB 的权重文件）
peft_model.save_pretrained("./tiny_lora")

# 推理方式一：加载到新基座上（不合并，多任务热插拔模式）
base = TinyLM()
loaded = PeftModel.from_pretrained(base, "./tiny_lora")

# 推理方式二：合并回基座（推理零开销，单文件部署）
merged = loaded.merge_and_unload()