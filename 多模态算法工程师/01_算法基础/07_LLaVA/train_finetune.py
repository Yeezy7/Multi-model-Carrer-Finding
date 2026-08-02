# LoRA 微调（省显存，7B 模型可跑在 12GB 显卡上）

import argparse
import torch
from torch.utils.data import DataLoader
from models.llava import LlavaLlamaForCausalLM
from models.dataset import LlavaDataset
from tqdm import tqdm
import wandb

def train(args):
    # 初始化模型（支持 4-bit 量化进一步省显存）
    model = LlavaLlamaForCausalLM(
        vision_model_name=args.vision_model_name,
        llm_name=args.llm_name,
        mm_hidden_mult=args.mm_hidden_mult,
        load_in_4bit=args.load_in_4bit
    )
    
    # 加载预训练的投影层
    pretrained_state = torch.load(args.pretrained_projector_path, map_location="cpu")
    if "mm_projector" in pretrained_state:
        # 从完整 checkpoint 中提取投影层
        model.mm_projector.load_state_dict(pretrained_state["mm_projector"])
    else:
        # 直接加载投影层权重
        model.mm_projector.load_state_dict(pretrained_state)

    # 视觉编码器始终冻结
    for p in model.vision_encoder.parameters():
        p.requires_grad = False

    # LoRA 训练：只训练 LoRA 适配器 + 投影层，LLM 主体不动
    if args.use_lora:
        model.apply_lora(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout
        )
    else:
        # 全量微调（显存需求大：~60GB for 7B）
        for p in model.llm.parameters():
            p.requires_grad = True
    for p in model.mm_projector.parameters():
        p.requires_grad = True
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if not args.load_in_4bit:  # 4-bit 模型已自动放到 GPU
        model.to(device)

    # 打印显存信息
    if torch.cuda.is_available():
        print(f"\n[显存] 当前占用: {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB")
        print(f"[显存] 峰值占用: {torch.cuda.max_memory_allocated() / 1024 ** 3:.2f} GB\n")

    dataset = LlavaDataset(
        data_path=args.data_path,
        image_folder=args.image_folder,
        tokenizer=model.tokenizer,
        image_processor=model.image_processor,
        mode="finetune"
    )
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=lambda x: x
    )

    # 只优化可训练参数（LoRA 适配器 + 投影层）
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate, weight_decay=0.01)

    trainable_count = sum(p.numel() for p in trainable_params)
    print(f"\n[训练] 总可训练参数: {trainable_count:,}")
    print(f"[训练] batch_size: {args.batch_size}, lr: {args.learning_rate}\n")

    for epoch in range(args.epochs):
        epoch_loss = 0.0
        num_batches = 0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{args.epochs}"):
            images = torch.stack([b['image'] for b in batch]).to(device)
            input_ids = torch.stack([b['input_ids'] for b in batch]).to(device)
            labels = torch.stack([b['labels'] for b in batch]).to(device)
            
            optimizer.zero_grad()
            loss, _ = model(images=images, input_ids=input_ids, labels=labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1
            wandb.log({"finetune_loss": loss.item()})

        avg_loss = epoch_loss / max(num_batches, 1)
        print(f"\nEpoch {epoch + 1} 平均 Loss: {avg_loss:.4f}")

        # 每隔 epoch 保存一次 LoRA 权重（很小，几 MB）
        if args.use_lora:
            lora_save_path = f"{args.save_path}/llava_lora_epoch{epoch + 1}"
            model.llm.save_pretrained(lora_save_path)  # 只保存 LoRA 适配器
            torch.save(model.mm_projector.state_dict(),
                      f"{lora_save_path}/mm_projector.bin")
            print(f"[保存] LoRA 权重已保存到: {lora_save_path}")
        else:
            torch.save(model.state_dict(),
                      f"{args.save_path}/llava_finetuned_epoch{epoch + 1}.pt")
            print(f"[保存] 完整模型已保存")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLaVA LoRA 微调")
    parser.add_argument("--vision_model_name", type=str,
                        default="/home/flj/flj/Multi-model-Carrer-Finding/pre-trained-model/vit-base-patch16-224")
    parser.add_argument("--llm_name", type=str, default="lmsys/vicuna-7b-v1.5")
    parser.add_argument("--mm_hidden_mult", type=float, default=2.0)
    parser.add_argument("--pretrained_projector_path", type=str,
                        default="/home/flj/flj/Multi-model-Carrer-Finding/多模态算法工程师/07_LLaVA/ckpts/llava_pretrained_1.pt")
    parser.add_argument("--save_path", type=str,
                        default="/home/flj/flj/Multi-model-Carrer-Finding/多模态算法工程师/07_LLaVA/ckpts")
    parser.add_argument("--data_path", type=str,
                        default="/home/flj/flj/Multi-model-Carrer-Finding/多模态算法工程师/07_LLaVA/data/llava-1.5.jsonl")
    parser.add_argument("--image_folder", type=str,
                        default="/home/flj/flj/Multi-model-Carrer-Finding/多模态算法工程师/07_LLaVA/data/images")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--epochs", type=int, default=10)

    # LoRA 相关参数
    parser.add_argument("--use_lora", action="store_true", default=True,
                        help="使用 LoRA 微调（省显存，7B 可跑在 12GB 显卡）")
    parser.add_argument("--load_in_4bit", action="store_true", default=True,
                        help="4-bit 量化加载（QLoRA，进一步省显存）")
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha（通常 r*2）")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")

    args = parser.parse_args()

    # 检查依赖
    if args.use_lora or args.load_in_4bit:
        try:
            import peft
            import bitsandbytes  # noqa: F401
        except ImportError:
            print("\n[错误] 缺少依赖，请先安装:")
            print("pip install peft bitsandbytes accelerate\n")
            exit(1)

    wandb.init(project="llava-finetune", config=vars(args))
    train(args)