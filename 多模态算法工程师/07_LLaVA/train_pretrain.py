# 只训练投影层，冻结视觉编码和LLM

import torch
from torch.utils.data import DataLoader
from models.llava import LlavaLlamaForCausalLM
from models.dataset import LlavaDataset
from tqdm import tqdm
import wandb
import argparse

def train(args):
    # 初始化
    model = LlavaLlamaForCausalLM(
        vision_model_name=args.vision_model_name,
        llm_name=args.llm_name,
        mm_hidden_mult=args.mm_hidden_mult
    )

    # 冻结视觉编码器和 LLM
    for p in model.vision_encoder.parameters():
        p.requires_grad = False
    for p in model.llm.parameters():
        p.requires_grad = False
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    # 数据集和数据加载器
    dataset = LlavaDataset(
        data_path=args.data_path,
        image_folder=args.image_folder,
        tokenizer=model.tokenizer,
        image_processor=model.image_processor,
        mode=args.mode
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=lambda x: x) # collate 需自行定义

    optimizer = torch.optim.AdamW(model.mm_projector.parameters(), lr=args.learning_rate, weight_decay=0.01)
    
    for epoch in range(args.epochs):
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            images = [b['image'] for b in batch].to(device)
            input_ids = torch.stack([b['input_ids'] for b in batch]).to(device)
            labels = torch.stack([b['labels'] for b in batch]).to(device)
            
            optimizer.zero_grad()
            loss, _ = model(images=images, input_ids=input_ids, labels=labels)
            loss.backward()
            optimizer.step()
            wandb.log({"pretrain_loss": loss.item()})
    
    # 保存模型
    torch.save(model.state_dict(), args.save_path + "/llava_pretrained" + f"_{epoch+1}" + ".pt")


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="LLaVA Pretraining")
    parser.add_argument("--vision_model_name", type=str, default="/home/flj/flj/Multi-model-Carrer-Finding/pre-trained-model/vit-base-patch16-224")
    parser.add_argument("--llm_name", type=str, default="lmsys/vicuna-7b-v1.5")
    parser.add_argument("--mm_hidden_mult", type=float, default=2.0)
    parser.add_argument("--data_path", type=str, default="data/llava-1.5.jsonl") # 官方 LLaVA-1.5 数据集 jsonl 文件
    parser.add_argument("--image_folder", type=str, default="data/images") 
    parser.add_argument("--mode", type=str, choices=["pretrain", "finetune"], default="pretrain")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--save_path", type=str, default="/home/flj/flj/Multi-model-Carrer-Finding/多模态算法工程师/07_LLaVA/ckpts")
    
    args = parser.parse_args()
    train(args)