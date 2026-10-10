"""
阶段一训练：只训练 Q-Former
"""

import torch
from torch.utils.data import DataLoader
from models.blip2 import Blip2ForConditionalGeneration
from dataset import BLIP2Dataset
from tqdm import tqdm
import argparse


def train(args):
    model = Blip2ForConditionalGeneration(
        vision_model_name=args.vision_model_name,
        llm_name=args.llm_name,
        qformer_num_queries=32,
        qformer_hidden_size=768,
        qformer_num_layers=6,
        qformer_num_heads=12,
    )
    # 冻结视觉塔 与 LLM
    for p in model.vision_model.parameters():
        p.requires_grad = False
    for p in model.llm.parameters():
        p.requires_grad = False
    # Q-Former 和投影层默认可训练
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    data = BLIP2Dataset(args.data_path, 
                        args.image_folder,
                        tokenizer=model.llm_tokenizer,
                        image_processor=model.image_processor,
                        mode='pretrain')
    dataloader = DataLoader(data, batch_size=args.batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(list(model.qformer.parameters()) + list(model.llm_proj.parameters()), lr=args.learning_rate)
    
    model.train()
    for epoch in range(args.num_epochs):
        epoch_loss = 0.0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.num_epochs}"):
            pixel_values = batch['pixel_values'].to(device)
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            # 前向传播
            loss, _ = model(pixel_values, input_ids, labels=labels)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1} Loss: {avg_loss:.4f}")
    
    # 保存模型
    torch.save(model.qformer.state_dict(), "blip2_qformer.pth")
    torch.save(model.llm_proj.state_dict(), "blip2_llm_proj.pth")


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--data_path", type=str, default="data/blip2_data.json")
    parser.add_argument("--image_folder", type=str, default="data/images")
    parser.add_argument("--vision_model_name", type=str, default="openai/clip-vit-large-patch14")
    parser.add_argument("--llm_name", type=str, default="gpt2")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    args = parser.parse_args()
    
    train(args)