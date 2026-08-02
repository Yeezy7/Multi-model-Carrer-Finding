from torch.utils.data import Dataset
import json
from PIL import Image
import os
import torch
import pandas as pd
from io import BytesIO
import base64
from transformers import AutoTokenizer, AutoProcessor
import random

class SiglipDataset(Dataset):
    def __init__(self, text_data_path, image_data_path, tokenizer, processor, max_seq_length=64):
        super().__init__()
        self.text_data_path = text_data_path
        self.image_data_path = image_data_path
        self.tokenizer = tokenizer
        self.processor = processor
        self.max_seq_length = max_seq_length
        with open(self.text_data_path, 'r', encoding='utf-8') as f:
            self.datas = []
            lines = f.readlines()
            for line in lines:
                line = json.loads(line)
                for image_id in line['image_ids']:
                    self.datas.append({'image_id': image_id, 'text': line['text']})
        random.shuffle(self.datas)  # 打乱数据顺序
        
        self.images = pd.read_csv(self.image_data_path, sep='\t', header=None)
    
    def __getitem__(self, index):
        sample = self.datas[index]
        image_id = sample['image_id']
        text = sample['text']
        tok = self.tokenizer(text, max_length=self.max_seq_length, padding='max_length', truncation=True) # 对文本进行编码
        input_ids = tok['input_ids']
        attention_mask = tok['attention_mask']
        image_base64 = self.images[self.images[0] == image_id][1].values[0] # 根据 image_id 获取对应的 base64 编码的图像数据
        image_bytes = base64.b64decode(image_base64) # 解码 base64 编码的图像数据
        
        image = Image.open(BytesIO(image_bytes)).convert("RGB") # 将图像数据转换为 RGB 格式的 PIL 图像对象
        pixel_values = self.processor(images=image, return_tensors="pt")['pixel_values'].squeeze(0) # 对图像进行处理，得到模型所需的输入格式
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'pixel_values': pixel_values
        }

    def __len__(self):
        return len(self.datas)

class MyDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    def __call__(self, features):
        input_ids = [f['input_ids'] for f in features]
        attention_mask = [f['attention_mask'] for f in features]
        pixel_values = [f['pixel_values'] for f in features]
        return {
            'input_ids': torch.tensor(input_ids),
            'attention_mask': torch.tensor(attention_mask),
            'pixel_values': torch.stack(pixel_values)
        }

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained('/home/flj/flj/Multi-model-Carrer-Finding/pre-trained-model/chinese-roberta-wwm-ext')
    processor = AutoProcessor.from_pretrained("/home/flj/flj/Multi-model-Carrer-Finding/pre-trained-model/vit-base-patch16-224")
    dataset = SiglipDataset(
        text_data_path='/home/flj/flj/Multi-model-Carrer-Finding/多模态算法工程师/04_SigLIP/dataset/MUGE/train_texts.jsonl',
        image_data_path='/home/flj/flj/Multi-model-Carrer-Finding/多模态算法工程师/04_SigLIP/dataset/MUGE/train_imgs.tsv', 
        tokenizer=tokenizer,
        processor=processor,
        max_seq_length=64)

    print(len(dataset))
    print(dataset[2])