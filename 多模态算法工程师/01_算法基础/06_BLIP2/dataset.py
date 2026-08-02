import json
from PIL import Image
from torch.utils.data import Dataset


class BLIP2Dataset(Dataset):
    def __init__(self, data_path, image_folder, tokenizer, image_processor, mode='pretrain'):
        with open(data_path, "r") as f:
            self.data = json.load(f)
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.mode = mode

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image = Image.open(f"{self.image_folder}/{item['image_id']}").convert("RGB")
        if self.mode == 'pretrain':
            # 随机选择一种预训练任务（图文匹配、图文对比、caption），为简化，只做caption
            caption = item['caption']
            # 构造文本输入，格式如 "a photo of {caption}"
            text = f"a photo of {caption}"
        else:
            # 指令微调格式，例如 VQA 数据
            question = item['question']
            answer = item['answer']
            text = f"Question: {question} Answer: {answer}"
        
        inputs = self.tokenizer(text, return_tensors='pt', padding='max_length',
                                truncation=True, max_length=128)
        return {
            'pixel_values': self.image_processor(image, return_tensors='pt')['pixel_values'][0],
            'input_ids': inputs['input_ids'][0],
            'labels': inputs['input_ids'][0].clone()  # 简易处理，实际上应该掩码掉问题部分
        }