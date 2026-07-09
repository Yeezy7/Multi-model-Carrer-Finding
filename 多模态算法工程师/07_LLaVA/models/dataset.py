import copy
import json
import torch
from PIL import Image
from torch.utils.data import Dataset

# Vicuna v1.5 对话模板（基于 LLaMA-2）
SEP = " "
BEGIN_INST = "<s>[INST] "
END_INST = " [/INST]"
BEGIN_SYS = "<<SYS>>\n"
END_SYS = "\n<</SYS>>\n\n"
DEFAULT_SYSTEM = "You are a helpful, respectful and honest assistant."


def create_plain_prompt(question, answer):
    """预训练阶段使用：简单的 QA 格式（caption 任务）"""
    
    return f"{BEGIN_INST}{question}{END_INST} {answer} </s>"

def create_conversation_prompt(sources, system_msg=None):
    """
    指令微调阶段：多轮对话
    sources: 列表 [{"from": "human", "value": "..."}, {"from": "gpt", "value": "..."}]
    返回: 完整 tokenize 后的 input_ids 和 labels
    """
    text = ""
    if system_msg is None:
        system_msg = DEFAULT_SYSTEM
        
    # 第一轮对话前加入系统消息
    text += f"<s>[INST] <<SYS>>\n{system_msg}\n<</SYS>>\n\n"
    for i, turn in enumerate(sources):
        if turn["from"] == "human":
            if i == 0:
                text += turn["value"] + " [/INST]"
            else:
                text += f"<s>[INST] {turn['value']} [/INST]"
        elif turn["from"] == "gpt":
            text += f" {turn['value']} </s>"
        else:
            raise ValueError(f"未知的对话角色: {turn['from']}")
    
    return text

class LlavaDataset(Dataset):
    """
    LLaVA 数据集
    每个样本包含：
        - 图像路径或 PIL 图像
        - 文本 prompt（包含 <image> 占位符）
        - 可选的标签（用于计算 loss）
    """
    def __init__(self, data_path, image_folder, tokenizer, image_processor, mode='pretrain'):
        """
        ddata_path: json 文件路径，格式为官方 LLaVA-1.5 使用的结构
        mode: 'pretrain' 或 'finetune'
        """
        with open(data_path, "r") as f:
            self.data = json.load(f)
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.mode = mode

    def __len__(self):
        return len(self.data)

    def tokenize_and_mask(self, text, max_len=512):
        """
        Tokenize 并生成 labels：只对 assistant 回答部分计算损失，其他设为 -100
        原理：[/INST] 之后到 </s> 之前是回答，其他都是 prompt
        """
        # Step 1: 先整体 tokenize，得到 input_ids
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=max_len)
        input_ids = inputs.input_ids[0]
        labels = input_ids.clone()
        labels.fill_(-100)  # 先全设为 -100

        # Step 2: 分段追踪，找到回答部分的 token 位置
        # 文本结构示例: "<image>\n<s>[INST] 问题 [/INST] 回答 </s>"
        # 策略：按 '[/INST]' 分段，[/INST] 后面到 '</s>' 之间是回答
        cursor = 0  # token 位置指针
        segments = text.split('[/INST]')
        for i, seg in enumerate(segments):
            if i == 0:
                # 第一段：<image>\n<s>[INST] 问题  → 这是 prompt，跳过
                part = seg + '[/INST]'
                part_ids = self.tokenizer(part, add_special_tokens=False,
                                          return_tensors='pt').input_ids[0]
                cursor += len(part_ids)
            else:
                # 后续段：[/INST] 之后是回答 + 可能的下一轮 prompt
                if '</s>' in seg:
                    # 回答部分：从当前位置到 '</s>'
                    ans_part, remaining = seg.split('</s>', 1)
                    ans_part = ans_part.strip() + ' </s>'
                    ans_ids = self.tokenizer(ans_part, add_special_tokens=False,
                                             return_tensors='pt').input_ids[0]
                    ans_len = len(ans_ids)
                    # 将回答部分的 label 填回（不是 -100）
                    if cursor + ans_len <= len(labels):
                        labels[cursor:cursor + ans_len] = input_ids[cursor:cursor + ans_len]
                    cursor += ans_len
                    # 剩余部分是下一轮 prompt：<s>[INST] ... [/INST]
                    if '[/INST]' in remaining:
                        prompt_part = remaining.rsplit('[/INST]', 1)[0] + '[/INST]'
                        prompt_ids = self.tokenizer(prompt_part, add_special_tokens=False,
                                                    return_tensors='pt').input_ids[0]
                        cursor += len(prompt_ids)
                else:
                    # 没有 </s> 结尾，整段当回答
                    ans_ids = self.tokenizer(seg, add_special_tokens=False,
                                             return_tensors='pt').input_ids[0]
                    ans_len = len(ans_ids)
                    if cursor + ans_len <= len(labels):
                        labels[cursor:cursor + ans_len] = input_ids[cursor:cursor + ans_len]
                    cursor += ans_len

        # Step 3: pad 到固定长度
        pad_len = max_len - len(input_ids)
        if pad_len > 0:
            pad_ids = torch.full((pad_len,), self.tokenizer.pad_token_id, dtype=input_ids.dtype)
            pad_labels = torch.full((pad_len,), -100, dtype=labels.dtype)
            input_ids = torch.cat([input_ids, pad_ids], dim=0)
            labels = torch.cat([labels, pad_labels], dim=0)

        return input_ids, labels

    def __getitem__(self, idx):
        item = self.data[idx]
        # 加载图像
        image_file = item["image"]
        image = Image.open(f"{self.image_folder}/{image_file}").convert('RGB')

        if self.mode == 'pretrain':
            # 图文对形式：一轮 QA
            conv = item['conversations']
            question = conv[0]['value'].replace('<image>', '').strip()
            answer = conv[1]['value']
            prompt = create_plain_prompt(question, answer)
            prompt = self.tokenizer.image_token + "\n" + prompt
            input_ids, labels = self.tokenize_and_mask(prompt)
            return {
                'image': self.image_processor(image, return_tensors='pt')['pixel_values'][0],
                'input_ids': input_ids,
                'labels': labels
            }
        elif self.mode == 'finetune':
            # 多轮对话
            conv = item['conversations']
            # 确保第一句 human 中有 <image> 占位符
            if '<image>' not in conv[0]['value']:
                conv[0]['value'] = '<image>\n ' + conv[0]['value']
            prompt = create_conversation_prompt(conv)
            input_ids, labels = self.tokenize_and_mask(prompt)
            return {
                'image': self.image_processor(image, return_tensors='pt')['pixel_values'][0],
                'input_ids': input_ids,
                'labels': labels
            }