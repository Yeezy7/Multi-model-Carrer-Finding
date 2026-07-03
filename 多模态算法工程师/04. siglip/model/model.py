from transformers import PreTrainedModel, PretrainedConfig, AutoModel, AutoTokenizer, AutoProcessor

from torch import nn
from transformers.utils import ModelOutput
import torch
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class SiglipOutput(ModelOutput):
    """
    用于 Siglip 模型的输出类
    """
    loss: torch.FloatTensor = None
    logits_per_text: torch.FloatTensor = None
    logits_per_image: torch.FloatTensor = None
    text_embeds: torch.FloatTensor = None
    image_embeds: torch.FloatTensor = None
    
    
class Siglipconfig(PreTrainedModel):
    model_type = "siglip"
    def __init__(self, 
                 vision_model_name_or_path: str = "openai/clip-vit-base-patch32",
                 text_model_name_or_path: str = "openai/clip-vit-base-patch32",):
        
        super(self, Siglipconfig).__init__()
        self.vision_model_name_or_path = vision_model_name_or_path
        self.text_model_name_or_path = text_model_name_or_path


class SiglipModel(PreTrainedModel):
    model_type = "siglip"
    def __init__(self, config: Siglipconfig):
        
        super(self, SiglipModel).__init__()
        self.vision_model = AutoModel.from_pretrained(config.vision_model_name_or_path)
        self.process = AutoProcessor.from_pretrained(config.vision_model_name_or_path)
        self.text_model = AutoModel.from_pretrained(config.text_model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(config.text_model_name_or_path)
        self.t = nn.Parameter(torch.randn(1)) # 可学习的温度参数
        self.b = nn.Parameter(torch.randn(1)) # 可学习的偏置参数
    
    def forward(self, input_ids, attention_mask, pixel_values):
        
        text_outputs = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        vision_outputs = self.vision_model(pixel_values=pixel_values)
        
        vision_features = vision_outputs[1] # pooler_output
        text_features = text_outputs[1] # pooler_output
        
        vision_features = vision_features / vision_features.norm(p=2, dim=-1, keepdim=True) # L2标准化
        text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True) # L2标准化    
        
        # 计算文本和图像之间的相似度，并应用可学习的温度参数和偏置参数
        logits_per_text = torch.matmul(text_features, vision_features.t()) * self.t.exp() + self.b
        logits_per_image = logits_per_text.t()
        
        b = logits_per_text.size(0) # 获取批量大小
        eye = torch.eye(b, device=logits_per_text.device) # 创建一个单位矩阵作为标签
        labels = 2*eye - torch.ones_like(logits_per_text, device=logits_per_text.device) # 将单位矩阵转换为标签矩阵，1表示匹配，-1表示不匹配
        loglik = F.logsigmoid(labels * logits_per_text) # 计算对数似然
        nll = -torch.sum(loglik, dim=-1) # 计算负对数似然
        loss = nll.mean() # 计算平均损失
        
        return SiglipOutput(loss=loss, logits_per_text=logits_per_text, logits_per_image=logits_per_image, text_embeds=text_features, image_embeds=vision_features)
        
        
        
        