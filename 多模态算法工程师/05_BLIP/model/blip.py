"""
BLIP 模型
"""

from model.vit import VisionTransformer
from model.med import BertConfig, BertModel, BertMHeadModel
from transformers import BertTokenizer

import torch
from torch import nn
import torch.nn.functional as F

import os
from urllib.parse import urlparse
from timm.models.hub import download_cached_file

class BLIP_Base(nn.Module):
    def __init__(self, 
                 med_config = 'configs/med_config.json',
                 image_size = 224, 
                 vit = 'base',
                 vit_grad_ckpt = False,
                 vit_ckpt_layer = 0):
        super().__init__()
        
        self.visual_encoder, vision_width = create_vit(vit, image_size, vit_grad_ckpt, vit_ckpt_layer)
        self.tokenizer = init_tokenizer()
        med_config = BertConfig.from_json_file(med_config)
        med_config.encoder_width = vision_width
        self.text_encoder = BertModel(config=med_config, add_pooling_layer=False)

    def forward(self, image, caption, mode):
        assert mode in ['image', 'text', 'multimodal'] # "mode parameter must be image, text, or multimodal"
        text = self.tokenizer(caption, return_tensors="pt").to(image.device)
                
        
        
        