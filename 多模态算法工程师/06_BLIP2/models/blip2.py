import torch
from models.qformer import Blip2QFormer, QFormer

import math
import torch.nn as nn
from transformers import (
    AutoModel, AutoTokenizer, AutoModelForCausalLM,
    CLIPVisionModel, CLIPImageProcessor, BertConfig, BertModel
)


class Blip2ForConditionalGeneration(nn.Module):
    def __init__(self, vision_model_name, llm_name, qformer_num_queries=32,
                 qformer_hidden_size=768, qformer_num_layers=6, qformer_num_heads=12,
                 llm_projection_dim=None):
        super().__init__()
        # 视觉塔（冻结）
        self.vision_model = CLIPVisionModel.from_pretrained(vision_model_name)
        self.vision_model.requires_grad_(False)
        self.image_processor = CLIPImageProcessor.from_pretrained(vision_model_name)
        vision_dim = self.vision_model.config.hidden_size # 1024
        
        # Q-Former
        self.qformer = QFormer(
            num_queries=qformer_num_queries,
            d_model=vision_dim,
            nhead=qformer_num_heads,
            num_layers=qformer_num_layers,
            vision_dim=vision_dim,
            dim_feedforward=vision_dim * 4,
            dropout=0.1,
        )
        
        # LLM
        self.llm = AutoModelForCausalLM.from_pretrained(llm_name)
        self.llm_tokenizer = AutoTokenizer.from_pretrained(llm_name)
        self.llm_tokenizer.pad_token = self.llm_tokenizer.eos_token
        llm_hidden = self.llm.config.hidden_size
        
        # 投影层：从 Q-Former 输出到 LLM 嵌入维度
        self.llm_proj = nn.Linear(qformer_hidden_size, llm_hidden) 
        if llm_projection_dim is not None:
            # 可选，先投影到更小维度再映射
            self.llm_proj = nn.Sequential(
                nn.Linear(qformer_hidden_size, llm_projection_dim),
                nn.ReLU(),
                nn.Linear(llm_projection_dim, llm_hidden)
            )
        
    def encode_image(self, images):
        """
        images: PIL image list or tensor
        returns: (B, num_queries, llm_hidden) 投影后的 query 表示
        """     
        if isinstance(images, list):
           pixel_values = self.image_processor(images, return_tensors="pt")["pixel_values"]
        else:
            pixel_values = images
        device = next(self.vision_model.parameters()).device
        pixel_values = pixel_values.to(device, dtype=self.vision_model.dtype)
        with torch.no_grad():
            vision_outputs = self.vision_model(pixel_values, output_hidden_states=True)
            # 取倒数第二层特征
            image_embeds = vision_outputs.hidden_states[-2][:, 1:]  # (B, num_patches, 1024)
        query_outputs = self.qformer(image_embeds)  # (B, num_queries, qformer_hidden)
        # 投影到 LLM 空间
        query_outputs = self.llm_proj(query_outputs)  # (B, num_queries, llm_hidden)
        return query_outputs

    def forward(self, images, input_ids, labels=None):
        """
        images: PIL image list
        input_ids: (B, seq_len) token ids
        labels: (B, seq_len) token ids
        returns: loss, logits
        """
        # 获取视觉 soft prompts
        image_tokens = self.encode_image(images)  # (B, num_queries, llm_hidden)
        # 获取文本 embeddings  
        text_embeds = self.llm.get_input_embeddings()(input_ids)  # (B, seq_len, llm_hidden)
        # 拼接视觉和文本 embeddings 
        # [视觉_token1, 视觉_token2, ..., 视觉_token32, 文本_token1, 文本_token2, ...]
        inputs_embeds = torch.cat([image_tokens, text_embeds], dim=1) # (B, num_queries + seq_len, llm_hidden)
        # 更新 attention mask: 前面 num_queries 个 token 为 1
        attn_mask = torch.ones(inputs_embeds.size()[:2], device=inputs_embeds.device, dtype=torch.long)
        
        # labels 需要在前面填充 -100，使得不计算视觉 token 的损失
        if labels is not None:
            batch_size, seq_len = labels.size()
            # 生成 -100 填充的标签
            prefix_labels = torch.full((batch_size, self.qformer.num_queries), -100, device=labels.device, dtype=torch.long)
            labels = torch.cat([prefix_labels, labels], dim=1)  # (B, num_queries + seq_len)
        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attn_mask,
            labels=labels,
            return_dict=True
        )
        return outputs.loss, outputs.logits