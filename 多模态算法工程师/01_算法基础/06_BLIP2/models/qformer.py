import torch
import torch.nn as nn
from transformers import (BertConfig, BertModel)

class QFormerConfig(BertConfig):
    def __init__(self, num_query_tokens=32, **kwargs):
        super().__init__(**kwargs)
        self.num_query_tokens = num_query_tokens # 可学习 query 数量
        
class Blip2QFormer(nn.Module):
    def __init__(self, config: QFormerConfig):
        super().__init__()
        self.config = config
        # 可学习的 query embeddings
        self.query_tokens = nn.Parameter(torch.zeros(1, config.num_query_tokens, config.hidden_size))
        nn.init.trunc_normal_(self.query_tokens, std=0.02) # 正态初始化
        # 基座 BERT 编码器，我们只是用其中的Transformer层，不用 embedding
        self.bert = BertModel(config, add_pooling_layer=False)
        # 如果 config 中隐藏大小与视觉特征维度不匹配，需要加一层投影
        self.vision_proj = nn.Linear(config.hidden_size, config.hidden_size) if config.hidden_size != config.hidden_size else nn.Identity()
        self.layer_norm = nn.LayerNorm(config.hidden_size)
        
    def forward(self, image_embeds, attention_mask=None):
        """
        image_embeds: [batch_size, seq_len, hidden_size]
        attention_mask: [batch_size, seq_len]
        """
        batch_size = image_embeds.size(0)
        # 投影视觉特征到 Q-Former 隐藏维度
        image_embeds = self.vision_proj(image_embeds)
        # 复制 query tokens 到 batch_size
        query_tokens = self.query_tokens.expand(batch_size, -1, -1) # [batch_size, num_query_tokens, hidden_size]

        # 准备 BERT 输入：将 query tokens 作为 input_ids 对应的 embedding，同时提供 encoder_hidden_states
        # 手动构建 inputs_embeds 并调用 BERT 的 encoder
        # 首先获取 BERT embedding 层
        inputs_embeds = query_tokens  # (B, num_queries, hidden)
        # 扩展 attention mask：query 之间双向注意力，query 与 image 交叉注意力
        # 通过 cross_attention_mask 控制，此处先忽略，直接调用 BERT
        if attention_mask is not None:
            query_attention_mask = torch.ones(batch_size, self.config.num_query_tokens).to(attention_mask.device)
            attention_mask = torch.cat([query_attention_mask, attention_mask], dim=1) # [batch_size, num_query_tokens + seq_len]
        
        # 通过 BERT 编码器
        encoder_outputs = self.bert.encoder(
            inputs_embeds=inputs_embeds, 
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=attention_mask
        )
        last_hidden_state = encoder_outputs.last_hidden_state # [batch_size, num_query_tokens + seq_len, hidden_size]
        # 取出 query tokens 的输出
        last_hidden_state = last_hidden_state[:, :self.config.num_query_tokens] # [batch_size, num_query_tokens, hidden_size]
        last_hidden_state = self.layer_norm(last_hidden_state)
        return last_hidden_state
        

class QFormerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = nn.GELU()
    
    def forward(self, tgt, memory, tgt_mask=None, memory_key_padding_mask=None):
        # self-attention on queries
        tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        # Cross-attention to image
        tgt2 = self.cross_attn(tgt, memory, memory, key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        # FFN
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt
    
class QFormer(nn.Module):
    def __init__(self, num_queries, d_model, nhead, num_layers, vision_dim, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.num_queries = num_queries
        self.query_tokens = nn.Parameter(torch.zeros(1, num_queries, d_model))
        nn.init.trunc_normal_(self.query_tokens, std=0.02)
        self.vision_proj = nn.Linear(vision_dim, d_model) if vision_dim != d_model else nn.Identity()
        self.layers = nn.ModuleList([
            QFormerDecoderLayer(d_model, nhead, dim_feedforward, dropout) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, image_embeds, image_mask=None):
        B = image_embeds.size(0)
        image_embeds = self.vision_proj(image_embeds)
        tgt = self.query_tokens.expand(B, -1, -1) # [B, num_queries, d_model]
        for layer in self.layers:
            tgt = layer(tgt, image_embeds, memory_key_padding_mask=image_mask)
        return self.norm(tgt) # [B, num_queries, d_model]