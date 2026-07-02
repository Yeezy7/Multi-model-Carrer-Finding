from functools import partial
from typing import OrderedDict
import torch
from torch import nn
from torch.nn import functional as F



class PatchEmbed(nn.Module):
    """
    2D Image to Patch Embedding
    将二维图像切成不重叠patch，并将每个patch展平后映射到一个向量空间中，形成patch embedding。
    """
    def __init__(self, img_size=224, patch_size=16, in_c=3, embed_dim=768, norm_layer=None):
        super(PatchEmbed, self).__init__()
        
        # 统一成（H，W）和（P，P）形式，便于后续计算
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        self.img_size = img_size        # 输入图像尺寸，例如（224，224） 
        self.patch_size = patch_size    # patch 尺寸，例如（16，16）
        
        # 网格大小，一行多少个 patch，一列多少个 patch
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])  # (H/P, W/P)
        
        # patch 总数量 N = 14*14 = 196
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        
        # 关键：用conv2d等价实现 "切 patch + 线性投影"
        # 输入通道 in_c=3，输出通道 embed_dim=0
        # kernel = stride = patch_size => 不重叠地覆盖每个patch
        # 输出特征图形状，[B, D, H/P, W/P]，例如 [B, 768, 14, 14]
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)
        
        # 可选的归一化层（有些实现会在 patch embedding 后加一个 LN/BN)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
        
    def forward(self, x):
        # x: [B, C, H, W] 例如：[B, 3, 224, 224]
        B, C, H, W = x.shape
        
        # ViT 原始实现通常固定训练分辨率
        # 如果想支持任意分辨率，这里一般不 assert，而是动态计算grid_size，并对pos_embed插值
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        
        # 1) proj：[B, 3, 224, 224] -> [B, 768, 14, 14]
        # 2) flatten(2): 把空间维度展平
        #     [B, 768, 14, 14] -> [B, 768, 196]
        # 3) transpose(1, 2): 把 token 维度放到中间，得到序列形式
        #     [B, 768, 196] -> [B, 196, 768]
        x = self.proj(x).flatten(2).transpose(1, 2)
        
        # norm: 保持形状不变 [B, 196, 768]
        x = self.norm(x)
        
        return x

class Attention(nn.Module):
    """
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop_ratio=0, proj_drop_ratio=0):
        super(Attention, self).__init__()
        
        self.num_heads = num_heads
        head_dim = dim // num_heads # 每个 head 的维度
        self.scale = qk_scale or head_dim ** -0.5  # 缩放因子，默认为 1/sqrt(d_k)，用于稳定训练

        # 一次线性层同时生成Q，K，V
        # 输入 [B, N, C] -> 输出 [B, N, 3C]
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)  # 注意力权重的dropout
        
        # 多头 concat 后再做一次输出投影
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)  # 输出的dropout
    
    def forward(self, x):
        # x: [B, N, C]，其中 N = num_patches + 1（加上class token）
        B, N, C = x.shape
        
        # 1) 生成 qkv: [B, N, 3C]
        # 2) reshape: [B, N, 3C] -> [B, N, 3, num_heads, head_dim]
        # 3) permute: [B, N, 3, num_heads, head_dim] -> [3, B, num_heads, N, head_dim]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # q,k,v: [B, num_heads, N, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2] # make torchscript happy (cannot use tensor as tuple)
        
        # 加权求和得到每个 token 的新表示：
        # attn @ v: [B, num_heads, N, head_dim]
        # transpose: [B, num_heads, N, head_dim] -> [B, N, num_heads, head_dim]
        # reshape: [B, N, num_heads, head_dim] -> [B, N, C]
        attn = (q @ k.transpose(-2, -1)) * self.scale # 缩放
        attn = attn.softmax(dim=-1) # softmax 得到注意力权重
        attn = self.attn_drop(attn) # dropout
        
        # 输出投影 （对应Wo): [B, N, C] -> [B, N, C]
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
        

class MLP(nn.Module):
    """
    ViT / Transformer 中的 前馈神经网络
    结构：Linear -> GELU -> Dropout -> Linear -> Dropout
    注意：对每个 token 独立作用，不改变 token 数 N
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        
        # 默认保持输入输出维度一致（残差连接要求）
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        # 第 1 层：升维（C -> hidden_features）
        self.fc1 = nn.Linear(in_features, hidden_features)
        # 激活：GELU （ViT/BERT 常用）
        self.act = act_layer()
        # 第 2 层：降维（hidden_features -> C）
        self.fc2 = nn.Linear(hidden_features, out_features)
        # dropout：在 fc1 后和 fc2 后都会用一次
        self.drop = nn.Dropout(drop)
        
    def forward(self, x):
        # x: [B, N, C]
        x = self.fc1(x) # 升维
        x = self.act(x) # 激活
        x = self.drop(x) # dropout
        
        x = self.fc2(x) # 降维 [B, N, hidden] -> [B, N, C]
        x = self.drop(x) # 正则化
        return x


class Block(nn.Module):
    
    def __init__(self, dim, # token维度 D（如 768） 
                 num_heads, # 注意力头数 h
                 mlp_ratio=4., # MLP 隐藏层维度 / token维度 比例
                 qkv_bias=False,
                 qk_scale=None,
                 drop_ratio=0., # MLP输出 / attention输出的dropout
                 attn_drop_ratio=0., # attention权重的dropout
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super(Block, self).__init__()
        
        # 1） 第一个 LN：给 attention 子层做归一化
        self.norm1 = norm_layer(dim)
        
        # 2） 多头注意力子层
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=drop_ratio)
        
        # 3) 第二个 LN：给 MLP 子层做归一化
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio) # MLP 隐藏层维度
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop_ratio)
        
    def forward(self, x):
        # x: [B, N, D]
        # ---- Attention 子层----
        # LN：[B, N, D] -> [B, N, D]
        # MSA: [B, N, D] -> [B, N, D]
        # Residual: x + branch
        x = x + self.attn(self.norm1(x))
        
        # ---- MLP 子层----
        # LN：[B, N, D] -> [B, N, D]
        # MLP: [B, N, D] -> [B, N, D
        x = x + self.mlp(self.norm2(x))
        return x
    

class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_c=3, num_classes=1000, embed_dim=768, depth=12, num_heads=12,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, representation_size=None, drop_ratio=0., attn_drop_ratio=0., embed_layer=PatchEmbed, norm_layer=None, act_layer=None,
                 ):
        super(VisionTransformer, self).__init__()
        
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # 这里num_features主要给分类用
        
        # 默认LayerNorm （eps=1e-6）和 GELU
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)    
        act_layer = act_layer or nn.GELU
        
        if representation_size is None:
            representation_size = embed_dim
        
        # 1) Patch Embedding: 将图像切成patch并映射到向量空间
        # 输出： [B, N, D]，M=196 其中 num_patches = (H/P)*(W/P)
        self.patch_embed = embed_layer(img_size=img_size, patch_size=patch_size, in_c=in_c, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        
        # 2) 可学习 cls token：shape [1, 1, D]，在 forward 时会 expand 到 [B, 1, D]
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # 3) 可学习绝对位置编码：shape [1, N + num_tokens, D]
        # 标准 ViT：[1, 197, 768]
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        
        # 对加了位置编码后的序列做dropout，防止过拟合
        self.pos_drop = nn.Dropout(p=drop_ratio)
        
        # 5) Transformer Encoder: 堆叠 depth 个 Block
        self.blocks = nn.ModuleList(*[
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, norm_layer=norm_layer, act_layer=act_layer)
            for _ in range(depth)
        ])
        
        # 最后再做一次 LayerNorm
        self.norm = norm_layer(embed_dim)
        
        # 加一层 Linear + Tanh
        self.num_features = representation_size
        self.pre_logits = nn.Sequential(OrderedDict([
            ('fc', nn.Linear(embed_dim, representation_size)),
            ('act', nn.Tanh())
        ]))
        
        # 分类头：把特征映射到类别数 K
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        
        # Weight init: pos_embed, cls_token, head
        nn.init.trunc_normal_(self.pos_embed, std=.02)
        nn.init.trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)
    
    def forward_features(self, x):
        # [B, C, H, W] -> [B, N, D]
        x = self.patch_embed(x)
        
        # [1, 1, D] -> [B, 1, D]
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        
        # [B, N+1, D]
        x = torch.cat((cls_token, x), dim=1)
        
        # 加位置编码
        x = self.pos_drop(x + self.pos_embed)
        
        # Encoder
        x = self.blocks(x)
        x = self.norm(x)
        
        # 取 cls
        return self.pre_logits(x[:, 0])
    
    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x
        
