"""
LLaVA 模型
"""

import torch
import torch.nn as nn
from transformers import CLIPVisionModel, CLIPImageProcessor, AutoModelForCausalLM, AutoTokenizer

try:
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    HAS_PEFT = True
except ImportError:
    HAS_PEFT = False


# LLaVA 模型的 MLP 层
class LlavaMLIP(nn.Module):
    def __init__(self, in_dim=1024, out_dim=4096, hidden_mult=2):
        """
        hidden_mult: 隐藏层维度倍数
        """
        
        super().__init__()

        hidden_dim = int(in_dim * hidden_mult)  # 隐藏层维度
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim)
        )
    
    def forward(self, x):
        return self.mlp(x)

# LLaVA 模型的 LlamaForCausalLM 层，用于生成文本
class LlavaLlamaForCausalLM(nn.Module):
    def __init__(self, vision_model_name, llm_name, mm_hidden_mult=2, load_in_4bit=False):
        super().__init__()
        
        # 视觉编码器（始终冻结）
        self.vision_encoder = CLIPVisionModel.from_pretrained(vision_model_name)
        self.vision_encoder.requires_grad_(False)
        self.image_processor = CLIPImageProcessor.from_pretrained(vision_model_name)

        # 大语言模型（支持 4-bit 量化节省显存）
        self.load_in_4bit = load_in_4bit
        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16
            )
            self.llm = AutoModelForCausalLM.from_pretrained(
                llm_name, quantization_config=bnb_config, device_map="auto"
            )
        else:
            self.llm = AutoModelForCausalLM.from_pretrained(llm_name)
        self.tokenizer = AutoTokenizer.from_pretrained(llm_name)
        # 解决 LLaMA  tokenizer 没有 pad_token的问题
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 扩展词表：添加 <image> token
        self.image_token = "<image>"
        self.tokenizer.add_tokens([self.image_token], special_tokens=True)
        self.llm.resize_token_embeddings(len(self.tokenizer))  # 调整词嵌维度
        self.image_token_id = self.tokenizer.convert_tokens_to_ids([self.image_token])  # 获取 <image> token 的 ID

        # 投影层
        vision_hidden = self.vision_encoder.config.hidden_size  # 1024
        llm_hidden = self.llm.config.hidden_size  # 4096
        self.mm_projector = LlavaMLIP(vision_hidden, llm_hidden, mm_hidden_mult)  # 投影层

        # 图像特征选取倒数第二层
        self.vision_select_layer = -2

        # LoRA 标记
        self.is_lora_enabled = False

    def apply_lora(self, r=8, lora_alpha=16, lora_dropout=0.05, target_modules=None):
        """
        给 LLM 附加 LoRA 适配器（大幅节省显存，7B 可跑在 12GB 显卡）
        r: LoRA rank，越大精度越高但参数越多（推荐 8-64）
        lora_alpha: 通常设为 r*2
        target_modules: 对哪些模块加 LoRA，None 自动检测
        """
        if not HAS_PEFT:
            raise ImportError("请先安装 peft: pip install peft bitsandbytes")

        # 准备模型（4-bit 模型需要特殊处理）
        if self.load_in_4bit:
            self.llm = prepare_model_for_kbit_training(self.llm)

        # 自动检测目标模块（针对 LLaMA 架构）
        if target_modules is None:
            target_modules = ["q_proj", "v_proj"]  # LLaMA 标准目标模块

        # 配置 LoRA
        lora_config = LoraConfig(
            r=r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM"
        )

        # 附加 LoRA
        self.llm = get_peft_model(self.llm, lora_config)
        self.is_lora_enabled = True

        # 打印可训练参数信息
        trainable_params = sum(p.numel() for p in self.llm.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.llm.parameters())
        print(f"\n[LoRA] 可训练参数: {trainable_params:,} / {total_params:,} "
              f"({100 * trainable_params / total_params:.2f}%)")
        return self

    def encode_images(self, images):
        """
        images: 一批 PIL Image 或已处理好的像素张量 (B, C, H, W)
        return: 视觉 token 序列 (B, num_patches, llm_hidden)
        """
        if not isinstance(images, torch.Tensor):
            pixel_values = self.image_processor(images, return_tensors="pt")["pixel_values"]
        else:
            pixel_values = images
        pixel_values = pixel_values.to(self.vision_encoder.device, dtype=self.vision_encoder.dtype)

        with torch.no_grad():
            outputs = self.vision_encoder(pixel_values, output_hidden_states=True)  # output_hidden_states 返回所有隐藏层的输出

            # hidden_states[-2] shape: (B, 1+num_patches, 1024)
            # 去掉 CLS token，得到 (B, num_patches, 1024)
            selected = outputs.hidden_states[self.vision_select_layer][:, 1:]
        return self.mm_projector(selected)

    def prepare_inputs_embeds(self, input_ids, images, labels=None):
        """
        构建混合的 inputs_embeds，将 <image> 占位符替换为对应的图像 token。
        input_ids: (B, seq_len) 文本 token 序列
        images: list of PIL images or tensor batch
        labels: (B, seq_len) 可选，用于计算 loss
        返回: inputs_embeds (B, seq_len + num_img_tokens - 1, llm_hidden)
               attention_mask (更新后的)
               labels (更新后的，可选)
        """
        # 获取文本 embeddings
        text_embeds = self.llm.get_input_embeddings()(input_ids)  # (B, seq_len, llm_hidden)

        # 编码图像
        image_features = self.encode_images(images)  # (B, num_patches, llm_hidden)
        num_patches = image_features.size(1)

        # 找到 <image> token 的位置并替换
        batch_size, seq_len = input_ids.shape
        mask = input_ids == self.image_token_id  # (B, seq_len) 1/0 mask 在文本 token 序列中标记出 <image> 占位符 token 的位置。

        # 每个样本中 <image> 数量应该一致（这里假设每样本一张图，一个 <image> 占位）
        # 更严谨的做法需考虑多个图像，此处简化处理
        # 计算新的 seq_len：原长 - 占位符个数 + 每个占位替换后的 patch 数
        num_image_tokens = mask.sum(dim=1)  # (B,)
        # 确保每个样本都有且仅有一个 <image> token（训练数据需保证）
        new_seq_lens = seq_len - num_image_tokens + num_image_tokens * num_patches  # 计算将 <image> 占位符替换为真正的图像 patch embeddings 后，每条序列的新长度。
        max_len = new_seq_lens.max()

        batch_embeds = []
        batch_labels = []
        for b in range(batch_size):
            # 找到该样本中 <image> 的索引
            img_indices = mask[b].nonzero(as_tuple=True)[0]  #
            if len(img_indices) != 1:
                raise ValueError(f"样本 {b} 中 <image> token 数量不为 1，而是为 {len(img_indices)}")

            idx = img_indices[0]
            # 拼接：左侧文本 embeds + 图像 tokens + 右侧文本 embeds
            left = text_embeds[b, :idx]
            right = text_embeds[b, idx + 1:]
            combined = torch.cat([left, image_features[b], right], dim=0)
            # 填充到 max_len
            if combined.size(0) < max_len:
                pad_len = max_len - combined.size(0)
                pad_tensor = torch.zeros(pad_len, combined.size(1), device=combined.device, dtype=combined.dtype)
                combined = torch.cat([combined, pad_tensor], dim=0)
            batch_embeds.append(combined)

            # 同步处理 labels：在 <image> 位置插入 num_patches 个 -100，padding 部分也设为 -100
            if labels is not None:
                label_left = labels[b, :idx]
                image_labels = torch.full((num_patches,), -100, device=labels.device, dtype=labels.dtype)
                label_right = labels[b, idx + 1:]
                combined_labels = torch.cat([label_left, image_labels, label_right], dim=0)
                # 填充到 max_len
                if combined_labels.size(0) < max_len:
                    pad_len = max_len - combined_labels.size(0)
                    pad_labels = torch.full((pad_len,), -100, device=combined_labels.device, dtype=combined_labels.dtype)
                    combined_labels = torch.cat([combined_labels, pad_labels], dim=0)
                batch_labels.append(combined_labels)

        inputs_embeds = torch.stack(batch_embeds, dim=0)

        # 更新 attention_mask
        new_attention_mask = torch.ones(batch_size, max_len, device=inputs_embeds.device, dtype=torch.long)
        for b in range(batch_size):
            actual_len = seq_len - num_image_tokens[b] + num_image_tokens[b] * num_patches
            new_attention_mask[b, actual_len:] = 0  # 填充部分为 0

        if labels is not None:
            new_labels = torch.stack(batch_labels, dim=0)
            return inputs_embeds, new_attention_mask, new_labels
        return inputs_embeds, new_attention_mask

    def forward(self, images, input_ids, labels=None):
        """
        images: 批处理后的 PIL 图像或张量
        input_ids: (B, seq_len) 文本 token 序列，包含 <image> 占位符
        labels: (B, seq_len) 用于计算 loss
        """
        if labels is not None:
            inputs_embeds, attention_mask, labels = self.prepare_inputs_embeds(input_ids, images, labels)
        else:
            inputs_embeds, attention_mask = self.prepare_inputs_embeds(input_ids, images)

        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels
        )

        return outputs.loss, outputs.logits