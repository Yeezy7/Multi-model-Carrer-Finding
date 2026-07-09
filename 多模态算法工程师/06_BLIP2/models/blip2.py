import torch
from models.qformer import Blip2QFormer, QFormerConfig

import math
import torch.nn as nn
from transformers import (
    AutoModel, AutoTokenizer, AutoModelForCausalLM,
    CLIPVisionModel, CLIPImageProcessor, BertConfig, BertModel
)


class Blip2ForConditionalGeneration(nn.Module):
    