import requests
from transformers import AutoProcessor, AutoModel, AutoTokenizer
import torch
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt



processor = AutoProcessor.from_pretrained("/home/flj/flj/Multi-model-Carrer-Finding/pre-trained-model/vit-base-patch16-224")
tokenizer = AutoTokenizer.from_pretrained('/home/flj/flj/Multi-model-Carrer-Finding/pre-trained-model/chinese-roberta-wwm-ext')
model = AutoModel.from_pretrained("/home/flj/flj/Multi-model-Carrer-Finding/pre-trained-model/siglip-base-patch16-224")


url = "http://images.cocodataset.org/val2017/000000039769.jpg"
response = requests.get(url, timeout=30)
response.raise_for_status()
image = Image.open(BytesIO(response.content)).convert("RGB")

plt.figure(figsize=(8, 8))
plt.imshow(image)
plt.axis("off")
plt.tight_layout()
plt.show()

image.save("sample_image.jpg")
print("图片已保存到 sample_image.jpg")