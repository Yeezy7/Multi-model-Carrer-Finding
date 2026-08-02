from models.llava import LlavaLlamaForCausalLM
from PIL import Image
import torch
import argparse


def test(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = LlavaLlamaForCausalLM(
        vision_model_name=args.vision_model_name,
        llm_name=args.llm_name,
        mm_hidden_mult=args.mm_hidden_mult
    )
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    model.eval()

    # 加载图像
    image = Image.open(args.image_path).convert("RGB")
    
    # 准备输入文本
    question = args.question
    
    prompt = f"<image>\n {question}"
    input_ids = model.tokenizer.encode(prompt, return_tensors="pt")
    
    # 进行推理
    with torch.no_grad():
        loss, logits = model([image], input_ids=input_ids, labels=None)
        
        # 自回归生成
        output_ids = model.llm.generate(
            input_embeds=model.llm.get_input_embeddings()(input_ids.to(device)),
            input_ids=input_ids.to(device),
            max_length=50,
            num_beams=5,
            early_stopping=True
        )
        response = model.tokenizer.decode(output_ids[0])
    
    print("Question:", question)
    print("Output Text:", response) 


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Test LLaVA Model")
    parser.add_argument("--vision_model_name", type=str, required=True, help="Name of the vision model")
    parser.add_argument("--llm_name", type=str, required=True, help="Name of the language model")
    parser.add_argument("--mm_hidden_mult", type=int, default=4, help="Multiplier for hidden dimensions in multimodal layers")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model checkpoint")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the input image")
    parser.add_argument("--question", type=str, required=True, help="Question for the model")

    args = parser.parse_args()
    test(args)