


def test(model, image, prompt, max_length=50):
    model.eval()
    pixel_values = model.image_processor([image], return_tensors="pt")["pixel_values"].to(device)
    # 先编码图像得到 soft prompt
    image_tokens = model.encode_images(pixel_values)  # (1, num_queries, llm_hidden)
    # 将文本 token 化
    input_ids = model.llm_tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
    # 获取文本 embedding
    text_embeds = model.llm.get_input_embeddings()(input_ids)
    # 拼接
    inputs_embeds = torch.cat([image_tokens, text_embeds], dim=1)
    # 生成
    outputs = model.llm.generate(
        inputs_embeds=inputs_embeds,
        max_length=max_length,
        do_sample=False
    )
    return model.llm_tokenizer.decode(outputs[0], skip_special_tokens=True)


if __name__ == "__main__":
    
    test(model, image, prompt)
