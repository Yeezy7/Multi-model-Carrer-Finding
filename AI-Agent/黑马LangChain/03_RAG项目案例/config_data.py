
md5_path = "./md5.txt"

# chroma参数
chroma_collection_name="rag"
persist_directory="./chroma_db"


# spliter参数
chunk_size = 1000
chunk_overlap = 100
separators = ["\n\n", "\n", ".", "!", "?", "。", "！", "？", " ", ""]
max_split_char_number = 1000    # 文本分割的阈值 

# 相似度检索阈值
similarity_threshold = 1        # 检索返回匹配的文档数量    

embedding_model_name = "text-embedding-v4"  # 嵌入模型名称
chat_model_name = "qwen3-max"       # 聊天模型名称

storage_path = "./chat_history"  # 历史消息存储路径

session_config = {
        "configurable": {
            "session_id": "user_001",
        }
    }