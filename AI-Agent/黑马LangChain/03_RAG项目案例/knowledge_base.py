"""知识库"""

import os
import config_data as config
import hashlib
from langchain_chroma import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from datetime import datetime

def check_md5(md_str: str):
    """检查传入的md5字符串是否已经被处理过了"""
    if not os.path.exists(config.md5_path):
        open(config.md5_path, 'w', encoding='utf-8').close()  # 如果文件不存在，则创建一个空文件
        return False
    for line in open(config.md5_path, 'r', encoding='utf-8'):
        if line.strip() == md_str:
            return True
    return False

def save_md5(md_str: str, filename: str):
    """将传入的md5字符串，保存在本地文件中"""
    with open(config.md5_path, 'a', encoding='utf-8') as f:
        f.write(f"{md_str}\n")


def get_string_md5(input_str: str, encoding='utf-8'):
    """将传入的字符串换为md5字符串"""
    # 将字符串转换为bytes字节数组
    str_bytes = input_str.encode(encoding=encoding)
    
    md5_obj = hashlib.md5()         # 得到m5对象
    md5_obj.update(str_bytes)       # 更新内容（传入即将要转换的字节数组）
    md5_hex = md5_obj.hexdigest()   # 得到md5的十六进制字符串
    return md5_hex


class KnowledgeBaseService(object):
    def __init__(self):
        os.makedirs(config.persist_directory, exist_ok=True)  # 如果目录不存在，则创建目录
        self.chroma = Chroma(  # 向量存储的实例 Chroma向量库对象
            collection_name=config.chroma_collection_name,      # 数据库的表名
            embedding_function=DashScopeEmbeddings(model="text-embedding-v4"),
            persist_directory=config.persist_directory          # 数据库的存储目录
        )      
        self.spliter = RecursiveCharacterTextSplitter(  # 文本分割器的对象
            chunk_size=config.chunk_size,       # 分割后的文本段最大长度
            chunk_overlap=config.chunk_overlap, # 连续文本段之间的字符重叠量
            separators=config.separators,       # 自然段落划分的符号
            length_function=len                 # 计算文本长度的函数
        )     
        
    def upload_by_str(self, data, filename):
        """将传入的字符串数据，进行向量化，上传到向量存储中"""
        # 先得到传入字符串的md5值
        md5_hex = get_string_md5(data)
        if check_md5(md5_hex):
            print(f"文件 {filename} 已经被处理过了，跳过上传")
            return "[失败] 文件已经被处理过了，跳过上传"

        # 判断 文件是否超过阈值
        if len(data) > config.max_split_char_number:
            # 如果传入的字符串长度大于阈值，则进行文本分割
            knowledge_chunks: list[str] = self.spliter.split_text(data)
        else:
            knowledge_chunks: list[str] = [data]
        
        # 上传到向量存储中
        self.chroma.add_texts(   # 元数据，标记每个文本段的来源文件名
            texts=knowledge_chunks, 
            metadatas=[{
                "source": filename,
                "create_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "operator": "爹"
            } for _ in range(len(knowledge_chunks))]
        )
        
        # 将md5值保存到本地文件中，表示该文件已经被处理过了
        save_md5(md5_hex, filename)
        
        return "[成功] 内容已经成功载入向量库"

    
if __name__ == "__main__":
    r1 = get_string_md5("测试字符串")
    r2 = get_string_md5("测试字符串")
    r3 = get_string_md5("测试字符串11")
    
    # print(r1)
    # print(r2)
    # print(r3)
    
    # print(check_md5("1f3ca051028d1d1e95a6f4e269d727ab"))
    service = KnowledgeBaseService()
    print(service.upload_by_str("周杰伦2", "test.txt"))
