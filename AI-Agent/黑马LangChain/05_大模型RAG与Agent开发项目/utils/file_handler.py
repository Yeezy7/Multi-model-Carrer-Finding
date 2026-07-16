"""处理文件相关"""

import os
import hashlib
from utils.logger_handler import logger   
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader


def get_file_md5_hex(file_path: str) -> str:
    """获取文件的MD5哈希值"""
    if not os.path.exists(file_path):
        logger.error(f"[md5计算] 文件不存在: {file_path}")
        return None
    
    if not os.path.isfile(file_path):
        logger.error(f"[md5计算] 路径{file_path}不是文件")
        return None
    
    md5_obj = hashlib.md5()
    chunk_size = 4096   # 4KB, 避免文件过大爆内存
    try:
        with open(file_path, "rb") as f:   # 必须二进制读取，否则会报错
            while chunk := f.read(chunk_size):
                md5_obj.update(chunk)
            """
            chunk = f.read(chunk_size)
            while chunk:
                md5_obj.update(chunk)
                chunk = f.read(chunk_size)
            """
            return md5_obj.hexdigest()   # 返回16进制字符串
    except Exception as e:
        logger.error(f"[md5计算] 计算MD5时发生错误: {e}")
        return None


def listdir_with_allowed_type(path: str, allowed_types: tuple[str]) -> tuple[str]:
    """返回文件夹内的文件列表（允许的文件后缀）"""
    if not os.path.isdir(path):
        logger.error(f"[listdir_with_allowed_type] 路径{path}不是文件夹")
        return []

    result = []
    for f in os.listdir(path):   
        if f.endswith(allowed_types):   # 以允许的后缀结尾的文件
            result.append(os.path.join(path, f))
    return tuple(result)    # 返回元组，避免被修改


def pdf_loader(file_path: str, passwd: str = None) -> list[Document]:
    """加载PDF文件"""
    return PyPDFLoader(file_path=file_path, password=passwd).load()  # 全量加载


def txt_loader(file_path: str) -> list[Document]:
    """加载TXT文件"""
    return TextLoader(file_path=file_path, encoding="utf-8").load()