"""向量存储服务 - 用于存储和检索向量
"""
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


from langchain_chroma import Chroma
from utils.config_handler import chroma_conf
from model.factory import embedding_model
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils.logger_handler import logger
from utils.path_tool import get_abs_path
from utils.file_handler import txt_loader, pdf_loader, listdir_with_allowed_type, get_file_md5_hex
from langchain_core.documents import Document


class VectorStoreService:
    def __init__(self):
        self.vector_store = Chroma(
            collection_name=chroma_conf["collection_name"],
            embedding_function=embedding_model,
            persist_directory=chroma_conf["persist_directory"]
        )
        self.spliter = RecursiveCharacterTextSplitter(
            chunk_size=chroma_conf["chunk_size"],
            chunk_overlap=chroma_conf["chunk_overlap"],
            separators=chroma_conf["separators"],
            length_function=len
        )
    
    def get_retriever(self):
        """获取向量检索器"""
        return self.vector_store.as_retriever(search_kwargs={"k": chroma_conf["k"]})
    
    def load_document(self) -> None:
        """
        从数据文件夹内读取数据文件，转为向量存入向量库，
        要计算文件的MD5做去重
        """

        def check_md5_hex(md5_for_check: str) -> bool:
            if not os.path.exists(get_abs_path(chroma_conf["md5_hex_store"])):
                # 如果文件不存在，创建一个空文件    
                open(get_abs_path(chroma_conf["md5_hex_store"]), "w", encoding="utf-8").close()  # 创建空文件
                return False
            with open(get_abs_path(chroma_conf["md5_hex_store"]), "r", encoding="utf-8") as f:
                for line in f.readlines():
                    line = line.strip()  # 去掉换行符
                    if line == md5_for_check:
                        return True  # 已存在
            return False  # 没有处理过，说明不存在
        
        def save_md5_hex(md5_for_check: str) -> None:
            with open(get_abs_path(chroma_conf["md5_hex_store"]), "a", encoding="utf-8") as f:
                f.write(md5_for_check + "\n")  # 写入文件，并换行
                
        def get_file_documents(read_path: str):
            if read_path.endswith("txt"):
                return txt_loader(read_path)
            elif read_path.endswith("pdf"):
                return pdf_loader(read_path)
            else:
                logger.warning(f"[load_document] 不支持的文件类型: {read_path}")
                return []
            
        allowed_files_path: tuple[str] = listdir_with_allowed_type(
            path=get_abs_path(chroma_conf["data_path"]),
            allowed_types=tuple(chroma_conf["allow_knowledge_file_type"])
        )
        
        for path in allowed_files_path:
            # 获取文件的MD5
            md5_hex = get_file_md5_hex(path)
            if check_md5_hex(md5_hex):
                logger.info(f"[load_document] 文件已存在知识库中，跳过: {path}")
                continue
            
            # 读取文件内容
            try:
                documents: list[Document]= get_file_documents(path)
                if not documents:
                    logger.warning(f"[load_document] 文件内容为空: {path}")
                    continue
                
                split_document: list[Document] = self.spliter.split_documents(documents)  # 分割文档
                if not split_document:
                    logger.warning(f"[load_document] 分割文档为空: {path}")
                    continue
                # 将分割后的文档添加到向量库
                self.vector_store.add_documents(split_document)  # 添加到向量库
                logger.info(f"[load_document] 成功加载文件到知识库: {path}")
                save_md5_hex(md5_hex)  # 保存MD5到文件
            except Exception as e:
                # exc_info=True 会记录详细的报错堆栈，如果为False，则只会记录报错信息
                logger.error(f"[load_document] 读取文件内容时出错: {path}, 错误: {str(e)}", exc_info=True)
                continue
        
        
if __name__ == "__main__":
    
    
    
    vector_store_service = VectorStoreService()
    vector_store_service.load_document()
    retriever = vector_store_service.get_retriever()
    results = retriever.invoke("迷路")
    
    for r in results:
        print(r.page_content)
        print("="*50)
