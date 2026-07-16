"""
基于Streamlit完成WEB网页上传服务

Streamlit：当WEB页面元素发生变化，则代码重新执行一遍
"""

import streamlit as st
from knowledge_base import KnowledgeBaseService
import time

# 添加网页标题
st.title("知识库更新服务")

# 添加文件上传功能
uploader_file = st.file_uploader(
    label="请上传文件", 
    type=["txt", "pdf", "docx"], 
    accept_multiple_files=False     # False 表示仅接受一个文件的上传
)

if "service" not in st.session_state:
    st.session_state.service = KnowledgeBaseService()

if uploader_file is not None:
    # 提取文件的信息
    file_name = uploader_file.name
    file_type = uploader_file.type
    file_size = uploader_file.size / 1024 # 转换为KB

    st.subheader(f"文件名：{file_name}")
    st.write(f"格式：{file_type} | 大小：{file_size:.2f} KB")
    
    # get_value -> bytes -> decode('utf-8') -> str
    file_content = uploader_file.getvalue().decode('utf-8')  # 获取文件内容的字节数据并解码为字符串
    # st.write(file_content)  # 显示文件内容
    with st.spinner("载入知识库中..."):
        time.sleep(1)
        result = st.session_state["service"].upload_by_str(file_content, file_name)
        st.write(result)