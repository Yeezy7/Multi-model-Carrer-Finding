
import os
import sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(PROJECT_ROOT)    
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from langchain_core.tools import tool
from rag.rag_service import RagSummarizeService
import random
from utils.path_tool import get_abs_path
from utils.logger_handler import logger
from utils.config_handler import agent_conf


rag_service = RagSummarizeService()

user_ids = ["1001", "1002", "1003", "1004", "1005"]  # 模拟的用户ID列表
month_arr = ["2025-01", "2025-02", "2025-03", "2025-04", "2025-05", "2025-06",
             "2025-07", "2025-08", "2025-09", "2025-10", "2025-11", "2025-12", ]

external_data = {}

@tool(description="RAG总结")
def rag_summarize(question: str) -> str:
    """RAG总结"""
    return rag_service.rag_summarize(question)


@tool(description="获取指定城市的天气信息，以消息字符串的形式返回")
def get_weather(city: str) -> str:
    """获取天气信息"""
    # 这里可以调用天气API获取天气信息
    return f"{city}的天气是晴天，温度25摄氏度。"


@tool(description="获取用户所在城市")
def get_user_location() -> str:
    """获取用户所在城市"""
    # 这里可以调用IP定位API获取用户所在城市
    return random.choice(["北京", "上海", "广州", "深圳"])


@tool(description="获取用户ID")
def get_user_id() -> str:
    return random.choice(user_ids)  # 随机返回一个用户ID


@tool(description="获取当前月份，以纯字符串的形式返回")
def get_current_month() -> str:
    return random.choice(month_arr)


@tool(description="从外部系统中获取指定用户在指定月份的使用记录，以纯字符串形式返回， 如果未检索到返回空字符串")
def fetch_external_data(user_id: str, month: str) -> str:
    """从外部系统中获取指定用户在指定月份的使用记录"""
    generate_external_data()  # 确保外部数据已加载
    try:
        return external_data[user_id][month]
    except KeyError:
        logger.warning(f"[fetch_external_data] 未找到用户 {user_id} 在月份 {month} 的记录")
        return ""
    
def generate_external_data():
    """生成外部系统的使用记录"""
    if external_data:
        return 
    try:
        with open(get_abs_path(agent_conf["external_data_file"]), "r", encoding="utf-8") as f:
            for line in f.readlines()[1:]:  # 跳过表头
                arr: list[str] = line.strip().split(",")
                user_id: str = arr[0].replace('"', "")  # 去掉引号
                feature: str = arr[1].replace('"', "")  
                efficiency: str = arr[2].replace('"', "") 
                consumables: str = arr[3].replace('"', "")
                comparison: str = arr[4].replace('"', "") 
                time: str = arr[5].replace('"', "")  
                if user_id not in external_data:
                    external_data[user_id] = {}
                external_data[user_id][time] = {
                    "特征": feature,
                    "效率": efficiency,
                    "耗材": consumables,
                    "对比": comparison
                }
    except FileNotFoundError:
        logger.error("外部数据文件不存在")


@tool(description="无入参，无返回值，调用后触发中间件自动为报告生成的场景动态注入上下文信息，为后续提示词切换提示上下文信息")
def fill_context_for_report():
    return "fill_context_for_report已调用"
    
    
def list_tools() -> list:
    """列出所有工具函数"""
    return [
        rag_summarize,
        get_weather,
        get_user_location,
        get_user_id,
        get_current_month,
        fetch_external_data,
        fill_context_for_report
    ]
    

if __name__ == "__main__":
    print(fetch_external_data("1001", "2025-02"))
    