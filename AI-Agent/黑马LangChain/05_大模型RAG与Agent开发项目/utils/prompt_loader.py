from utils.logger_handler import logger
from utils.config_handler import prompts_conf
from utils.path_tool import get_abs_path


def load_system_prompt() -> str:
    """加载系统提示词"""
    try:
        system_prompt_path = get_abs_path(prompts_conf["main_prompt_path"])
    except KeyError as e:
        logger.error(f"[load_system_prompt] 配置文件中缺少键: {e}")
        return e
    
    try:
        return open(system_prompt_path, "r", encoding="utf-8").read()
    except Exception as e:
        logger.error(f"[load_system_prompt] 加载系统提示词失败: {e}")
        return e
            

def load_rag_summarize_prompt() -> str:
    """加载RAG摘要提示词"""
    try:
        rag_summarize_prompt_path = get_abs_path(prompts_conf["rag_summarize_prompt_path"])
    except KeyError as e:
        logger.error(f"[load_rag_summarize_prompt] 配置文件中缺少键: {e}")
        return e
    
    try:
        return open(rag_summarize_prompt_path, "r", encoding="utf-8").read()
    except Exception as e:
        logger.error(f"[load_rag_summarize_prompt] 加载RAG摘要提示词失败: {e}")
        return e
            
            
def load_report_prompt() -> str:
    """加载报告提示词"""
    try:
        report_prompt_path = get_abs_path(prompts_conf["report_prompt_path"])
    except KeyError as e:
        logger.error(f"[load_report_prompt] 配置文件中缺少键: {e}")
        return e
    
    try:
        return open(report_prompt_path, "r", encoding="utf-8").read()
    except Exception as e:
        logger.error(f"[load_report_prompt] 加载报告提示词失败: {e}")
        return e
                        
                        
if __name__ == "__main__":
    print(load_system_prompt())
    # print(load_rag_summarize_prompt())
    # print(load_report_prompt())