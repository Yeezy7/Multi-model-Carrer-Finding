"""为整个工程提供统一的绝对路径"""

import os


def get_project_root() -> str:
    """获取工程所在的根目录"""
    current_file =  os.path.abspath(__file__)
    # 先获取文件所在的文件夹绝对路径
    current_dir = os.path.dirname(current_file)  # 获取当前文件所在目录
    # 获取工程的根目录
    project_root = os.path.dirname(current_dir)  # 获取当前文件所在目录的上一级目录
    return project_root


def get_abs_path(relative_path: str) -> str:
    """获取相对路径对应的绝对路径"""
    project_root = get_project_root()
    abs_path = os.path.join(project_root, relative_path)
    return abs_path


if __name__ == "__main__":
    print(get_project_root())
    print(get_abs_path("data"))