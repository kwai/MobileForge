import os
from dotenv import load_dotenv

# 获取 .env 文件的路径，为当前文件夹的父目录
dotenv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
assert os.path.exists(dotenv_path), f"{dotenv_path} not found"
load_dotenv(dotenv_path=dotenv_path, verbose=True, override=True)
print("环境变量已加载")
