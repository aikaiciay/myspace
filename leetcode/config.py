import os

# ================= 基础配置 =================
PDF_PATH = r"C:\Users\26219\Desktop\me\优秀论文\我的资源\0.优秀参考论文赏析\基于SpringBoot的...沉市场交易平台的设计与实现_贾志勇.pdf"
IMAGE_DIR = "extracted_images"

# 确保图片目录存在
os.makedirs(IMAGE_DIR, exist_ok=True)

# ================= 大模型 API 配置 =================
API_KEY = "sk-204a633da0fd44a78a4da9ea6032b6fc" 
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
VISION_MODEL = "qwen-vl-plus"
DB_DIR = "./my_vector_db"