import os
import base64
from openai import OpenAI
from config import API_KEY, BASE_URL, VISION_MODEL
import concurrent.futures

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def get_image_description(image_path):
    """调用视觉模型获取单张图片的描述"""
    base64_image = encode_image(image_path)
    print(f"📡 [Vision] 呼叫模型 {VISION_MODEL} 解析图片: {os.path.basename(image_path)}...")
    
    response = client.chat.completions.create(
        model=VISION_MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "你是一个严谨的科研助手。请详细描述这张图片的内容。如果是系统架构图，请说明其模块关系；如果是流程图，请简述流程；如果是数据图表，请说明其趋势。"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]
            }
        ],
        max_tokens=800,
        stream=True
    )
    
    final_content = ""
    for chunk in response:
        delta = chunk.choices[0].delta
        if getattr(delta, "content", None):
            final_content += delta.content
    return final_content

def process_all_images(image_dir, limit=None):
    """🚀 [并发加速版] 处理目录下的图片，返回描述列表。"""
    print(f"📦 [Vision] 启动多线程图片解析引擎...")
    
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if limit:
        image_files = image_files[:limit]
        
    # 定义处理单张图片的“工人函数”
    def _worker(filename):
        img_path = os.path.join(image_dir, filename)
        try:
            desc = get_image_description(img_path)
            print(f"  ✅ [Vision-Thread] 解析完成: {filename} ({desc[:15]}...)")
            return f"【图片说明：{filename}】\n{desc}"
        except Exception as e:
            print(f"  ❌ [Vision-Thread] 失败 {filename}: {e}")
            return None

    image_descriptions = []
    # 💥 核心：开启 5 个线程的线程池同时发请求
    # max_workers=5 意味着同时有 5 张图在被处理
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        # map 会自动把任务分发给 5 个线程，并且保留原本图片的顺序
        results = executor.map(_worker, image_files)
        
    # 收集成功的结果
    for res in results:
        if res:
            image_descriptions.append(res)
            
    print(f"🏁 [Vision] 所有图片多线程解析完毕！")
    return image_descriptions

if __name__ == "__main__":
    # 【独立测试区】直接运行此文件，只测试图片解析，可传入 limit=1 省钱
    from config import IMAGE_DIR
    print("开始独立测试多模态模块...")
    results = process_all_images(IMAGE_DIR, limit=1)
    if results:
        print("\n--- 提取结果预览 ---")
        print(results[0])