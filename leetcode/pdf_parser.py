import pymupdf4llm
from config import PDF_PATH, IMAGE_DIR

def extract_markdown_and_images(pdf_path=PDF_PATH, image_dir=IMAGE_DIR):
    """解析 PDF，返回 Markdown 文本，并保存图片到本地"""
    print(f"📄 [Parser] 开始解析 PDF: {pdf_path}")
    try:
        md_text = pymupdf4llm.to_markdown(
            doc=pdf_path,
            write_images=True,
            image_path=image_dir
        )
        print("✅ [Parser] PDF 解析与图片提取完成！")
        return md_text
    except Exception as e:
        print(f"❌ [Parser] PDF 解析失败: {e}")
        return None

if __name__ == "__main__":
    # 【独立测试区】直接运行此文件，只测试解析功能，不消耗 API
    test_md = extract_markdown_and_images()
    if test_md:
        print("\n--- Markdown 预览 ---")
        print(test_md[:300] + "...\n(截断显示)")