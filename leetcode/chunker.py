from langchain_text_splitters import MarkdownTextSplitter
from langchain.docstore.document import Document

def create_chunks(md_text, image_descriptions):
    """将 Markdown 正文和图片描述切分为 LangChain Documents"""
    print("✂️ [Chunker] 开始智能切块...")
    markdown_splitter = MarkdownTextSplitter(chunk_size=500, chunk_overlap=50)
    
    # 1. 切分主体 Markdown 正文
    text_chunks = markdown_splitter.create_documents([md_text])
    
    # 2. 将图片描述转换为 Document 对象列表
    img_docs = [
        Document(page_content=img_desc, metadata={'source': 'image_description'}) 
        for img_desc in image_descriptions
    ]
    
    # 3. 优雅地将整个列表直接拼接到 text_chunks 尾部
    text_chunks.extend(img_docs) 
    
    print(f"✅ [Chunker] 切块完成，共生成 {len(text_chunks)} 个 Chunk。")
    return text_chunks