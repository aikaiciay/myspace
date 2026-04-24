import streamlit as st
import os
import tempfile
from openai import OpenAI
import shutil

# 导入你的底层核心模块
from search_engine import RAGSearchEngine
from pdf_parser import extract_markdown_and_images
from vision_helper import process_all_images
from chunker import create_chunks
from config import API_KEY, IMAGE_DIR

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# ================= 1. 页面与全局设置 =================
st.set_page_config(page_title="工业级多模态 RAG", page_icon="🏢", layout="wide")
client = OpenAI(api_key=API_KEY, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")

# 初始化全局状态变量
if "engine" not in st.session_state:
    st.session_state.engine = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# ================= 2. 侧边栏：文件上传与处理引擎 =================
with st.sidebar:
    st.image("https://img.alicdn.com/tfs/TB1..50QpXXXXX7XpXXXXXXXXXX-900-900.png", width=50) # 随便加个高级的Logo
    st.title("⚙️ 知识库管理")
    st.markdown("请先上传需要解析的 PDF 论文。")
    
    uploaded_file = st.file_uploader("📁 上传 PDF 文件", type=["pdf"])
    
    if st.button("🚀 开始构建知识库", use_container_width=True) and uploaded_file is not None:
        if st.session_state.engine is None:
            # 1. 将前端上传的文件暂存到本地临时目录
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_pdf_path = tmp_file.name
            
            if os.path.exists(IMAGE_DIR):
                shutil.rmtree(IMAGE_DIR)  # 物理超度旧图片
            os.makedirs(IMAGE_DIR)        # 重新建个干净的空文件夹
            
            if os.path.exists("./my_vector_db"):
                shutil.rmtree("./my_vector_db") # 物理超度旧的向量库（防止数据混入）
            
            # 2. 调用你的后端流水线并展示漂亮的进度条
            with st.status("正在执行工业级数据流水线...", expanded=True) as status:
                st.write("📄 正在抽取 PDF 文本与图片...")
                md_text = extract_markdown_and_images(tmp_pdf_path, IMAGE_DIR)
                
                st.write("👁️ 正在呼叫视觉大模型解析图片 (演示仅限前2张)...")
                img_descs = process_all_images(IMAGE_DIR)
                
                st.write("✂️ 正在进行语义切块...")
                chunks = create_chunks(md_text, img_descs)
                
                st.write("📦 正在存入 ChromaDB 向量库并构建 BM25...")
                # 动态初始化引擎！填补之前的 NoneType 漏洞
                engine = RAGSearchEngine(chunks=chunks)
                engine.build_index()
                
                # 存入 session_state，防止页面刷新后丢失
                st.session_state.engine = engine
                status.update(label="✅ 知识库构建完成！", state="complete", expanded=False)
            
            st.success("知识库已就绪，您可以开始提问了！")
        else:
            st.info("知识库已经构建完毕，无需重复构建。")

# ================= 3. 主界面：多模态对话与溯源 =================
st.title("📚 基于大模型的多模态 RAG 检索分析系统")
st.markdown("""
<style>
    .reportview-container .main .block-container{ max-width: 1000px; }
    .source-box { background-color: #f0f2f6; padding: 10px; border-radius: 5px; font-size: 0.9em; }
</style>
""", unsafe_allow_html=True)

# 展示历史对话
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "source" in msg:
            with st.expander("🔍 查看底层检索溯源 (双路召回+BGE精排)"):
                st.markdown(f"<div class='source-box'>{msg['source']}</div>", unsafe_allow_html=True)

# 处理用户输入
if prompt := st.chat_input("在下方输入问题（请先在左侧上传并构建知识库）"):
    if st.session_state.engine is None:
        st.warning("⚠️ 请先在左侧侧边栏上传 PDF 并点击构建知识库！")
    else:
        # 1. 记录并显示用户问题
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 2. 系统检索与回答
        with st.chat_message("assistant"):
            with st.spinner("🕵️ 正在向量库中深度检索并进行降噪重排..."):
                retrieved_docs = st.session_state.engine.search(prompt, top_k=2)
                context_str = "\n\n---\n\n".join([doc.page_content for doc in retrieved_docs])
                
            with st.spinner("🧠 大模型正在基于知识库思考..."):
                system_prompt = f"""你是一个严谨的科研助手。请严格基于以下【参考片段】回答用户问题。
【参考片段】：
{context_str}
【严格指令】：
1. 如果参考片段中包含了 Markdown 表格或图片描述，请仔细核对数据。
2. 如果参考片段中的信息完全无法回答用户问题，你必须回答：“⚠️ 文档中未涉及此信息，为防止幻觉，我拒绝回答。” 绝不允许你自己编造！"""



                
            llm_messages = [{"role": "system", "content": system_prompt}]
            
            # 提取最近的 4 条历史对话（也就是最近 2 轮），防止上下文太长爆 Token
            # 注意：API 只认识 role 和 content，所以我们要过滤掉自定义的 source 字段
            for msg in st.session_state.messages[-5:]:
                llm_messages.append({"role": msg["role"], "content": msg["content"]})

            # 🎯 核心优化 2：开启流式打字机输出
            message_placeholder = st.empty() # 创建一个空占位符
            full_response = ""
            
            response = client.chat.completions.create(
                model="qwen-plus",
                messages=llm_messages, # 传入带有历史记忆的消息列表
                temperature=0.1,
                stream=True # 开启流式输出开关！
            )
            
            # 循环接收大模型吐出来的每一个字
            for chunk in response:
                if chunk.choices[0].delta.content is not None:
                    full_response += chunk.choices[0].delta.content
                    # 动态更新占位符，加上一个闪烁的光标 ▌ 显得更逼真
                    message_placeholder.markdown(full_response + "▌")
            
            # 输出结束后，把光标去掉，显示最终结果
            message_placeholder.markdown(full_response)
            
            # 渲染底部的溯源面板
            with st.expander("🔍 查看底层检索溯源 (双路召回+BGE精排)"):
                st.markdown(f"<div class='source-box'>{context_str}</div>", unsafe_allow_html=True)

        # 3. 记录助手回答到全局状态中（保存为下一次的记忆）
        st.session_state.messages.append({"role": "assistant", "content": full_response, "source": context_str})