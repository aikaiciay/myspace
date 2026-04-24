import os
# 防崩溃咒语
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.retrievers import BM25Retriever
from config import API_KEY, DB_DIR

class RAGSearchEngine:
    def __init__(self, chunks=None):
        self.embeddings = DashScopeEmbeddings(model="text-embedding-v3",dashscope_api_key=API_KEY)
        self.chunks = chunks
        self.vector_db = None
        self.bm25_retriever = None
        self.reranker = None

    def build_index(self):
        """将 chunks 存入向量数据库并初始化 BM25"""
        print("📦 [Index] 正在构建向量数据库 (ChromaDB)...")
        self.vector_db = Chroma.from_documents(
            documents=self.chunks,
            embedding=self.embeddings,
            persist_directory=DB_DIR
        )
        
        print("🚀 [Index] 正在初始化 BM25 检索器...")
        self.bm25_retriever = BM25Retriever.from_documents(self.chunks)
        self.bm25_retriever.k = 5
        print("✅ [Index] 索引构建完成！")

    def get_reranker(self):
        """延迟加载 Reranker，防止内存冲突"""
        if self.reranker is None:
            print("⚖️ [Search] 正在加载 BGE-Reranker 模型...")
            from FlagEmbedding import FlagReranker
            self.reranker = FlagReranker('BAAI/bge-reranker-base', use_fp16=True)
        return self.reranker

    def search(self, query, top_k=2):
        """双路检索 + Rerank 重排"""
        # 1. 双路召回
        print(f"🕵️ [Search] 正在检索: '{query}'")
        vector_docs = self.vector_db.similarity_search(query, k=5)
        bm25_docs = self.bm25_retriever.invoke(query)
        
        # 2. 去重
        unique_chunks = {doc.page_content: doc for doc in vector_docs + bm25_docs}
        candidates = list(unique_chunks.values())
        
        # 3. Rerank 重排
        reranker = self.get_reranker()
        pairs = [[query, doc.page_content] for doc in candidates]
        scores = reranker.compute_score(pairs)
        
        # 4. 排序并返回
        scored_docs = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        return [doc for doc, score in scored_docs[:top_k]]