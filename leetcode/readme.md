# 🚀 工业级多模态 RAG 检索分析系统 (Multi-Modal Hybrid RAG)

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32.0-red)
![Qwen-VL](https://img.shields.io/badge/Model-Qwen_VL_Plus-green)
![BGE](https://img.shields.io/badge/Reranker-BGE_Reranker-yellow)

## 📖 项目简介
本项目是一个基于**多模态大模型**与**双路召回引擎**的检索增强生成（RAG）系统。旨在解决传统 RAG 系统在处理复杂 PDF 论文时“图表丢失”和“专有名词检索不准”的痛点。

本项目完全独立开发，涵盖了从底层 PDF 多模态解析、并发优化、向量与稀疏双路建库，到在线交叉重排、流式打字机交互的完整工业级流水线。

## ✨ 核心特性 (Core Features)

* **👁️ 多模态深度解析 (Multi-Modal Parser)**
  * 剥离 PDF 文本与图片，引入**多线程并发**调用 Qwen-VL 视觉大模型，实现对复杂架构图、数据折线图的精准描述提取，解析速度提升数倍。
* **🔍 双路混合召回引擎 (Hybrid Retrieval)**
  * 构建 `ChromaDB (稠密向量)` + `BM25 (稀疏关键词)` 双路索引。
  * 完美兼顾大模型对“自然语言的语义泛化”与“专业编号（如表1、图2）的精准匹配”。
* **⚖️ BGE 交叉重排降噪 (Rerank)**
  * 召回结果统一输入 `BGE-Reranker` 模型进行二次交叉打分，有效剔除低质量上下文，提升最终喂给 LLM 的知识纯度。
* **🛡️ 防幻觉与会话记忆 (Anti-Hallucination)**
  * 组合 `多轮对话历史` + `Top-K 上下文` 构建动态 Prompt。
  * 设定极其严格的**“无信源拒答”**指令，从工程物理层面上阻断大模型的发散幻觉。
* **🖥️ 响应式 Web 交互 (Streamlit UI)**
  * 采用左右分栏大厂级 UI 设计，支持动态执行进度条展示。
  * 具备 LLM 流式打字机输出体验，并在底部提供 100% 透明的**底层检索溯源面板**。

## 🗺️ 系统架构图 (System Architecture)

![[架构图.png]]

## 📸 运行效果展示 (Demo)

![[Pasted image 20260424220906.png]]

## 🚀 快速启动 (Quick Start)

**1. 克隆项目与安装依赖**
```bash
git clone [https://github.com/你的用户名/你的项目名.git](https://github.com/你的用户名/你的项目名.git)
cd 你的项目名
pip install -r requirements.txt