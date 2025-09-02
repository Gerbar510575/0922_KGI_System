# apps/rag-service/step_back_rag.py
# -*- coding: utf-8 -*-
"""
Step-Back 提問 + 兩路檢索 + RAG（完全使用 Gemini：生成 + Embeddings）
- 預設 backend="qdrant"：kfh_docs_gemini（Gemini 向量）
- 可選 backend="chroma"：臨時索引 (PDF/URL)，亦用 Gemini 向量
回傳：answer、step_back_question、兩路 contexts
"""

import os, bs4
from typing import List, Dict, Any
#from dotenv import load_dotenv

# LangChain：文件/切片/提示/鏈
#from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
#from langchain.text_splitter import RecursiveCharacterTextSplitter
#from langchain_core.documents import Document
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Vectorstores
#from langchain_community.vectorstores import Chroma
#from langchain_community.vectorstores import Qdrant as LCQdrant
#from qdrant_client import QdrantClient

# 從既有模組沿用 Gemini 封裝與 Qdrant 連線設定（皆以 google.genai 實作）
#from .multi_query_rag import (
    #GeminiLLM, GeminiEmbeddings,
    #QDRANT_HOST, QDRANT_PORT, COLLECTION
#)

from .multi_query_rag import GeminiLLM
from .retrieval_utils import (
    build_qdrant_retriever,
    build_chroma_retriever,
    join_docs,
    collect_context_items,
)

# 讀取 .env 並強制檢查金鑰（此模組必須使用 Gemini）
#load_dotenv()
#if not os.getenv("GENAI_API_KEY"):
    #raise ValueError("請在 .env 設定 GENAI_API_KEY，且此模組要求必須使用 Gemini。")

# ---------- Step-Back 問題生成鏈（Gemini 文字生成） ----------
def build_step_back_chain(llm: GeminiLLM):
    tmpl = (
        "You are an expert at world knowledge. Step back and paraphrase the user question "
        "into a more generic question that is easier to answer.\n\n"
        "Examples:\n"
        "Input: Could the members of The Police perform lawful arrests?\n"
        "Output: what can the members of The Police do?\n\n"
        "Input: Jan Sindel’s was born in what country?\n"
        "Output: what is Jan Sindel’s personal history?\n\n"
        "New question: {question}\n"
        "Return only the rephrased step-back question."
    )
    return PromptTemplate.from_template(tmpl) | llm | StrOutputParser()

# ---------- 最終回答鏈（Gemini 文字生成） ----------
def build_response_chain(llm: GeminiLLM):
    tmpl = (
        "You are an expert of world knowledge. Your response should be comprehensive and must not "
        "contradict the provided context (ignore irrelevant context).\n\n"
        "# Normal Context\n{normal_ctx}\n\n"
        "# Step-Back Context\n{step_ctx}\n\n"
        "# Original Question: {question}\n# Answer:"
    )
    return PromptTemplate.from_template(tmpl) | llm | StrOutputParser()

# ---------- 對外：一次完成 Step-Back RAG ----------
def run_step_back_rag(
    question: str,
    backend: str = "qdrant",
    topk_normal: int = 4,
    topk_step: int = 4,
    urls: List[str] | None = None,
    temperature: float = 0.0,
    model: str = "models/gemini-2.5-flash",
) -> Dict[str, Any]:
    llm = GeminiLLM(model=model, temperature=temperature)
    retr_normal = build_qdrant_retriever(top_k=topk_normal) if backend=="qdrant" else build_chroma_retriever(urls=urls, top_k=topk_normal)
    retr_step   = build_qdrant_retriever(top_k=topk_step)   if backend=="qdrant" else build_chroma_retriever(urls=urls, top_k=topk_step)

    step_q = build_step_back_chain(llm).invoke({"question": question}).strip()
    normal_docs = retr_normal.get_relevant_documents(question)
    step_docs   = retr_step.get_relevant_documents(step_q)

    normal_ctx = join_docs(normal_docs, top_k=topk_normal)
    step_ctx   = join_docs(step_docs,   top_k=topk_step)
    answer = build_response_chain(llm).invoke({"normal_ctx": normal_ctx, "step_ctx": step_ctx, "question": question})

    return {
        "backend": backend,
        "step_back_question": step_q,
        "answer": answer,
        "normal_contexts": collect_context_items(normal_docs, top_k=topk_normal),
        "step_back_contexts": collect_context_items(step_docs,   top_k=topk_step),
    }

