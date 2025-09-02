# apps/rag-service/hyde_rag.py
# -*- coding: utf-8 -*-
"""
HyDE (Hypothetical Document Embeddings) + RAG（完全使用 Gemini：生成 + Embeddings）
- 先用 Gemini 生成「假想回答段落」（HyDE Passage）
- 再用該段落做檢索 → 取前 N 段 context → 回答原問題（Gemini 生成）
- 預設 backend="qdrant"：kfh_docs_gemini（Gemini 向量）
- 可選 backend="chroma"：臨時索引 PDF/URL，亦用 Gemini 向量
回傳：answer、hyde_passage、contexts（含 source/片段）
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

# 讀取 .env 並強制檢查金鑰（此模組必須使用 Gemini）
#load_dotenv()
#if not os.getenv("GENAI_API_KEY"):
    #raise ValueError("請在 .env 設定 GENAI_API_KEY，且此模組要求必須使用 Gemini。")

from .multi_query_rag import GeminiLLM
from .retrieval_utils import (
    build_qdrant_retriever,
    build_chroma_retriever,
    collect_context_items,
)

# ---------- HyDE Passage 生成（Gemini 文字生成） ----------
def build_hyde_chain(llm: GeminiLLM):
    tmpl = (
        "Please write a concise, neutral, fact-style passage that would answer the question.\n"
        "Avoid speculation; focus on definitional/encyclopedic content.\n\n"
        "Question: {question}\n"
        "Passage:"
    )
    return PromptTemplate.from_template(tmpl) | llm | StrOutputParser()

# ---------- 最終 RAG 回答（Gemini 文字生成） ----------
def build_final_rag_chain(llm: GeminiLLM):
    tmpl = (
        "Answer the question strictly based on the provided context. "
        "If the answer is not contained in the context, say you don't know.\n\n"
        "Context:\n{context}\n\nQuestion: {question}"
    )
    return PromptTemplate.from_template(tmpl) | llm | StrOutputParser()

# ---------- 對外主入口 ----------
def run_hyde_rag(
    question: str,
    backend: str = "qdrant",
    topk: int = 4,
    urls: List[str] | None = None,
    ctx_topn: int = 4,
    temperature: float = 0.0,
    model: str = "models/gemini-2.5-flash",
) -> Dict[str, Any]:
    llm = GeminiLLM(model=model, temperature=temperature)
    retriever = build_qdrant_retriever(top_k=topk) if backend=="qdrant" else build_chroma_retriever(urls=urls, top_k=topk)

    hyde_passage = build_hyde_chain(llm).invoke({"question": question}).strip()
    docs = retriever.get_relevant_documents(hyde_passage)

    context_text = "\n\n---\n\n".join([d.page_content for d in docs[:ctx_topn]])
    answer = build_final_rag_chain(llm).invoke({"context": context_text, "question": question})

    return {
        "backend": backend,
        "hyde_passage": hyde_passage,
        "answer": answer,
        "contexts": collect_context_items(docs, top_k=ctx_topn),
    }

