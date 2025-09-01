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
from dotenv import load_dotenv

# LangChain：文件/切片/提示/鏈
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Vectorstores
from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores import Qdrant as LCQdrant
from qdrant_client import QdrantClient

# 從既有模組沿用 Gemini 封裝與 Qdrant 連線設定（皆以 google.genai 實作）
from .multi_query_rag import (
    GeminiLLM, GeminiEmbeddings,
    QDRANT_HOST, QDRANT_PORT, COLLECTION
)

# 讀取 .env 並強制檢查金鑰（此模組必須使用 Gemini）
load_dotenv()
if not os.getenv("GENAI_API_KEY"):
    raise ValueError("請在 .env 設定 GENAI_API_KEY，且此模組要求必須使用 Gemini。")

# ---------- Retrievers ----------
def build_qdrant_retriever(top_k: int = 4):
    cli = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    emb = GeminiEmbeddings()
    vs = LCQdrant(client=cli, collection_name=COLLECTION, embeddings=emb)
    return vs.as_retriever(search_kwargs={"k": top_k})

def build_chroma_retriever(
    docs_dir: str = "data/docs",
    urls: List[str] | None = None,
    chunk_size: int = 800,
    chunk_overlap: int = 200,
    top_k: int = 4,
):
    docs: List[Document] = []
    if os.path.isdir(docs_dir):
        for n in os.listdir(docs_dir):
            if n.lower().endswith(".pdf"):
                docs.extend(PyPDFLoader(os.path.join(docs_dir, n)).load())
    if urls:
        loader = WebBaseLoader(
            web_paths=tuple(urls),
            bs_kwargs=dict(parse_only=bs4.SoupStrainer(class_=("post-content","post-title","post-header")))
        )
        docs.extend(loader.load())
    if not docs:
        raise RuntimeError("Chroma 構建失敗：沒有文件（請確認 data/docs 或 urls）")
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_documents(docs)
    vs = Chroma.from_documents(documents=chunks, embedding=GeminiEmbeddings())
    return vs.as_retriever(search_kwargs={"k": top_k})

# ---------- 工具 ----------
def _join_docs(docs: List[Document], top_k: int = 4) -> str:
    return "\n\n---\n\n".join(d.page_content for d in docs[:top_k])

def _collect_context_items(docs: List[Document], top_k: int = 4):
    items = []
    for d in docs[:top_k]:
        meta = d.metadata if isinstance(d.metadata, dict) else {}
        items.append({
            "source": meta.get("source") or meta.get("file_path") or "unknown",
            "chunk": d.page_content[:800]
        })
    return items

# ---------- Step-Back 問題生成鏈（Gemini 文字生成） ----------
def build_step_back_chain(llm: GeminiLLM):
    """
    輸入：{question} → 輸出：更一般化、易回答的 step-back 問題（單行字串）
    """
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
    prompt = PromptTemplate.from_template(tmpl)
    return prompt | llm | StrOutputParser()

# ---------- 最終回答鏈（Gemini 文字生成） ----------
def build_response_chain(llm: GeminiLLM):
    """
    輸入：{'normal_ctx': str, 'step_ctx': str, 'question': str} → 輸出：答案字串
    """
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
    backend: str = "qdrant",       # "qdrant" | "chroma"
    topk_normal: int = 4,
    topk_step: int = 4,
    urls: List[str] | None = None,
    temperature: float = 0.0,
    model: str = "models/gemini-2.5-flash",
) -> dict:
    """
    回傳：
      {
        "backend": "...",
        "step_back_question": "...",
        "answer": "...",
        "normal_contexts": [{"source":"...", "chunk":"..."}*],
        "step_back_contexts": [{"source":"...", "chunk":"..."}*]
      }
    """
    llm = GeminiLLM(model=model, temperature=temperature)

    # 選 retriever
    if backend == "qdrant":
        retr_normal = build_qdrant_retriever(top_k=topk_normal)
        retr_step   = build_qdrant_retriever(top_k=topk_step)
    else:
        retr_normal = build_chroma_retriever(urls=urls, top_k=topk_normal)
        retr_step   = build_chroma_retriever(urls=urls, top_k=topk_step)

    # 1) 生成 step-back 問題
    step_q = (build_step_back_chain(llm)).invoke({"question": question}).strip()

    # 2) 兩路檢索
    normal_docs = retr_normal.get_relevant_documents(question)
    step_docs   = retr_step.get_relevant_documents(step_q)

    # 3) 拼接 context 並作答
    normal_ctx = _join_docs(normal_docs, top_k=topk_normal)
    step_ctx   = _join_docs(step_docs,   top_k=topk_step)
    answer = (build_response_chain(llm)).invoke({
        "normal_ctx": normal_ctx,
        "step_ctx":   step_ctx,
        "question":   question
    })

    return {
        "backend": backend,
        "step_back_question": step_q,
        "answer": answer,
        "normal_contexts": _collect_context_items(normal_docs, top_k=topk_normal),
        "step_back_contexts": _collect_context_items(step_docs,   top_k=topk_step),
    }
