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
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from multi_query_rag import GeminiLLM
from retrieval_utils import (
    build_qdrant_retriever,
    build_chroma_retriever,
    join_docs,
    collect_context_items,
)


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
    topk_normal: int = 3,
    topk_step: int = 3,
    urls: List[str] | None = None,
    model: str = "models/gemini-2.5-flash",
    temperature: float = 0.0,
    doc_type: str | None = None,
) -> Dict[str, Any]:
    llm = GeminiLLM(model=model, temperature=temperature)
    retr_normal = (
        build_qdrant_retriever(top_k=topk_normal, doc_type=doc_type)
        if backend == "qdrant" else build_chroma_retriever(urls=urls, top_k=topk_normal)
    )
    retr_step = (
        build_qdrant_retriever(top_k=topk_step, doc_type=doc_type)
        if backend == "qdrant" else build_chroma_retriever(urls=urls, top_k=topk_step)
    )

    step_q = build_step_back_chain(llm).invoke({"question": question}).strip()
    normal_docs = retr_normal.get_relevant_documents(question)
    step_docs = retr_step.get_relevant_documents(step_q)

    normal_ctx = join_docs(normal_docs, top_k=topk_normal)
    step_ctx = join_docs(step_docs, top_k=topk_step)
    answer = build_response_chain(llm).invoke(
        {"normal_ctx": normal_ctx, "step_ctx": step_ctx, "question": question}
    )

    return {
        "answer": answer,
        "step_back_question": step_q,
        "normal_contexts": collect_context_items(normal_docs),
        "step_back_contexts": collect_context_items(step_docs),
        "backend": backend,
    }


