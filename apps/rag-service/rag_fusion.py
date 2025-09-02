# apps/rag-service/rag_fusion.py
# -*- coding: utf-8 -*-
"""
RAG-Fusion (Reciprocal Rank Fusion) with Gemini + Qdrant
- 預設 backend="qdrant"：使用既有 kfh_docs_gemini（Gemini embeddings）
- 可選 backend="chroma"：臨時索引 (PDF/URL)，亦用 Gemini embeddings
"""

import os, json, bs4
from typing import List, Tuple, Dict, Any
from operator import itemgetter

from dotenv import load_dotenv

# LangChain Core
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
#from langchain.text_splitter import RecursiveCharacterTextSplitter

from multi_query_rag import GeminiLLM
from retrieval_utils import (
    build_qdrant_retriever,
    build_chroma_retriever,
    reciprocal_rank_fusion,
)


# ===== 多視角查詢生成（RAG-Fusion 的「相關查詢」）=====
def build_rag_fusion_query_chain(llm: GeminiLLM):
    template = (
        "You are a helpful assistant that generates multiple search queries for a single input.\n"
        "Generate 4 search queries related to: {question}\n"
        "Output:\n"
    )
    prompt = PromptTemplate.from_template(template)
    return prompt | llm | StrOutputParser() | (lambda x: [q.strip() for q in x.split("\n") if q.strip()])

# ===== RAG-Fusion 鏈 =====
def build_final_rag_fusion_chain(retriever, llm: GeminiLLM, top_n_context: int = 8):
    gen_queries = build_rag_fusion_query_chain(llm)
    retrieval_chain = gen_queries | retriever.map() | reciprocal_rank_fusion

    prompt = PromptTemplate.from_template(
        "Answer the following question based on this context:\n\n{context}\n\nQuestion: {question}"
    )

    def ctx_builder(x: Dict[str, Any]) -> str:
        fused = retrieval_chain.invoke(x)[:top_n_context]
        return "\n\n---\n\n".join([d[0].page_content for d in fused])

    final_chain = (
        {"context": ctx_builder, "question": itemgetter("question")}
        | prompt
        | llm
        | StrOutputParser()
    )
    return final_chain, retrieval_chain

# ===== 對外呼叫 =====
def run_rag_fusion(
    question: str,
    backend: str = "qdrant",
    top_k: int = 3,
    urls: List[str] | None = None,
    top_n_context: int = 8,
    model: str = "models/gemini-2.5-flash",
    temperature: float = 0.0,
) -> Dict[str, Any]:
    llm = GeminiLLM(model=model, temperature=temperature)
    retriever = build_qdrant_retriever(top_k=top_k) if backend == "qdrant" else build_chroma_retriever(urls=urls, top_k=top_k)
    final_chain, retrieval_chain = build_final_rag_fusion_chain(retriever, llm, top_n_context=top_n_context)

    fused = retrieval_chain.invoke({"question": question})
    ans = final_chain.invoke({"question": question})

    ctx = [
        {
            "source": (d.metadata or {}).get("source") or (d.metadata or {}).get("file_path") or "unknown",
            "score": float(score),
            "chunk": d.page_content[:800],
        }
        for d, score in fused[:top_n_context]
    ]
    return {"answer": ans, "contexts": ctx, "backend": backend}
