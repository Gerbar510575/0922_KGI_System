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
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Vectorstores & Loaders
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_community.vectorstores import Qdrant as LCQdrant
from qdrant_client import QdrantClient

# 從既有 multi_query_rag 重用 Gemini 封裝與環境常數
from .multi_query_rag import (
    GeminiLLM, GeminiEmbeddings,
    QDRANT_HOST, QDRANT_PORT, COLLECTION
)

load_dotenv()  # 僅為保險，確保 .env 已讀取


# ===== 多視角查詢生成（RAG-Fusion 的「相關查詢」）=====
def build_rag_fusion_query_chain(llm: GeminiLLM):
    """
    輸入 {"question": "..."}，輸出 list[str]（4 條相關查詢）
    """
    template = (
        "You are a helpful assistant that generates multiple search queries based on a single input query.\n"
        "Generate multiple search queries related to: {question}\n"
        "Output (4 queries):"
    )
    prompt = PromptTemplate.from_template(template)
    chain = (
        prompt
        | llm
        | StrOutputParser()
        | (lambda x: [q.strip() for q in x.split("\n") if q.strip()])
    )
    return chain


# ===== RRF 融合 =====
def reciprocal_rank_fusion(results: List[List[Document]], k: int = 60) -> List[Tuple[Document, float]]:
    """
    results: 多組檢索排序結果；每組是 list[Document]（已依相似度排序）
    回傳：[(Document, fused_score)]，依 score 由大到小
    """
    fused_scores: Dict[str, float] = {}
    for docs in results:
        for rank, doc in enumerate(docs):
            key = json.dumps(doc.dict(), ensure_ascii=False)
            fused_scores.setdefault(key, 0.0)
            fused_scores[key] += 1.0 / (rank + k)

    reranked = [
        (Document(**json.loads(key)), score)
        for key, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    ]
    return reranked


# ===== Retrievers =====
def build_qdrant_retriever(top_k: int = 3):
    """
    使用 Qdrant 既有 collection（kfh_docs_gemini）+ GeminiEmbeddings
    """
    cli = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    emb = GeminiEmbeddings()
    vs = LCQdrant(client=cli, collection_name=COLLECTION, embeddings=emb)
    return vs.as_retriever(search_kwargs={"k": top_k})

def build_chroma_retriever(
    docs_dir: str = "data/docs",
    urls: List[str] | None = None,
    chunk_size: int = 800,
    chunk_overlap: int = 200,
    top_k: int = 3,
):
    """
    臨時索引（Chroma）：讀取本機 PDF 與/或指定 URLs 後切片建庫；嵌入用 GeminiEmbeddings
    """
    docs: List[Document] = []

    # 本機 PDF
    if os.path.isdir(docs_dir):
        for name in os.listdir(docs_dir):
            if name.lower().endswith(".pdf"):
                docs.extend(PyPDFLoader(os.path.join(docs_dir, name)).load())

    # 網頁
    if urls:
        loader = WebBaseLoader(
            web_paths=tuple(urls),
            bs_kwargs=dict(parse_only=bs4.SoupStrainer(class_=("post-content","post-title","post-header")))
        )
        docs.extend(loader.load())

    if not docs:
        raise RuntimeError("Chroma 構建失敗：沒有可用文件（請確認 data/docs 或 urls）")

    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_documents(docs)
    vs = Chroma.from_documents(documents=chunks, embedding=GeminiEmbeddings())
    return vs.as_retriever(search_kwargs={"k": top_k})


# ===== RAG-Fusion 鏈 =====
def build_final_rag_fusion_chain(retriever, llm: GeminiLLM, top_n_context: int = 8):
    """
    回傳：
      - final_chain：輸入 {"question": str} → 融合後答案（str）
      - retrieval_chain：輸入 {"question": str} → RRF 後 [(Document, score)]
    """
    gen_queries = build_rag_fusion_query_chain(llm)
    retrieval_chain = gen_queries | retriever.map() | reciprocal_rank_fusion

    # 把前 N 段內容串成 context（可依需要調整）
    def ctx_builder(x: Dict[str, Any]) -> str:
        fused = retrieval_chain.invoke(x)[:top_n_context]
        return "\n\n---\n\n".join([d[0].page_content for d in fused])

    prompt = PromptTemplate.from_template(
        "Answer the following question based on this context:\n\n{context}\n\nQuestion: {question}"
    )

    final_chain = (
        {
            "context": ctx_builder,
            "question": itemgetter("question")
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return final_chain, retrieval_chain


# ===== 對外呼叫 =====
def run_rag_fusion(
    question: str,
    backend: str = "qdrant",        # "qdrant" | "chroma"
    top_k: int = 3,
    urls: List[str] | None = None,
    top_n_context: int = 8
) -> dict:
    """
    回傳：{"answer": str, "contexts": [ {source, score, chunk}* ], "backend": "..."}
    """
    llm = GeminiLLM()
    retriever = (
        build_qdrant_retriever(top_k=top_k)
        if backend == "qdrant"
        else build_chroma_retriever(urls=urls, top_k=top_k)
    )

    final_chain, retrieval_chain = build_final_rag_fusion_chain(retriever, llm, top_n_context=top_n_context)
    fused = retrieval_chain.invoke({"question": question})
    ans = final_chain.invoke({"question": question})

    ctx = [
        {
            "source": (d.metadata or {}).get("source") or (d.metadata or {}).get("file_path") or "unknown",
            "score": float(score),
            "chunk": d.page_content[:800]
        }
        for d, score in fused[:top_n_context]
    ]

    return {"answer": ans, "contexts": ctx, "backend": backend}
