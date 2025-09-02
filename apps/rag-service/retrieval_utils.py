# apps/rag-service/retrieval_utils.py
# -*- coding: utf-8 -*-
"""
共用檢索工具：
- build_qdrant_retriever / build_chroma_retriever（皆使用 GeminiEmbeddings）
- get_unique_union：多輪檢索的 Document 去重
- reciprocal_rank_fusion：RRF 融合排序
- join_docs / collect_context_items：將 Documents 整理成文字/引用清單
"""

import os, json, bs4
from typing import List, Tuple, Dict, Any
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Qdrant as LCQdrant
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from qdrant_client import QdrantClient

# 從現有模組沿用 GeminiEmbeddings（內部以 google.genai 實作）
from .multi_query_rag import GeminiEmbeddings

load_dotenv()
QDRANT_HOST = os.getenv("QDRANT_HOST", "qdrant")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
COLLECTION  = os.getenv("QDRANT_COLLECTION", "kfh_docs_gemini")

# ---------------- Retrievers ----------------
def build_qdrant_retriever(top_k: int = 3):
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
    docs: List[Document] = []
    # 本機 PDF
    if os.path.isdir(docs_dir):
        for name in os.listdir(docs_dir):
            if name.lower().endswith(".pdf"):
                docs.extend(PyPDFLoader(os.path.join(docs_dir, name)).load())
    # 指定網頁
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

# ---------------- Utilities ----------------
def get_unique_union(doc_lists: List[List[Document]]) -> List[Document]:
    """將多輪檢索的文件去重（以 Document.dict() 序列化為 key）"""
    flat_json = [
        json.dumps(doc.dict(), ensure_ascii=False)
        for sub in doc_lists for doc in sub
    ]
    unique_json = list(set(flat_json))
    return [Document(**json.loads(s)) for s in unique_json]

def reciprocal_rank_fusion(results: List[List[Document]], k: int = 60) -> List[Tuple[Document, float]]:
    """RRF：融合多個已排序結果，回 [(Document, fused_score)]"""
    fused: Dict[str, float] = {}
    for docs in results:
        for rank, doc in enumerate(docs):
            key = json.dumps(doc.dict(), ensure_ascii=False)
            fused.setdefault(key, 0.0)
            fused[key] += 1.0 / (rank + k)
    reranked = [
        (Document(**json.loads(key)), score)
        for key, score in sorted(fused.items(), key=lambda x: x[1], reverse=True)
    ]
    return reranked

def join_docs(docs: List[Document], top_k: int = 4) -> str:
    return "\n\n---\n\n".join(d.page_content for d in docs[:top_k])

def collect_context_items(docs: List[Document], top_k: int = 4) -> List[Dict[str, Any]]:
    items = []
    for d in docs[:top_k]:
        meta = d.metadata if isinstance(d.metadata, dict) else {}
        items.append({
            "source": meta.get("source") or meta.get("file_path") or "unknown",
            "chunk": d.page_content[:800]
        })
    return items
