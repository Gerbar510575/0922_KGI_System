# apps/rag-service/multi_query_rag.py
# -*- coding: utf-8 -*-
import os, json, bs4
from typing import List, Tuple, Dict, Any
from operator import itemgetter
from dotenv import load_dotenv

from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
#from langchain.text_splitter import RecursiveCharacterTextSplitter
#from langchain_community.vectorstores import Chroma
#from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader

import google.genai as genai
from google.genai import types as gtypes
from google.genai import types

from qdrant_client import QdrantClient
from langchain_community.vectorstores import Qdrant as LCQdrant

# 共用檢索工具
from .retrieval_utils import (
    build_qdrant_retriever,
    build_chroma_retriever,
    get_unique_union,
    collect_context_items,
)

# ===== 讀取 API key 並初始化 Gemini =====
load_dotenv()
api_key = os.getenv("GENAI_API_KEY")
if not api_key:
    raise ValueError("請在 .env 設定 GENAI_API_KEY 以使用 Gemini")
_gclient = genai.Client(api_key=api_key)

QDRANT_HOST = os.getenv("QDRANT_HOST", "qdrant")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
COLLECTION  = os.getenv("QDRANT_COLLECTION", "kfh_docs_gemini")

# -------- Gemini LLM --------
class GeminiLLM:
    """封裝 Gemini 生成模型，讓物件可直接被 LCEL 呼叫"""
    def __init__(self, model: str = "models/gemini-2.5-flash", temperature: float = 0.0):
        self.model = model
        self.temperature = temperature

    def __call__(self, prompt: str) -> str:
        resp = _gclient.models.generate_content(model=self.model, contents=prompt)
        return resp.text or ""

# ===== Embeddings 封裝（供 Chroma/Qdrant 用）=====
class GeminiEmbeddings:
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        vecs = []
        for t in texts:
            r = _gclient.models.embed_content(
                model="models/text-embedding-004",
                contents=t,
                config=types.EmbedContentConfig(task_type="retrieval_document"),
            )
            vecs.append(r.embeddings[0].values)
        return vecs

    def embed_query(self, text: str) -> List[float]:
        r = _gclient.models.embed_content(
            model="models/text-embedding-004",
            contents=text,
            config=types.EmbedContentConfig(task_type="retrieval_query"),
        )
        return r.embeddings[0].values

# ===== Multi-Query 產生鏈 =====
def build_multi_query_chain(llm: GeminiLLM):
    template = (
        "You are an AI assistant. Generate five different versions of the given user question "
        "to retrieve relevant documents from a vector database. Provide them line by line.\n"
        "Original question: {question}"
    )
    prompt = PromptTemplate.from_template(template)
    return prompt | llm | StrOutputParser() | (lambda x: [q.strip() for q in x.split("\n") if q.strip()])

# -------- Final RAG Chain --------
def build_final_rag_chain(retriever, llm: GeminiLLM):
    rag_template = (
        "Answer the following question based on this context:\n\n"
        "{context}\n\n"
        "Question: {question}"
    )
    prompt = PromptTemplate.from_template(rag_template)

    # 多視角 → 檢索（map）→ 去重 → 拼 context → 生成
    generate_queries = build_multi_query_chain(llm)
    retrieval_chain = generate_queries | retriever.map() | get_unique_union
    final_chain = (
        {
            "context": lambda x: "\n\n---\n\n".join([d.page_content for d in retrieval_chain.invoke(x)[:8]]),
            "question": itemgetter("question"),
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return final_chain, retrieval_chain

# ===== 對外主入口 =====
def run_multi_query_rag(
    question: str,
    backend: str = "qdrant",    # "qdrant" | "chroma"
    top_k: int = 3,
    urls: List[str] | None = None,
    model: str = "models/gemini-2.5-flash",
    temperature: float = 0.0,
) -> Dict[str, Any]:
    llm = GeminiLLM(model=model, temperature=temperature)
    retriever = build_qdrant_retriever(top_k=top_k) if backend == "qdrant" else build_chroma_retriever(urls=urls, top_k=top_k)
    final_chain, retrieval_chain = build_final_rag_chain(retriever, llm)

    docs: List[Document] = retrieval_chain.invoke({"question": question})
    ans = final_chain.invoke({"question": question})
    ctx = collect_context_items(docs, top_k=8)

    return {"answer": ans, "contexts": ctx, "backend": backend}
