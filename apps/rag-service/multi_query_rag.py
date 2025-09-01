# apps/rag-service/multi_query_rag.py
# -*- coding: utf-8 -*-
import os, json, bs4
from typing import List, Tuple
from operator import itemgetter
from dotenv import load_dotenv

from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader

import google.genai as genai
from google.genai import types as gtypes

from qdrant_client import QdrantClient
from langchain_community.vectorstores import Qdrant as LCQdrant

load_dotenv()
api_key = os.getenv("GENAI_API_KEY")
if not api_key:
    raise ValueError("請在 .env 設定 GENAI_API_KEY 以使用 Gemini")
_gclient = genai.Client(api_key=api_key)

QDRANT_HOST = os.getenv("QDRANT_HOST", "qdrant")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
COLLECTION  = os.getenv("QDRANT_COLLECTION", "kfh_docs_gemini")

# -------- Gemini LLM / Embeddings 封裝 --------
class GeminiLLM:
    def __init__(self, model="models/gemini-2.5-flash", temperature=0.0):
        self.model = model; self.temperature = temperature
    def __call__(self, prompt: str) -> str:
        r = _gclient.models.generate_content(model=self.model, contents=prompt)
        return r.text or ""

class GeminiEmbeddings:
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        out = []
        for t in texts:
            r = _gclient.models.embed_content(
                model="models/text-embedding-004",
                contents=t,
                config=gtypes.EmbedContentConfig(task_type="retrieval_document"),
            )
            out.append(r.embeddings[0].values)
        return out
    def embed_query(self, text: str) -> List[float]:
        r = _gclient.models.embed_content(
            model="models/text-embedding-004",
            contents=text,
            config=gtypes.EmbedContentConfig(task_type="retrieval_query"),
        )
        return r.embeddings[0].values

# -------- Multi-Query 生成 & 去重 --------
def build_multi_query_chain(llm) -> callable:
    tmpl = (
        "You are an AI language model assistant. Generate five different versions of the "
        "user question to improve vector search recall. Separate with newlines. "
        "Original question: {question}"
    )
    return (PromptTemplate.from_template(tmpl) | llm | StrOutputParser() | (lambda x: [q for q in x.split("\n") if q.strip()]))

def get_unique_union(doc_lists: List[List[Document]]) -> List[Document]:
    flat = [json.dumps(d.dict(), ensure_ascii=False) for sub in doc_lists for d in sub]
    uniq = list(set(flat))
    return [Document(**json.loads(s)) for s in uniq]

# -------- Retrievers --------
def build_qdrant_retriever(top_k=3):
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    emb = GeminiEmbeddings()
    vs = LCQdrant(client=client, collection_name=COLLECTION, embeddings=emb)
    return vs.as_retriever(search_kwargs={"k": top_k})

def build_chroma_retriever(docs_dir="data/docs", urls: List[str]=None, chunk_size=800, chunk_overlap=200, top_k=3):
    docs: List[Document] = []
    if os.path.isdir(docs_dir):
        for n in os.listdir(docs_dir):
            if n.lower().endswith(".pdf"):
                docs.extend(PyPDFLoader(os.path.join(docs_dir, n)).load())
    if urls:
        loader = WebBaseLoader(web_paths=tuple(urls), bs_kwargs=dict(parse_only=bs4.SoupStrainer(class_=("post-content","post-title","post-header"))))
        docs.extend(loader.load())
    if not docs:
        raise RuntimeError("Chroma 構建失敗：沒有文件")
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(docs)
    vs = Chroma.from_documents(documents=chunks, embedding=GeminiEmbeddings())
    return vs.as_retriever(search_kwargs={"k": top_k})

# -------- Final RAG Chain --------
def build_final_rag_chain(retriever, llm) -> Tuple[callable, callable]:
    gen_queries = build_multi_query_chain(llm)
    retrieval_chain = gen_queries | retriever.map() | get_unique_union
    prompt = PromptTemplate.from_template("Answer the question using the context:\n\n{context}\n\nQuestion: {question}")
    final_chain = ({"context": retrieval_chain, "question": itemgetter("question")} | prompt | llm | StrOutputParser())
    return final_chain, retrieval_chain

def run_multi_query_rag(question: str, backend: str="qdrant", top_k: int=3, urls: List[str]=None) -> dict:
    llm = GeminiLLM()
    retriever = build_qdrant_retriever(top_k) if backend=="qdrant" else build_chroma_retriever(urls=urls, top_k=top_k)
    final_chain, retrieval_chain = build_final_rag_chain(retriever, llm)
    docs = retrieval_chain.invoke({"question": question})
    ans  = final_chain.invoke({"question": question})
    ctx = [{"source": (d.metadata or {}).get("source") or (d.metadata or {}).get("file_path") or "unknown",
            "page_content": d.page_content[:800]} for d in docs]
    return {"answer": ans, "contexts": ctx, "backend": backend}

