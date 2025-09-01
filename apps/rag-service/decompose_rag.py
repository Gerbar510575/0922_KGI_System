# apps/rag-service/decompose_rag.py
# -*- coding: utf-8 -*-
"""
Decomposition + RAG（Gemini + Qdrant/Chroma）
- 預設 backend="qdrant"：連既有 kfh_docs_gemini（Gemini 向量）
- 可選 backend="chroma"：臨時索引 PDF/URL（亦用 Gemini 向量）
回傳：final_answer、sub_questions、contexts（每題前N段）
"""
import os, json, bs4
from typing import List, Tuple, Dict, Any
from operator import itemgetter
from dotenv import load_dotenv

# LangChain 基件
try:
    from langchain import hub  # 可無網路；拉取失敗會 fallback
    _HUB_OK = True
except Exception:
    _HUB_OK = False

from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Vectorstores & loaders
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_community.vectorstores import Qdrant as LCQdrant
from qdrant_client import QdrantClient

# 從 multi_query_rag 重用 Gemini 封裝與 Qdrant 連線常數
from .multi_query_rag import (
    GeminiLLM, GeminiEmbeddings,
    QDRANT_HOST, QDRANT_PORT, COLLECTION
)

load_dotenv()

# ---------- Retrievers ----------
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
    if os.path.isdir(docs_dir):
        for name in os.listdir(docs_dir):
            if name.lower().endswith(".pdf"):
                docs.extend(PyPDFLoader(os.path.join(docs_dir, name)).load())
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

# ---------- 子問題分解 ----------
def build_decomposition_chain(llm: GeminiLLM):
    tmpl = (
        "You are a helpful assistant that generates multiple sub-questions related to an input question.\n"
        "Break the input into smaller, answerable sub-questions.\n"
        "Input: {question}\n"
        "Output (3 queries, one per line):"
    )
    return (PromptTemplate.from_template(tmpl) | llm | StrOutputParser()
            | (lambda x: [q.strip() for q in x.split("\n") if q.strip()]))

# ---------- RAG Prompt（Hub or fallback） ----------
def load_rag_prompt_from_hub_or_fallback():
    if _HUB_OK:
        try:
            return hub.pull("rlm/rag-prompt")
        except Exception:
            pass
    return PromptTemplate.from_template(
        "Use ONLY the given context to answer the question. "
        "If the answer is not in the context, say you don't know.\n\n"
        "Context:\n{context}\n\nQuestion: {question}"
    )

# ---------- 工具 ----------
def _docs_to_context_text(docs: List[Document], topn: int = 4) -> str:
    return "\n\n---\n\n".join([d.page_content for d in docs[:topn]])

def _collect_context_items(docs: List[Document], topn: int = 4):
    items = []
    for d in docs[:topn]:
        meta = d.metadata if isinstance(d.metadata, dict) else {}
        items.append({
            "source": meta.get("source") or meta.get("file_path") or "unknown",
            "chunk": d.page_content[:800]
        })
    return items

def format_qa_pairs_block(questions: List[str], answers: List[str]) -> str:
    lines = []
    for i, (q, a) in enumerate(zip(questions, answers), start=1):
        lines.append(f"Question {i}: {q}\nAnswer {i}: {a}\n")
    return "\n".join(lines).strip()

def build_decomposition_prompt():
    return PromptTemplate.from_template(
        "Here is the question you need to answer:\n---\n{question}\n---\n\n"
        "Existing Q&A pairs:\n---\n{q_a_pairs}\n---\n\n"
        "Additional context:\n---\n{context}\n---\n\n"
        "Use the above to answer:\n{question}"
    )

# ---------- 流程 A：逐題累積 Q/A ----------
def answer_sub_questions_with_accumulated_qa(
    sub_questions: List[str], retriever, llm: GeminiLLM, topn_ctx: int = 4
) -> Tuple[str, List[Dict[str, Any]]]:
    q_a_pairs = ""
    ctx_per_subq: List[Dict[str, Any]] = []
    decomp_prompt = build_decomposition_prompt()

    for sq in sub_questions:
        docs = retriever.get_relevant_documents(sq)
        ctx_text = _docs_to_context_text(docs, topn=topn_ctx)
        rag_chain = (
            {
                "context": lambda _: ctx_text,
                "question": itemgetter("question"),
                "q_a_pairs": itemgetter("q_a_pairs")
            }
            | decomp_prompt | llm | StrOutputParser()
        )
        ans = rag_chain.invoke({"question": sq, "q_a_pairs": q_a_pairs})
        q_a_pairs = (q_a_pairs + "\n---\n" + f"Question: {sq}\nAnswer: {ans}").strip()
        ctx_per_subq.append({"sub_question": sq, "contexts": _collect_context_items(docs, topn=topn_ctx)})
    return q_a_pairs, ctx_per_subq

# ---------- 流程 B：各題平行 RAG 後再綜整 ----------
def retrieve_and_rag_each(
    main_question: str, retriever, llm: GeminiLLM, sub_question_chain, topn_ctx: int = 4
) -> Tuple[List[str], List[str], List[Dict[str, Any]]]:
    sub_questions = sub_question_chain.invoke({"question": main_question}) or [main_question]
    prompt_rag = load_rag_prompt_from_hub_or_fallback()
    answers, ctx_per_subq = [], []

    for sq in sub_questions:
        docs = retriever.get_relevant_documents(sq)
        ctx_text = _docs_to_context_text(docs, topn=topn_ctx)
        ans = (prompt_rag | llm | StrOutputParser()).invoke({"context": ctx_text, "question": sq})
        answers.append(ans)
        ctx_per_subq.append({"sub_question": sq, "contexts": _collect_context_items(docs, topn=topn_ctx)})
    return answers, sub_questions, ctx_per_subq

def synthesize_final_answer(llm: GeminiLLM, context_block: str, main_question: str) -> str:
    prompt = PromptTemplate.from_template(
        "Here is a set of Q+A pairs:\n\n{context}\n\n"
        "Synthesize a concise, well-structured answer to the question: {question}"
    )
    return (prompt | llm | StrOutputParser()).invoke({"context": context_block, "question": main_question})

# ---------- 對外主入口 ----------
def run_decompose_rag(
    question: str,
    backend: str = "qdrant",        # "qdrant" | "chroma"
    top_k: int = 3,
    urls: List[str] | None = None,
    strategy: str = "accumulate",   # "accumulate" | "parallel"
    topn_context_per_subq: int = 4
) -> dict:
    llm = GeminiLLM()
    retriever = build_qdrant_retriever(top_k=top_k) if backend=="qdrant" else build_chroma_retriever(urls=urls, top_k=top_k)
    decompose_chain = build_decomposition_chain(llm)

    if strategy == "parallel":
        answers, sub_questions, ctxs = retrieve_and_rag_each(
            main_question=question, retriever=retriever, llm=llm,
            sub_question_chain=decompose_chain, topn_ctx=topn_context_per_subq
        )
        qa_block = format_qa_pairs_block(sub_questions, answers)
        final = synthesize_final_answer(llm, context_block=qa_block, main_question=question)
        return {
            "strategy": "parallel",
            "final_answer": final,
            "sub_questions": sub_questions,
            "answers": answers,
            "contexts": ctxs,
            "backend": backend
        }
    else:
        sub_questions = decompose_chain.invoke({"question": question}) or [question]
        qas, ctxs = answer_sub_questions_with_accumulated_qa(
            sub_questions=sub_questions, retriever=retriever, llm=llm, topn_ctx=topn_context_per_subq
        )
        final = synthesize_final_answer(llm, context_block=qas, main_question=question)
        return {
            "strategy": "accumulate",
            "final_answer": final,
            "sub_questions": sub_questions,
            "qa_pairs": qas,
            "contexts": ctxs,
            "backend": backend
        }

