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
#from dotenv import load_dotenv

# LangChain 基件
try:
    from langchain import hub
    _HUB_OK = True
except Exception:
    _HUB_OK = False

from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from multi_query_rag import GeminiLLM
from retrieval_utils import (
    build_qdrant_retriever,
    build_chroma_retriever,
    join_docs,
    collect_context_items,
)

def build_decomposition_chain(llm: GeminiLLM):
    tmpl = (
        "You are a helpful assistant that generates multiple sub-questions for an input question.\n"
        "Break it into 3 smaller, answerable sub-questions.\n"
        "Input: {question}\n"
        "Output (one per line):"
    )
    return PromptTemplate.from_template(tmpl) | llm | StrOutputParser() | (lambda x: [q.strip() for q in x.split("\n") if q.strip()])

def load_rag_prompt_from_hub_or_fallback():
    if _HUB_OK:
        try:
            return hub.pull("rlm/rag-prompt")
        except Exception:
            pass
    return PromptTemplate.from_template(
        "Use ONLY the given context to answer the question. If not present, say you don't know.\n\n"
        "Context:\n{context}\n\nQuestion: {question}"
    )

def build_decomposition_prompt():
    return PromptTemplate.from_template(
        "Here is the question you need to answer:\n---\n{question}\n---\n\n"
        "Existing Q&A pairs:\n---\n{q_a_pairs}\n---\n\n"
        "Additional context:\n---\n{context}\n---\n\n"
        "Use the above to answer:\n{question}"
    )

def answer_sub_questions_with_accumulated_qa(sub_questions: List[str], retriever, llm: GeminiLLM, topn_ctx: int = 4) -> Tuple[str, List[Dict[str, Any]]]:
    q_a_pairs = ""
    ctx_per_subq: List[Dict[str, Any]] = []
    prompt = build_decomposition_prompt()

    for sq in sub_questions:
        docs = retriever.get_relevant_documents(sq)
        ctx_text = join_docs(docs, top_k=topn_ctx)
        rag_chain = (
            {"context": lambda _: ctx_text, "question": itemgetter("question"), "q_a_pairs": itemgetter("q_a_pairs")}
            | prompt
            | llm
            | StrOutputParser()
        )
        ans = rag_chain.invoke({"question": sq, "q_a_pairs": q_a_pairs})
        q_a_pairs = (q_a_pairs + "\n---\n" + f"Question: {sq}\nAnswer: {ans}").strip()
        ctx_per_subq.append({"sub_question": sq, "contexts": collect_context_items(docs, top_k=topn_ctx)})
    return q_a_pairs, ctx_per_subq

def retrieve_and_rag_each(main_question: str, retriever, llm: GeminiLLM, sub_question_chain, topn_ctx: int = 4) -> Tuple[List[str], List[str], List[Dict[str, Any]]]:
    sub_questions = sub_question_chain.invoke({"question": main_question}) or [main_question]
    prompt_rag = load_rag_prompt_from_hub_or_fallback()
    answers, ctx_per_subq = [], []

    for sq in sub_questions:
        docs = retriever.get_relevant_documents(sq)
        ctx_text = join_docs(docs, top_k=topn_ctx)
        ans = (prompt_rag | llm | StrOutputParser()).invoke({"context": ctx_text, "question": sq})
        answers.append(ans)
        ctx_per_subq.append({"sub_question": sq, "contexts": collect_context_items(docs, top_k=topn_ctx)})
    return answers, sub_questions, ctx_per_subq

def format_qa_pairs_block(questions: List[str], answers: List[str]) -> str:
    lines = []
    for i, (q, a) in enumerate(zip(questions, answers), start=1):
        lines.append(f"Question {i}: {q}\nAnswer {i}: {a}\n")
    return "\n".join(lines).strip()

def synthesize_final_answer(llm: GeminiLLM, context_block: str, main_question: str) -> str:
    prompt = PromptTemplate.from_template(
        "Here is a set of Q+A pairs:\n\n{context}\n\nSynthesize a concise, well-structured answer to: {question}"
    )
    return (prompt | llm | StrOutputParser()).invoke({"context": context_block, "question": main_question})

def run_decompose_rag(..., doc_type: str | None = None) -> Dict[str, Any]:
    llm = GeminiLLM(model=model, temperature=temperature)
    retriever = (
        build_qdrant_retriever(top_k=top_k, doc_type=doc_type)
        if backend=="qdrant" else build_chroma_retriever(urls=urls, top_k=top_k)
    )
    decompose_chain = build_decomposition_chain(llm)

    if strategy == "parallel":
        answers, sub_questions, ctxs = retrieve_and_rag_each(...)
        qa_block = format_qa_pairs_block(sub_questions, answers)
        final = synthesize_final_answer(llm, context_block=qa_block, main_question=question)
        return {"answer": final, "sub_questions": sub_questions, "answers": answers, "contexts": ctxs, "backend": backend}
    else:
        sub_questions = decompose_chain.invoke({"question": question}) or [question]
        qas, ctxs = answer_sub_questions_with_accumulated_qa(...)
        final = synthesize_final_answer(llm, context_block=qas, main_question=question)
        return {"answer": final, "sub_questions": sub_questions, "qa_pairs": qas, "contexts": ctxs, "backend": backend}


