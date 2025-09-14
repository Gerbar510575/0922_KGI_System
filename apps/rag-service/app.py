import os
import re
import chromadb
from fastapi import FastAPI
from pydantic import BaseModel
from google import genai
from google.genai import types
from google.api_core import retry
from typing import Optional, Dict, Any

# ---------------- Paths ----------------
CHROMA_DIR = os.getenv("CHROMA_DIR", "/app/chroma_db")
DB_NAME    = os.getenv("DB_NAME", "funds")

# ---------------- Models ----------------
GEMINI_MODEL    = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GENAI_API_KEY   = os.getenv("GENAI_API_KEY", "")

# ---------------- Retrieval Defaults ----------------
TOP_K     = int(os.getenv("TOP_K", 5))
SHOW_CHAR = int(os.getenv("SHOW_CHAR", 200))

# ---------------- Init ----------------
app = FastAPI(title="Fund RAG API", version="4.0.0")

# ---------------- Google Gemini Client ----------------
genai_client = genai.Client(api_key=GENAI_API_KEY)
EMBED_MODEL = "models/gemini-embedding-001"

# ---------------- Embedding Function ----------------
is_retriable = lambda e: (
    isinstance(e, genai.errors.APIError) and getattr(e, "code", None) in {429, 503}
)

class GeminiEmbeddingFunction(chromadb.EmbeddingFunction):
    def __init__(self, document_mode=True):
        self.document_mode = document_mode

    @retry.Retry(predicate=is_retriable)
    def __call__(self, inputs: list[str]) -> list[list[float]]:
        task_type = "retrieval_document" if self.document_mode else "retrieval_query"
        response = genai_client.models.embed_content(
            model=EMBED_MODEL,
            contents=inputs,
            config=types.EmbedContentConfig(task_type=task_type),
        )
        return [e.values for e in response.embeddings]

# ---------------- ChromaDB ----------------
try:
    chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
    embed_fn = GeminiEmbeddingFunction(document_mode=True)
    db = chroma_client.get_collection(DB_NAME, embedding_function=embed_fn)
except Exception as e:
    raise RuntimeError(f"❌ Failed to init ChromaDB: {e}")

# ---------------- Query Embedding ----------------
def encode_query(text: str) -> list[float]:
    resp = genai_client.models.embed_content(
        model=EMBED_MODEL,
        contents=text,
        config=types.EmbedContentConfig(task_type="retrieval_query"),
    )
    return resp.embeddings[0].values

# ---------------- Request / Response ----------------
class QueryRequest(BaseModel):
    query: str
    top_k: int = TOP_K
    fund_code: Optional[str] = None
    doc_type: Optional[str] = None  # "月報" or "公開說明書"

class QueryResponse(BaseModel):
    answer: str
    passages: list

# ---------------- Metadata-aware Filter ----------------
FUND_RE  = re.compile(r"\bG0\d{2}\b", re.IGNORECASE)
MONTH_RE = re.compile(r"(20\d{2})[-/年]?(0?[1-9]|1[0-2])")

DOC_TYPE_HINTS = {
    "月報": ["月報", "七月", "八月", "前十大", "持股", "產業", "地區", "績效", "報酬"],
    "公開說明書": ["公開說明書", "簡式", "費用", "經理費", "保管費", "手續費", "RR", "風險等級", "投資範圍"],
}

def decide_doc_type(q: str) -> str | None:
    s = q.strip().lower()
    for dt, kws in DOC_TYPE_HINTS.items():
        if any(k.lower() in s for k in kws):
            return dt
    return None

def build_where_clause(query: str, override: Optional[str] = None) -> Dict[str, Any] | None:
    where: Dict[str, Any] = {}

    # fund_code
    m = FUND_RE.search(query)
    if m:
        where["fund_code"] = {"$eq": m.group(0).upper()}

    # asof_date
    m = MONTH_RE.search(query)
    if m:
        y, mo = m.group(1), m.group(2).zfill(2)
        ym = f"{y}-{mo}"
        where["asof_date"] = {"$contains": ym}

    # doc_type
    dt = override or decide_doc_type(query)
    if dt:
        where["doc_type"] = {"$eq": dt}

    return where if where else None

# ---------------- Retrieval ----------------
def vector_search(query_vec, topk: int, where: dict | None = None):
    return db.query(query_embeddings=[query_vec], n_results=topk, where=where)

# ---------------- Answer synthesize ----------------
ANSWER_PROMPT_TEMPLATE = """
請只根據提供的脈絡回答用戶問題，並在每一條重點後加上【來源】，包含
(fund_code, fund_name, doc_type, asof_date)。
若脈絡不足請明確說不知道，不要臆測。

脈絡：
{context}

問題：{question}

回答（條列式）：
"""

def synthesize_answer(context_txt: str, q: str) -> str:
    prompt = ANSWER_PROMPT_TEMPLATE.format(context=context_txt, question=q)
    try:
        resp = genai_client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
        return (resp.text or "").strip()
    except Exception as e:
        return f"(⚠️ generation failed: {e})"

# ---------------- Main RAG ----------------
def answer_with_rag(query_text: str, top_k: int = TOP_K, show_char: int = SHOW_CHAR, fund_code: str | None = None, doc_type: str | None = None):
    if not query_text.strip():
        return "⚠️ Query is empty.", []

    # where 條件 (支援多欄位 AND)
    clauses = []
    base_where = build_where_clause(query_text, override=doc_type)
    if base_where:
        clauses.append(base_where)
    if fund_code:
        clauses.append({"fund_code": {"$eq": fund_code}})
    if doc_type:
        clauses.append({"doc_type": {"$eq": doc_type}})

    if len(clauses) == 1:
        where = clauses[0]
    elif len(clauses) > 1:
        where = {"$and": clauses}
    else:
        where = None

    # Step 1. Embedding
    query_emb = encode_query(query_text)

    # Step 2. Retrieval
    raw = vector_search(query_emb, topk=top_k, where=where)

    results = []
    if raw.get("ids") and raw["ids"][0]:
        for i in range(len(raw["ids"][0])):
            similarity = 1 - raw["distances"][0][i]
            meta = raw["metadatas"][0][i]
            snippet = raw["documents"][0][i][:show_char] + ("..." if len(raw["documents"][0][i]) > show_char else "")
            results.append(
                {
                    "rank": i + 1,
                    "id": raw["ids"][0][i],
                    "similarity": round(similarity, 4),
                    "metadata": meta,
                    "snippet": snippet,
                }
            )

    # Step 3. Gemini Answer
    def _label(m: dict) -> str:
        return f"[{m.get('fund_code','?')} {m.get('fund_name','?')} {m.get('doc_type','?')} {m.get('asof_date','?')}]"

    contexts = "\n".join([f"{_label(r['metadata'])} {r['snippet']}" for r in results])
    answer = synthesize_answer(contexts, query_text)

    return answer, results

# ---------------- API Routes ----------------
@app.post("/query", response_model=QueryResponse)
def query_fund(request: QueryRequest):
    answer, passages = answer_with_rag(
        request.query,
        top_k=request.top_k,
        fund_code=request.fund_code,
        doc_type=request.doc_type,
    )
    return {"answer": answer, "passages": passages}

@app.get("/health")
def health():
    return {"status": "ok"}


