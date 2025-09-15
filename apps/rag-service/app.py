import os
import re
import json
import chromadb
from fastapi import FastAPI
from pydantic import BaseModel
from google import genai
from google.genai import types
from google.api_core import retry
from typing import Optional, Dict, Any, List, Tuple

# ---------------- Paths ----------------
CHROMA_DIR = os.getenv("CHROMA_DIR", "/app/chroma_db")
DB_NAME    = os.getenv("DB_NAME", "funds")

# ---------------- Models ----------------
GEMINI_MODEL  = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GENAI_API_KEY = os.getenv("GENAI_API_KEY", "")
EMBED_MODEL   = os.getenv("EMBED_MODEL", "models/gemini-embedding-001")

# ---------------- Retrieval Defaults ----------------
TOP_K     = int(os.getenv("TOP_K", 8))
SHOW_CHAR = int(os.getenv("SHOW_CHAR", 320))
MAX_STRUCTURED_PAIRS = int(os.getenv("MAX_STRUCTURED_PAIRS", 60))
MAX_PAIRS_PER_DOC    = int(os.getenv("MAX_PAIRS_PER_DOC", 20))

# ---------------- Init ----------------
app = FastAPI(title="Fund RAG API (content-only embed + structured answer)", version="6.2.0")

# ---------------- Google Gemini Client ----------------
genai_client = genai.Client(api_key=GENAI_API_KEY)

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
    doc_type: Optional[str] = None

class QueryResponse(BaseModel):
    answer: str
    passages: list

# ---------------- Metadata-aware Filter ----------------
FUND_RE  = re.compile(r"\bG0\d{2}\b", re.IGNORECASE)
MONTH_RE = re.compile(r"(20\d{2})[-/年]?(0?[1-9]|1[0-2])")

DOC_TYPE_HINTS = {
    "月報": ["月報", "持股", "產業", "績效", "規模"],
    "公開說明書": ["公開說明書", "簡式", "費用", "經理費", "保管費", "RR", "投資範圍"],
}
FEES_KWS = ["費用", "經理費", "保管費", "手續費", "申購", "贖回"]

def decide_doc_type(q: str) -> str | None:
    s = q.strip().lower()
    for dt, kws in DOC_TYPE_HINTS.items():
        if any(k.lower() in s for k in kws):
            return dt
    return None

def build_where_clause(query: str, override: Optional[str] = None) -> Dict[str, Any] | None:
    where: Dict[str, Any] = {}

    m = FUND_RE.search(query)
    if m:
        where["fund_code"] = {"$eq": m.group(0).upper()}

    m = MONTH_RE.search(query)
    if m:
        y, mo = m.group(1), m.group(2).zfill(2)
        ym = f"{y}-{mo}"
        where["asof_date"] = {"$contains": ym}

    dt = override or decide_doc_type(query)
    if dt:
        where["doc_type"] = {"$eq": dt}

    return where if where else None

# ---------------- Retrieval ----------------
def vector_search(query_vec, topk: int, where: dict | None = None):
    return db.query(query_embeddings=[query_vec], n_results=topk, where=where)

# ---------------- Structured Extract ----------------
JSONISH_SUFFIX = "_json"
CORE_ID_KEYS = {"fund_code", "fund_name", "doc_type", "asof_date"}
SKIP_KEYS = {"content"}

def parse_jsonish(v: str):
    try:
        return json.loads(v)
    except Exception:
        return None

def is_long_text(v: str, limit: int = 400) -> bool:
    return isinstance(v, str) and len(v) > limit

def label_from_meta(m: dict) -> str:
    return f"({m.get('fund_code','?')}, {m.get('fund_name','?')}, {m.get('doc_type','?')}, {m.get('asof_date','?')})"

def kv_pairs_from_meta(m: dict, limit_pairs: int) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    for k, v in m.items():
        if k in CORE_ID_KEYS or k in SKIP_KEYS:
            continue
        if not v:
            continue
        s = str(v)
        if not is_long_text(s):
            pairs.append((k, s))
    return pairs[:limit_pairs]

def extract_structured_block(results: list[dict]) -> str:
    lines: List[str] = []
    total = 0
    for r in results:
        m = r["metadata"]
        label = label_from_meta(m)
        kvs = kv_pairs_from_meta(m, MAX_PAIRS_PER_DOC)
        for k, v in kvs:
            lines.append(f"- {k}：{v} 【來源：{label}】")
            total += 1
            if total >= MAX_STRUCTURED_PAIRS:
                return "\n".join(lines)
    return "\n".join(lines)

# ---------------- Answer synthesize ----------------
ANSWER_PROMPT_TEMPLATE = """
你是一位基金文件助理。請根據提供的脈絡回答用戶問題，
**用條列式、摘要化**輸出，避免逐條 dump 所有資料。
每個回答重點後加上【來源：(fund_code, fund_name, doc_type, asof_date)】。

規則：
1. 若有「結構化欄位」，優先使用其中的數據。
2. 避免重複相同內容。
3. 整理成對一般投資人可讀的摘要。
4. 若脈絡不足請回答「不知道」。

結構化欄位（若有）：  
{structured}

脈絡摘錄（僅供補充）：  
{context}

問題：{question}

回答（條列式摘要）：  
"""

def synthesize_answer(context_txt: str, q: str, structured: str = "") -> str:
    prompt = ANSWER_PROMPT_TEMPLATE.format(context=context_txt, question=q, structured=structured)
    resp = genai_client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
    return (resp.text or "").strip()

# ---------------- Main RAG ----------------
def answer_with_rag(query_text: str, top_k: int = TOP_K, show_char: int = SHOW_CHAR,
                    fund_code: str | None = None, doc_type: str | None = None):
    if not query_text.strip():
        return "⚠️ Query is empty.", []

    where = build_where_clause(query_text, override=doc_type)
    if fund_code:
        where = {"$and": [where or {}, {"fund_code": {"$eq": fund_code}}]} if where else {"fund_code": {"$eq": fund_code}}
    if doc_type:
        where = {"$and": [where or {}, {"doc_type": {"$eq": doc_type}}]} if where else {"doc_type": {"$eq": doc_type}}

    # Step 1. Embedding
    query_emb = encode_query(query_text)

    # Step 2. Retrieval
    raw = vector_search(query_emb, topk=top_k, where=where)

    results = []
    if raw.get("ids") and raw["ids"][0]:
        for i in range(len(raw["ids"][0])):
            similarity = 1 - raw["distances"][0][i]
            meta = raw["metadatas"][0][i]
            doc = raw["documents"][0][i]
            snippet = doc[:show_char] + ("..." if len(doc) > show_char else "")
            results.append({
                "rank": i + 1,
                "id": raw["ids"][0][i],
                "similarity": round(similarity, 4),
                "metadata": meta,
                "snippet": snippet,
            })

    structured_block = extract_structured_block(results)
    contexts = "\n".join([f"[{r['metadata'].get('fund_code','?')}] {r['snippet']}" for r in results])
    answer = synthesize_answer(contexts, query_text, structured=structured_block)

    return answer, results

# ---------------- API Routes ----------------
@app.post("/query", response_model=QueryResponse)
def query_fund(request: QueryRequest):
    answer, passages = answer_with_rag(request.query, top_k=request.top_k,
                                       fund_code=request.fund_code, doc_type=request.doc_type)
    return {"answer": answer, "passages": passages}

@app.get("/health")
def health():
    return {"status": "ok"}





