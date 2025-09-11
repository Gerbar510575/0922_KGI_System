# app.py
import chromadb
from fastapi import FastAPI
from pydantic import BaseModel
#from sentence_transformers import SentenceTransformer
from google import genai
from config import (
    CHROMA_DIR,
    #EMBEDDING_MODEL,
    GEMINI_MODEL,
    GENAI_API_KEY,
    TOP_K,
    SHOW_CHAR,
)

# ---------------- Init ----------------
app = FastAPI(title="Fund RAG API", version="1.0.0")

#embed_model = SentenceTransformer(EMBEDDING_MODEL)
genai_client = genai.Client(api_key=GENAI_API_KEY)
chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
db = chroma_client.get_collection("funds")

# ---------------- Request / Response ----------------
class QueryRequest(BaseModel):
    query: str
    top_k: int = TOP_K


class QueryResponse(BaseModel):
    answer: str
    passages: list


# ---------------- Heuristics ----------------
CATEGORY_KEYWORDS = {
    "prospectus_short": [
        "費用",
        "經理費",
        "保管費",
        "手續費",
        "買回費",
        "申購",
        "配息",
        "RR",
        "風險等級",
    ],
    "monthly_report": [
        "前十大",
        "持股",
        "產業",
        "國家",
        "績效",
        "報酬",
        "淨值",
        "資料日期",
        "月報",
        "月",
    ],
}


def decide_doc_type_pref(q: str) -> str | None:
    """根據 query 判斷應偏好檢索的文件類型"""
    q = q.strip().lower()
    for doc_type, keywords in CATEGORY_KEYWORDS.items():
        if any(k.lower() in q for k in keywords):
            return doc_type
    return None


# ---------------- Retrieval ----------------
def vector_search(query_vec, topk: int, doc_type: str | None = None, backend="chroma"):
    """向量檢索，支援多 backend (目前預設 chroma)"""
    if backend == "chroma":
        where = {"doc_type": doc_type} if doc_type else None
        return db.query(query_embeddings=[query_vec], n_results=topk, where=where)

    # 預留: 若要支援 qdrant，可以在這裡加
    raise ValueError(f"Unknown backend: {backend}")


# ---------------- Answer synthesize ----------------
ANSWER_PROMPT_TEMPLATE = """
請只根據提供的脈絡回答用戶問題，並在末行以【來源】列出文件名與頁碼；
若脈絡不足請明確說不知道，不要臆測。

脈絡：
{context}

問題：{question}

回答：
"""


def synthesize_answer(context_txt: str, q: str) -> str:
    """用 Gemini 生成答案"""
    prompt = ANSWER_PROMPT_TEMPLATE.format(context=context_txt, question=q)
    try:
        resp = genai_client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
        return (resp.text or "").strip()
    except Exception as e:
        return f"(⚠️ generation failed: {e})"


# ---------------- Main RAG ----------------
def answer_with_rag(query_text: str, top_k: int = TOP_K, show_char: int = SHOW_CHAR):
    """檢索 (embedding) + Gemini 生成答案"""
    if not query_text.strip():
        return "⚠️ Query is empty.", []

    # Step 1. 查詢 embedding
    q = f"query: {query_text}"
    query_emb = embed_model.encode(q, normalize_embeddings=True).tolist()

    # Step 2. 檢索 (有 doc_type 偏好)
    doc_type_pref = decide_doc_type_pref(query_text)
    raw = vector_search(query_emb, topk=top_k, doc_type=doc_type_pref, backend="chroma")

    results = []
    for i in range(len(raw["ids"][0])):
        similarity = 1 - raw["distances"][0][i]
        results.append(
            {
                "rank": i + 1,
                "id": raw["ids"][0][i],
                "similarity": round(similarity, 4),
                "metadata": raw["metadatas"][0][i],
                "snippet": raw["documents"][0][i][:show_char] + "...",
            }
        )

    # Step 3. Prompt
    contexts = "\n".join([f"[{r['rank']}] {r['snippet']}" for r in results])
    answer = synthesize_answer(contexts, query_text)

    return answer, results


# ---------------- API Routes ----------------
@app.post("/query", response_model=QueryResponse)
def query_fund(request: QueryRequest):
    """前端呼叫 API，輸入 query → 回答 + 檢索片段"""
    answer, passages = answer_with_rag(request.query, top_k=request.top_k)
    return {"answer": answer, "passages": passages}


