# app.py
import chromadb
from fastapi import FastAPI
from pydantic import BaseModel
from google import genai
from config import (
    CHROMA_DIR,
    EMBEDDING_MODEL,
    GEMINI_MODEL,
    GENAI_API_KEY,
    TOP_K,
    SHOW_CHAR,
)

# --- HuggingFace (只 import 實際需要的東西) ---
from transformers import AutoTokenizer, AutoModel
from torch import no_grad
from torch.nn.functional import normalize

# ---------------- Init ----------------
app = FastAPI(title="Fund RAG API", version="1.0.0")

CHROMA_DIR = "/app/chroma_db"
DB_NAME = "funds"

# Load HuggingFace embedding model
tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)
hf_model = AutoModel.from_pretrained(EMBEDDING_MODEL)

# Google Gemini
genai_client = genai.Client(api_key=GENAI_API_KEY)

# ChromaDB
chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
db = chroma_client.get_collection(DB_NAME)


# ---------------- Embedding ----------------
def encode_query(text: str) -> list[float]:
    """Encode query text into embedding using HuggingFace BGE-M3"""
    text = f"query: {text}"  # 官方建議加前綴
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)

    with no_grad():
        outputs = hf_model(**inputs)
        emb = outputs.last_hidden_state[:, 0, :]  # CLS token
        emb = normalize(emb, p=2, dim=1)          # L2 normalize

    return emb.squeeze(0).tolist()


# ---------------- Request / Response ----------------
class QueryRequest(BaseModel):
    query: str
    top_k: int = TOP_K


class QueryResponse(BaseModel):
    answer: str
    passages: list


# ---------------- Heuristics ----------------
CATEGORY_KEYWORDS = {
    "prospectus_short": ["費用", "經理費", "保管費", "手續費", "買回費", "申購", "配息", "RR", "風險等級"],
    "monthly_report": ["前十大", "持股", "產業", "國家", "績效", "報酬", "淨值", "資料日期", "月報", "月"],
}


def decide_doc_type_pref(q: str) -> str | None:
    q = q.strip().lower()
    for doc_type, keywords in CATEGORY_KEYWORDS.items():
        if any(k.lower() in q for k in keywords):
            return doc_type
    return None


# ---------------- Retrieval ----------------
def vector_search(query_vec, topk: int, doc_type: str | None = None, backend="chroma"):
    if backend == "chroma":
        where = {"doc_type": doc_type} if doc_type else None
        return db.query(query_embeddings=[query_vec], n_results=topk, where=where)
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
    prompt = ANSWER_PROMPT_TEMPLATE.format(context=context_txt, question=q)
    try:
        resp = genai_client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
        return (resp.text or "").strip()
    except Exception as e:
        return f"(⚠️ generation failed: {e})"


# ---------------- Main RAG ----------------
def answer_with_rag(query_text: str, top_k: int = TOP_K, show_char: int = SHOW_CHAR):
    if not query_text.strip():
        return "⚠️ Query is empty.", []

    # Step 1. Embedding
    query_emb = encode_query(query_text)

    # Step 2. Retrieval
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

    # Step 3. Gemini Answer
    contexts = "\n".join([f"[{r['rank']}] {r['snippet']}" for r in results])
    answer = synthesize_answer(contexts, query_text)

    return answer, results


# ---------------- API Routes ----------------
@app.post("/query", response_model=QueryResponse)
def query_fund(request: QueryRequest):
    answer, passages = answer_with_rag(request.query, top_k=request.top_k)
    return {"answer": answer, "passages": passages}
