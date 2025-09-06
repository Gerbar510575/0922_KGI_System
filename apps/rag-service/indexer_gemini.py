# -*- coding: utf-8 -*-
import os, uuid, re, hashlib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from dotenv import load_dotenv

import pdfplumber
from pdfminer.high_level import extract_text  # fallback
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct

import google.genai as genai
from google.genai import types as gtypes

load_dotenv()
api_key = os.getenv("GENAI_API_KEY")
if not api_key:
    raise ValueError("請在 .env 設定 GENAI_API_KEY")
client_g = genai.Client(api_key=api_key)

QDRANT_HOST = os.getenv("QDRANT_HOST", "qdrant")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
COLLECTION  = os.getenv("QDRANT_COLLECTION", "kfh_docs_gemini")

DOCS_DIR = Path(os.getenv("DOCS_DIR", "data/docs"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))
EMBED_WORKERS = int(os.getenv("EMBED_WORKERS", "8"))

RE_MONTHLY = re.compile(r"(月報|月報告)")
RE_PROSPECTUS = re.compile(r"(簡式公開說明書|公開說明書)")

def normalize_spaces(s: str) -> str:
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\u3000", " ", s)  # 全形空白
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

def detect_doc_type(name: str, text0: str) -> str:
    name = name.lower()
    if RE_MONTHLY.search(name) or RE_MONTHLY.search(text0):
        return "monthly_report"
    if RE_PROSPECTUS.search(name) or RE_PROSPECTUS.search(text0):
        return "prospectus_short"
    return "other"

def parse_asof_date(text: str, doc_type: str) -> str | None:
    # 月報常見：資料日期 : 2025/7/31
    m = re.search(r"(資料日期)\s*[:：]\s*(\d{4})[\/\-年](\d{1,2})[\/\-月](\d{1,2})", text)
    if m:
        y, mo, d = int(m.group(2)), int(m.group(3)), int(m.group(4))
        try:
            return datetime(y, mo, d).date().isoformat()
        except Exception:
            pass
    # 簡式公開說明書常見：刊印日期：114 年 7 月 31 日（民國年）
    m = re.search(r"(刊印日期)\s*[:：]\s*(\d{2,3})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日", text)
    if m:
        roc, mo, d = int(m.group(2)), int(m.group(3)), int(m.group(4))
        y = roc + 1911
        try:
            return datetime(y, mo, d).date().isoformat()
        except Exception:
            pass
    return None

def parse_meta(text: str) -> dict:
    meta = {}
    # 風險等級 RR\d
    m = re.search(r"RR\s*([1-5])", text, re.IGNORECASE)
    if m: meta["risk_level"] = f"RR{m.group(1)}"
    # 經理費 / 保管費
    mf = re.search(r"經理費\s*（?每年）?\s*([0-9\.]+)\s*%?", text)
    if mf: meta["management_fee_pct"] = float(mf.group(1))
    mc = re.search(r"保管費\s*（?每年）?\s*([0-9\.]+)\s*%?", text)
    if mc: meta["custody_fee_pct"] = float(mc.group(1))
    return meta

def sentence_chunks(s: str, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    # 以句號/換行做粗切再累積，避免硬切句
    units = re.split(r"(?<=[。！？\.\?\!])\s+|\n+", s)
    buf, out = "", []
    for u in units:
        if not u: continue
        if len(buf) + len(u) + 1 <= size:
            buf = (buf + " " + u).strip()
        else:
            if buf:
                out.append(buf)
            if len(u) > size:  # 單句超長，強制切
                for i in range(0, len(u), size - overlap):
                    out.append(u[i:i+size])
                buf = ""
            else:
                # 重疊
                if out:
                    last = out[-1]
                    overlap_part = last[-overlap:] if len(last) > overlap else last
                    buf = (overlap_part + " " + u).strip()
                else:
                    buf = u
    if buf:
        out.append(buf)
    return out

def extract_text_by_page(pdf_path: Path) -> list[tuple[int, str]]:
    pages = []
    try:
        with pdfplumber.open(str(pdf_path)) as pdf:
            for i, p in enumerate(pdf.pages, start=1):
                t = p.extract_text() or ""
                t = normalize_spaces(t)
                if t:
                    pages.append((i, t))
    except Exception:
        # fallback: 全文抽取（無頁資訊）
        t = extract_text(str(pdf_path)) or ""
        t = normalize_spaces(t)
        if t:
            pages = [(1, t)]
    return pages

def embed_one(text: str) -> list[float]:
    r = client_g.models.embed_content(
        model="models/text-embedding-004",
        contents=text,
        config=gtypes.EmbedContentConfig(task_type="retrieval_document"),
    )
    return r.embeddings[0].values

def embed_batch_parallel(texts: list[str], workers=EMBED_WORKERS) -> list[list[float]]:
    vecs = [None] * len(texts)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(embed_one, t): i for i, t in enumerate(texts)}
        for f in as_completed(futs):
            i = futs[f]
            vecs[i] = f.result()
    return vecs

def main():
    pdfs = sorted(DOCS_DIR.glob("*.pdf"))
    if not pdfs:
        raise RuntimeError(f"找不到 {DOCS_DIR}/*.pdf")

    # 先以第一段推維度
    probe = embed_one("probe for dimension")
    dim = len(probe)

    qc = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    qc.recreate_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )

    points = []
    for pdf in pdfs:
        pages = extract_text_by_page(pdf)
        if not pages: 
            continue

        # 判斷 doc_type / asof_date / meta
        first_text = pages[0][1]
        doc_type = detect_doc_type(pdf.name, first_text)
        asof = parse_asof_date("\n".join(t for _, t in pages), doc_type)
        meta_common = parse_meta("\n".join(t for _, t in pages))
        fund_name = re.sub(r"\.pdf$", "", pdf.name)

        for page, text in pages:
            for ch in sentence_chunks(text):
                ch = ch.strip()
                if not ch:
                    continue
                # 去重（同一份文件同一頁同一內容）
                digest = hashlib.sha1(f"{pdf.name}:{page}:{ch}".encode("utf-8")).hexdigest()
                points.append(PointStruct(
                    id=digest,
                    vector=embed_one(ch),  # 小量逐呼叫；大量可換成 batch 並行
                    payload={
                        "chunk": ch,
                        "source": pdf.name,
                        "page": page,
                        "doc_type": doc_type,
                        "asof_date": asof,           # YYYY-MM-DD or None
                        "fund_name": fund_name,
                        **meta_common                # 例如 risk_level / fee%
                    }
                ))

    # 若資料量很大，建議改為批次 embed_batch_parallel 再 upsert
    qc.upsert(collection_name=COLLECTION, points=points)
    print(f"[OK] Indexed {len(points)} chunks -> collection={COLLECTION}, dim={dim}")

if __name__ == "__main__":
    main()


