# apps/rag-service/indexer_gemini.py
# -*- coding: utf-8 -*-
import os, uuid, numpy as np
from pathlib import Path
from dotenv import load_dotenv
from pdfminer.high_level import extract_text

import google.genai as genai
from google.genai import types as gtypes

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct

load_dotenv()
api_key = os.getenv("GENAI_API_KEY")
if not api_key:
    raise ValueError("請在 .env 設定 GENAI_API_KEY")
client_g = genai.Client(api_key=api_key)

QDRANT_HOST = os.getenv("QDRANT_HOST", "qdrant")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
COLLECTION  = os.getenv("QDRANT_COLLECTION", "kfh_docs_gemini")

def embed_batch(texts):
    # 逐段呼叫（保守作法，便於錯誤處理；你也可自行做 batch）
    vecs = []
    for t in texts:
        r = client_g.models.embed_content(
            model="models/text-embedding-004",
            contents=t,
            config=gtypes.EmbedContentConfig(task_type="retrieval_document")
        )
        vecs.append(r.embeddings[0].values)
    return vecs

def main():
    # 1) 掃描 PDF → 簡單分段
    docs_dir = Path("data/docs")
    pdfs = sorted(docs_dir.glob("*.pdf"))
    if not pdfs:
        raise RuntimeError("找不到 data/docs/*.pdf")

    chunks, sources = [], []
    for pdf in pdfs:
        text = extract_text(str(pdf))
        if not text: 
            continue
        # 簡單 slicing（可改為更精細的段落/表格切分）
        parts = [text[i:i+800] for i in range(0, len(text), 600)]
        chunks.extend(parts)
        sources.extend([pdf.name]*len(parts))

    if not chunks:
        raise RuntimeError("沒有可用文字片段，請檢查 PDF 內容")

    # 2) 取得向量維度（以第一段推得）
    probe = client_g.models.embed_content(
        model="models/text-embedding-004",
        contents=chunks[0][:2000],
        config=gtypes.EmbedContentConfig(task_type="retrieval_document")
    ).embeddings[0].values
    dim = len(probe)

    # 3) 重建 Qdrant collection
    qc = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    qc.recreate_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )

    # 4) 向量化並 upsert
    vecs = embed_batch(chunks)
    points = []
    for ch, src, v in zip(chunks, sources, vecs):
        points.append(PointStruct(
            id=str(uuid.uuid4()),
            vector=list(v),
            payload={"chunk": ch, "source": src}
        ))
    qc.upsert(collection_name=COLLECTION, points=points)
    print(f"[OK] Indexed {len(points)} chunks ??collection={COLLECTION}, dim={dim}")

if __name__ == "__main__":
    main()

