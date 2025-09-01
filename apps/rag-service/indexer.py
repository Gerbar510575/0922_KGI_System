# indexer.py
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
from pdfminer.high_level import extract_text
from pathlib import Path
import numpy as np, uuid

COL = "kfh_docs" # Qdrant儲存PointStruct物件的collection
cli = QdrantClient(host="qdrant", port=6333)
emb = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

cli.recreate_collection(COL, vectors_config=VectorParams(size=384, distance=Distance.COSINE))

pts = [] #除了PDF文件檔案，其他種類的資料可以如何被INDEXING?
for pdf in Path("data/docs").glob("*.pdf"):
    text = extract_text(str(pdf))
    chunks = [text[i:i+800] for i in range(0, len(text), 600)]
    vecs = emb.encode(chunks)
    for ch, v in zip(chunks, vecs):
        pts.append(PointStruct(id=str(uuid.uuid4()), vector=np.asarray(v).tolist(),
                               payload={"chunk": ch, "source": pdf.name}))
cli.upsert(COL, points=pts)
print(f"Indexed {len(pts)} chunks")
