# rag/qa_server.py
from fastapi import FastAPI, Query
import json, os, torch
from haystack.dataclasses import Document
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.retrievers.in_memory import (
    InMemoryBM25Retriever,
    InMemoryEmbeddingRetriever,
)
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.components.rankers import SentenceTransformersSimilarityRanker
from haystack.utils import ComponentDevice

app = FastAPI(title="RAG QA")

# 向量相似度用 cosine（与 BGE 默认匹配）
store = InMemoryDocumentStore(embedding_similarity_function="cosine")

# 载入索引（JSONL -> Document）
store_path = "rag/index_bge/store.jsonl"
if os.path.exists(store_path):
    docs_to_load = []
    with open(store_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            docs_to_load.append(Document.from_dict(json.loads(line)))
    if docs_to_load:
        store.write_documents(docs_to_load)

# 组件初始化
bm25 = InMemoryBM25Retriever(document_store=store, top_k=8)

device = ComponentDevice.from_str("cuda:0" if torch.cuda.is_available() else "cpu")
embedder = SentenceTransformersTextEmbedder(model="BAAI/bge-m3", device=device)

# 注意：InMemoryEmbeddingRetriever 需要在 run() 时传 query_embedding
dense = InMemoryEmbeddingRetriever(document_store=store, top_k=20)

# 新版相似度重排器（替代旧的 TransformersSimilarityRanker）
reranker = SentenceTransformersSimilarityRanker(
    model="BAAI/bge-reranker-base",  # 如需更强可换 large；base 下载更快更省显存
    top_k=5,
    device=device,
)

# 统一预热，避免 “wasn't warmed up” 报错（尤其在 --reload 时）
@app.on_event("startup")
async def _warmup():
    embedder.warm_up()
    reranker.warm_up()

@app.get("/ask")
def ask(q: str = Query(..., description="question")):
    # 1) 关键词检索
    bm25_docs = bm25.run(query=q)["documents"]

    # 2) 语义检索：先把 query 向量化，再查向量库
    q_emb = embedder.run(text=q)["embedding"]                # 返回键是 "embedding"
    dense_docs = dense.run(query_embedding=q_emb)["documents"]

    # 3) 合并候选并用交叉编码器重排
    candidates = (bm25_docs + dense_docs)[:20]
    reranked = reranker.run(query=q, documents=candidates)["documents"]

    contexts = [d.content for d in reranked][:5]
    return {"query": q, "contexts": contexts}

@app.get("/health")
def health():
    return {"status": "ok"}
