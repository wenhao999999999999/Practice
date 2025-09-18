import argparse, os, glob, json
import torch
from haystack.components.converters import MarkdownToDocument, PyPDFToDocument, TextFileToDocument
from haystack.components.preprocessors import DocumentCleaner, DocumentSplitter
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack.utils import ComponentDevice

def load_docs(d):
    docs, md, txt, pdf = [], MarkdownToDocument(), TextFileToDocument(), PyPDFToDocument()
    for p in glob.glob(os.path.join(d, "**", "*"), recursive=True):
        pl = p.lower()
        if pl.endswith(".md"):
            docs.extend(md.run(sources=[p])["documents"])
        elif pl.endswith(".txt"):
            docs.extend(txt.run(sources=[p])["documents"])
        elif pl.endswith(".pdf"):
            docs.extend(pdf.run(sources=[p])["documents"])
    return docs

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs_dir", default="data/docs")
    ap.add_argument("--index_dir", default="rag/index_bge")
    args = ap.parse_args()

    os.makedirs(args.index_dir, exist_ok=True)

    # 1) 读入文档
    docs = load_docs(args.docs_dir)

    # 2) 清洗 + 切分
    cleaner = DocumentCleaner()
    splitter = DocumentSplitter(split_by="word", split_length=300, split_overlap=40)
    docs = cleaner.run(documents=docs)["documents"]
    docs = splitter.run(documents=docs)["documents"]

    # 3) 选择设备（优先用 GPU）
    device = ComponentDevice.from_str("cuda:0" if torch.cuda.is_available() else "cpu")

    # 4) 嵌入（初始化时传 device；可 warm_up 预下载/加载）
    embedder = SentenceTransformersDocumentEmbedder(
        model="BAAI/bge-m3",
        device=device,
        batch_size=32,            # 可按显存调整
        progress_bar=True
    )
    embedder.warm_up()
    docs = embedder.run(documents=docs)["documents"]

    # 5) 写入内存向量库并导出 JSONL
    store = InMemoryDocumentStore()
    store.write_documents(docs)

    out_path = os.path.join(args.index_dir, "store.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for d in store.filter_documents() or []:
            f.write(json.dumps(d.to_dict(), ensure_ascii=False) + "\n")

    print(f"Indexed {len(docs)} chunks -> {out_path}")
