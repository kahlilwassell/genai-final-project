"""
Build a FAISS index from PDFs/MD/TXT in data/raw/.

Usage:
  python -m src.ingest.build_index

Outputs:
  data/index/faiss_index/ (FAISS index + metadata)
"""

import os
from pathlib import Path

from dotenv import load_dotenv, find_dotenv
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
INDEX_DIR = ROOT / "data" / "index" / "faiss_index"


def load_env() -> None:
    # Prefer a local .env; if not present, try walking up.
    load_dotenv(find_dotenv(usecwd=True, raise_error_if_not_found=False))


def load_documents():
    if not RAW_DIR.exists():
        raise FileNotFoundError(f"Raw data folder not found: {RAW_DIR}")

    loaders = [
        DirectoryLoader(str(RAW_DIR), glob="**/*.pdf", loader_cls=PyPDFLoader),
        DirectoryLoader(str(RAW_DIR), glob="**/*.md", loader_cls=TextLoader),
        DirectoryLoader(str(RAW_DIR), glob="**/*.txt", loader_cls=TextLoader),
    ]

    docs = []
    for loader in loaders:
        try:
            docs.extend(loader.load())
        except Exception as exc:  # pragma: no cover - logging only
            print(f"[warn] Loader failed: {exc}")

    if not docs:
        raise RuntimeError(f"No documents found in {RAW_DIR}. Add PDFs/MD/TXT first.")

    return docs


def chunk_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len,
    )
    return splitter.split_documents(documents)


def build_and_save_index(chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(INDEX_DIR))
    return INDEX_DIR


def main():
    load_env()
    print(f"[info] Loading documents from {RAW_DIR}")
    documents = load_documents()
    print(f"[info] Loaded {len(documents)} documents")

    print("[info] Chunking documents")
    chunks = chunk_documents(documents)
    print(f"[info] Total chunks: {len(chunks)}")

    print(f"[info] Building FAISS index at {INDEX_DIR}")
    out_path = build_and_save_index(chunks)
    print(f"[done] Index written to {out_path}")


if __name__ == "__main__":
    main()
