"""
Helpers to load the FAISS index and run similarity search.
"""

from pathlib import Path
from typing import List

from dotenv import load_dotenv, find_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

ROOT = Path(__file__).resolve().parents[2]
INDEX_DIR = ROOT / "data" / "index" / "faiss_index"


def load_env() -> None:
    load_dotenv(find_dotenv(usecwd=True, raise_error_if_not_found=False))


def load_vectorstore() -> FAISS:
    load_env()
    if not INDEX_DIR.exists():
        raise FileNotFoundError(f"FAISS index not found at {INDEX_DIR}. Run build_index.py first.")
    embeddings = OpenAIEmbeddings()
    return FAISS.load_local(str(INDEX_DIR), embeddings, allow_dangerous_deserialization=True)


def retrieve(query: str, k: int = 4) -> List[Document]:
    vs = load_vectorstore()
    return vs.similarity_search(query, k=k)
