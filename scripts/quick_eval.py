"""
Quick retrieval sanity check against data/eval/questions.jsonl.
Prints top sources per question and shows domain metadata.
"""

import json
from pathlib import Path

from dotenv import load_dotenv, find_dotenv

from src.ingest.retriever import retrieve


def main():
    load_dotenv(find_dotenv(usecwd=True, raise_error_if_not_found=False))
    questions_path = Path(__file__).resolve().parents[1] / "data" / "eval" / "questions.jsonl"
    if not questions_path.exists():
        raise FileNotFoundError(f"Eval file missing: {questions_path}")

    with questions_path.open() as f:
        records = [json.loads(line) for line in f if line.strip()]

    for rec in records:
        qid = rec.get("id")
        q = rec.get("question", "")
        domain = rec.get("domain")  # optional
        print(f"\n[{qid}] {q} (domain={domain})")
        try:
            docs = retrieve(q, k=2, domain=domain)
            for i, d in enumerate(docs, 1):
                snippet = d.page_content.replace("\n", " ")
                src = d.metadata.get("source")
                dom = d.metadata.get("domain")
                print(f"  {i}. {snippet}\n     source: {src}\n     domain: {dom}")
        except Exception as exc:
            print("  error:", exc)


if __name__ == "__main__":
    main()
