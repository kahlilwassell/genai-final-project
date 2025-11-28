"""
Quick retrieval sanity check against data/eval/questions.jsonl.
Prints top sources per question.
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
        print(f"\n[{qid}] {q}")
        try:
            docs = retrieve(q, k=2)
            for i, d in enumerate(docs, 1):
                snippet = d.page_content[:200].replace("\n", " ")
                print(f"  {i}. {snippet}\n     source: {d.metadata.get('source')}")
        except Exception as exc:
            print("  error:", exc)


if __name__ == "__main__":
    main()
