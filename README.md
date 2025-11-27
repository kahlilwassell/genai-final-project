# Agentic Run Coach (Final Project)

This repository will host the final project for the generative AI class: an agentic running coach that combines RAG, LangGraph, and safety guardrails to build personalized training plans.

## Getting started
1. Create and activate a virtual environment (already present at `.venv/` if you want to reuse).
2. Install dependencies: `pip install -r requirements.txt`.
3. Add any required API keys to a local `.env` file (e.g., `OPENAI_API_KEY`).

## Project status
Currently just initialized. Code, data ingestion scripts, and Streamlit UI will be added next.

## Two-week plan (day by day)
- **Day 1** — Lock scope: single-runner MVP, goal race/date, target weekly mileage. Write safety rules (max % mileage increase, long-run caps, illness/injury disclaimer). Sketch agent graph (planner/adjuster/safety).
- **Day 2** — Collect corpus (Daniels/Pfitzinger PDFs, safety docs); place in `data/raw/`. Export a sample run-log CSV (date, distance, time, RPE) to use in dev. Define weather input format (API or stub CSV).
- **Day 3** — Write ingestion script to load PDFs/MD/TXT, chunk, embed, store FAISS index to `data/index/`. Start with simple chunking (1k/100 overlap). Add `.env` with API key.
- **Day 4** — Build retrieval helper and base prompt focused on training guidance. Create a 10–15 question eval set with expected source snippets; run a quick relevance check and tweak chunking if needed.
- **Day 5** — Implement LangGraph skeleton: nodes for Planner, Adjuster, SafetyGuard, Router; wire retriever tool. Planner outputs weekly plan JSON; Adjuster takes today’s context and weather/fatigue inputs.
- **Day 6** — Implement SafetyGuard logic (caps, progressive overload checks) and guardrail messaging. Add Nutrition node placeholder if time permits. Test the graph with a few manual calls.
- **Day 7** — Build Streamlit UI skeleton: inputs (race/date, current mileage, recent long run, injury flags, fatigue slider, upload run-log CSV), buttons for “Generate plan” and “Adjust today.” Display plan table + “Why this?” citations.
- **Day 8** — Add retrieval trace and source display in UI. Ensure safety warnings surface in the UI. Improve prompt few-shots for concise prescriptions (distance + pace guidance).
- **Day 9** — Evaluation pass: scripted checks for unsafe asks, out-of-domain questions, hot-weather adjustments; add hallucination guard (require ≥2 retrieved chunks or fallback). Log runs (inputs/retrieval/outputs/safety flags) to JSON/CSV.
- **Day 10** — Tune retrieval (chunk size/overlap, synonym glossary for tempo/threshold). Refine Adjuster behavior for weather/heat/humidity and fatigue scaling.
- **Day 11** — Polish UX: better table formatting, consistent tone, small defaults, helpful error messages (missing index, missing API key). Add minimal tests/helpers to smoke-test graph functions.
- **Day 12** — Documentation pass: expand README with architecture diagram/flow description, setup steps, env vars, eval results, safety rules, limitations, and future work. Add short demo script/gif if possible.
- **Day 13** — Full end-to-end dry run on sample runner; fix any logic/UI bugs; verify safety checks trigger correctly.
- **Day 14** — Final QA and submission packaging: clean logs, ensure instructions are clear, prepare submission summary (concepts covered: RAG, LangGraph agents/tools, guardrails, eval).

## Current repo scaffold
- `data/raw/` — place PDFs/MD/TXT training corpus and run-log CSVs.
- `data/index/` — FAISS index output.
- `data/eval/questions.jsonl` — seed eval questions for retrieval sanity checks.
- `src/ingest/build_index.py` — script to build FAISS from `data/raw/`.
- `src/ingest/retriever.py` — helper to load the index and run similarity search.
