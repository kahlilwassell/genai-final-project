# Agentic Run Coach (Final Project)

Agentic running coach that combines RAG, LangGraph, safety guardrails, and goal-pace logic to build personalized training plans.

## Getting started
1. Create and activate a virtual environment (already present at `.venv/` if you want to reuse).
2. Install dependencies: `pip install -r requirements.txt`.
3. Add any required API keys to a local `.env` file (e.g., `OPENAI_API_KEY`).

## Current status
- Data ingestion => FAISS index with domain tags (plans/safety/fueling/biomech).
- LangGraph agent with domain-specific retrieval tools + safety/heat tools; plan/adjust/QA entrypoints.
- Streamlit UI with three tabs: (1) Setup profile & generate full plan (daily through race day, long run/tempo/interval enforced), (2) Ask the coach (general corpus QA), (3) Adjust a session (weather/fatigue/injury aware).
- Goal time parsing + pace overview; blocks unrealistic goals (e.g., sub-2:00 marathon).
- Safety: heuristic caps (weekly/long-run), heat slowdowns, LLM safety review.

## Key features
- RAG over running corpus (plans, safety, fueling, biomechanics) with domain-filtered retrieval tools.
- LangGraph agent + tools: plan generation, session adjustment, general QA; safety/heat utilities.
- Full-plan generation: daily schedule to race day with enforced structure (1 long run on chosen day, 1 tempo, 1 interval, easy days between).
- Goal time => pace overview; validates unrealistic goals and surfaces target paces.
- Safety guardrails: weekly/long-run caps, heat slowdowns, injury flag, LLM safety review.
- Streamlit UI: profile setup, coach Q&A (profile optional), session adjustor.

## Responsible AI & requirements checklist
- **RAG grounding**: all answers cite retrieved corpus chunks; domain-tagged retrieval to reduce drift.
- **Agents/Tools**: LangGraph with multiple tools (plans/safety/fueling/biomech retrieval, safety limits, heat adjust).
- **Safety**: heuristic caps, heat adjustments, injury flag, LLM safety review; goal-time validation to block unrealistic inputs.
- **Responsible use**: app declines or flags incomplete plans; warns when corpus is weak; avoids hallucinated physics/out-of-domain answers in prompts.
- **Customization**: uses the corpus + profile context (goal race/date/time, mileage, preferred days) to tailor plans and paces.

## Current repo scaffold
- `data/raw/` — place PDFs/MD/TXT training corpus and run-log CSVs.
- `data/index/` — FAISS index output.
- `data/eval/questions.jsonl` — seed eval questions for retrieval sanity checks.
- `src/ingest/build_index.py` — script to build FAISS from `data/raw/`.
- `src/ingest/retriever.py` — helper to load the index and run similarity search.
- `src/graph/coach_graph.py` — LangGraph scaffold (agent + retrieval tool + safety review helper + domain tagging).
- `src/ui/app.py` — Streamlit UI shell for the coach.
- `scripts/quick_eval.py` — CLI to sanity-check retrieval over eval questions.

## Running things
- Build index (from repo root): `source .venv/bin/activate && python -m src.ingest.build_index`
- Quick retrieval sanity check:
  ```
  source .venv/bin/activate
  python - <<'PY'
  from src.ingest.retriever import retrieve
  docs = retrieve("Define a tempo run", k=2)
  for d in docs:
      print(d.metadata.get("source"), d.page_content[:200])
  PY
  ```
- Eval retrieval set: `source .venv/bin/activate && python scripts/quick_eval.py`
- Streamlit UI: `source .venv/bin/activate && PYTHONPATH=. streamlit run src/ui/app.py`
