# Applied AI / ML Engineering — Take-Home Assignment

Submission by Juhi.

## Contents
- `p1_rag/`: Problem 1, a cost-efficient RAG application on ChromaDB.
- `p2_judge/`: Problem 2, an LLM-as-judge evaluation pipeline.

Each folder has its own README with setup and run commands. Secrets are read from environment variables (see `.env.example`); nothing is hardcoded.

## Problem 1: Cost-efficient RAG
- Vector store: ChromaDB (persistent, local); embeddings: all-MiniLM-L6-v2 (384-dim, CPU).
- Ingests PDF/HTML/MD/TXT, chunk size 800 and overlap 120 characters by default, idempotent re-ingest (hash IDs, delete and upsert).
- FastAPI `/query` endpoint with a metadata filter, cited answers, a no-context refusal, and per-query latency/chunk/token logging.
- Evaluation (`eval/results.json`): Hit Rate, Recall, MRR, nDCG, context precision, faithfulness, relevance, EM/F1, p50/p95 latency. Cost comparison at 100K / 1M / 10M vectors with stated assumptions (`eval/cost.py`).
- Limitations: tiny 4-document corpus (10 chunks), so retrieval scores are optimistic; one false refusal from the similarity threshold; prices are assumptions.

## Problem 2: LLM-as-judge
- Pointwise reference-based scoring with a 5-criterion rubric and calibration anchors; pairwise A/B judged in both orders to detect position bias.
- Structured JSON verdicts with robust parsing (fence stripping, repair retry, parse-error flag); every judge prompt and raw reply is logged to `logs/judge_calls.jsonl`.
- Judge and generator are configured independently (`JUDGE_*` and `GEN_*`); the judge is from a different model family than the generator.
- Implemented: suite report, A/B comparison with sign test, bias probes (position, verbosity, confidently-wrong), judge validation (agreement, Cohen's kappa, test-retest).
- Status: the suite report was run. The A/B comparison, bias probes and judge validation are implemented but were not run because the Gemini free-tier quota (20 requests/day) was exhausted.
