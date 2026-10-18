# Problem 1 — Cost-efficient RAG (ChromaDB)
## Run
    pip install -r requirements.txt && cp .env.example .env   # fill LLM_API_KEY
    python -m rag.ingest data/                  # defaults: chunk=800 chars, overlap=120
    uvicorn app:app --port 8000
    curl -s localhost:8000/query -H 'content-type: application/json' -d '{"question":"...","k":5,"doctype":"pdf"}'
    python -m rag.ingest data/                  # re-run: vector count must be unchanged
    python -m eval.run_eval --k 5               # -> eval/results.json
    python -m eval.cost                         # cost table
## Discussion  (WRITE THIS YOURSELF)
- When would I switch back to managed?
- Was retrieval or generation the weak link? (cite eval/results.json)
