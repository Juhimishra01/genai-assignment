from typing import Optional
from fastapi import FastAPI
from pydantic import BaseModel
from rag.pipeline import answer
from rag.store import collection

app = FastAPI(title="Cost-efficient RAG")

class Q(BaseModel):
    question: str
    k: int = 5
    source: Optional[str] = None     # metadata filter
    doctype: Optional[str] = None    # metadata filter: pdf|html|md|txt

@app.post("/query")
def query(q: Q):
    f = {k: v for k, v in {"source": q.source, "doctype": q.doctype}.items() if v}
    where = None if not f else (f if len(f) == 1 else {"$and": [{k: v} for k, v in f.items()]})
    r = answer(q.question, q.k, where)
    r.pop("retrieved")
    return r

@app.get("/stats")
def stats(): return {"vectors": collection().count()}
