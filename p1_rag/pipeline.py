import json, os, time
from . import config as C
from .store import collection, embed
from .llm import chat

NO_CTX = "I don't know based on the provided documents."
SYSTEM = ("You answer ONLY from the numbered context chunks. Cite every claim like [1] or [2][3]. "
          f"If the context does not contain the answer, reply exactly: {NO_CTX}")

def retrieve(q, k=None, where=None):
    k = k or C.TOP_K
    r = collection().query(query_embeddings=embed([q]), n_results=k, where=where,
                           include=["documents", "metadatas", "distances"])
    return [{"id": i, "text": d, "meta": m, "sim": round(1 - dist, 4)}
            for i, d, m, dist in zip(r["ids"][0], r["documents"][0], r["metadatas"][0], r["distances"][0])]

def answer(q, k=None, where=None):
    t0 = time.perf_counter()
    hits = retrieve(q, k, where)
    t_ret = (time.perf_counter() - t0) * 1000
    usable = [h for h in hits if h["sim"] >= C.MIN_SIM]
    out = {"question": q, "retrieved": hits, "retrieval_ms": round(t_ret, 2)}
    if not usable:   # no-relevant-context branch: skip the LLM entirely
        out.update(answer=NO_CTX, citations=[], no_context=True, llm_ms=0, tokens={"prompt": 0, "completion": 0})
    else:
        ctx = "\n\n".join(f"[{i+1}] (source: {h['meta']['source']}, p{h['meta']['page']})\n{h['text']}" for i, h in enumerate(usable))
        t1 = time.perf_counter()
        text, tok = chat([{"role": "system", "content": SYSTEM},
                          {"role": "user", "content": f"Context:\n{ctx}\n\nQuestion: {q}"}],
                         C.LLM_BASE_URL, C.LLM_API_KEY, C.LLM_MODEL)
        out.update(answer=text, no_context=(NO_CTX in text), llm_ms=round((time.perf_counter() - t1) * 1000, 2), tokens=tok,
                   citations=[{"n": i + 1, "id": h["id"], "source": h["meta"]["source"], "page": h["meta"]["page"]} for i, h in enumerate(usable)])
    log = {"q": q, "retrieval_ms": out["retrieval_ms"], "llm_ms": out["llm_ms"], "chunks": len(hits),
           "chunks_used": len(usable), "tokens": out["tokens"], "no_context": out["no_context"]}
    os.makedirs(os.path.dirname(C.LOG_PATH) or ".", exist_ok=True)
    with open(C.LOG_PATH, "a") as f: f.write(json.dumps(log) + "\n")
    print("QUERY_LOG", json.dumps(log))
    return out
