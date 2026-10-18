"""python -m eval.run_eval [--k 5] -> eval/results.json (retrieval + answer + latency layers)"""
import argparse, json, re, statistics
from rag import config as C
from rag.store import collection
from rag.pipeline import answer, NO_CTX
from rag.llm import chat
from eval.metrics import *

def judge_json(prompt):
    txt, _ = chat([{"role": "user", "content": prompt}], C.EVAL_LLM_BASE_URL, C.EVAL_LLM_API_KEY, C.EVAL_LLM_MODEL)
    m = re.search(r"\{.*\}", txt, re.S)
    try: return json.loads(m.group(0))
    except Exception: return None

def faithfulness(ans, ctx):
    p = ("Split the ANSWER into atomic factual claims. For each, say whether the CONTEXT supports it.\n"
         f"CONTEXT:\n{ctx}\n\nANSWER:\n{ans}\n\n"
         'Return ONLY JSON: {"claims":[{"claim":"...","supported":true}]}')
    j = judge_json(p)
    if not j or not j.get("claims"): return None
    return sum(bool(c.get("supported")) for c in j["claims"]) / len(j["claims"])

def relevance(q, ans):
    p = (f"QUESTION: {q}\nANSWER: {ans}\n\nRate 1-5 how directly the answer addresses the question "
         '(1=off-topic, 5=fully and directly). Return ONLY JSON: {"score":int}')
    j = judge_json(p)
    return (j["score"] - 1) / 4 if j and "score" in j else None

def mean(xs):
    xs = [x for x in xs if x is not None]
    return round(statistics.mean(xs), 4) if xs else None

def main(k):
    qs = json.load(open("eval/questions.json"))
    all_chunks = collection().get(include=["documents", "metadatas"])
    rows, abst_ok, abst_n = [], 0, 0
    for q in qs:
        r = answer(q["question"], k)
        if not q["answerable"]:
            abst_n += 1; abst_ok += int(r["no_context"]); 
            rows.append({"id": q["id"], "answerable": False, "abstained": r["no_context"], "answer": r["answer"]})
            continue
        n_rel_total = sum(is_relevant(d, m["source"], q["gold_source"], q["gold_evidence"])
                          for d, m in zip(all_chunks["documents"], all_chunks["metadatas"]))
        flags = [is_relevant(h["text"], h["meta"]["source"], q["gold_source"], q["gold_evidence"]) for h in r["retrieved"]]
        used_ctx = "\n\n".join(h["text"] for h in r["retrieved"] if h["sim"] >= C.MIN_SIM)
        rows.append({"id": q["id"], "answerable": True, "n_rel_total": n_rel_total, **retrieval_metrics(flags, n_rel_total, k),
                     "retrieval_ms": r["retrieval_ms"], "llm_ms": r["llm_ms"], "answer": r["answer"],
                     "em": exact_match(r["answer"], q["gold_answer"]) if q["gold_answer"] else None,
                     "f1": f1(r["answer"], q["gold_answer"]) if q["gold_answer"] else None,
                     "faithfulness": faithfulness(r["answer"], used_ctx) if not r["no_context"] else None,
                     "relevance": relevance(q["question"], r["answer"]) if not r["no_context"] else None})
    A = [x for x in rows if x["answerable"]]
    ret = [x["retrieval_ms"] for x in A]
    res = {"k": k, "n_answerable": len(A), "n_unanswerable": abst_n, "embed_model": C.EMBED_MODEL,
           "retrieval": {m: mean([x[m] for x in A]) for m in ["hit", "recall", "mrr", "ndcg", "context_precision", "precision_at_k"]},
           "answer": {m: mean([x[m] for x in A]) for m in ["faithfulness", "relevance", "em", "f1"]},
           "no_context_accuracy": (abst_ok / abst_n) if abst_n else None,
           "latency_ms": {"retrieval_p50": pct(ret, 50), "retrieval_p95": pct(ret, 95),
                          "e2e_p95": pct([x["retrieval_ms"] + x["llm_ms"] for x in A], 95)},
           "per_question": rows}
    json.dump(res, open("eval/results.json", "w"), indent=2)
    print(json.dumps({k_: v for k_, v in res.items() if k_ != "per_question"}, indent=2))

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--k", type=int, default=C.TOP_K)
    main(ap.parse_args().k)
