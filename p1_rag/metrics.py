import math, re, string, collections
import numpy as np

def norm(s): return re.sub(r"\s+", " ", s.lower()).strip()

def is_relevant(chunk_text, chunk_source, gold_source, gold_evidence):
    """Relevant chunk = comes from gold_source AND contains the gold evidence phrase."""
    return chunk_source == gold_source and norm(gold_evidence) in norm(chunk_text)

def retrieval_metrics(rel_flags, n_rel_total, k):
    rel = rel_flags[:k]
    hit = 1.0 if any(rel) else 0.0
    recall = (sum(rel) / n_rel_total) if n_rel_total else 0.0
    rr = next((1.0 / (i + 1) for i, r in enumerate(rel) if r), 0.0)
    dcg = sum(r / math.log2(i + 2) for i, r in enumerate(rel))
    idcg = sum(1 / math.log2(i + 2) for i in range(min(n_rel_total, k)))
    ndcg = dcg / idcg if idcg else 0.0
    prec = sum(rel) / k
    # rank-weighted context precision (RAGAS-style): mean of precision@i at relevant ranks
    cp = [sum(rel[:i + 1]) / (i + 1) for i, r in enumerate(rel) if r]
    ctx_prec = sum(cp) / len(cp) if cp else 0.0
    return {"hit": hit, "recall": recall, "mrr": rr, "ndcg": ndcg, "precision_at_k": prec, "context_precision": ctx_prec}

def _toks(s):
    s = s.lower().translate(str.maketrans("", "", string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return s.split()

def exact_match(pred, gold): return float(_toks(pred) == _toks(gold))

def f1(pred, gold):
    p, g = _toks(pred), _toks(gold)
    common = collections.Counter(p) & collections.Counter(g)
    ns = sum(common.values())
    if not p or not g or ns == 0: return 0.0
    pr, rc = ns / len(p), ns / len(g)
    return 2 * pr * rc / (pr + rc)

def pct(xs, q): return float(np.percentile(xs, q)) if xs else float("nan")
