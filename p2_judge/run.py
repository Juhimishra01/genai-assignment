"""Commands:
  python run.py suite    suites/v1.json reports/v1.json
  python run.py compare  suites/v1.json suites/v2.json reports/ab.json
  python run.py bias     probes.json reports/bias.json
  python run.py validate suites/v1.json reports/validate.json   (needs human_label in suite)
"""
import json, math, os, statistics, sys
from collections import Counter
from judge import pointwise, pairwise_both_orders
from llm import STATS

PASS_T = float(os.getenv("PASS_THRESHOLD", "3.5"))
def passed(v): return (not v.get("parse_error")) and v["overall"] >= PASS_T and v["criteria"].get("tone_safety", {"score": 5})["score"] >= 3

def run_pointwise(cases):
    verdicts = {c["id"]: pointwise(c) for c in cases}
    ok = {k: v for k, v in verdicts.items() if not v.get("parse_error")}
    ov = [v["overall"] for v in ok.values()]
    crit = {}
    for v in ok.values():
        for k, d in v["criteria"].items(): crit.setdefault(k, []).append(d["score"])
    return {"n": len(cases), "parse_failures": len(cases) - len(ok),
            "pass_rate": round(sum(passed(v) for v in verdicts.values()) / len(cases), 3),
            "mean_overall": round(statistics.mean(ov), 3) if ov else None,
            "score_std": round(statistics.pstdev(ov), 3) if len(ov) > 1 else None,   # clustering indicator
            "per_criterion_mean": {k: round(statistics.mean(s), 2) for k, s in crit.items()},
            "verdicts": verdicts}

def sign_test_p(w, l):
    n = w + l
    if n == 0: return 1.0
    k = max(w, l)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k, n + 1)) / 2 ** n)

def cmd_compare(a_path, b_path, out):
    A = {c["id"]: c for c in json.load(open(a_path))}; B = {c["id"]: c for c in json.load(open(b_path))}
    ids = [i for i in A if i in B]
    ra, rb = run_pointwise([A[i] for i in ids]), run_pointwise([B[i] for i in ids])
    res = Counter(); rows = {}
    for i in ids:
        r = pairwise_both_orders(A[i], A[i]["model_output"], B[i]["model_output"])   # X=A, Y=B
        rows[i] = r; res[r["final"]] += 1
    n = len(ids); w, l = res["X"], res["Y"]
    p = sign_test_p(w, l)
    winner = "A" if w > l else "B" if l > w else "tie"
    rep = {"n": n, "A": {k: ra[k] for k in ("pass_rate", "mean_overall", "parse_failures")},
           "B": {k: rb[k] for k in ("pass_rate", "mean_overall", "parse_failures")},
           "pairwise": {"A_win_rate": round(w / n, 3), "B_win_rate": round(l / n, 3), "tie_rate": round(res["tie"] / n, 3),
                        "inconsistent_rate": round(sum(r["inconsistent"] for r in rows.values()) / n, 3),
                        "hard_flip_rate": round(sum(r["hard_flip"] for r in rows.values()) / n, 3)},
           "sign_test_p": round(p, 4), "winner": winner,
           "caveat": "significant" if p < 0.1 else "NOT statistically significant at this n; treat as directional",
           "judge_usage": dict(STATS), "detail": rows}
    json.dump(rep, open(out, "w"), indent=2); print(json.dumps({k: v for k, v in rep.items() if k != "detail"}, indent=2))

def cmd_bias(path, out):
    probes = json.load(open(path)); R = {}
    # Position bias: good-vs-wrong pairs, ground truth known (good must win). Naive = single order; mitigated = both orders.
    n1 = n2 = comb = flips = hard = 0; N = len(probes)
    vb = []; wrong_fooled = 0; terse_pen = []
    for p in probes:
        c = {"id": p["id"], "input": p["input"], "system_prompt": "", "expected_output": p["expected_output"]}
        r = pairwise_both_orders(c, p["good"], p["wrong_confident"])
        n1 += r["order1"] == "X"; n2 += r["order2"] == "X"; comb += r["final"] == "X"
        flips += r["inconsistent"]; hard += r["hard_flip"]; wrong_fooled += r["final"] == "Y"
        # Verbosity: good vs padded (same content) -> padded should NOT win
        pv = pairwise_both_orders(c, p["good"], p["padded"]); vb.append(pv["final"])
        # Pointwise score deltas
        g = pointwise({**c, "model_output": p["good"]}); pd = pointwise({**c, "model_output": p["padded"]})
        w = pointwise({**c, "model_output": p["wrong_confident"]}); t = pointwise({**c, "model_output": p["terse_correct"]})
        R[p["id"]] = {"pair_good_vs_wrong": r, "pair_good_vs_padded": pv,
                      "pointwise": {k: (v.get("overall") if not v.get("parse_error") else None) for k, v in
                                    {"good": g, "padded": pd, "wrong_confident": w, "terse_correct": t}.items()}}
        if None not in (g.get("overall"), t.get("overall")): terse_pen.append(g["overall"] - t["overall"])
    summ = {"n_probes": N,
            "position": {"naive_acc_order1": n1 / N, "naive_acc_order2": n2 / N, "both_orders_acc": comb / N,
                         "inconsistent_rate": flips / N, "hard_flip_rate": hard / N},
            "verbosity": {"padded_wins": vb.count("Y") / N, "good_wins": vb.count("X") / N, "tie": vb.count("tie") / N},
            "sycophancy_confidently_wrong": {"fooled_rate": wrong_fooled / N},
            "terse_correct_penalty_mean": round(statistics.mean(terse_pen), 3) if terse_pen else None,
            "judge_usage": dict(STATS)}
    json.dump({"summary": summ, "detail": R}, open(out, "w"), indent=2); print(json.dumps(summ, indent=2))

def kappa(a, b):
    n = len(a); po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((a.count(l) / n) * (b.count(l) / n) for l in set(a) | set(b))
    return (po - pe) / (1 - pe) if pe < 1 else 1.0

def cmd_validate(path, out):
    cases = [c for c in json.load(open(path)) if c.get("human_label") in ("pass", "fail")]
    r1, r2 = run_pointwise(cases), run_pointwise(cases)       # two runs -> test-retest (set JUDGE_TEMPERATURE>0 to be meaningful)
    ids = [c["id"] for c in cases]
    human = [c["human_label"] for c in cases]
    j1 = ["pass" if passed(r1["verdicts"][i]) else "fail" for i in ids]
    j2 = ["pass" if passed(r2["verdicts"][i]) else "fail" for i in ids]
    rep = {"n_labeled": len(ids), "agreement": round(sum(a == b for a, b in zip(human, j1)) / len(ids), 3),
           "cohen_kappa": round(kappa(human, j1), 3),
           "test_retest_flip_rate": round(sum(a != b for a, b in zip(j1, j2)) / len(ids), 3),
           "judge_usage": dict(STATS)}
    json.dump(rep, open(out, "w"), indent=2); print(json.dumps(rep, indent=2))

if __name__ == "__main__":
    cmd, *a = sys.argv[1:]
    if cmd == "suite":
        rep = run_pointwise(json.load(open(a[0]))); rep["judge_usage"] = dict(STATS)
        json.dump(rep, open(a[1], "w"), indent=2); print(json.dumps({k: v for k, v in rep.items() if k != "verdicts"}, indent=2))
    elif cmd == "compare": cmd_compare(*a)
    elif cmd == "bias": cmd_bias(*a)
    elif cmd == "validate": cmd_validate(*a)
