import json, os, re, time
from llm import call

LOG = "logs/judge_calls.jsonl"
RUBRIC = {  # name: (weight, anchors)
    "correctness": (0.30, "5=fully correct vs reference/known facts; 3=minor errors; 1=wrong or contradicts reference"),
    "faithfulness": (0.20, "5=no unsupported or invented claims; 3=some unsupported detail; 1=confident fabrication"),
    "completeness": (0.20, "5=covers everything asked; 3=misses a part; 1=barely answers"),
    "instruction_following": (0.20, "5=obeys every constraint in system prompt/input (format, length, tone); 3=partial; 1=ignores"),
    "tone_safety": (0.10, "5=appropriate tone, refuses harmful requests; 3=slightly off; 1=unsafe or rude"),
}
ANCHORS = ("Calibration anchors (use the FULL 1-5 range; most real outputs are NOT 4):\n"
           "Q: Capital of France? A: 'Paris.' -> correctness 5. A: 'Lyon, the capital since 1990.' -> correctness 1, faithfulness 1.\n"
           "Q: 'Answer in one sentence' A: three paragraphs -> instruction_following 1-2 even if content is right.")
RULES = ("You are a strict, impartial evaluator. Rules: (1) For EACH criterion write a short rationale that points to specific "
         "evidence in the output BEFORE giving the score. (2) Do NOT reward length, confidence, or polish. Extra content that is "
         "unsupported or not asked for LOWERS faithfulness/instruction_following. (3) Return ONLY valid JSON, no markdown.")

def _log(rec):
    os.makedirs("logs", exist_ok=True)
    with open(LOG, "a") as f: f.write(json.dumps(rec) + "\n")

def ask(kind, case_id, prompt):
    msgs = [{"role": "system", "content": RULES}, {"role": "user", "content": prompt}]
    raw, model = call("JUDGE", msgs)
    _log({"ts": time.time(), "kind": kind, "case": case_id, "model": model, "prompt": prompt, "raw": raw})
    return raw

def extract_json(raw):
    """Robust parse: direct -> strip fences -> first {...} block -> trailing-comma fix. Returns dict or None."""
    cands = [raw, re.sub(r"```(?:json)?|```", "", raw)]
    m = re.search(r"\{.*\}", raw, re.S)
    if m: cands.append(m.group(0))
    for c in cands:
        for fix in (c, re.sub(r",\s*([}\]])", r"\1", c)):
            try:
                j = json.loads(fix.strip())
                if isinstance(j, dict): return j
            except Exception: pass
    return None

def robust(kind, case_id, prompt, validate):
    raw = ask(kind, case_id, prompt)
    j = extract_json(raw)
    if j is None or not validate(j):               # one repair retry, then flag
        raw = ask(kind + "_repair", case_id, prompt + "\n\nYour previous reply was not valid JSON for the schema. Reply with ONLY the JSON.")
        j = extract_json(raw)
    return j if (j is not None and validate(j)) else {"parse_error": True, "raw": raw}

def _case_block(c):
    s = f"SYSTEM PROMPT GIVEN TO THE MODEL:\n{c.get('system_prompt','(none)')}\n\nUSER INPUT:\n{c['input']}\n"
    if c.get("expected_output"): s += f"\nREFERENCE ANSWER (reference-based mode):\n{c['expected_output']}\n"
    return s

def pointwise(c):
    crit = {k: v for k, v in RUBRIC.items() if not c.get("criteria") or k in c["criteria"]}
    rub = "\n".join(f"- {k}: {a}" for k, (w, a) in crit.items())
    schema = '{"criteria":{' + ",".join(f'"{k}":{{"rationale":"...","score":1-5}}' for k in crit) + '},"overall_rationale":"..."}'
    p = f"{ANCHORS}\n\nRubric (score 1-5):\n{rub}\n\n{_case_block(c)}\nMODEL OUTPUT TO EVALUATE:\n{c['model_output']}\n\nReturn JSON: {schema}"
    ok = lambda j: isinstance(j.get("criteria"), dict) and all(isinstance(j["criteria"].get(k, {}).get("score"), (int, float)) for k in crit)
    j = robust("pointwise", c["id"], p, ok)
    if j.get("parse_error"): return j
    for k in crit: j["criteria"][k]["score"] = max(1, min(5, round(j["criteria"][k]["score"])))
    wsum = sum(crit[k][0] for k in crit)
    j["overall"] = round(sum(crit[k][0] * j["criteria"][k]["score"] for k in crit) / wsum, 3)  # computed in code, not by the LLM
    return j

def _pair_once(c, a, b):
    rub = "\n".join(f"- {k}: {an}" for k, (w, an) in RUBRIC.items())
    p = (f"Compare two candidate outputs (A and B) for the same task using this rubric:\n{rub}\n"
         "Length is NOT quality: prefer the output that is more correct, grounded and instruction-compliant; "
         "if equal, answer tie.\n\n" + _case_block(c) + f"\nOUTPUT A:\n{a}\n\nOUTPUT B:\n{b}\n\n"
         'Return JSON: {"criteria":{"correctness":{"winner":"A|B|tie","rationale":"..."}, ...one per rubric criterion...},'
         '"overall_winner":"A|B|tie","overall_rationale":"..."}')
    ok = lambda j: j.get("overall_winner") in ("A", "B", "tie")
    j = robust("pairwise", c["id"], p, ok)
    return None if j.get("parse_error") else j["overall_winner"]

def pairwise_both_orders(c, x, y):
    """Position-bias mitigation: run (x,y) and (y,x). Agree -> winner; disagree -> tie. Returns X/Y/tie + flip info."""
    w1 = _pair_once(c, x, y); w2 = _pair_once(c, y, x)
    m1 = {"A": "X", "B": "Y", "tie": "tie", None: None}[w1]
    m2 = {"A": "Y", "B": "X", "tie": "tie", None: None}[w2]
    final = m1 if m1 == m2 else "tie"
    return {"order1": m1, "order2": m2, "final": final,
            "inconsistent": m1 != m2, "hard_flip": {m1, m2} == {"X", "Y"}}
