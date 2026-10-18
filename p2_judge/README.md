# Problem 2 — LLM-as-Judge pipeline
Mode: pointwise reference-based (rubric in judge.py) + pairwise A/B (both orders) for comparisons.
## Run
    pip install -r requirements.txt && cp .env.example .env     # JUDGE_* and GEN_* from DIFFERENT families
    python gen.py cases.json prompts/v1.txt suites/v1.json
    python gen.py cases.json prompts/v2.txt suites/v2.json
    # hand-label human_label ("pass"/"fail") in suites/v1.json
    python run.py suite    suites/v1.json reports/v1.json
    python run.py compare  suites/v1.json suites/v2.json reports/ab.json
    python run.py bias     probes.json reports/bias.json
    JUDGE_TEMPERATURE=0.7 python run.py validate suites/v1.json reports/validate.json
Every judge prompt + raw reply is appended to logs/judge_calls.jsonl.
## Discussion (WRITE THIS YOURSELF)
- How biased before vs after mitigation? Would you let it gate a release?
