"""python gen.py cases.json prompts/v1.txt suites/v1.json  -> fills model_output with the GENERATOR model."""
import json, sys
from llm import call
cases, prompt_file, out = sys.argv[1:4]
sp = open(prompt_file).read().strip()
res = []
for c in json.load(open(cases)):
    txt, _ = call("GEN", [{"role": "system", "content": sp}, {"role": "user", "content": c["input"]}], temperature=0.7)
    res.append({**c, "system_prompt": sp, "model_output": txt, "human_label": None})  # fill human_label by hand: "pass"/"fail"
json.dump(res, open(out, "w"), indent=2); print("wrote", out)
