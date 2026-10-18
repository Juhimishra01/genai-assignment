import os, re, time
from dotenv import load_dotenv
from openai import OpenAI
load_dotenv()
STATS = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0}
_last = {}

def call(prefix, messages, temperature=None):
    t = float(os.getenv(f"{prefix}_TEMPERATURE", "0")) if temperature is None else temperature
    gap = float(os.getenv(f"{prefix}_MIN_INTERVAL", "0"))
    wait = gap - (time.time() - _last.get(prefix, 0))
    if wait > 0: time.sleep(wait)
    cl = OpenAI(base_url=os.getenv(f"{prefix}_BASE_URL") or None, api_key=os.getenv(f"{prefix}_API_KEY"), max_retries=0)
    model = os.getenv(f"{prefix}_MODEL")
    for attempt in range(6):
        try:
            _last[prefix] = time.time()
            r = cl.chat.completions.create(model=model, messages=messages, temperature=t)
            break
        except Exception as e:
            if attempt == 5: raise
            print(f"[retry {attempt+1}] {type(e).__name__}: waiting", flush=True); time.sleep(20 * (attempt + 1))
    _last[prefix] = time.time()
    STATS["calls"] += 1
    if r.usage:
        STATS["prompt_tokens"] += r.usage.prompt_tokens
        STATS["completion_tokens"] += r.usage.completion_tokens
    txt = r.choices[0].message.content or ""
    return re.sub(r"<think>.*?</think>", "", txt, flags=re.S).strip(), model
