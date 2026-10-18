from openai import OpenAI
def chat(messages, base_url, api_key, model, temperature=0.0):
    r = OpenAI(base_url=base_url, api_key=api_key).chat.completions.create(
        model=model, messages=messages, temperature=temperature)
    u = r.usage
    return r.choices[0].message.content, {"prompt": u.prompt_tokens, "completion": u.completion_tokens}
