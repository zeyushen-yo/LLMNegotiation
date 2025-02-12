import os
import openai

api_key = os.getenv("OPENAI_API_KEY", "")

if api_key != "":
    openai.api_key = api_key
else:
    print("Warning: OPENAI_API_KEY is not set")

def completions_gpt(**kwargs):
    return openai.chat.completions.create(**kwargs)

def get_output(prompt, model, temperature, max_tokens):
    messages = [{"role": "user", "content": prompt}]
    res = completions_gpt(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens, n=1)
    return res.choices[0].message.content

# requires the score to be within a box
def get_score(text):
    match = re.search(r'\\boxed\{(.+?)\}', text)
    return float(match.group(1).strip())