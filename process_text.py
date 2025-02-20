import re

# requires the score to be within a box
def get_score(text):
    match = re.search(r'\\boxed\{(.+?)\}', text)
    score_str = match.group(1).strip()
    try:
        score = float(score_str)
    except ValueError:
        raise ValueError(f"Extracted content '{score_str}' is not a valid float.")
    
    return score

# extracts the content after prefix
def extract_from_text(text, prefix):
    pattern = re.escape(prefix) + r'\s*(.*)'
    match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
    return match.group(1).strip()