import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from typing import List, Dict

def load_llm(llm_path: str):
    tokenizer = AutoTokenizer.from_pretrained(llm_path)
    model = AutoModelForCausalLM.from_pretrained(llm_path)
    model.eval()
    if torch.cuda.is_available():
        model.cuda()
    return tokenizer, model

def simulate_negotiation(
    setting: str,
    tokenizer,
    model,
    num_turns: int = 6,
    max_new_tokens: int = 1024,
    do_sample: bool = True,
    top_k: int = 50,
    top_p: float = 0.95,
    temperature: float = 1
):
    # Dummy for now, to add mcts
    conversation = []
    scenario_line = f"Scenario: {setting}\n"
    conversation_history = scenario_line
    speakers = ["Agent A", "Agent B"]

    for turn in range(num_turns):
        speaker = speakers[turn % 2]
        model_input_text = conversation_history + f"{speaker}:"

        input_ids = tokenizer.encode(model_input_text, return_tensors="pt")
        if torch.cuda.is_available():
            input_ids = input_ids.cuda()

        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature
            )

        new_tokens = output_ids[0][input_ids.shape[1]:]
        generated_text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        utterance = f"{speaker}: {generated_text}"
        conversation.append(utterance)
        conversation_history += utterance + "\n"

    return conversation


def extract_prompt_completion_pairs(conversation):
    pairs = []
    for i in range(1, len(conversation)):
        prompt = "\n".join(conversation[:i])
        completion = conversation[i]            
        pairs.append({"prompt": prompt, "completion": completion})
    return pairs


def generate_data_mcts(
    llm_path: str,
    output_file: str,
    max_new_tokens: int = 1024,
    do_sample: bool = True,
    top_k: int = 50,
    top_p: float = 0.95,
    temperature: float = 1
):
    tokenizer, model = load_llm(llm_path)
    with open(path, "r") as f:
        settings = json.load(f)

    all_pairs = []
    for setting in settings:
        conversation = simulate_negotiation(setting, max_new_tokens, do_sample, top_k, top_p, temperature)
        pairs = extract_prompt_completion_pairs(conversation)
        all_pairs.extend(pairs)

    with open(output_path, "w") as f:
        for pair in data_pairs:
            json_line = json.dumps(pair)
            f.write(json_line + "\n")