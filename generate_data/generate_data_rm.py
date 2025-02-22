import os
import json
import random
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from generate_data.tree import TreeNode, expand_node, get_all_paths
from prompts import negotiation_start_prompt, negotiation_respond_prompt
from misc import load_llm

# generate negotiation data with fine-tuned model and reference model for reward modeling
# assume rl model is agent1 and reference model is agent2
def generate_data_rm(rl_llm_path, ref_model, ref_tokenizer, rm_dataset_path, settings_path, 
                     model, max_new_tokens, do_sample, temperature, span, max_depth, agents):
    print("Loading RL model and tokenizer from:", rl_llm_path)
    rl_model, rl_tokenizer = load_llm(rl_llm_path)

    with open(settings_path, "r") as f:
        settings = json.load(f)

    os.makedirs(os.path.dirname(rm_dataset_path), exist_ok=True)
    num_training_examples = 0
    with open(rm_dataset_path, "w") as f_rm:
        for setting in settings:
            initial_agent = random.choice([agents[0], agents[1]])
            root = TreeNode(message="[Negotiation Starts]", agent="System", parent=None)

            expand_node(node=root, depth=0, max_depth=max_depth, span=span, current_agent=initial_agent, setting=setting, current_conversation=[f"{root.agent}: {root.message}"],
                        judge_model=model, rl_model=rl_model, rl_tokenizer=rl_tokenizer, ref_model=ref_model, ref_tokenizer=ref_tokenizer, 
                        max_new_tokens=max_new_tokens, do_sample=do_sample, temperature=temperature, agents=agents)
            
            for path in get_all_paths(root):
                conversation_so_far = ""
                # For a path [v1, v2, ..., vt] generate examples:
                # (v1, v2), (v1+v2, v3), ..., (v1+...+v_{t-1}, vt)
                for i in range(len(path) - 1):
                    current_node = path[i]
                    next_node = path[i + 1]
                    conversation_so_far += f"{current_node.agent}: {current_node.message}\n"
                    training_example = {
                        "prompt": conversation_so_far.strip(),
                        "completion": f"{next_node.agent}: {next_node.message}",
                        "reward": next_node.reward[next_node.agent]
                    }
                    f_rm.write(json.dumps(training_example) + "\n")
                    num_training_examples += 1

    print(f"Saved RM training data with {num_training_examples} examples to {rm_dataset_path}")
