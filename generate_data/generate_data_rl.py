import os
import json
import random
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tree import TreeNode, expand_node, get_all_paths

# generate negotiation data with gpt for rl
def generate_data_rl(rl_dataset_file, settings_path, max_new_tokens, 
                    do_sample, temperature, span, depth):
    os.makedirs(os.path.dirname(rm_dataset_path), exist_ok=True)
    num_training_examples = 0
    with open(rm_dataset_path, "w") as f_rm:
        for setting in settings:
            initial_agent = random.choice(["rl", "ref"])
            root = TreeNode(message="", agent="system", parent=None)
            expand_node(root, depth=0, max_depth=max_depth, k=k, agent=initial_agent,
                        setting=setting, rl_model=rl_model, rl_tokenizer=rl_tokenizer,
                        ref_model=ref_model, ref_tokenizer=ref_tokenizer)
            
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
                        "reward": next_node.reward
                    }
                    f_rm.write(json.dumps(training_example) + "\n")
                    num_training_examples += 1

    print(f"Saved RL training data with {num_training_examples} examples to {rm_dataset_path}")
