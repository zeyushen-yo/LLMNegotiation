import os
import json
import random
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from gpt import get_output

# generate negotiation data with gpt for rl
def generate_data_rl(rl_dataset_file, settings_path, model, max_new_tokens, 
                    do_sample, temperature, span, depth, agents):
    os.makedirs(os.path.dirname(rm_dataset_path), exist_ok=True)
    num_training_examples = 0
    with open(rm_dataset_path, "w") as f_rm:
        for setting in settings:
            initial_agent = random.choice([agents[0], agents[1]])
