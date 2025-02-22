import os
import json
import random
from gpt import get_output
from prompts import negotiation_start_prompt, negotiation_respond_prompt, reward_judge_prompt
from process_text import extract_from_text
from misc import generate_model_response

def generate_data_rl(ref_model, ref_tokenizer, rl_dataset_path, settings_path, model, 
                     num_samples, max_new_tokens, do_sample, temperature, max_depth, agents):
    """
    Generates linear (instead of tree-structured) negotiation conversation data using gpt for reinforcement learning.
    """
    os.makedirs(os.path.dirname(rl_dataset_path), exist_ok=True)
    
    num_training_examples = 0
    
    with open(settings_path, "r") as f:
        settings = json.load(f)
    
    with open(rl_dataset_path, "w") as f_rl:
        for setting in settings:
            for _ in range(num_samples):
                current_agent = random.choice(agents)
                conversation = []
                conversation.append("System: [Negotiation Starts]")
                
                for turn in range(max_depth):
                    response_text = generate_model_response(current_agent=current_agent, setting=setting, model=ref_model, 
                                                            tokenizer=ref_tokenizer, current_conversation=conversation, 
                                                            max_new_tokens=max_new_tokens, do_sample=do_sample, 
                                                            temperature=temperature, agents=agents)
                    
                    conversation_line = f"{current_agent}: {response_text}"
                    conversation.append(conversation_line)
                    
                    if "NEGOTIATION END" in response_text:
                        break
                    
                    current_agent = agents[1] if current_agent == agents[0] else agents[0]
                
                full_conversation = "\n".join(conversation)
                # For a conversation with turns [v1, v2, v3, ..., vt] we create examples:
                # (v1), (v1+v2), ... , (v1+...+v_{t-1})
                conversation_so_far = ""
                for i in range(1, len(conversation)):
                    conversation_so_far += conversation[i - 1] + "\n"
                    completion_line = conversation[i].strip()
                    agent_label = completion_line.split(":", 1)[0]
                    training_example = {"prompt": conversation_so_far.strip()}
                    f_rl.write(json.dumps(training_example) + "\n")
                    num_training_examples += 1
    
    print(f"Saved RL training data with {num_training_examples} examples to {rl_dataset_path}")