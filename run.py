import yaml
import os
import argparse
from datasets import load_dataset

from reward_trainer import train_reward_model
from grpo_trainer import run_rl_finetuning
# from generate_data_mcts import generate_data_mcts
from generate_data.generate_data_rm import generate_data_rm
from generate_data.generate_data_rl import generate_data_rl
from generate_data.generate_settings import generate_settings

def parse_args():
    parser = argparse.ArgumentParser(description="Iterative Reward Model Training & RL Fine-Tuning")

    # path
    parser.add_argument("--reward_train_config", type=str, required=True, help="Path to the YAML configuration file for reward training.")
    parser.add_argument("--grpo_train_config", type=str, required=True, help="Path to the YAML configuration file for grpo training.")
    parser.add_argument("--settings_path", type=str, required=True, help="Path to negotiation settings.")

    # parameters
    parser.add_argument("--num_settings", type=int, default=3, help="Number of settings to generate.")
    parser.add_argument("--num_iterations", type=int, default=3, help="Number of iterations of rl.")
    parser.add_argument("--model", type=str, default='o3-mini', choices=['o3-mini', 'o1-mini', 'o1'], help="Which gpt model to use.")
    parser.add_argument("--num_samples_for_each_setting_rl", type=int, default=10, help="Number of samples to generate for each setting for RL.")
    parser.add_argument("--max_new_tokens", type=int, default=1024, help="Maximum number of new tokens in data generation.")
    parser.add_argument('--do_sample', action='store_true', help="Enable sampling-based text generation (greedy decoding if omitted)")
    parser.add_argument("--temperature", type=float, default=1, help="Sampling temperature (higher is more random).")
    parser.add_argument("--span", type=int, default=3, help="Number of children of each non-leaf node in mcts.")
    parser.add_argument("--depth", type=int, default=6, help="Maximum depth of the tree in mcts.")

    return parser.parse_args()


def main():
    args = parse_args()
    agents = ['A', 'B']

    print("[+] Generating negotiation settings...")
    generate_settings(num_settings=args.num_settings, 
                      settings_path=args.settings_path, 
                      model=args.model,
                      max_new_tokens=args.max_new_tokens,
                      agents=agents)

    with open(args.reward_train_config, "r") as f:
        reward_train_config = yaml.safe_load(f)
    with open(args.grpo_train_config, "r") as f:
        grpo_train_config = yaml.safe_load(f)

    current_rm_path = reward_train_config.get("other_args", {})["model_name_or_path"]
    current_llm_path = grpo_train_config.get("other_args", {})["model_name_or_path"]
    start_llm_path = current_llm_path

    print("[+] Generating data for RL...")
    generate_data_rl(rl_dataset_file=reward_train_config.train_file, 
                     settings_path=args.settings_path, 
                     model=args.model,
                     num_samples=args.num_samples_for_each_setting_rl,
                     max_new_tokens=args.max_new_tokens, 
                     do_sample=args.do_sample, 
                     temperature=args.temperature, 
                     depth=args.depth,
                     agents=agents)

    for i in range(args.num_iterations):
        print(f"\n===== ITERATION {i+1}/{args.num_iterations} =====")

        print("[+] Training reward model...")
        current_rm_path = train_reward_model(current_rm_path, reward_train_config)

        print("[+] RL fine-tuning the LLM...")
        current_llm_path = run_rl_finetuning(current_llm_path, current_rm_path, grpo_train_config)

        print("[+] Generating new data with the RL-finetuned LLM...")
        new_data = generate_data_rm(rl_llm_path=current_llm_path, 
                                    reference_llm_path=start_llm_path, 
                                    rm_dataset_file=reward_train_config.train_file, 
                                    settings_path=args.settings_path, 
                                    model=args.model,
                                    max_new_tokens=args.max_new_tokens, 
                                    do_sample=args.do_sample, 
                                    temperature=args.temperature, 
                                    span=args.span, 
                                    depth=args.depth
                                    agents=agents)

    print("\nDone! Final LLM is in:", current_llm_path)


if __name__ == "__main__":
    main()