import yaml
import os
import argparse
from datasets import load_dataset

from reward_trainer import train_reward_model
from grpo_trainer import run_rl_finetuning
from generate_data_mcts import generate_data_mcts

def parse_args():
    parser = argparse.ArgumentParser(description="Iterative Reward Model Training & RL Fine-Tuning")

    # path
    parser.add_argument("--reward_train_config", type=str, required=True, help="Path to the YAML configuration file for reward training.")
    parser.add_argument("--grpo_train_config", type=str, required=True, help="Path to the YAML configuration file for grpo training.")
    parser.add_argument("--settings", type=str, required=True, help="Path to negotiation settings.")

    # parameters
    parser.add_argument("--num_iterations", type=int, default=3, help="Number of iterations of rl.")
    parser.add_argument("--max_new_tokens", type=int, default=1024, help="Maximum number of tokens in data generation.")
    parser.add_argument('--do_sample', action='store_true', help="Enable sampling-based text generation (greedy decoding if omitted)")
    parser.add_argument("--temperature", type=float, default=1, help="Sampling temperature (higher is more random).")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k sampling limit.")
    parser.add_argument("--top_p", type=float, default=0.95, help="Nucleus sampling probability (higher is more diverse).")

    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.reward_train_config, "r") as f:
        reward_train_config = yaml.safe_load(f)
    with open(args.grpo_train_config, "r") as f:
        grpo_train_config = yaml.safe_load(f)

    current_rm_path = reward_train_config.get("other_args", {})["model_name_or_path"]
    current_llm_path = grpo_train_config.get("other_args", {})["model_name_or_path"]

    for i in range(args.num_iterations):
        print(f"\n===== ITERATION {i+1}/{args.num_iterations} =====")

        print("[+] Training reward model...")
        current_rm_path = train_reward_model(current_rm_path, reward_train_config)

        print("[+] RL fine-tuning the LLM...")
        current_llm_path = run_rl_finetuning(current_llm_path, current_rm_path, grpo_train_config)

        print("[+] Generating new data with the RL-finetuned LLM...")
        new_data = generate_data_mcts(llm_path=current_llm_path, output_file=args.rm_dataset_path, max_new_tokens=args.max_new_tokens, do_sample=args.do_sample, top_k=args.top_k, top_p=args.top_p, temperature=args.temperature)

    print("\nDone! Final LLM is in:", current_llm_path)


if __name__ == "__main__":
    main()