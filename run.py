import os
import argparse
from datasets import load_dataset

from reward_trainer import train_reward_model
from GRPO_trainer import run_rl_finetuning
from generate_data_MCTS import generate_data_MCTS

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Iterative Reward Model Training & RL Fine-Tuning")

    # File paths
    parser.add_argument("--rm_dataset_path", type=str, required=True, help="Path to JSONL reward model dataset.")
    parser.add_argument("--rl_dataset_path", type=str, required=True, help="Path to JSONL RL dataset.")
    parser.add_argument("--high_level_settings", type=str, required=True, help="Path to high-level negotiation settings.")
    parser.add_argument("--base_rm_path", type=str, required=True, help="Path to the base reward model.")
    parser.add_argument("--base_llm_path", type=str, required=True, help="Path to the base LLM.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save trained models.")

    # Training hyperparameters
    parser.add_argument("--num_iterations", type=int, default=3, help="Number of training iterations.")
    parser.add_argument("--rm_epochs", type=int, default=1, help="Reward model training epochs.")
    parser.add_argument("--rm_batch_size", type=int, default=4, help="Reward model training batch size.")
    parser.add_argument("--rl_epochs", type=int, default=1, help="RL fine-tuning epochs.")
    parser.add_argument("--rl_batch_size", type=int, default=2, help="RL fine-tuning batch size.")
    parser.add_argument("--max_new_tokens", type=int, default=1024, help="Maximum number of tokens in data generation.")
    parser.add_argument('--do_sample', action='store_true', help="Enable sampling-based text generation (greedy decoding if omitted)")
    parser.add_argument("--temperature", type=float, default=1, help="Sampling temperature (higher is more random).")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k sampling limit.")
    parser.add_argument("--top_p", type=float, default=0.95, help="Nucleus sampling probability (higher is more diverse).")

    return parser.parse_args()


def main():
    """Main function that executes the iterative training loop."""
    args = parse_args()

    current_llm_path = args.base_llm_path
    current_rm_path = args.base_rm_path

    for i in range(args.num_iterations):
        print(f"\n===== ITERATION {i+1}/{args.num_iterations} =====")

        print("[+] Loading Reward Model (RM) dataset from JSON...")
        rm_dataset = load_dataset("json", data_files=args.rm_dataset_path, split="train")

        rm_output_dir = os.path.join(args.output_dir, "Qwen2.5-0.5B-Instruct_reward_model")
        print("[+] Training reward model...")
        current_rm_path = train_reward_model(
            model_name_or_path=current_rm_path,
            train_dataset=rm_dataset,
            output_dir=rm_output_dir,
            num_epochs=args.rm_epochs,
            batch_size=args.rm_batch_size,
        )

        print("[+] Loading RL dataset from JSON...")
        rl_dataset = load_dataset("json", data_files=args.rl_dataset_path, split="train")

        llm_output_dir = os.path.join(args.output_dir, "DeepSeek-R1-Distill-Llama-8B_negotiation")
        print("[+] RL fine-tuning the LLM...")
        current_llm_path = run_rl_finetuning(
            base_llm=current_llm_path,
            reward_model_path=trained_rm_path,
            rl_dataset=rl_dataset,
            output_dir=llm_output_dir,
            num_epochs=args.rl_epochs,
            batch_size=args.rl_batch_size,
        )

        print("[+] Generating new data with the RL-finetuned LLM...")
        # new_data = generate_data_MCTS(llm_path=current_llm_path, output_file=args.rm_dataset_path, max_new_tokens=args.max_new_tokens, do_sample=args.do_sample, top_k=args.top_k, top_p=args.top_p, temperature=args.temperature)

    print("\nDone! Final LLM is in:", current_llm_path)


if __name__ == "__main__":
    main()