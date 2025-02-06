import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

def run_rl_finetuning(
    base_llm: str,
    reward_model_path: str,
    rl_dataset,
    output_dir: str,
    num_epochs: int = 1,
    batch_size: int = 2,
):
    """
    Args:
        base_llm (str): HF model ID or local path to your base LLM.
        reward_model_path (str): local path to the trained reward model.
        rl_dataset (Dataset): a dataset of prompts to run RL on.
        output_dir (str): directory to save the RL-finetuned model.
        num_epochs (int): training epochs.
        batch_size (int): batch size for RL.
    Returns:
        str: path to the RL-finetuned model.
    """

    rm_tokenizer = AutoTokenizer.from_pretrained(reward_model_path)
    rm_model = AutoModelForSequenceClassification.from_pretrained(reward_model_path)
    rm_model.cuda()
    rm_model.eval()

    def reward_func(completions, prompts, **kwargs):
        """
        completions: List[str] of responses from the LLM
        prompts:     List[str] of original prompt strings
        """
        texts = [p + "\n" + c for p, c in zip(prompts, completions)]
        with torch.no_grad():
            inputs = rm_tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt"
            ).to("cuda")
            outputs = rm_model(**inputs)
            # shape is [batch_size, 1], so squeeze
            rewards = outputs.logits.squeeze(-1)
        return rewards.cpu().tolist()  # convert to floats

    # GRPO expects a column "query" (the prompt)
    if "prompt" in rl_dataset.column_names:
        rl_dataset = rl_dataset.rename_column("prompt", "query")

    training_args = GRPOConfig(
        model_name=base_llm, 
        reward_funcs=reward_func,
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        num_train_epochs=num_epochs,
        logging_steps=10,
    )

    trainer = GRPOTrainer(
        model=base_llm,
        args=training_args,
        train_dataset=rl_dataset,
    )

    trainer.train()

    trainer.save_model(output_dir)
    return output_dir
