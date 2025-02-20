import torch
import torch.nn as nn
import os
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer, DataCollatorWithPadding

class RewardModelTrainer(Trainer): 
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        if isinstance(outputs, dict):
            preds = outputs.get("logits", outputs.get("reward", outputs.get("scores")))
        else:
            preds = outputs[0]
        
        loss = torch.nn.functional.mse_loss(preds.squeeze(), labels.squeeze().float(), reduction="mean")
        return (loss, outputs) if return_outputs else loss

def train_reward_model(current_rm_path, config):
    """
    Train a pointwise reward model (num_labels=1) on train_dataset, which should
    contain ['prompt', 'completion', 'reward'] columns.
    """
    training_args_dict = config.get("training_args", {})
    other_args = config.get("other_args", {})
    training_args = TrainingArguments(**training_args_dict)
    output_dir = training_args_dict["output_dir"]

    num_labels = other_args["num_labels"]
    tokenizer = AutoTokenizer.from_pretrained(current_rm_path)
    rm_model = AutoModelForSequenceClassification.from_pretrained(current_rm_path, num_labels=num_labels)
    
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id or 0
    rm_model.config.pad_token_id = tokenizer.pad_token_id    

    train_dataset = load_dataset("json", data_files=other_args["dataset_name_or_path"], split="train")

    def tokenize_fn(examples):
        texts = [q + "\n" + r for q, r in zip(examples["prompt"], examples["completion"])]
        tokens = tokenizer(texts, truncation=True, padding="max_length", max_length=1024)
        tokens["labels"] = examples["reward"]
        return tokens

    ds_tokenized = train_dataset.map(tokenize_fn, batched=True)
    ds_tokenized = ds_tokenized.remove_columns(["prompt", "completion", "reward"])
    ds_tokenized.set_format("torch")
    data_collator = DataCollatorWithPadding(tokenizer)

    trainer = RewardModelTrainer(
        model=rm_model,
        args=training_args,
        train_dataset=ds_tokenized,
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    trainer.train()
    os.makedirs(output_dir, exist_ok=True)
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model and tokenizer for reward model saved to {output_dir}")

    return output_dir