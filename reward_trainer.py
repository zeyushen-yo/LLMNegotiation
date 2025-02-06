import torch
import torch.nn as nn
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)

def train_reward_model(
    model_name_or_path: str,
    train_dataset: Dataset,
    output_dir: str,
    num_epochs: int = 1,
    batch_size: int = 4,
) -> str:
    """
    Train a pointwise reward model (num_labels=1) on train_dataset, which should
    contain ['prompt', 'response', 'reward'] columns.
    """

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    rm_model = AutoModelForSequenceClassification.from_pretrained(
        model_name_or_path,
        num_labels=1
    )
    rm_model.config.pad_token_id = tokenizer.pad_token_id

    def tokenize_fn(examples):
        texts = [q + "\n" + r for q, r in zip(examples["prompt"], examples["response"])]
        tokens = tokenizer(texts, truncation=True, padding="max_length", max_length=128)
        tokens["labels"] = examples["reward"]
        return tokens

    # Map over dataset
    ds_tokenized = train_dataset.map(tokenize_fn, batched=True)
    # remove old columns
    ds_tokenized = ds_tokenized.remove_columns(["prompt", "response", "reward"])
    ds_tokenized.set_format("torch")

    # Custom trainer to do MSE on single output
    class RewardModelTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False):
            labels = inputs.pop("labels").float()
            outputs = model(**inputs)
            logits = outputs.logits.squeeze(-1)
            loss_fct = nn.MSELoss()
            loss = loss_fct(logits, labels)
            return (loss, outputs) if return_outputs else loss

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        do_train=True,
        per_device_train_batch_size=batch_size,
        num_train_epochs=num_epochs,
        logging_steps=20,
    )

    trainer = RewardModelTrainer(
        model=rm_model,
        args=training_args,
        train_dataset=ds_tokenized,
        tokenizer=tokenizer,
    )

    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    return output_dir