import torch
import os
import yaml
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
from trl import GRPOTrainer, GRPOConfig, ModelConfig, ScriptArguments, TrlParser
from peft import LoraConfig, get_peft_model

def build_reward_function(reward_model_path):
    rm_tokenizer = AutoTokenizer.from_pretrained(reward_model_path)
    rm_model = AutoModelForSequenceClassification.from_pretrained(reward_model_path)
    rm_model.cuda()
    rm_model.eval()

    def reward_func(prompts, completions, **kwargs):
        texts = [f"{prompt}\n{completion}" for prompt, completion in zip(prompts, completions)]
        inputs = rm_tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        inputs = {k: v.cuda() for k, v in inputs.items()}
        with torch.no_grad():
            outputs = rm_model(**inputs)
            scores = outputs.logits.squeeze(-1)  # shape: (batch_size,), one score per input
        return scores.tolist()
    
    return reward_func

def run_rl_finetuning(current_llm_path, reward_model_path, config):
    training_args_dict = config.get("training_args", {})
    other_args = config.get("other_args", {})
    output_dir = training_args_dict["output_dir"]
    training_args = GRPOConfig(**training_args_dict)

    tokenizer = AutoTokenizer.from_pretrained(
        current_llm_path,
        revision=other_args["model_revision"]
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_dataset = load_dataset("json", data_files=other_args["dataset_name_or_path"], split="train")
    
    reward_func = build_reward_function(reward_model_path)

    model = AutoModelForCausalLM.from_pretrained(
        current_llm_path,
        torch_dtype=other_args["torch_dtype"],
        device_map="auto"
    )

    lora_config = LoraConfig(r=other_args["r"], 
        lora_alpha=other_args["alpha"], 
        lora_dropout=other_args["dropout"],
        bias="none"
    )
    model = get_peft_model(model, lora_config)

    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        reward_funcs=reward_func
    )
    
    trainer.train()
    
    os.makedirs(output_dir, exist_ok=True)
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model and tokenizer for llm saved to {output_dir}")
    
    return output_dir