import os
import argparse
import yaml
from datasets import load_dataset
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification
from trl import AutoModelForCausalLMWithValueHead, GRPOConfig, GRPOTrainer
from peft import LoraConfig

def prepare_model_and_tokenizer(base_model_name, lora_rank, lora_alpha, lora_dropout, 
                                load_4bit=False, use_cpu_offload=False):
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=None,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM"
    )
    device_map = None
    if load_4bit or use_cpu_offload:
        device_map = {"": 0}
    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        base_model_name,
        load_in_4bit=load_4bit,
        device_map=device_map,
        peft_config=lora_config
    )
    return model, tokenizer

def build_reward_function(reward_model_path):
    rm_tokenizer = AutoTokenizer.from_pretrained(reward_model_path)
    rm_model = AutoModelForSequenceClassification.from_pretrained(reward_model_path)
    rm_model.cuda()
    rm_model.eval()

    def reward_func(prompts, completions):
        texts = [f"{prompt}\n{completion}" for prompt, completion in zip(prompts, completions)]
        inputs = rm_tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        inputs = {k: v.cuda() for k, v in inputs.items()}
        with torch.no_grad():
            outputs = rm_model(**inputs)
            scores = outputs.logits.squeeze(-1)  # shape: (batch_size,), one score per input
        return scores.tolist()

def run_rl_finetuning(
    base_llm,
    reward_model_path,
    rl_dataset,
    output_dir,
    num_epochs,
    batch_size,
    num_generations,
    max_completion_length,
    per_device_train_batch_size,
    gradient_accumulation_steps,
    learning_rate,
    logging_steps,
    lora_r,
    lora_alpha,
    lora_dropout,
    load_4bit
):
    model, tokenizer = prepare_model_and_tokenizer(
        base_model_name=base_llm,
        lora_rank=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
        load_4bit=load_4bit
    )
    
    grpo_config = GRPOConfig(
        num_generations=num_generations,
        max_completion_length=max_completion_length,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        logging_steps=logging_steps,
        output_dir=output_dir
    )
    
    reward_func = build_reward_function(reward_model_path)
    
    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        train_dataset=rl_dataset,
        reward_funcs=reward_func,
        tokenizer=tokenizer
    )
    
    trainer.train()
    
    os.makedirs(output_dir, exist_ok=True)
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model and tokenizer saved to {output_dir}")
    
    return output_dir