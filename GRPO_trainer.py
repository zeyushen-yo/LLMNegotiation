import os
import yaml
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from trl import AutoModelForCausalLMWithValueHead, GRPOTrainer, GRPOConfig, ModelConfig, ScriptArguments, TrlParser
from peft import LoraConfig

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

def run_rl_finetuning(current_llm_path, config):
    output_dir = config["output_dir"]
    training_args_dict = config.get("training_args", {})
    other_args = config.get("other_args", {})
    training_args = GRPOConfig(**training_args_dict)

    tokenizer = AutoTokenizer.from_pretrained(
        current_llm_path,
        revision=other_args.model_revision,
        trust_remote_code=other_args.trust_remote_code
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_dataset = load_dataset(other_args.dataset_id_or_path, split="train")
    
    reward_func = build_reward_function(reward_model_path)

    model = AutoModelForCausalLM.from_pretrained(
        current_llm_path,
        torch_dtype=other_args["torch_dtype"],
        device_map="auto"
    )

    lora_config = LoraConfig(r=config["r"], 
        lora_alpha=config["lora_alpha"], 
        lora_dropout=config["lora_dropout"],
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