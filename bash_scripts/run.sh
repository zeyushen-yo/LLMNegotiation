python3.11 -B run.py \
    --config /home/zs7353/LLMNegotiation/config/grpo_train_config.yaml \
    --reward_train_config /home/zs7353/LLMNegotiation/config/reward_train_config.yaml \
    --high_level_settings /home/zs7353/LLMNegotiation/negotiation_data/high_level_settings.jsonl \
    --num_iterations 3 \
    --max_new_tokens 1024 \
    --do_sample \
    --temperature 1 \
    --top_k 50 \
    --top_p 0.95 \