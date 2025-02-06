python3.11 -B run.py \
    --rm_dataset_path /home/zs7353/LLMNegotiation/negotiation_data/rm_dataset.jsonl \
    --rl_dataset_path /home/zs7353/LLMNegotiation/negotiation_data/rl_dataset.jsonl \
    --high_level_settings /home/zs7353/LLMNegotiation/negotiation_data/high_level_settings.jsonl \
    --base_rm_path /scratch/gpfs/zs7353/Qwen2.5-0.5B-Instruct \
    --base_llm_path /scratch/gpfs/zs7353/DeepSeek-R1-Distill-Llama-8B \
    --output_dir /scratch/gpfs/zs7353/LLMNegotiation \
    --num_iterations 3 \
    --rm_epochs 1 \
    --rm_batch_size 4 \
    --rl_epochs 1 \
    --rl_batch_size 2 \
    --max_new_tokens 1024 \
    --do_sample \
    --temperature 1 \
    --top_k 50 \
    --top_p 0.95 \
    ${@}