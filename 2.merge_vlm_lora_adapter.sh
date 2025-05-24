python 2.merge_vlm_lora_adapter.py \
    --model_name_or_path /work/checkpoints/hf/Qwen2.5-VL-3B-Instruct \
    --adapter_dir /work/code/vlm_sft/Qwen2.5-VL-3B-Instruct-qlora-sft-ko-1.5k \
    --output_dir /work/code/vlm_sft/Qwen2.5-VL-3B-Instruct-qlora-sft-ko-1.5k/merged_model \
    --torch_dtype bfloat16