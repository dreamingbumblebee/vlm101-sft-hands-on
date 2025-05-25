python 2.merge_vlm_lora_adapter.py \
    --model_name_or_path /work/checkpoints/hf/Qwen2.5-VL-3B-Instruct \
    --adapter_dir /work/vlm101-sft-hands-on/qwen2.5-3b-qlora-sft-ko-1.5k \
    --output_dir /work/vlm101-sft-hands-on/qwen2.5-3b-qlora-sft-ko-1.5k/merged_model \
    --torch_dtype bfloat16