
source ./.env
huggingface-cli login --token $HF_TOKEN


wandb login $WANDB_API_KEY
export WANDB_PROJECT="qwen2.5-vlm-sft"

# accelerate configuration saved at /root/.cache/huggingface/accelerate/default_config.yaml
model_name_or_path=/work/checkpoints/hf/Qwen2.5-VL-3B-Instruct
output_dir=/work/vlm101-sft-hands-on/qwen2.5-3b-qlora-sft-ko-1.5k


accelerate launch \
    --config_file=./deepspeed_zero3.yaml \
    1.run_vlm_lora_sft.py \
    --dataset_name /work/dataset/vlm/KoLLaVA-Instruct-1.5k \
    --model_name_or_path $model_name_or_path \
    --output_dir ${output_dir} \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --bf16 \
    --torch_dtype bfloat16 \
    --gradient_checkpointing
