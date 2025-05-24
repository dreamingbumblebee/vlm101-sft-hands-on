# Copyright 2025 DreamingBumblebee. All rights reserved.
# Licensed under the Apache License, Version 2.0.
# LoRA 병합 및 추론 스크립트

"""
# 사용 예시:
python run_llava_next_merge.py \
    --model_name_or_path /work/checkpoints/hf/llava-v1.6-vicuna-7b-hf \
    --output_dir llava-next-7B-qlora-sft-ko-1.5k \
    --torch_dtype bfloat16
"""

import os
import gc
import time
import torch
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor, AutoProcessor
from peft import PeftModel
from dataclasses import dataclass, field
from typing import Optional
from trl import TrlParser, SFTConfig

# 유틸리티 함수

def is_main_process() -> bool:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank() == 0
    else:
        return True

def print_once(*args, **kwargs):
    if is_main_process():
        print(*args, **kwargs)

def clear_memory():
    for var in ['inputs', 'model', 'processor', 'trainer', 'peft_model', 'bnb_config']:
        if var in globals():
            del globals()[var]
    time.sleep(2)
    gc.collect()
    time.sleep(2)
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    time.sleep(2)
    gc.collect()
    time.sleep(2)
    if is_main_process():
        print(f"GPU 할당된 메모리: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"GPU 예약된 메모리: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")

# 시스템 메시지
SYSTEM_MESSAGE = """You are a Vision Language Model specialized in Korean Language.\nYou are given a question and an image.\nYou need to answer the question based on the image.\nYou need to answer in Korean.\nYou need to answer in a detailed manner.\nYou need to answer in a way that is easy to understand.\n"""

# 커맨드 라인 인자 정의
@dataclass
class MyArguments(SFTConfig):
    model_name_or_path: Optional[str] = field(default=None)
    adapter_dir: Optional[str] = field(default=None)
    output_dir: Optional[str] = field(default=None)
    torch_dtype: Optional[str] = field(default=None)

def main():
    clear_memory()
    parser = TrlParser((MyArguments,))
    sft_args, = parser.parse_args_and_config()

    # 1. LoRA와 기본 모델 병합 및 저장
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        sft_args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(sft_args.model_name_or_path, use_fast=True)
    if processor.tokenizer.chat_template is None:
        processor.tokenizer.chat_template = processor.chat_template

    peft_model = PeftModel.from_pretrained(model, f"{sft_args.adapter_dir}")
    print_once(f"peft_model: {peft_model}")
    merged_model = peft_model.merge_and_unload()
    print_once(f"merged_model: {merged_model}")

    # accelerator 없이 모델 저장
    merged_model.save_pretrained(
        f"{sft_args.output_dir}",
        safe_serialization=True
    )
    processor.save_pretrained(f"{sft_args.output_dir}")
    print_once("병합된 모델 및 프로세서 저장 완료")

    # 2. 샘플 데이터에 대한 추론 수행
    image_url = "https://i.namu.wiki/i/i4MLtzbUOetZyyjiL84EDMjAKemVCIbgAv4fxeNJBj-NT1j9W_9LHwzcIE6dzzBr65ew6IA1YbzVHPGSf4bz_g.webp"
    conversation = [
        {
            "role": "system",
            "content": [{"type": "text", "text": SYSTEM_MESSAGE}],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_url},
                {"type": "text", "text": "이 이미지에서는 뭐가 보이니?"},
            ],
        },
    ]

    # 병합된 모델 로드 및 추론
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        f"{sft_args.output_dir}",
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    print_once(f"model.device: {model.device}")

    # 추론 함수 (이미지 URL 지원)
    from PIL import Image
    import requests
    from io import BytesIO
    from qwen_vl_utils import process_vision_info
    
    def generate_text_from_sample(model, processor, sample, max_new_tokens=256, device="cuda"):
        text_input = processor.apply_chat_template(
            sample[1:2],  # 시스템 메시지 제외
            tokenize=False,
            add_generation_prompt=True
        )
        image_inputs, _ = process_vision_info(sample)
        model_inputs = processor(
            text=[text_input],
            images=image_inputs,
            return_tensors="pt",
        )
        model_inputs = {k: v.to(model.device) for k, v in model_inputs.items()}
        print(f"model_inputs:{model_inputs}")
        generated_ids = model.generate(**model_inputs, max_new_tokens=max_new_tokens)
        trimmed_generated_ids = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(model_inputs["input_ids"], generated_ids)
        ]
        output_text = processor.batch_decode(
            trimmed_generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )
        return output_text[0]

    result = generate_text_from_sample(model, processor, conversation, max_new_tokens=256, device=model.device)
    print_once("샘플 추론 결과:")
    print_once(result)

if __name__ == "__main__":
    main()