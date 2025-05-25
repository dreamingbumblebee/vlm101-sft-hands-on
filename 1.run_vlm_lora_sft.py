# Copyright 2025 DreamingBumblebee. All rights reserved.
# Licensed under the Apache License, Version 2.0.
# Simplified and adapted for LoRA SFT training with WandB integration.

"""
pip install accelerate transformers datasets trl peft bitsandbytes wandb

# Example launch:
accelerate launch \
    --config_file=./deepspeed_zero3.yaml \
    1.run_vlm_lora_sft.py \
    --dataset_name /work/dataset/vlm/KoLLaVA-Instruct-1.5k \
    --model_name_or_path /work/checkpoints/hf/Qwen2.5-VL-3B-Instruct \
    --output_dir /work/vlm101-sft-hands-on/qwen2.5-3b-qlora-sft-ko-1.5k \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --bf16 \
    --torch_dtype bfloat16 \
    --gradient_checkpointing

"""

import os
import gc
import time
import torch
from datasets import load_dataset
from transformers import AutoProcessor, BitsAndBytesConfig
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor
from trl import TrlParser, SFTTrainer, SFTConfig
from peft import LoraConfig, get_peft_model, PeftModel, prepare_model_for_kbit_training
import wandb
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any
from dotenv import load_dotenv
from qwen_vl_utils import process_vision_info
from accelerate import Accelerator

# 분산 학습 관련 유틸리티 함수
def is_main_process() -> bool:
    """현재 프로세스가 main rank (rank 0)인지 확인"""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank() == 0
    else:
        return True

def get_world_size() -> int:
    """전체 rank 수(world size)를 반환"""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_world_size()
    else:
        return 1

def print_once(*args, **kwargs):
    """rank 0에서만 print하는 함수"""
    if is_main_process():
        print(*args, **kwargs)

# 메모리 관리 함수
def clear_memory():
    """GPU 메모리를 정리하는 함수"""
    # 전역 변수 정리
    for var in ['inputs', 'model', 'processor', 'trainer', 'peft_model', 'bnb_config']:
        if var in globals():
            del globals()[var]
    
    # 메모리 정리 과정
    time.sleep(2)
    gc.collect()
    time.sleep(2)
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    time.sleep(2)
    gc.collect()
    time.sleep(2)

    # 메모리 사용량 출력
    if is_main_process():
        print(f"GPU 할당된 메모리: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"GPU 예약된 메모리: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")

# 시스템 메시지 정의
SYSTEM_MESSAGE = """You are a Vision Language Model specialized in Korean Language.
You are given a question and an image.
You need to answer the question based on the image.
You need to answer in Korean.
You need to answer in a detailed manner.
You need to answer in a way that is easy to understand.
"""

# 데이터 포맷팅 함수
def format_data(sample):
    """데이터 샘플을 챗봇 상호작용 구조로 변환"""
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": SYSTEM_MESSAGE}],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": sample["images"]},
                {"type": "text", "text": sample['questions'].replace('<image>', '')},
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": sample["answers"]}],
        },
    ]

# 샘플에서 텍스트 생성 함수
def generate_text_from_sample(model, processor, sample, max_new_tokens=1024, device="cuda"):
    """모델을 사용하여 샘플로부터 텍스트 생성"""
    # 대화 템플릿 적용
    text_input = processor.apply_chat_template(
        sample[1:2],  # 시스템 메시지 제외
        tokenize=False,
        add_generation_prompt=True
    )

    # 이미지 처리
    image_inputs, _ = process_vision_info(sample)

    # 모델 입력 준비
    model_inputs = processor(
        text=[text_input],
        images=image_inputs,
        return_tensors="pt",
    )
    
    # 입력을 모델 디바이스로 이동
    model_inputs = {k: v.to(model.device) for k, v in model_inputs.items()}

    # 텍스트 생성
    generated_ids = model.generate(**model_inputs, max_new_tokens=max_new_tokens)

    print(f"model_inputs:{model_inputs}")
    # 입력 id를 제외한 생성된 id 추출
    trimmed_generated_ids = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    # 생성된 텍스트 디코딩
    output_text = processor.batch_decode(
        trimmed_generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )

    return output_text[0]

# 데이터 콜레이터 정의
def collate_fn(examples, processor):
    """배치 데이터 처리를 위한 콜레이터 함수"""
    # 텍스트와 이미지 전처리
    texts = [processor.apply_chat_template(example, tokenize=False) for example in examples]
    image_inputs = [process_vision_info(example)[0] for example in examples]
    
    # 텐서로 변환
    batch = processor(text=texts, images=image_inputs, return_tensors="pt", padding=True)
    
    # 라벨 설정 및 마스킹
    labels = batch["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100
    
    # 이미지 토큰 마스킹
    image_tokens = [processor.tokenizer.convert_tokens_to_ids(processor.image_token)]
    for image_token_id in image_tokens:
        labels[labels == image_token_id] = -100
    
    batch["labels"] = labels
    return batch

def get_messages_from_image_url(prompt, image_url):
    
    # 이미지를 URL로 지정
    default_image_url = "https://i.namu.wiki/i/i4MLtzbUOetZyyjiL84EDMjAKemVCIbgAv4fxeNJBj-NT1j9W_9LHwzcIE6dzzBr65ew6IA1YbzVHPGSf4bz_g.webp"
    # 시스템 메시지와 사용자 메시지 정의
    messages = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": SYSTEM_MESSAGE,  # 시스템 메시지
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_url if image_url else default_image_url},  # 이미지 메시지
                {"type": "text", "text": prompt if prompt else "이 이미지에 대해서 설명해봐"},  # 이미지에 대한 설명 요청
            ],
        }
    ]
    
    return messages

# 커맨드 라인 인자 정의
@dataclass
class MyArguments(SFTConfig):
    dataset_name: Optional[str] = field(default=None)
    model_name_or_path: Optional[str] = field(default=None)
    torch_dtype: Optional[str] = field(default=None)

def main():
    # 메모리 초기화
    clear_memory()
    
    # 1. 인자 파싱 및 설정
    parser = TrlParser((MyArguments,))
    sft_args, = parser.parse_args_and_config()
    
    # 학습 설정
    sft_args.num_train_epochs = 2
    sft_args.per_device_train_batch_size = 4
    sft_args.per_device_eval_batch_size = 1
    sft_args.gradient_accumulation_steps = 4
    
    # 옵티마이저 설정
    sft_args.optim = "adamw_torch_fused"
    sft_args.learning_rate = 2e-04
    sft_args.lr_scheduler_type = "constant"
    
    # 로깅 및 평가 설정
    sft_args.logging_steps = 5
    sft_args.eval_steps = 40
    sft_args.eval_strategy = "steps"
    sft_args.save_strategy = "steps"
    sft_args.save_steps = 40
    sft_args.metric_for_best_model = "eval_loss"
    sft_args.greater_is_better = False
    sft_args.load_best_model_at_end = True
    
    # HF Hub 및 보고 설정
    sft_args.push_to_hub = False
    sft_args.report_to = "wandb"
    sft_args.run_name = sft_args.output_dir
    
    # 데이터셋 설정
    sft_args.dataset_text_field = ""
    sft_args.dataset_kwargs = {"skip_prepare_dataset": True}
    sft_args.remove_unused_columns = False
    sft_args.label_names = ["labels"]
    
    print_once(f"sft_args: {sft_args}")

    # 2. 데이터셋 로드 및 분할
    raw_dataset = load_dataset(sft_args.dataset_name)
    dataset = raw_dataset['train'].train_test_split(test_size=0.1, seed=42)
    
    # 데이터셋 전처리
    train_dataset = [format_data(sample) for sample in dataset['train']]
    eval_dataset = [format_data(sample) for sample in dataset['test']]
    print_once(f"train_dataset[0]: {train_dataset[0]}")
    
    # 3. 모델 및 프로세서 로드
    # 양자화 설정
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_storage=torch.bfloat16,
    )
    
    # 모델 로드
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        sft_args.model_name_or_path,
        torch_dtype=torch.bfloat16,  # 모델 파라미터의 기본 데이터 타입 지정
        quantization_config=bnb_config  # 위에서 정의한 양자화 설정 적용
    )
    
    model.config.use_cache = False  # gradient checkpointing 활성화
    
    # 프로세서 로드
    processor = AutoProcessor.from_pretrained(sft_args.model_name_or_path, use_fast=True)
    
    if processor.tokenizer.chat_template is None:
        processor.tokenizer.chat_template = processor.chat_template
    
    # 4. LoRA 설정
    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.05,
        r=8,
        bias="none",
        target_modules=["q_proj", "v_proj"],  # LoRA를 적용할 모듈 이름 (여기서는 q_proj와 v_proj)
        task_type="CAUSAL_LM",
    )
    
    # 모델 준비
    model = prepare_model_for_kbit_training(model)
    peft_model = get_peft_model(model, peft_config)
    
    if is_main_process():
        peft_model.print_trainable_parameters()

    # 5. 트레이너 설정 및 학습
    # 콜레이터 래퍼 함수
    def collator(examples):
        return collate_fn(examples, processor)
    
    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset if sft_args.eval_strategy != 'no' else None,
        data_collator=collator,
        processing_class=processor.tokenizer,
        peft_config=peft_config,
    )

    # 학습 실행
    trainer.train()
    print_once("학습 완료")

    # 모델 저장
    trainer.save_model(sft_args.output_dir)
    processor.save_pretrained(sft_args.output_dir)
    print_once("모델 저장 완료")    
    
    # 추론 결과 출력
    result = generate_text_from_sample(model, 
                                       processor, 
                                       get_messages_from_image_url("이 이미지에 대해서 설명해봐", 
                                                                   "https://i.namu.wiki/i/i4MLtzbUOetZyyjiL84EDMjAKemVCIbgAv4fxeNJBj-NT1j9W_9LHwzcIE6dzzBr65ew6IA1YbzVHPGSf4bz_g.webp"), 
                                       max_new_tokens=256, 
                                       device=model.device)
    print_once(result)

if __name__ == "__main__":
    main()
