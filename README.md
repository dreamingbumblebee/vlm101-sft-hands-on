# vlm101-sft-hands-on

This repository provides a hands-on pipeline for supervised fine-tuning (SFT) of Vision-Language Models (VLMs) in Korean, using LoRA (Low-Rank Adaptation) and Qwen2.5-VL as the base model. The workflow includes training, merging adapters, converting to GGUF format, quantization, and pushing to the HuggingFace Hub.

---

## Table of Contents

- [Features](#features)
- [Requirements](#requirements)
- [Quickstart](#quickstart)
  - [1. Download Checkpoints and Dataset](#1-download-checkpoints-and-dataset)
  - [2. Environment Setup](#2-environment-setup)
  - [3. Training (LoRA SFT)](#3-training-lora-sft)
  - [4. Merge LoRA Adapter](#4-merge-lora-adapter)
  - [5. Convert to GGUF & Quantize](#5-convert-to-gguf--quantize)
  - [6. Push to HuggingFace Hub](#6-push-to-hf-hub)
- [Configuration](#configuration)
- [Scripts Overview](#scripts-overview)
- [References](#references)

---

## Features

- **LoRA-based SFT** for Qwen2.5-VL and similar VLMs.
- **Distributed training** with DeepSpeed Zero-3 and Accelerate.
- **WandB integration** for experiment tracking.
- **Adapter merging** and export to HuggingFace format.
- **Conversion to GGUF** for efficient inference and quantization.
- **Easy upload** to HuggingFace Hub.

---

## Requirements

- Python 3.8+
- CUDA-enabled GPU(s)
- [HuggingFace CLI](https://huggingface.co/docs/huggingface_hub/guides/cli)
- [WandB](https://wandb.ai/)
- [llama.cpp](https://github.com/dreamingbumblebee/llama.cpp) (for GGUF conversion)

Install Python dependencies:

```bash
pip install -r requirements.txt
```

**requirements.txt:**
```
transformers
trl
datasets
bitsandbytes
peft
qwen-vl-utils
wandb
accelerate
deepspeed
jupyter
```

---

## Quickstart
### 0. Make directories

```bash
bash 0.mkdir_dir.sh
```

### 1. Download Checkpoints and Dataset

Download the base model and dataset using the provided script:

```bash
python download_ckpt_data.py
```

This will download:
- Qwen2.5-VL-3B-Instruct (base model)
- KoLLaVA-Instruct-1.5k (Korean VLM dataset)

### 2. Environment Setup

Set your environment variables in a `.env` file:

```env
HF_TOKEN=your_huggingface_token
WANDB_API_KEY=your_wandb_api_key
```

Login to HuggingFace and WandB:

```bash
source .env
huggingface-cli login --token $HF_TOKEN
wandb login $WANDB_API_KEY
```

### 3. Training (LoRA SFT)

Run the training script with DeepSpeed and Accelerate:

```bash
bash 1.run_vlm_lora_sft.sh
```

This will:
- Train a LoRA adapter on the specified dataset and model.
- Save outputs to a directory named after the base model.

**Key script:** `1.run_vlm_lora_sft.py`  
**Config:** `deepspeed_zero3.yaml`

### 4. Merge LoRA Adapter

After training, merge the LoRA adapter with the base model:

```bash
bash 2.merge_vlm_lora_adapter.sh
```

This will produce a merged model in the `merged_model` subdirectory.

### 5. Convert to GGUF & Quantize

Convert the merged model to GGUF format and quantize for efficient inference:

```bash
bash 3.convert_lora_to_gguf_with_quantization.sh
```

- Installs and uses `llama.cpp` for conversion and quantization.
- Outputs quantized models in the `gguf` subdirectory.

### 6. Push to HF Hub

Upload the merged model to the HuggingFace Hub:

```bash
bash 4.push_to_hf_hub.sh
```

---

## Configuration

- **Distributed Training:** Controlled via `deepspeed_zero3.yaml`
- **Model/Dataset Paths:** Set in the shell scripts or via command-line arguments.
- **WandB Project:** Set via `WANDB_PROJECT` environment variable.

---

## Scripts Overview

- `download_ckpt_data.py` — Downloads base model and dataset from HuggingFace.
- `1.run_vlm_lora_sft.py` — Main training script for LoRA SFT.
- `1.run_vlm_lora_sft.sh` — Shell script to launch training with Accelerate/DeepSpeed.
- `2.merge_vlm_lora_adapter.py` — Merges LoRA adapter with base model.
- `2.merge_vlm_lora_adapter.sh` — Shell script for merging.
- `3.convert_lora_to_gguf_with_quantization.sh` — Converts merged model to GGUF and quantizes.
- `4.push_to_hf_hub.sh` — Uploads merged model to HuggingFace Hub.
- `deepspeed_zero3.yaml` — DeepSpeed Zero-3 configuration for distributed training.

---

## References

- [Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct)
- [KoLLaVA-Instruct-1.5k](https://huggingface.co/datasets/kihoonlee/KoLLaVA-Instruct-1.5k)
- [llama.cpp (dreamingbumblebee fork)](https://github.com/dreamingbumblebee/llama.cpp)
- [HuggingFace Transformers](https://github.com/huggingface/transformers)
- [WandB](https://wandb.ai/)

---

## Notes

- For custom datasets or models, adjust the paths in the shell scripts or pass them as arguments.
- The pipeline is designed for Korean VLM SFT but can be adapted for other languages or models with minor changes.
- For troubleshooting, check the logs and ensure all environment variables are set correctly.

---
