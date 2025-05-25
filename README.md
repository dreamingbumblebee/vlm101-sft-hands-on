# VLM101 SFT Hands-on

이 프로젝트는 Vision-Language Models (VLMs)의 Supervised Fine-Tuning (SFT)을 위한 종합적인 실습 파이프라인입니다. Qwen2.5-VL을 기본 모델로 사용하여 LoRA (Low-Rank Adaptation) 기반 학습, MCP (Model Context Protocol) 서버, 그리고 Ollama를 통한 배포까지 포함합니다.

---

## 📋 목차

- [프로젝트 개요](#프로젝트-개요)
- [주요 기능](#주요-기능)
- [시스템 아키텍처](#시스템-아키텍처)
- [요구사항](#요구사항)
- [빠른 시작](#빠른-시작)
- [상세 가이드](#상세-가이드)
- [MCP 서버](#mcp-서버)
- [Ollama 배포](#ollama-배포)
- [문제 해결](#문제-해결)
- [참고 자료](#참고-자료)

---

## 🎯 프로젝트 개요

이 프로젝트는 다음과 같은 완전한 VLM 학습 및 배포 파이프라인을 제공합니다:

### 🔄 학습 파이프라인
- **LoRA 기반 SFT**: Qwen2.5-VL 모델의 효율적인 미세조정
- **분산 학습**: DeepSpeed Zero-3와 Accelerate를 통한 대규모 학습
- **실험 추적**: WandB를 통한 학습 과정 모니터링
- **한국어 특화**: KoLLaVA-Instruct-1.5k 데이터셋 활용

### 🚀 배포 파이프라인
- **MCP 서버**: 도구 연동을 위한 Model Context Protocol 서버
- **Ollama 배포**: 로컬 추론을 위한 Ollama 모델 변환
- **GGUF 변환**: 효율적인 추론을 위한 양자화 및 포맷 변환

---

## ✨ 주요 기능

### 🎓 학습 기능
- **LoRA 기반 SFT**: 메모리 효율적인 파라미터 효율적 미세조정
- **분산 학습**: DeepSpeed Zero-3를 통한 대규모 GPU 활용
- **WandB 통합**: 실시간 학습 과정 추적 및 시각화
- **한국어 특화**: 한국어 VLM 데이터셋 활용

### 🔧 MCP 서버
- **Weather MCP Server**: 실시간 날씨 정보 제공
- **Stock MCP Server**: 주식 데이터 및 차트 정보
- **MCP Proxy**: 여러 MCP 서버 통합 관리
- **Open WebUI 연동**: 웹 기반 AI 채팅 인터페이스

### 🐳 Ollama 배포
- **GGUF 변환**: llama.cpp를 통한 효율적인 모델 변환
- **양자화**: Q4_K_M 양자화를 통한 메모리 최적화
- **로컬 추론**: Ollama를 통한 간편한 로컬 실행

---

## 🏗️ 시스템 아키텍처

```
┌─────────────────────────────────────────────────────────────────┐
│                        VLM101 SFT Pipeline                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐          │
│  │   Training  │    │   MCP       │    │   Ollama    │          │
│  │   Pipeline  │    │   Servers   │    │   Deploy    │          │
│  └─────────────┘    └─────────────┘    └─────────────┘          │
│         │                   │                   │               │
│         ▼                   ▼                   ▼               │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐          │
│  │ LoRA SFT    │    │ Weather     │    │ GGUF        │          │
│  │ DeepSpeed   │    │ Stock       │    │ Quantize    │          │
│  │ WandB       │    │ Proxy       │    │ Ollama      │          │
│  └─────────────┘    └─────────────┘    └─────────────┘          │
│         │                   │                   │               │
│         ▼                   ▼                   ▼               │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐          │
│  │ Merged      │    │ Open WebUI  │    │ Local       │          │
│  │ Model       │    │ Integration │    │ Inference   │          │
│  └─────────────┘    └─────────────┘    └─────────────┘          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📋 요구사항

### 시스템 요구사항
- **Python**: 3.8+
- **GPU**: CUDA 지원 GPU (최소 40GB VRAM 권장)
- **메모리**: 최소 16GB RAM
- **저장공간**: 최소 50GB 여유 공간

### 필수 소프트웨어
- [HuggingFace CLI](https://huggingface.co/docs/huggingface_hub/guides/cli)
- [WandB](https://wandb.ai/)
- [Docker](https://www.docker.com/) (MCP 서버용)
- [Ollama](https://ollama.ai/) (로컬 추론용)
- [llama.cpp](https://github.com/dreamingbumblebee/llama.cpp) (GGUF 포멧 변환용)

### Python 의존성 설치

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
python-dotenv
```

---

## 🚀 빠른 시작

### 0. 디렉토리 생성

```bash
bash 0.mkdir_dir.sh
```

### 1. 체크포인트 및 데이터셋 다운로드

```bash
python download_ckpt_data.py
```

다운로드되는 항목:
- **Qwen2.5-VL-3B-Instruct**: 기본 VLM 모델
- **KoLLaVA-Instruct-1.5k**: 한국어 VLM 데이터셋

### 2. 환경 설정

`.env` 파일에 환경 변수 설정:

```env
HF_TOKEN=your_huggingface_token
WANDB_API_KEY=your_wandb_api_key
```

로그인:

```bash
source .env
huggingface-cli login --token $HF_TOKEN
wandb login $WANDB_API_KEY
```

### 3. LoRA SFT 학습

```bash
bash 1.run_vlm_lora_sft.sh
```

### 4. LoRA 어댑터 병합

```bash
bash 2.merge_vlm_lora_adapter.sh
```

### 5. GGUF 변환 및 양자화

```bash
bash 3.convert_lora_to_gguf_with_quantization.sh
```

### 6. HuggingFace Hub 업로드

```bash
bash 4.push_to_hf_hub.sh
```

---

## 📚 상세 가이드

### 학습 파이프라인

#### 1. LoRA SFT 학습 (`1.run_vlm_lora_sft.py`)

**주요 특징:**
- 4-bit 양자화 (BitsAndBytes)
- DeepSpeed Zero-3 분산 학습
- WandB 실험 추적
- 한국어 특화 시스템 메시지

**학습 설정:**
```python
# LoRA 설정
lora_alpha=16
lora_dropout=0.05
r=8
target_modules=["q_proj", "v_proj"]

# 학습 설정
per_device_train_batch_size=4
gradient_accumulation_steps=4
learning_rate=2e-04
num_train_epochs=2
```

#### 2. 어댑터 병합 (`2.merge_vlm_lora_adapter.py`)

LoRA 어댑터를 기본 모델과 병합하여 독립 실행 가능한 모델 생성.

#### 3. GGUF 변환 (`3.convert_lora_to_gguf_with_quantization.sh`)

- llama.cpp를 사용한 GGUF 포맷 변환
- Q4_K_M 양자화로 메모리 최적화
- 멀티모달 프로젝션 모델 포함

---

## 🔧 MCP 서버

### MCP 서버 실행

```bash
# 전체 MCP 시스템 시작
cd mcp
./start_docker_compose.sh

# 또는 개별 서비스 실행
docker compose --profile tools up -d
```

### 서비스 접속

- **Open WebUI**: http://localhost:3000
- **MCP Proxy**: http://localhost:8000
- **Weather MCP**: http://localhost:8002
- **Stock MCP**: http://localhost:8003

### MCP 서버 구성

#### Weather MCP Server
- **기능**: 날씨 예보 및 경보 조회
- **API**: `get_alerts`, `get_forecast`
- **데이터 소스**: National Weather Service, OpenWeatherMap

#### Stock MCP Server
- **기능**: 실시간 주식 데이터 및 차트
- **API**: 주식 정보, 차트 데이터
- **데이터 소스**: Yahoo Finance, Alpha Vantage

#### MCP Proxy
- **기능**: 여러 MCP 서버 통합 관리
- **내장 서버**: Fetch, Time (한국 시간대)

### MCP 서버 사용 예시

```python
import requests

# 날씨 정보 조회
response = requests.post(
    "http://localhost:8002/get_forecast",
    json={"latitude": 37.5665, "longitude": 126.9780}
)

# 주식 정보 조회
response = requests.post(
    "http://localhost:8003/get_stock_info",
    json={"symbol": "AAPL"}
)
```

---

## 🐳 Ollama 배포

### Ollama 모델 생성

```bash
cd ollama

# GGUF 파일을 Ollama 모델로 변환
ollama create qwen2.5-vlm-sft -f Modelfile

# 모델 실행
ollama run qwen2.5-vlm-sft
```

### Modelfile 예시

```dockerfile
FROM /path/to/model.gguf
TEMPLATE """{{ if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}{{ if .Prompt }}<|im_start|>user
{{ .Prompt }}<|im_end|>
<|im_start|>assistant
{{ end }}"""
PARAMETER stop "<|im_end|>"
PARAMETER stop "<|im_start|>"
PARAMETER stop "user"
PARAMETER stop "assistant"
PARAMETER stop "system"
```

### Ollama API 사용

```python
import requests

# 이미지와 함께 추론
response = requests.post(
    "http://localhost:11434/api/generate",
    json={
        "model": "qwen2.5-vlm-sft",
        "prompt": "이 이미지를 설명해주세요",
        "images": ["base64_encoded_image"]
    }
)
```

---

## 🐛 문제 해결

### 학습 관련 문제

#### 메모리 부족
```bash
# 배치 크기 줄이기
--per_device_train_batch_size 2
--gradient_accumulation_steps 8

# 더 강한 양자화
bnb_4bit_quant_type="nf4"
bnb_4bit_compute_dtype=torch.float16
```

#### 분산 학습 오류
```bash
# DeepSpeed 설정 확인
accelerate config

# 단일 GPU로 실행
accelerate launch --num_processes 1 1.run_vlm_lora_sft.py
```

### MCP 서버 문제

#### 포트 충돌
```bash
# 포트 사용 확인
netstat -tulpn | grep :800

# 다른 포트로 실행
docker compose -f docker-compose.yml -p mcp-custom up -d
```

#### 서버 연결 실패
```bash
# 로그 확인
docker logs mcp-proxy
docker logs weather-mcp
docker logs stock-mcp

# 컨테이너 재시작
docker compose restart
```

### Ollama 문제

#### 모델 로드 실패
```bash
# 모델 정보 확인
ollama list

# 모델 재생성
ollama rm qwen2.5-vlm-sft
ollama create qwen2.5-vlm-sft -f Modelfile
```

#### GGUF 변환 오류
```bash
# llama.cpp 빌드 확인
cd llama.cpp
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release
```

---

## 📁 프로젝트 구조

```
vlm101-sft-hands-on/
├── 📁 mcp/                          # MCP 서버 관련
│   ├── 📁 mcp-proxy/               # MCP 프록시 서버
│   ├── 📁 mcp-servers/             # MCP 서버들
│   │   ├── 📁 weather/             # 날씨 MCP 서버
│   │   └── 📁 stock/               # 주식 MCP 서버
│   ├── docker-compose.yml          # Docker Compose 설정
│   └── README.md                   # MCP 서버 문서
├── 📁 ollama/                      # Ollama 배포 관련
│   ├── 📁 llama.cpp/               # llama.cpp (포크)
│   ├── 📁 qwen2.5-3b-qlora-sft-ko-1.5k-with-gguf/
│   └── quantize_and_build_ollama.sh
├── 📄 0.mkdir_dir.sh               # 디렉토리 생성
├── 📄 1.run_vlm_lora_sft.py        # LoRA SFT 학습 스크립트
├── 📄 1.run_vlm_lora_sft.sh        # 학습 실행 스크립트
├── 📄 2.merge_vlm_lora_adapter.py  # 어댑터 병합
├── 📄 2.merge_vlm_lora_adapter.sh  # 병합 실행 스크립트
├── 📄 3.convert_lora_to_gguf_with_quantization.sh
├── 📄 4.push_to_hf_hub.sh          # HF Hub 업로드
├── 📄 deepspeed_zero3.yaml         # DeepSpeed 설정
├── 📄 download_ckpt_data.py        # 모델/데이터셋 다운로드
├── 📄 requirements.txt              # Python 의존성
└── 📄 README.md                     # 이 파일
```

---

## 📚 참고 자료

### 모델 및 데이터셋
- [Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct)
- [KoLLaVA-Instruct-1.5k](https://huggingface.co/datasets/kihoonlee/KoLLaVA-Instruct-1.5k)

### 기술 문서
- [HuggingFace Transformers](https://github.com/huggingface/transformers)
- [PEFT (Parameter-Efficient Fine-Tuning)](https://github.com/huggingface/peft)
- [TRL (Transformer Reinforcement Learning)](https://github.com/huggingface/trl)
- [DeepSpeed](https://github.com/microsoft/DeepSpeed)

### MCP 관련
- [Model Context Protocol](https://modelcontextprotocol.io/)
- [MCPO](https://github.com/jlowin/mcpo)
- [Open WebUI](https://github.com/open-webui/open-webui)

### Ollama 관련
- [Ollama](https://ollama.ai/)
- [llama.cpp (dreamingbumblebee fork)](https://github.com/dreamingbumblebee/llama.cpp)

### 실험 추적
- [WandB](https://wandb.ai/)

---

## 🤝 기여하기

1. 이슈 생성 또는 기존 이슈 확인
2. 포크 생성
3. 기능 브랜치 생성 (`git checkout -b feature/amazing-feature`)
4. 변경사항 커밋 (`git commit -m 'Add amazing feature'`)
5. 브랜치에 푸시 (`git push origin feature/amazing-feature`)
6. Pull Request 생성

---

## 📄 라이선스

이 프로젝트는 Apache License 2.0 하에 배포됩니다.

---

## 📝 변경 이력

- **v2.0.0**: MCP 서버 및 Ollama 배포 기능 추가
- **v1.5.0**: GGUF 변환 및 양자화 기능 추가
- **v1.4.0**: WandB 통합 및 실험 추적
- **v1.3.0**: DeepSpeed Zero-3 분산 학습 지원
- **v1.2.0**: 한국어 특화 시스템 메시지
- **v1.1.0**: LoRA 기반 SFT 구현
- **v1.0.0**: 초기 릴리스

---

## 📞 지원

문제가 발생하거나 질문이 있으시면:
1. [GitHub Issues](https://github.com/your-repo/vlm101-sft-hands-on/issues)에서 이슈 생성
2. 프로젝트 문서 확인
3. 참고 자료 섹션의 링크 참조

---
