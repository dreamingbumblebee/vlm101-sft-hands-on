# llamacpp 설치
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
git clone https://github.com/dreamingbumblebee/llama.cpp.git
cd llama.cpp
# pip install -r requirements.txt

# 모델 변환
# merged_model_dir=<vlm_sft_model_dir>
# python convert_hf_to_gguf.py --outfile model.gguf --outtype f16 $merged_model_dir
# python convert_hf_to_gguf.py --mmproj --outfile model_mmproj.gguf --outtype f16 $merged_model_dir 

# llama.cpp CPU 빌드 
# https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md#cpu-build
brew install cmake
cmake -B build
cmake --build build --config Release

# 업로드할 ollama 디렉토리 생성
mkdir -p $SCRIPT_DIR/ollama_upload
cp $SCRIPT_DIR/qwen2.5-3b-qlora-sft-ko-1.5k-with-gguf/gguf/*.gguf $SCRIPT_DIR/ollama_upload

# 작성 시점 기준 LLM만 quantization 가능
$SCRIPT_DIR/llama.cpp/build/bin/llama-quantize $SCRIPT_DIR/ollama_upload/model.gguf $SCRIPT_DIR/ollama_upload/model-Q4_K_M.gguf Q4_K_M

cd $SCRIPT_DIR/ollama_upload

# 기존 Modelfile 참고를 위해 복사
ollama show qwen3:1.7b --modelfile > Modelfile

FILE="Modelfile"
BACKUP="${FILE}.bak"

# 백업 먼저 생성
cp "$FILE" "$BACKUP"

# 첫 번째 FROM 라인의 줄 번호 찾기
FROM_LINE=$(grep -n '^FROM ' "$FILE" | head -n 1 | cut -d: -f1)

if [ -z "$FROM_LINE" ]; then
  echo "FROM 줄을 찾을 수 없습니다."
  exit 1
fi
# Qunatization 모델을 쓰려면 model.gguf 대신 ./model-Q4_K_M.gguf 사용
# 새로운 내용 (따옴표에 주의)
NEW_BLOCK=$(cat <<'EOF'
FROM ./model-Q4_K_M.gguf
FROM ./model_mmproj.gguf

SYSTEM """You are a Vision Language Model specialized in Korean Language.
You are given a question and an image.
You need to answer the question based on the image.
You need to answer in Korean.
You need to answer in a way that is easy to understand.
You need to answer in a Vision Language Model tone.
"""
EOF
)

# 기존 파일을 앞, 중간(대체), 뒤로 나눠서 다시 작성
{
  head -n $((FROM_LINE - 1)) "$FILE"
  echo "$NEW_BLOCK"
  tail -n +"$((FROM_LINE + 1))" "$FILE"
} > "${FILE}.tmp" && mv "${FILE}.tmp" "$FILE"

echo "✅ 첫 번째 FROM 줄을 성공적으로 교체했습니다. 백업: $BACKUP"

# 모델 생성
ollama create dreamingBumblebee/qwen2.5vl-3b-qlora-ko-1.5k-quant-test -f Modelfile


# # 모델 실행
ollama run dreamingBumblebee/qwen2.5vl-3b-qlora-ko-1.5k-quant-test


# # 모델 푸시
# ollama push dreamingbumblebee/qwen2.5vl-3b-qlora-ko-1.5k:latest