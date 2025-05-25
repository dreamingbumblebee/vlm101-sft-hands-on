# install llama.cpp
git clone https://github.com/dreamingbumblebee/llama.cpp.git
cd llama.cpp
pip install -r requirements.txt

# convert llm model to gguf
merged_model_dir=/work/vlm101-sft-hands-on/qwen2.5-3b-qlora-sft-ko-1.5k/merged_model
echo "convert llm model to gguf"
python convert_hf_to_gguf.py --outfile model.gguf --outtype f16 $merged_model_dir

echo "convert mmproj model to gguf"
python convert_hf_to_gguf.py --mmproj --outfile model_mmproj.gguf --outtype f16 $merged_model_dir 


# quantize llm model to Q4_K_M
apt update && apt install -y cmake libcurl4-openssl-dev

# build llama.cpp
# ref: https://github.com/dreamingbumblebee/llama.cpp/blob/master/docs/build.md#cuda
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release

./build/bin/llama-quantize ./model.gguf ./model-Q4_K_M.gguf Q4_K_M
# ./build/bin/llama-quantize ./model_mmproj.gguf ./model_mmproj-Q4_K_M.gguf Q4_K_M # vision encoder 아직 안되는것으로 예상됨

mkdir -p $merged_model_dir/gguf
mv *.gguf $merged_model_dir/gguf/

echo "done"