source .env
huggingface-cli login --token $HF_TOKEN --add-to-git-credential

repo_id=qwen2.5-3b-qlora-sft-ko-1.5k-with-gguf
local_path=/work/vlm101-sft-hands-on/qwen2.5-3b-qlora-sft-ko-1.5k/merged_model

huggingface-cli repo create $repo_id --type model -y
huggingface-cli upload $repo_id $local_path .
