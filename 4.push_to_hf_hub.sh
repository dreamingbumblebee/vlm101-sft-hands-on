source .env
huggingface-cli login --token $HF_TOKEN --add-to-git-credential

repo_id=Qwen2.5-VL-3B-Instruct-qlora-sft-ko-1.5k
local_path=/work/code/vlm_sft/Qwen2.5-VL-3B-Instruct-qlora-sft-ko-1.5k/merged_model

huggingface-cli repo create $repo_id --type model -y
huggingface-cli upload $repo_id $local_path .
