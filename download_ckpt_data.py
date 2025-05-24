from huggingface_hub import snapshot_download

snapshot_download(repo_id="Qwen/Qwen2.5-VL-3B-Instruct",
      local_dir="/work/checkpoints/hf/Qwen2.5-VL-3B-Instruct",
      local_dir_use_symlinks=False)

snapshot_download("kihoonlee/KoLLaVA-Instruct-1.5k", 
                repo_type="dataset",
                local_dir="/work/dataset/vlm/KoLLaVA-Instruct-1.5k",
                local_dir_use_symlinks=False)