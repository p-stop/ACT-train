"""Push the augmented datasets to private Hugging Face dataset repos (with the v3.0 tag lerobot looks for)."""
import sys
from pathlib import Path
from huggingface_hub import HfApi

api = HfApi()
for name in sys.argv[1:]:
    root = Path.home() / "zero-wam/PICO-ACT/dataset/lerobot" / name
    repo = f"<HF_USER>/{name}"
    api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
    print("uploading", root, "->", repo, flush=True)
    api.upload_large_folder(repo_id=repo, folder_path=str(root), repo_type="dataset",
                            ignore_patterns=["images/**", "*.tmp"], print_report=False)
    try:
        api.create_tag(repo, tag="v3.0", repo_type="dataset")
    except Exception as e:  # tag already there
        print("tag:", e)
    print("done", repo, "private:", api.repo_info(repo, repo_type="dataset").private, flush=True)
