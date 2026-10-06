"""Upload the staged dataset to the Hugging Face Hub.

Staging is built first (run once):
  uv run python src/vlm_report_framework/build_hf_dataset.py   # -> hf_dataset/

Then upload (you run this yourself; needs `hf auth login` beforehand):
  uv run python src/vlm_report_framework/upload_hf_dataset.py
  uv run python src/vlm_report_framework/upload_hf_dataset.py \\
      --repo shahedm2001/cataract-video-to-report --private

The repo holds Part I teacher-side data only (videos, clips, timelines,
YOLO instruments, teacher reports + memory, parquet indexes). Candidate
(small-model) outputs are never uploaded.
"""
from __future__ import annotations

import argparse
from pathlib import Path

REPO = Path.cwd()
STAGE = REPO / "hf_dataset"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="shahedm2001/cataract-video-to-report")
    ap.add_argument("--private", action="store_true",
                    help="create as private (default: public)")
    args = ap.parse_args()
    assert STAGE.exists(), "run build_hf_dataset.py first (hf_dataset/ missing)"

    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(args.repo, repo_type="dataset",
                    private=args.private, exist_ok=True)
    url = api.upload_folder(folder_path=str(STAGE), repo_id=args.repo,
                            repo_type="dataset")
    print(f"uploaded {STAGE} -> {url}")


if __name__ == "__main__":
    main()
