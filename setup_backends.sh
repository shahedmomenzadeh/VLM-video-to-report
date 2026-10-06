#!/usr/bin/env bash
# Create one isolated venv per local-VLM family (they need incompatible
# dependency versions). Pins mirror E:/VLM_Evaluation/run.sh.
#
#   bash setup_backends.sh                  # create all venvs
#   bash setup_backends.sh --backend hulumed # just one: qwen3vl | hulumed
#
# Mapping (see src/vlm_report_framework/backends/):
#   qwen3vl -> Qwen3-VL-2B (+SFT/GRPO) AND Lingshu-7B (same processor family)
#   hulumed -> Hulu-Med-4B / 7B (pinned transformers 4.51.2 + compat patch)
set -euo pipefail
cd "$(dirname "$0")"

BACKEND="${1:---all}"
if [ "$BACKEND" == "--backend" ]; then BACKEND="$2"; fi

make_venv() {
  local dir="$1"
  if [ ! -d "$dir" ]; then
    uv venv "$dir" --python 3.12
  else
    echo "reuse $dir"
  fi
}

if [ "$BACKEND" == "--all" ] || [ "$BACKEND" == "qwen3vl" ]; then
  echo ">>> venv: qwen3vl (+lingshu)"
  make_venv .venv-qwen3vl
  uv pip install --python .venv-qwen3vl/bin/python \
      torch torchvision \
      --index-url https://download.pytorch.org/whl/cu130
  uv pip install --python .venv-qwen3vl/bin/python \
      "git+https://github.com/huggingface/transformers.git" \
      "accelerate" \
      "bitsandbytes>=0.43.0" \
      "qwen-vl-utils[decord]" \
      "pandas" \
      "openai" \
      "tqdm" \
      "imageio"
fi

if [ "$BACKEND" == "--all" ] || [ "$BACKEND" == "hulumed" ]; then
  echo ">>> venv: hulumed"
  make_venv .venv-hulumed
  uv pip install --python .venv-hulumed/bin/python \
      torch torchvision \
      --index-url https://download.pytorch.org/whl/cu130
  uv pip install --python .venv-hulumed/bin/python \
      "transformers==4.51.2" \
      "accelerate==1.7.0" \
      "bitsandbytes>=0.43.0" \
      "ffmpeg-python" \
      "decord" \
      "opencv-python" \
      "Pillow" \
      "pandas" \
      "openai" \
      "tqdm" \
      "imageio"
fi

echo "done. Run candidates per backend, e.g.:"
echo "  .venv-qwen3vl/bin/python src/vlm_report_framework/candidate_reports.py --backend qwen3vl --model <path> --model-tag qwen3vl-2b-instruct --videos all"
echo "  .venv-hulumed/bin/python src/vlm_report_framework/candidate_reports.py --backend hulumed --model ZJU-AI4H/Hulu-Med-4B --model-tag hulumed-4b --videos all --frame-size 480 --temperature 0.6"
echo "  .venv-qwen3vl/bin/python src/vlm_report_framework/candidate_reports.py --backend lingshu --model lingshu-medical-mllm/Lingshu-7B --model-tag lingshu-7b --videos all"
