#!/usr/bin/env bash
# Qwen3-VL-2B family (base + SFT-stage2 + GRPO) on the 4 remaining videos.
#
# Run from a Kaggle notebook (GPU enabled) as:
#   !git clone https://github.com/shahedmomenzadeh/VLM-video-to-report
#   %cd VLM-video-to-report
#   !bash kaggle_run/qwen3vl-2b-remaining.sh
#
# Runs all three 2B models back-to-back on the 4 videos missing from the
# corpus (PH_0134/0136/0144/0146). After each model finishes, its weights are
# DELETED from disk (rm -rf) before the next model downloads — only one
# ~4GB model lives on disk at a time.
# Knobs (env): VIDEOS (default: the 4 missing), MAX_FRAMES (default 32),
#   OUTDIR (default /kaggle/working), HF_TOKEN (optional).
set -euo pipefail
cd "$(dirname "$0")/.."

# Force decord video reading in qwen-vl-utils: torchcodec chokes on
# sub-second clips and its hardcoded torchvision fallback calls
# io.read_video, removed in torchvision>=0.24 (Kaggle images).
export FORCE_QWENVL_VIDEO_READER=decord

VIDEOS="${VIDEOS:-PH_0134_2178_S1,PH_0136_2243_S1,PH_0144_2448_S1,PH_0146_2452_S1}"
MAX_FRAMES="${MAX_FRAMES:-32}"
OUTDIR="${OUTDIR:-/kaggle/working}"

echo ">>> GPU check"
if ! command -v nvidia-smi >/dev/null; then
  echo "ERROR: nvidia-smi not found -- this session has no GPU."
  echo "Enable one via Notebook settings: Settings > Accelerator > GPU, then rerun."
  exit 1
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo ">>> pip deps (torch first: Kaggle images usually ship CUDA torch)"
python -c "import torch, torchvision" 2>/dev/null \
  || pip install torch torchvision
pip install "transformers>=5.0" accelerate bitsandbytes qwen-vl-utils \
  decord pandas huggingface_hub
python -c "import torch; assert torch.cuda.is_available(), 'no CUDA'; \
  print('torch', torch.__version__, 'cuda OK')"

echo ">>> dataset: shahedm2001/cataract-video-to-report -> hf_dataset/"
python - <<'PY'
import os
from huggingface_hub import snapshot_download
snapshot_download(repo_id="shahedm2001/cataract-video-to-report",
                  repo_type="dataset", local_dir="hf_dataset",
                  token=os.environ.get("HF_TOKEN") or None)
PY

echo ">>> materialize repo layout (videos/timelines/instruments/clips)"
VIDEOS="$VIDEOS" python - <<'PY'
import os
from pathlib import Path
spec = os.environ["VIDEOS"]
if spec == "all":
    vids = sorted(p.stem for p in Path("hf_dataset/videos").glob("*.mp4"))
else:
    vids = [v.strip() for v in spec.split(",") if v.strip()]
for vid in vids:
    vd = Path("videos") / vid
    vd.mkdir(parents=True, exist_ok=True)
    for src, dst in [
            (f"hf_dataset/videos/{vid}.mp4", vd / f"{vid}.mp4"),
            (f"hf_dataset/timelines/{vid}.timeline.json", vd / f"{vid}.timeline.json"),
            (f"hf_dataset/instruments/{vid}_chunk_instruments.csv",
             Path("output-instruments") / f"{vid}_chunk_instruments.csv"),
            (f"hf_dataset/instruments/{vid}_instruments_clean.csv",
             Path("output-instruments") / f"{vid}_instruments_clean.csv")]:
        Path("output-instruments").mkdir(parents=True, exist_ok=True)
        dst.unlink(missing_ok=True)
        dst.symlink_to(Path(src).resolve())
    clipdir = Path("output-teacher/clips")
    clipdir.mkdir(parents=True, exist_ok=True)
    for c in sorted(Path("hf_dataset/clips").glob(f"{vid}__*.mp4")):
        link = clipdir / c.name
        link.unlink(missing_ok=True)
        link.symlink_to(c.resolve())
print("materialized:", vids)
PY

# model_id | local_dir | model_tag | essentials-only?
MODELS=(
  "Qwen/Qwen3-VL-2B-Instruct|models/Qwen3-VL-2B-Instruct|qwen3vl-2b-instruct|no"
  "shahedm2001/qwen3-vl-2b-cataract-sft-stage2|models/qwen3-vl-2b-cataract-sft-stage2|qwen3vl-2b-sft-stage2|yes"
  "shahedm2001/qwen3-vl-2b-cataract-grpo|models/qwen3-vl-2b-cataract-grpo|qwen3vl-2b-grpo|yes"
)

IFS=',' read -ra VLIST <<< "$VIDEOS"
for entry in "${MODELS[@]}"; do
  IFS='|' read -r MODEL_ID MODEL_DIR MODEL_TAG ESS <<< "$entry"
  echo ">>> model weights: $MODEL_ID"
  if [ "$ESS" = "yes" ]; then
    python - <<PY
import os
from huggingface_hub import snapshot_download
snapshot_download(repo_id="$MODEL_ID", local_dir="$MODEL_DIR",
                  allow_patterns=["config.json", "generation_config.json",
                                  "model.safetensors", "processor_config.json",
                                  "tokenizer.json", "tokenizer_config.json",
                                  "chat_template.jinja"],
                  token=os.environ.get("HF_TOKEN") or None)
PY
  else
    python - <<PY
import os
from huggingface_hub import snapshot_download
snapshot_download(repo_id="$MODEL_ID", local_dir="$MODEL_DIR",
                  token=os.environ.get("HF_TOKEN") or None)
PY
  fi
  for VID in "${VLIST[@]}"; do
    echo ">>> inference: $MODEL_TAG x s1,s2,s3 (video=$VID)"
    python src/vlm_report_framework/candidate_reports.py \
      --backend qwen3vl --model "$MODEL_DIR" --model-tag "$MODEL_TAG" \
      --videos "$VID" --settings s1,s2,s3 --max-frames "$MAX_FRAMES"
    echo ">>> audit $VID ($MODEL_TAG)"
    python src/vlm_report_framework/verify_candidates.py --model-tag "$MODEL_TAG" || true
    OUTZIP="$OUTDIR/${MODEL_TAG}-remaining.zip"
    echo ">>> checkpoint zip -> $OUTZIP ($VID done)"
    rm -f "$OUTZIP"
    zip -qr "$OUTZIP" "output-candidates/$MODEL_TAG"
    ls -lh "$OUTZIP"
  done
  echo ">>> deleting weights to free disk: $MODEL_DIR"
  rm -rf "$MODEL_DIR"
  df -h "$OUTDIR" | tail -n 1
done
echo "DONE. Download the *-remaining.zip files from the notebook output panel."
