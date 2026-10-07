#!/usr/bin/env bash
# Lingshu-7B candidate sweep on Kaggle (backend: lingshu, Qwen2.5-VL family).
#
# Run from a Kaggle notebook (GPU enabled, 16GB class recommended for 7B) as:
#   !git clone https://github.com/shahedmomenzadeh/VLM-video-to-report
#   %cd VLM-video-to-report
#   !bash kaggle_run/lingshu-7b.sh
#
# Knobs (env): VIDEOS (default: pilot trio), MAX_FRAMES (default 32),
#   OUTZIP, HF_TOKEN (optional). 4-bit quant is on; the backend halves
#   frames automatically if VRAM runs out (see n_frames_used per row).
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL_ID="lingshu-medical-mllm/Lingshu-7B"
MODEL_DIR="models/Lingshu-7B"
MODEL_TAG="lingshu-7b"
BACKEND="lingshu"
VIDEOS="${VIDEOS:-all}"
MAX_FRAMES="${MAX_FRAMES:-32}"
OUTZIP="${OUTZIP:-/kaggle/working/${MODEL_TAG}-results.zip}"

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

echo ">>> model weights: $MODEL_ID"
python - <<PY
import os
from huggingface_hub import snapshot_download
snapshot_download(repo_id="$MODEL_ID", local_dir="$MODEL_DIR",
                  token=os.environ.get("HF_TOKEN") or None)
PY

if [ "$VIDEOS" = "all" ]; then
  mapfile -t VLIST < <(ls hf_dataset/videos/*.mp4 | xargs -n1 basename | sed 's/\.mp4$//' | sort)
else
  IFS=',' read -ra VLIST <<< "$VIDEOS"
fi
for VID in "${VLIST[@]}"; do
echo ">>> video: $VID"
echo ">>> inference: $MODEL_TAG x s1,s2,s3 (video=$VID)"
python src/vlm_report_framework/candidate_reports.py \
  --backend "$BACKEND" --model "$MODEL_DIR" --model-tag "$MODEL_TAG" \
  --videos "$VID" --settings s1,s2,s3 --max-frames "$MAX_FRAMES"

echo ">>> audit $VID"
python src/vlm_report_framework/verify_candidates.py --model-tag "$MODEL_TAG" || true

echo ">>> checkpoint zip -> $OUTZIP ($VID done)"
rm -f "$OUTZIP"
zip -qr "$OUTZIP" "output-candidates/$MODEL_TAG"
ls -lh "$OUTZIP"
done
echo "DONE. Download $OUTZIP from the notebook output panel."
