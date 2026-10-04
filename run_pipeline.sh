#!/usr/bin/env bash
# End-to-end Part I pipeline: YOLO -> instruments -> cleaning -> teacher VLM.
#
#   bash run_pipeline.sh           # fresh run from scratch (wipes generated outputs)
#   RESUME=1 bash run_pipeline.sh  # keep existing outputs, resume where possible
#
# Stages:
#   1. segment_videos.py      YOLO segmentation -> output-segmented/<VID>/<VID>.avi
#   2. extract_instruments.py per-frame instrument CSVs -> output-instruments/*_raw.csv
#   3. clean_instruments.py   3-layer cleaning + chunk I(i,j) -> *_clean.csv, *_chunk_instruments.csv
#   4. teacher_reports.py     teacher VLM per chunk -> output-teacher/ (needs localhost:20128 up)
set -euo pipefail
cd "$(dirname "$0")"

VIDEOS="PH_0001_2931_S2 PH_0043_0096_S1 PH_0057_0239_S1"
PY=".venv/bin/python"

if [ "${RESUME:-0}" != "1" ]; then
  echo ">>> Fresh run: clearing generated outputs"
  rm -rf output-segmented output-instruments output-teacher
else
  echo ">>> Resume mode: keeping existing outputs"
fi
mkdir -p output-segmented output-instruments output-teacher

echo ">>> [1/4] YOLO segmentation (annotated videos)"
$PY src/vlm_report_framework/segment_videos.py

echo ">>> [2/4] Instrument extraction to CSV"
$PY src/vlm_report_framework/extract_instruments.py

echo ">>> [3/4] Cleaning + per-chunk aggregation"
$PY src/vlm_report_framework/clean_instruments.py

echo ">>> [4/4] Teacher reports (VLM, one video at a time)"
if ! curl -s -m 5 http://localhost:20128/v1/models > /dev/null; then
  echo "ERROR: VLM endpoint http://localhost:20128/v1 is not reachable. Aborting before stage 4."
  exit 1
fi
for v in $VIDEOS; do
  echo "--- video: $v"
  $PY src/vlm_report_framework/teacher_reports.py --video "$v" --all
done

echo ">>> Pipeline complete."
ls output-teacher/*_teacher_reports.jsonl
