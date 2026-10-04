#!/usr/bin/env bash
# End-to-end Part I pipeline: YOLO -> instruments -> cleaning -> teacher VLM.
#
#   bash run_pipeline.sh                                # default 3-video sample, all stages
#   bash run_pipeline.sh --all-videos                   # all videos/ dirs, all stages
#   bash run_pipeline.sh --all-videos --skip-segment    # all videos, no annotated AVIs
#   RESUME=1 bash run_pipeline.sh ...                   # keep existing outputs, resume
#
# Stages:
#   1. segment_videos.py      YOLO segmentation -> output-segmented/<VID>/<VID>.avi
#                             (skip with --skip-segment; per-video skip if AVI exists)
#   2. extract_instruments.py per-frame instrument CSVs -> output-instruments/*_raw.csv
#                             (per-video skip if raw CSV exists)
#   3. clean_instruments.py   3-layer cleaning + chunk I(i,j) -> *_clean.csv, *_chunk_instruments.csv
#   4. teacher_reports.py     teacher VLM per chunk -> output-teacher/ (needs localhost:20128 up;
#                             per-chunk skip if already in master table)
set -euo pipefail
cd "$(dirname "$0")"

ALL_VIDEOS=0
SKIP_SEGMENT=0
for arg in "$@"; do
  case "$arg" in
    --all-videos) ALL_VIDEOS=1 ;;
    --skip-segment) SKIP_SEGMENT=1 ;;
    *) echo "Unknown flag: $arg (expected --all-videos, --skip-segment)"; exit 1 ;;
  esac
done

if [ "$ALL_VIDEOS" = "1" ]; then
  mapfile -t VIDS < <(ls -d videos/PH_*/ | xargs -n1 basename | sort)
  VIDS_CSV=$(IFS=,; echo "${VIDS[*]}")
  echo ">>> Videos: ${#VIDS[@]} (all videos/ dirs)"
else
  VIDS=(PH_0001_2931_S2 PH_0043_0096_S1 PH_0057_0239_S1)
  VIDS_CSV="PH_0001_2931_S2,PH_0043_0096_S1,PH_0057_0239_S1"
  echo ">>> Videos: 3 (default sample)"
fi

PY=".venv/bin/python"

if [ "${RESUME:-0}" != "1" ] && [ "$ALL_VIDEOS" != "1" ]; then
  echo ">>> Fresh run: clearing generated outputs"
  rm -rf output-segmented output-instruments output-teacher
else
  echo ">>> Resume mode: keeping existing outputs (stages skip finished work)"
fi
mkdir -p output-segmented output-instruments output-teacher

if [ "$SKIP_SEGMENT" = "1" ]; then
  echo ">>> [1/4] SKIPPED (--skip-segment: no annotated videos saved)"
else
  echo ">>> [1/4] YOLO segmentation (annotated videos)"
  $PY src/vlm_report_framework/segment_videos.py --videos "$VIDS_CSV"
fi

echo ">>> [2/4] Instrument extraction to CSV"
$PY src/vlm_report_framework/extract_instruments.py --videos "$VIDS_CSV"

echo ">>> [3/4] Cleaning + per-chunk aggregation"
$PY src/vlm_report_framework/clean_instruments.py --videos "$VIDS_CSV"

echo ">>> [4/4] Teacher reports (VLM, one video at a time)"
if ! curl -s -m 5 http://localhost:20128/v1/models > /dev/null; then
  echo "ERROR: VLM endpoint http://localhost:20128/v1 is not reachable. Aborting before stage 4."
  exit 1
fi
for v in "${VIDS[@]}"; do
  echo "--- video: $v"
  $PY src/vlm_report_framework/teacher_reports.py --video "$v" --all
done

echo ">>> Pipeline complete."
ls output-teacher/*_teacher_reports.jsonl
