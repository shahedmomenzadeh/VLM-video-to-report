"""Stage the cataract-video-to-report HF dataset (Part I modalities).

Layout (repo: shahedm2001/cataract-video-to-report, public):
  videos/<VID>.mp4                        full surgeries (25)
  clips/<VID>__<chunk>.mp4                frozen VLM chunks (849)
  timelines/<VID>.timeline.json           phase segments
  instruments/<VID>_instruments_clean.csv per-frame clean detections
  instruments/<VID>_chunk_instruments.csv per-(chunk,class) tiers
  teacher_reports/<VID>_teacher_reports.jsonl  reference reports
  memory/<VID>_memory.jsonl               causal memory snapshots
  chunks.parquet                          master chunk index (849 rows)
  chunk_instruments.parquet               chunk x class detail rows
  splits.json                             video-level train/val/test
  README.md                               dataset card

Usage: uv run python src/vlm_report_framework/build_hf_dataset.py
Output: hf_dataset/ (git-ignored staging dir for upload_folder).
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd

REPO = Path.cwd()
STAGE = REPO / "hf_dataset"

# Video-level split (no chunk leakage across splits). Test = the pilot trio.
TEST = ["PH_0001_2931_S2", "PH_0043_0096_S1", "PH_0057_0239_S1"]
VAL = ["PH_0002_2932_S2", "PH_0064_0667_S1", "PH_0070_0882_S1"]


def split_of(vid: str, all_vids: list[str]) -> str:
    if vid in TEST:
        return "test"
    if vid in VAL:
        return "validation"
    assert vid in all_vids, vid
    return "train"


CARD = """---
license: other
task_categories: [video-text-to-text]
tags: [surgery, cataract, surgical-video, report-generation]
---

# cataract-video-to-report

Frozen video chunks of 25 cataract surgeries with phase timestamps,
YOLO instrument detections, and teacher-VLM reference reports — the Part I
dataset of the VLM-video-to-report framework
(https://github.com/shahedmomenzadeh/VLM-video-to-report).

Each surgery is split into phase chunks (`videos/<VID>/<VID>.timeline.json`
segments; segments over ~1 min are split into `*_c1/c2/...` subchunks, max
240 s). Every chunk has: the clip (`clips/`), phase label, instrument tiers
from a YOLO segmentation model (`instruments/`), and a teacher reference
report with causal memory (`teacher_reports/`, `memory/`).

`chunks.parquet` is the master index (one row per chunk); `splits.json`
assigns whole videos to train/validation/test (no chunk leakage).
Intended use: run candidate VLMs on identical `clips/` under
video-only / phase-informed / phase+instrument-informed conditions.
"""


def main() -> None:
    if STAGE.exists():
        shutil.rmtree(STAGE)
    for d in ["videos", "clips", "timelines", "instruments",
              "teacher_reports", "memory"]:
        (STAGE / d).mkdir(parents=True)

    vids = sorted(p.name for p in (REPO / "videos").glob("PH_*") if p.is_dir())
    chunk_rows, inst_frames = [], []
    for vid in vids:
        shutil.copy(REPO / "videos" / vid / f"{vid}.mp4", STAGE / "videos")
        shutil.copy(REPO / "videos" / vid / f"{vid}.timeline.json",
                    STAGE / "timelines")
        for name in (f"{vid}_instruments_clean.csv",
                     f"{vid}_chunk_instruments.csv"):
            shutil.copy(REPO / "output-instruments" / name,
                        STAGE / "instruments")
        shutil.copy(REPO / "output-teacher" / f"{vid}_teacher_reports.jsonl",
                    STAGE / "teacher_reports")
        shutil.copy(REPO / "output-teacher" / "memory" / f"{vid}_memory.jsonl",
                    STAGE / "memory")
        ci = pd.read_csv(REPO / "output-instruments" / f"{vid}_chunk_instruments.csv")
        inst_frames.append(ci)
        for line in open(REPO / "output-teacher" / f"{vid}_teacher_reports.jsonl"):
            r = json.loads(line)
            clip = f"clips/{vid}__{r['chunk_id']}.mp4"
            src = REPO / "output-teacher" / "clips" / f"{vid}__{r['chunk_id']}.mp4"
            assert src.exists(), src
            shutil.copy(src, STAGE / "clips")
            chunk_rows.append({
                "video_id": vid, "chunk_id": r["chunk_id"],
                "phase": r["phase"], "phase_name": r["phase_name"],
                "t_start": r["t_start"], "t_end": r["t_end"],
                "duration_s": round(r["t_end"] - r["t_start"], 2),
                "video_file": f"videos/{vid}.mp4", "clip_file": clip,
                "tiers_observed": r["tiers"]["observed"],
                "tiers_weak": r["tiers"]["weak"],
                "report": r["report"], "memory_update": r["memory_update"],
                "flags_add": r["flags_add"],
                "prompt_tokens": r["usage"]["total_tokens"] - r["usage"]["completion_tokens"],
                "completion_tokens": r["usage"]["completion_tokens"],
                "split": split_of(vid, vids),
            })
    chunks = pd.DataFrame(chunk_rows)
    chunks.to_parquet(STAGE / "chunks.parquet", index=False)
    pd.concat(inst_frames, ignore_index=True).to_parquet(
        STAGE / "chunk_instruments.parquet", index=False)
    splits = {s: sorted(chunks[chunks.split == s].video_id.unique())
              for s in ("train", "validation", "test")}
    (STAGE / "splits.json").write_text(json.dumps(splits, indent=2))
    (STAGE / "README.md").write_text(CARD)
    n_clips = len(list((STAGE / 'clips').glob('*.mp4')))
    print(f"staged: {len(vids)} videos, {n_clips} clips, "
          f"{len(chunks)} chunk rows, splits={ {k: len(v) for k, v in splits.items()} }")


if __name__ == "__main__":
    main()
