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

Each surgery is split into phase chunks. Every chunk has: the clip
(`clips/`), phase label (`timelines/`), instrument tiers from a YOLO
segmentation model (`instruments/`), and a teacher reference report with
causal memory (`teacher_reports/`, `memory/`).

`chunks.parquet` is the master index (one row per chunk); `splits.json`
assigns whole videos to train/validation/test (no chunk leakage).
Intended downstream use (not included here): run candidate VLMs on identical
`clips/` under video-only / phase-informed / phase+instrument-informed
conditions and score them against the teacher reports.

## How this dataset was generated

**Stage 1 — YOLO segmentation.** An Ultralytics segmentation model
(12 classes) runs over each full surgery on GPU, producing annotated video
plus per-frame detections. Tissue classes (Cornea, Pupil) are excluded —
only the 10 instrument classes are kept:
Cannula, Cap-Cystotome, Cap-Forceps, Forceps, I-A-Handpiece, Lens-Injector,
Phaco-Handpiece, Primary-Knife, Second-Instrument, Secondary-Knife.

**Stage 2 — Noise cleaning.** Raw detections pass three layers: confidence
floor (0.25) + tiny-box + dedupe, gap-tolerant track persistence, and
class-level support against phase priors (tiers only, never deletes).
Cleaned detections are aggregated per phase segment into observed (strong,
phase-plausible) vs weak (trace, verify-visually) tiers with frame counts
and median confidences — see `instruments/*_chunk_instruments.csv` and
`chunk_instruments.parquet`.

**Stage 3 — Chunking.** Timeline segments longer than ~1 min are split into
~1-min subchunks (`P05_o01_c1/c2/...`, 65 s rule); no chunk sent to the VLM
exceeds 240 s. Result: 849 chunks across 25 videos.

**Stage 4 — Teacher reports.** Each chunk is cut frame-accurately with
ffmpeg, sent as a single base64 MP4 to a teacher VLM
(`ag/gemini-3.8-flash`, temperature 0.2, ≤1024 tokens), and prompted with
chunk time window + phase + instrument tiers + memory (below). The model
must return JSON with `report` (Phase / Observed instruments / Candidate
but not confirmed / Anatomy visible / Actions-events), `memory_update`
(compressed state for the next chunk), and `flags_add` (persistent event
tags). Candidate instruments are always worded as unverified ("candidate",
never "definitely present"). Totals: 849/849 non-empty section-valid
reports, ~3.2M tokens.

## Memory system

Memory M(i,j) is built causally in temporal order — one dict per video,
updated once per chunk from that chunk's own outputs:

- `running_summary`: compressed state, one model-written sentence per
  chunk, tail-capped at ~1600 chars (full history, lossy).
- `prev_report`: verbatim report of the immediately previous chunk,
  Idle or not (exact recent context, one chunk).
- `prev_nonidle_report`: verbatim report of the most recent NON-IDLE
  chunk (scan-back: Idle chunks carry it forward). When the previous
  chunk is Idle, the prompt shows both verbatim blocks, explicitly
  labeled as transition vs substantive-action context.
- `phase_trail`: last 10 phase ids (Idle included).
- `instruments_seen`: class → last chunk id; Idle chunks never overwrite it.
- `flags`: append-only deduplicated event tags.
- `same_phase_continuation`: transient flag triggering a "describe what
  CHANGED" note for split-phase continuations.

Every step is snapshotted (`memory/<VID>_memory.jsonl`), so the full
memory trajectory is reproducible from chunk 1.

## Splits and license

Video-level splits (test = the exhaustively studied pilot trio).
License is currently `other` — confirm video sharing clearance before
redistributing beyond research use.
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
