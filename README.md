# VLM-video-to-report

Generate surgical reports from video using a vision-language model (VLM) pipeline.
Current stage: YOLO segmentation of cataract surgery videos (instrument + anatomy masks).

## Setup

Requires Python ≥ 3.13 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

## Usage

### 1. Segment videos with YOLO

Place your segmentation model at `yolo-model/model.pt` and input videos under
`videos/<VIDEO_ID>/<VIDEO_ID>.mp4` —
video/model/output folders are git-ignored (large + private). The
`videos/README.md` dataset doc is not pushed (it lives next to the ignored data).

Run segmentation on the configured video list:

```bash
uv run python src/vlm_report_framework/segment_videos.py
```

Annotated videos (masks + boxes) are written to `output-segmented/<VIDEO_ID>/`
(git-ignored).

### 2. Extract + clean instrument detections (Part I: `I_{i,j}`)

Per-frame instrument detections (CSV with class, confidence, frame, timestamp,
bbox) — tissues (Cornea, Pupil) excluded, 10 instrument classes only:

```bash
uv run python src/vlm_report_framework/extract_instruments.py
# -> output-instruments/<VIDEO_ID>_instruments_raw.csv (git-ignored)
```

Then clean noise and aggregate per phase subchunk:

```bash
uv run python src/vlm_report_framework/clean_instruments.py
# -> output-instruments/<VIDEO_ID>_instruments_clean.csv  (per-detection status)
# -> output-instruments/<VIDEO_ID>_chunk_instruments.csv  (Part I I_{i,j}: one row per chunk x instrument)
```

Cleaning is 3 layers (intrinsic evidence first; phase prior only tiers, never
deletes): **L1** confidence floor (0.25) + tiny-box + same-frame duplicate
removal → **L2** gap-tolerant (≤2-frame gap) track persistence, 1–2 frame
blips kept only if conf ≥ 0.6 → **L3** class-level support (drops hallucinated
classes, e.g. `Secondary-Knife` firing 3 frames in one video). Each phase
segment becomes one chunk; segments > ~1 min split into ~1-min subchunks.
Each chunk × instrument lands in one of two tiers for the teacher-VLM prompt:
**observed** (strong evidence + phase-plausible) vs **weak** (candidate but
not confirmed).

### 3. Teacher reference reports (Part I: `R_teacher(i,j)`)

The frontier VLM (`ag/gemini-3.8-flash` via OpenAI-compatible `localhost:20128/v1`)
receives per chunk: the **MP4 subclip** (base64, single blob — never frames),
phase label, tiered candidate instruments, and causal **memory** `M(i,j)`:

```bash
uv run python src/vlm_report_framework/teacher_reports.py --probe  # 1-chunk pipeline check
uv run python src/vlm_report_framework/teacher_reports.py          # smoke test (4 chunks)
uv run python src/vlm_report_framework/teacher_reports.py --all    # all chunks of the video
```

- Chunk ceiling: no send longer than **4 min** (all 25 videos' segments are
  ≤3.24 min; typical chunk ~1 min).
- Memory is background only (running summary ≤400 tok + verbatim previous
  report + phase trail + instruments-seen + event flags + same-phase flag),
  updated in the same VLM call (`report` / `memory_update` / `flags_add` JSON).
  Prompt forbids restating background as current observation.
- Outputs in `output-teacher/` (git-ignored): cached `clips/`, `reports/`
  (.md + .json), `memory/` chain snapshots, master `*_teacher_reports.jsonl`.

### Model

- Ultralytics YOLO **segmentation** (`yolo-model/model.pt`, git-ignored)
- Classes (12): `Cannula`, `Cap-Cystotome`, `Cap-Forceps`, `Cornea`, `Forceps`,
  `I-A-Handpiece`, `Lens-Injector`, `Phaco-Handpiece`, `Primary-Knife`, `Pupil`,
  `Second-Instrument`, `Secondary-Knife`

## Layout

```text
├── src/vlm_report_framework/
│   ├── segment_videos.py     # YOLO segmentation runner (streamed inference)
│   ├── extract_instruments.py # per-frame instrument detections -> CSV
│   ├── clean_instruments.py  # noise cleaning + per-chunk I_{i,j} aggregation
│   ├── teacher_prompt.py     # teacher prompt builder (P + I_ij + M_ij)
│   └── teacher_reports.py    # teacher VLM loop with causal memory
├── videos/                 # input videos, git-ignored
├── yolo-model/             # model.pt, git-ignored
├── output-segmented/       # annotated outputs, git-ignored
├── output-instruments/     # raw/clean/chunk CSVs, git-ignored
└── output-teacher/         # clips/reports/memory snapshots, git-ignored
```

## Notes

- Inference uses `stream=True` + CUDA (`device=0`) so per-frame mask results are
  not accumulated in RAM (avoids OOM/SIGBUS on long videos).
