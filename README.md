# VLM-video-to-report

Generate surgical reports from video using a vision-language model (VLM) pipeline.
Stage: full pilot pipeline — segmentation → instrument cleaning → teacher
references → tagged-text student reports → video-grounded evaluation
(see `judgement_overview.md`).

## Setup

Requires Python ≥ 3.13 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

## Usage

Full pipeline (YOLO → instruments → cleaning → teacher VLM), fresh from scratch:

```bash
bash run_pipeline.sh
```

Resume after an interruption (keeps existing outputs):

```bash
RESUME=1 bash run_pipeline.sh
```

Individual stages (see sections below) can also be run by hand.

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
- Memory is background only (v2.1 slim, timestamp-free: rolling STATE history
  ≤4 + last non-idle state + phase trail + instruments-seen + event flags +
  same-phase flag). Verbatim previous reports are **not** carried (echo
  guard). The model returns structured JSON fields only (`phase_id`,
  instrument/anatomy ids, `actions`, `memory_update`, `flags_add`); code
  renders the report text and all timestamps. Prompt forbids restating
  background as current observation.
- Outputs in `output-teacher/` (git-ignored): cached `clips/`, `reports/`
  (.md + .json), `memory/` chain snapshots, master `*_teacher_reports.jsonl`.

### 4. Candidate reports (Part II)

Smaller VLMs describe the same frozen chunks under two settings: `s1` (video
only — the model states its own phase) and `s3` (video + phase + tiered
instrument proposals). Students write plain tagged text (`PHASE`, `TOOLS`,
`UNCERTAIN_TOOLS`, `ANATOMY`, `EVENTS`, `STATE`, `FLAGS`); code canonicalizes
names and renders the same five-section wire text as the teacher, attaching
timestamps from the frozen chunk table. Prompts are time-blind, parsing is
zero-retry, per-chunk writes are resume-safe:

```bash
.venv-vlm/bin/python src/vlm_report_framework/candidate_reports.py \
    --model <local-path-or-hf-id> --model-tag qwen3vl-2b-grpo \
    --videos PH_0001_2931_S2 --settings s1,s3
# -> output-candidates/<tag>/{s1,s3}/ (git-ignored)
```

### 5. Evaluation (Part III)

Deterministic metrics (lexical vs teacher, reliability, s1 phase accuracy)
plus a teacher-blind video-grounded judge — one multimodal call per chunk
report, accuracy/coverage 0–3, resume-safe. Method and prompt:
`judgement_overview.md`.

```bash
# judge pass (parallel, resume-safe, skips unchanged reports)
.venv-vlm/bin/python src/vlm_report_framework/evaluation/batch.py \
    --videos PH_0001_2931_S2 --tags qwen3vl-2b-grpo --workers 8
# dashboard + timeline validation
.venv-vlm/bin/python src/vlm_report_framework/evaluation/run.py report --run-id v2slim
.venv-vlm/bin/python src/vlm_report_framework/evaluation/run.py timeline
# code-locked full-video report (add --polish for optional text polish)
.venv-vlm/bin/python src/vlm_report_framework/evaluation/compile_self.py \
    --tag qwen3vl-2b-grpo --setting s3 --videos PH_0001_2931_S2 --no-polish
# interactive viewer -> output-evaluation/<run-id>/viewer.html
.venv-vlm/bin/python src/vlm_report_framework/evaluation/viewer.py --run-id v2slim
```

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
│   ├── memory.py             # shared slim causal memory (teacher + students)
│   ├── report_render.py      # tagged-text parsing + canonical report rendering
│   ├── teacher_prompt.py     # teacher prompt builder (P + I_ij + M_ij)
│   ├── teacher_reports.py    # teacher VLM loop with causal memory
│   ├── candidate_prompt.py   # student prompts (s1/s3, tagged-text contract)
│   ├── candidate_reports.py  # student generation loop (resume-safe)
│   └── evaluation/           # tiers: reliability, reference, judge, compile, viewer
├── judgement_overview.md   # judge method, prompt, and pipeline documentation
├── videos/                 # input videos, git-ignored
├── yolo-model/             # model.pt, git-ignored
├── output-segmented/       # annotated outputs, git-ignored
├── output-instruments/     # raw/clean/chunk CSVs, git-ignored
├── output-teacher/         # clips/reports/memory snapshots, git-ignored
├── output-candidates/      # student per-chunk reports, git-ignored
├── output-evaluation/      # scores, judgments, dashboard, git-ignored
└── output-readable/        # compiled full-video reports, git-ignored
```

## Notes

- Inference uses `stream=True` + CUDA (`device=0`) so per-frame mask results are
  not accumulated in RAM (avoids OOM/SIGBUS on long videos).
