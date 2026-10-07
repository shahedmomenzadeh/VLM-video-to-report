# Vision-Language Models for Cataract Surgery Video Reporting: Data, Methods, and Evaluation

## Abstract

We describe a framework that converts full-length cataract surgery videos into
structured natural-language operative reports using vision-language models
(VLMs). Because manually authored reference reports are unavailable at scale,
a frontier VLM generates teacher reference reports for frozen video chunks,
conditioned on phase labels, instrument evidence from a YOLO segmentation
model, and a causal memory of preceding chunks. Candidate (student) models —
Qwen3-VL-2B variants (base, SFT, GRPO), Hulu-Med-4B, and Lingshu-7B — generate
reports for the identical chunks under three input conditions (video only,
video + phase, video + phase + instruments). Evaluation is teacher-referenced
but not teacher-dependent: automatic lexical agreement against the teacher
reference is paired with a teacher-blind, evidence-grounded LLM judge scored
at whole-video level. On a 3-video pilot (102 chunks), the GRPO-trained model
outperforms its base model under instrument-informed conditions on both
evaluation modes, while video-only phase identification remains at floor for
both. This document records the data pipeline, the report-generation method,
the memory system, and the evaluation protocol as implemented.

## 1. Objective and design

The task is video-to-report generation for cataract surgery: given a surgical
video, produce a factual operative report describing phases, instruments,
anatomy, and actions over time. The framework has three parts:

- **Part I — teacher references.** One reference report per video chunk from a
  frontier VLM, grounded in phase labels and instrument detections.
- **Part II — candidate reports.** The same frozen chunks are described by
  smaller VLMs under three ablated input conditions.
- **Part III — judging.** Candidates are scored against the teacher
  (lexical agreement) and against the surgical evidence with the teacher
  hidden (LLM factuality judge), at whole-video level.

The central experimental contrasts are
V < V+P < V+P+I (effect of procedural context) and base < SFT < GRPO
(effect of training), across model families.

## 2. Data and cohort

The corpus comprises **25 cataract surgery videos** (`videos/<VID>/<VID>.mp4`),
each with a phase timeline (`<VID>.timeline.json`) of timestamped segments
over **13 phases** (P01 Incision … P12 Tonifying-Antibiotics, P13 Idle),
with per-segment phase id, name, occurrence index, and start/end times.
The teacher corpus totals **849 chunks**; a 3-video pilot subset
(PH_0001_2931_S2, PH_0043_0096_S1, PH_0057_0239_S1; 102 chunks) is used for
model development and all reported pilot numbers. Video-level
train/validation/test splits (19/3/3 videos; test = the pilot trio) are
stored in `splits.json` so no chunk leaks across splits. The Part I dataset
is published on the Hugging Face Hub (`cataract-video-to-report`) containing
only teacher-side modalities: full videos, frozen clips, timelines, YOLO
instruments, teacher reports, memory snapshots, and parquet indexes.

## 3. Methods

### 3.1 Chunking: timeline segments to frozen VLM inputs

Timeline segments are the natural unit, but long phases (16 segments exceed
2 min, all P05 Phacoemulsification, max 194.3 s) exceed a VLM's effective
short-video window. Segments longer than ~1 min (`CHUNK_SPLIT_S = 65.0 s`)
are split into ~1-min subchunks (`P05_o01_c1/c2/…`), and no chunk sent to any
VLM exceeds a hard 240 s ceiling. Chunking is deterministic code
(`clean_instruments.make_chunks`), so every model and every setting sees
byte-identical temporal inputs. Teacher clips are cut frame-accurately with
ffmpeg (libx264 re-encode) and cached (`output-teacher/clips/`); candidates
reuse the same files and are never re-cut.

### 3.2 Instrument perception: YOLO segmentation plus three-layer cleaning

An Ultralytics segmentation model (`yolo-model/model.pt`, 12 classes) runs
over each full surgery on GPU. Of the 12 classes, Cornea and Pupil are
tissues and are excluded; the 10 instrument classes (Cannula, Cap-Cystotome,
Cap-Forceps, Forceps, I-A-Handpiece, Lens-Injector, Phaco-Handpiece,
Primary-Knife, Second-Instrument, Secondary-Knife) are kept with per-frame
bounding boxes, confidences, and relative areas
(`extract_instruments.py` → `*_instruments_raw.csv`).

Raw detections are noisy, so `clean_instruments.py` applies three layers,
using intrinsic evidence first and phase priors only as a soft flag that
never deletes:

- **L1, detection level:** drop confidence < 0.25, box area < 0.1% of frame,
  and same-frame same-class duplicates (keep max confidence).
- **L2, track level:** gap-tolerant runs (gap ≤ 2 frames); runs ≥ 3 frames
  persist, 1–2 frame runs survive only at confidence ≥ 0.6 (else *fleeting*).
- **L3, class level:** drop whole classes with < 0.5% frame coverage,
  max run < 5, and median confidence < 0.6 (*weak_class* — kills
  hallucinated classes).

Surviving detections are aggregated per chunk into two tiers:
**observed** (strong evidence — ≥ 8 frames, or ≥ 5 with run ≥ 3 or median
confidence ≥ 0.7 — *and* plausible for the phase under a surgical prior
table) vs **weak** (everything else: trace detections the VLM must verify
visually, never assume). The prior table (`EXPECTED`) encodes which
instruments plausibly appear in each phase; P13 Idle allows all classes
since it holds transitions. Output: `*_chunk_instruments.csv` with frame
counts, fractions, max runs, and median confidences per (chunk, class).

### 3.3 Teacher reference generation (Part I)

Each chunk is sent as a single base64 MP4 to a frontier VLM
(`ag/gemini-3.8-flash` via an OpenAI-compatible endpoint, temperature 0.2,
≤ 1024 tokens) with a prompt containing the chunk time window, the phase,
the tiered instrument block (observed vs weak, with an explicit
verify-visually safeguard and a never-assume instruction for the imperfect
perception model), and the causal memory state (§3.4). The model must return
JSON with `report` (fixed five-section text: Phase / Observed instruments /
Candidate but not confirmed / Anatomy visible / Actions-events),
`memory_update` (compressed carry-over state for the next chunk), and
`flags_add` (persistent event tags). Result: 849/849 non-empty,
section-valid reports (~3.2M tokens total), each with prompt, raw response,
usage, and latency archived per chunk.

### 3.4 The memory system

All report generation — teacher and every candidate setting — uses one
shared causal memory paradigm (`memory.py`), one dict per (video [, model,
setting]) chain advanced once per chunk in temporal order from that chain's
own outputs:

- `running_summary`: compressed state, one model-written sentence per chunk,
  tail-capped at ~1600 chars (full history, lossy);
- `prev_report`: verbatim previous-chunk report (capped at 1500 chars, above
  the longest legitimate report, as an anti-cascade guard);
- `prev_nonidle_report`: verbatim report of the most recent non-Idle chunk
  (scan-back: Idle chunks carry it forward), shown alongside the previous
  chunk when they differ — transition vs substantive-action context;
- `phase_trail`: last 10 phase ids; `instruments_seen`: class → last chunk
  id (Idle chunks never overwrite); `flags`: append-only deduplicated event
  tags; `same_phase_continuation`: transient hint triggering a "describe
  what changed" instruction for split-phase continuations.

Every step is snapshotted to JSONL, so each chain's full memory trajectory
is reproducible. Per-setting rules: s2/s3 trails use the given phase, s1
uses the model's self-stated phase (best-effort `PXX` parse); only s3 (and
the teacher) seed `instruments_seen` from the observed tier, since
self-reported instruments in s1/s2 are unverifiable and would corrupt the
chain.

### 3.5 Candidate generation (Part II): three conditions, five models

Each candidate model describes every frozen chunk under three conditions
with memory always on: **s1** video only (model must identify the phase
itself), **s2** video + phase name, **s3** video + phase + tiered
instruments. Sampling is fixed across models (temperature 0.7, ≤ 512
tokens, repetition penalty 1.05, ≤ 32 frames per chunk with clamp-to-clip
and halve-on-OOM retry, one JSON-reformat retry). A `VideoBackend`
interface isolates per-family video ingest and generation (Qwen3-VL,
Hulu-Med with a processor compat patch, Lingshu), each under its own
dependency venv, while prompts, memory, resume handling, and output schemas
stay shared. Evaluated models: Qwen3-VL-2B-Instruct (base), its SFT-stage2
and GRPO checkpoints, Hulu-Med-4B, and Lingshu-7B. An audit script verifies
every chain (timeline coverage/order, snapshot parity, summary health,
parse stats, frame ranges, s1 phase accuracy).

### 3.6 Evaluation (Part III): teacher-referenced, not teacher-dependent

The teacher is treated as a scalable pseudo-reference, never as ground
truth. Reports are stitched into whole-video documents (chunk headers kept
for error localization; consecutive Idle chunks collapsed so Idle text does
not dominate). Two modes:

- **Eval A — automatic reference agreement.** Candidate vs teacher
  video-docs: ROUGE-L and chrF (primary), METEOR (secondary), BLEU
  (report-only), CIDEr over the chunk corpus (video-level CIDEr degenerates
  on long templated documents — verified 0.0 even on self-match — so the
  corpus unit is chunks). Scored full-doc and per-section (Observed,
  Actions/events).
- **Eval B — evidence-grounded LLM judge, teacher-blind.** A judge model
  from a different family than the teacher sees the candidate video-doc
  plus timeline and aggregated tiers (video primary, tiers auxiliary) and
  scores 1–5 with anchors: groundedness, completeness,
  instrument_correctness, temporal_coherence, plus an echo flag for
  verbatim perception-metadata copying. A calibration study (teacher
  self-scoring ≈ 5s; three rubric variants) confirmed the scale works and
  selected the explicit-anchor v2 rubric. Focused order-swapped pairwise
  comparisons (s1-vs-s3; SFT-vs-GRPO at s3) and a human-expert subset for
  judge correlation complete the protocol.

## 4. Reproducibility

The pipeline is scripted end to end: `run_pipeline.sh` (YOLO → extraction
→ cleaning → teacher), `setup_backends.sh` (per-family venvs),
`candidate_reports.py`, `verify_candidates.py`, `video_documents.py`,
`score_lexical.py`, `judge_reports.py` (score/pairwise, prompt versions,
source filters, resume-safe), and `build_hf_dataset.py` /
`upload_hf_dataset.py` for the public dataset. All stages skip finished
work and rebuild memory from snapshots, so long GPU runs resume cleanly.

## 5. Pilot results (base 2B-Instruct vs GRPO, 102 chunks)

Lexical (Eval A, mean over 3 videos): GRPO wins cleanly only at s3
(ROUGE-L 0.548 vs 0.504, chrF 63.2 vs 58.7, BLEU 40.6 vs 38.3); s1/s2 are a
wash, and CIDEr plus the Observed-section score favor the base model —
both reward verbatim tier-text echoing, which the base model does more.
Video-only phase identification is floor for both (~7% exact phase-id
match; both stick on P01). LLM judge (Eval B, v2 means): GRPO ≥ base in all
12 (model × setting × criterion) cells, strictly better in 5, concentrated
in s2/s3 (e.g. s3 completeness 2.7 vs 1.7); absolute levels remain low
(~2/5), with late-phase collapse (P10/P12 → P09 confusion) as GRPO's main
residual failure. Interpretation: training transferred the *use* of given
procedural context, not video-only phase recognition — and lexical
agreement alone would have overstated base-model quality, which is why the
two-mode design matters.

## 6. Limitations

Teacher references inherit frontier-model errors and possible
self-preference when judge and teacher share a family (pipeline-grade
results use Gemini for both; paper-grade numbers require the independent
judge plus human correlation). Lexical metrics reward boilerplate mimicry.
Video-only phase recognition is unsolved in all tested 2B models. The 4B/7B
family comparison (SFT, Hulu-Med, Lingshu) is in progress.
