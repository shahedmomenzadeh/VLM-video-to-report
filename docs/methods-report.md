# Vision-Language Models for Cataract Surgery Video Reporting: Data, Methods, and Evaluation

## Abstract

We describe a framework that converts full-length cataract surgery videos into
structured natural-language operative reports using vision-language models
(VLMs). Because manually authored reference reports are unavailable at scale,
a frontier VLM generates teacher reference reports for frozen video chunks,
conditioned on phase labels, instrument proposals from a YOLO segmentation
model, and a causal memory of preceding chunks. Candidate (student) models —
Qwen3-VL-2B checkpoints (SFT-stage2, GRPO) in the current pilot, with
Qwen3-VL-2B-Instruct, Hulu-Med-4B, and Lingshu-7B from the earlier round —
describe the identical chunks under a two-point context ablation (video only;
video + phase + instrument proposals), writing time-blind tagged plain text
that code normalizes and renders, so timestamps and headers can never drift.
Evaluation is teacher-referenced but not teacher-dependent: deterministic
agreement metrics against the teacher reference are paired with a
teacher-blind, video-grounded LLM judge that watches each frozen clip and
grades the report's clinical content on two independent anchored 0–3 scales
(accuracy, coverage). On a 3-video pilot (102 chunks, every chunk judged),
the GRPO-trained checkpoint improves when given phase and instrument
propositions in both evaluation modes, while the SFT-stage2 checkpoint shows
the same lexical gains without a judged-quality gain — its hint-aided
reports still fabricate events the video does not show; video-only phase
identification remains at floor for both. This document records the data
pipeline, the report-generation method, the memory system, and the
evaluation protocol as implemented (v2.1).

## 1. Objective and design

The task is video-to-report generation for cataract surgery: given a surgical
video, produce a factual operative report describing phases, instruments,
anatomy, and actions over time. The framework has three parts:

- **Part I — teacher references.** One reference report per video chunk from a
  frontier VLM, grounded in phase labels and instrument detections and
  rendered deterministically from structured fields.
- **Part II — candidate reports.** The same frozen chunks are described by
  smaller VLMs under a two-point context ablation.
- **Part III — evaluation.** Candidates are scored in three layers:
  deterministic metrics against the teacher reference and the timeline, a
  teacher-blind video-grounded LLM judge that watches each clip, and a
  code-locked full-video compilation that assembles per-chunk reports into
  validated timelines.

The central experimental contrasts are
V < V+P+I (effect of procedural context) and base < SFT < GRPO (effect of
training), across model families. The v2.1 redesign reduced the context
ablation to its two endpoints (s1, s3) and dropped the phase-only middle rung
(v2 s2).

## 2. Data and cohort

The corpus comprises **25 cataract surgery videos** (`videos/<VID>/<VID>.mp4`),
each with a phase timeline (`<VID>.timeline.json`) of timestamped segments
over **13 phases** (P01 Incision … P12 Tonifying-Antibiotics, P13 Idle),
with per-segment phase id, name, occurrence index, and start/end times.
The teacher corpus totals **849 chunks**; a 3-video pilot subset
(PH_0001_2931_S2, PH_0043_0096_S1, PH_0057_0239_S1; 102 chunks: 29/34/39) is
used for model development and all reported pilot numbers. Video-level
train/validation/test splits (19/3/3 videos; test = the pilot trio) are
stored in `splits.json` so no chunk leaks across splits. The Part I dataset
is published on the Hugging Face Hub (`cataract-video-to-report`) under the
v2.1 contract (the earlier release is preserved as tag `v1-2026-10-06`),
containing only teacher-side modalities: full videos, frozen clips, timelines,
YOLO instruments, teacher reports, memory snapshots, and parquet indexes.

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
≤ 1024 tokens, 240 s send ceiling) with a **time-blind** prompt — the chunk
*duration* only, never the wall clock — containing the phase, the tiered
instrument block (observed vs weak, with an explicit verify-visually
safeguard and a never-assume instruction for the imperfect perception
model), and the causal memory state (§3.4). The model returns **structured
JSON fields only**: `phase_id`, `instruments_visible` /
`instruments_uncertain` (subsets of a closed 10-class vocabulary),
`anatomy` (closed 8-term vocabulary), `actions` (1–5 observable bullets),
`memory_update` (one or two sentences of compressed state for the next
chunk), and `flags_add` (persistent event tags). The model is forbidden from
writing timestamps, headers, or display names; `report_render.py` assembles
the fixed five-section wire text (Phase / Observed instruments / Candidate
but not confirmed / Anatomy visible / Actions-events) from the fields, so
headers and timestamps are byte-identical across reports. Closed-vocabulary
and schema violations trigger correction retries (max 2 extra calls);
transient gateway errors retry with backoff (3 attempts total). The pilot
trio was regenerated under this contract: 102/102 reports, 0 errors, each
row archiving prompt, raw response, usage, and latency; the 25-video corpus
totals 849 teacher chunks generated under the earlier contract.

### 3.4 The memory system

All report generation — teacher and every candidate setting — shares one
causal memory paradigm (`memory.py`): one dict per (video [, model, setting])
chain, advanced once per chunk in temporal order from that chain's own
outputs. The v2.1 schema is deliberately slim and timestamp-free:

- `recent_states`: rolling list (max 4) of `{chunk_id, phase, state}`, where
  `state` is the model's own one-sentence end-of-chunk summary;
- `last_nonidle_state`: the most recent non-Idle state, pinned separately so
  Idle gaps cannot push away the last substantive action (Idle chunks never
  overwrite it);
- `phase_trail`: last 10 phase ids (Idle included) — the given phase in
  s3/teacher, the model's self-stated phase in s1;
- `instruments_seen`: class → last chunk id; only s3 and the teacher seed it
  from the observed tier, since self-reported instruments are unverifiable
  and would corrupt the chain;
- `flags`: append-only deduplicated event tags;
- `same_phase_continuation`: transient hint, recomputed before each prompt,
  asking for what started/ended/changed rather than a repeat description.

Full preceding reports are **not** carried forward: v2.1 removed the verbatim
`prev_report` / `prev_nonidle_report` / `running_summary` block (it invited
echo and was the main hallucination vector for 2B models; the timeline is
code-owned anyway). Legacy snapshots auto-migrate to the slim schema,
keeping only `phase_trail` / `instruments_seen` / `flags`. Every step is
snapshotted to JSONL, so each chain's memory trajectory is reproducible and
resumable.

### 3.5 Candidate generation (Part II): two conditions, tagged text

Each candidate model describes every frozen chunk under two conditions with
memory always on: **s1** video only (the model must identify the phase
itself), **s3** video + phase + tiered instrument proposals. Both prompts
are time-blind — they carry the chunk duration, never timestamps — and the
shared background block (§3.4) carries no times either.

v2.1 changed the student output contract from JSON to **plain tagged text**
(`PHASE` (s1 only), `TOOLS`, `UNCERTAIN_TOOLS`, `ANATOMY`, `EVENTS`,
`STATE`, `FLAGS`) with an explicit instruction never to write timestamps,
headers, or JSON. Parsing is total and zero-retry:
`report_render.parse_student_tags` canonicalizes tool and anatomy names via
synonym maps (e.g. "phaco probe" → `Phaco-Handpiece`) and renders the same
five-section wire text as the teacher, attaching timestamps from the frozen
chunk table. This removed the JSON-reformat retry loop of v2 and made
cross-model comparison fairer: prompt difficulty, not parser luck, dominates
the parse rate.

Sampling is fixed across models (temperature 0.7, ≤ 512 new tokens, ≤ 32
frames per chunk sampled at ≤ 1 fps with clamp-to-clip and halve-on-OOM
retry), and per-chunk writes are resume-safe. A `VideoBackend` interface
isolates per-family video ingest and generation (Qwen3-VL; Hulu-Med with a
processor compat patch; Lingshu), each under its own dependency venv, while
prompts, memory, scoring schemas, and storage stay shared. The v2.1 pilot
evaluates Qwen3-VL-2B checkpoints: SFT-stage2 and its GRPO-trained sibling
(102/102 chunks per setting each); the earlier round additionally covered
the 2B base/Instruct model, Hulu-Med-4B, and Lingshu-7B under the superseded
v1/v2 contracts.

### 3.6 Evaluation (Part III): teacher-referenced, not teacher-dependent

The teacher is treated as a scalable pseudo-reference, never as ground
truth. Evaluation has three layers; the judge is one of them, not the only
one.

**Tier 0 — deterministic (code only).** Chunk-level, joined on frozen chunk
ids: reliability screens (section presence, empty reports, degenerate
repetition, verbatim echo of perception metadata), reference agreement
(ROUGE-L and chrF over the rendered Actions/events section; instrument and
anatomy F1 over their canonical class sets), and s1 phase recognition
(accuracy, macro-F1, and confusion against the timeline). No LLM calls; every claim that can be
settled in code is.

**Tier J — video-grounded judge, teacher-blind.** One multimodal call per
chunk report: the judge watches the frozen clip together with the report's
clinical content, with code-generated phase headers and timestamps stripped,
and with no access to the teacher report, the model identity or setting, the
detector tiers, or any prior score (clips are referred to by neutral ids).
It returns two **independent anchored 0–3 grades** — accuracy and coverage —
plus up to three quoted issues with severity (minor | material) and a list
of details the clip cannot resolve (`unassessable`). The rubric is frozen
(`visual-1`): score caps are reserved for material errors (wrong surgical
step, invented implant or complication, instrument confusion that changes
the described action, concealment of a visible adverse event), while
synonyms, terse phrasing, and legitimate continuation are tolerated. Rows
are versioned by `report_hash` × rubric × judge model, so a regenerated
report is automatically re-judged and stale scores can never be reused;
errored rows are retried, identical rows skipped, and the pass is
resume-safe. The rubric is calibrated on a deliberately hard subset before
any model comparison is read (§5); method and prompt details live in
`judgement_overview.md`.

**Full-video layer.** Per-chunk reports are compiled into full-video
documents by a code-locked spine (`compile_self.py`): consecutive same-phase
chunks are grouped into timestamped blocks whose spans come from the frozen
chunk table, and the compiled text is re-validated against `make_chunks`
every time (no gaps, no overlaps). An optional text-only polish may rewrite
the bullets inside each block; a preservation check then compares every
polished block against the source bullets (synonym-aware content recall, no
invented instruments), and the block falls back to the verbatim bullets if
the check fails. `run.py timeline` re-runs the deterministic span validation
over the compiled corpus.

Earlier diagnostics from the claim-counting era — the candidate-blind
evidence note, per-claim support/contradiction counting, and five-label
temporal transition audits — remain runnable behind explicit flags
(`--with-evidence`, `--legacy-claims`, `--legacy-temporal`) as debugging
diagnostics only. Their known failure modes (an AI-written evidence note
treated as the authority for "factuality", uncalibrated atomic-claim
weighting) motivated moving the primary judge onto the video itself.

## 4. Reproducibility

The pipeline is scripted end to end: `run_pipeline.sh` (YOLO → extraction →
cleaning → teacher), `setup_backends.sh` (per-family venvs),
`candidate_reports.py` for students (all videos or a `--videos` list,
`--settings s1,s3`), and the `evaluation/` package for scoring: `run.py`
(`inventory`, `reliability`, `reference`, `phase`, `judge`, `timeline`,
`report`; `evidence`/`temporal` kept as legacy diagnostics), `batch.py` for
the parallel judge pass, `compile_self.py` for code-locked full-video
reports (optional polish), `compile_readable.py` for human-review
presentation, and `viewer.py` for the interactive dashboard. The public
teacher-side dataset is built and uploaded with `build_hf_dataset.py` /
`upload_hf_dataset.py`. All stages skip finished work and rebuild memory
from per-chunk snapshots, so long GPU runs resume cleanly; judge rows are
keyed by report hash, so regenerating a report invalidates exactly its own
score.

## 5. Pilot results (v2.1, pilot trio, 102 chunks)

Two Qwen3-VL-2B checkpoints (SFT-stage2, GRPO) × two settings (s1, s3), all
102 chunks per chain, teacher regenerated under the same v2.1 contract, and
every chunk report judged (408/408, 0 errors).

**Tier 0 (deterministic).** Per-chunk scores averaged per video, then over
the three videos; ROUGE-L and chrF are computed on the Actions/events
section, instrument and anatomy F1 over their canonical class sets. No chain
produced empty, echoed, or degenerate reports (0.0 rates). Scores are
chunk-level and are not comparable to earlier whole-document numbers.

| model | setting | ROUGE-L ↑ | instrF1 ↑ | anatomyF1 ↑ | nonIdle-ROUGE-L ↑ |
|---|---|---|---|---|---|
| qwen3vl-2b-grpo | s1 | 0.119 | 0.146 | 0.253 | 0.113 |
| qwen3vl-2b-grpo | s3 | 0.223 | 0.653 | 0.262 | 0.236 |
| qwen3vl-2b-sft-stage2 | s1 | 0.128 | 0.117 | 0.158 | 0.129 |
| qwen3vl-2b-sft-stage2 | s3 | 0.177 | 0.671 | 0.444 | 0.169 |

**Tier J (video-grounded judge), full trio.** All 102 chunks per chain:

| model | setting | accuracy ↑ | coverage ↑ | material ↓ |
|---|---|---|---|---|
| qwen3vl-2b-grpo | s1 | 0.95 | 1.05 | 0.71 |
| qwen3vl-2b-grpo | s3 | 1.40 | 1.33 | 0.66 |
| qwen3vl-2b-sft-stage2 | s1 | 1.04 | 1.10 | 0.68 |
| qwen3vl-2b-sft-stage2 | s3 | 1.14 | 1.16 | 0.75 |

**Context × training contrasts** (paired per-video, chunk-matched, 95%
bootstrap CI over videos). GRPO improves with context in both modes
(ROUGE-L +0.104 [0.062, 0.134], instrF1 +0.507 [0.448, 0.616]; judge
accuracy +0.44 [0.08, 0.91], coverage +0.28 [−0.03, 0.85]). SFT-stage2
improves lexically (ROUGE-L +0.049 [0.0003, 0.0768], instrF1 +0.554
[0.516, 0.626]) but shows no judged-quality gain (accuracy +0.10
[−0.17, 0.38], coverage +0.06 [−0.21, 0.41]) — its hint-aided reports still
fabricate plausible surgery (e.g. an iris prolapse, an already-centered
IOL) that the clip does not show. Video-only phase recognition is at floor
for both (mean accuracy 0.166 GRPO, 0.128 SFT-stage2). The deterministic
timeline check passes on every compiled full-video report on disk (34/34 at
time of writing), validating span coverage, gaps, and overlaps against the
frozen chunk tables.

**Interpretation, with a note on subsets.** Training transferred the *use*
of given procedural context: GRPO turns the phase and instrument
propositions into more accurate descriptions, and its advantage is visible
in both evaluation tiers. SFT-stage2 turns them into lexical overlap — its
instrument-F1 gain is as large as GRPO's — without a corresponding judged
gain, and its material-issue rate is the highest of the four chains: the
proposals reshape its wording more than its reading of the clip. A first
calibration pass on a deliberately varied 24-chunk subset had suggested
SFT-stage2 *degrades* under context (s1 ≫ s3); on the full trio that gap
vanished (s3−s1 accuracy +0.10 [−0.17, 0.38]). The subset had over-sampled
this chain's fabrication cases — which is why the larger full run was worth
doing: calibration subsets order the rubric, they do not estimate
model-level effects.

## 6. Limitations

Teacher references inherit frontier-model errors and, since the judge and
teacher currently share a model family (Gemini), a shared-family preference
cannot be excluded; pipeline-grade results use Gemini for both, and
paper-grade numbers require an independent judge family plus a human-expert
correlation subset. Lexical metrics reward boilerplate mimicry and are
computed against a single teacher pass. The video-grounded judge is itself a
model: grades are one-call outputs with sampling variance, the 0–3 scale is
ordinal, and borderline "material" calls (empty reports near idle
boundaries, faint instruments) need periodic human spot checks. Tier J
levels come from a single full-trio pass (all 408 chunk reports, 0 errors);
the earlier calibration-subset ordering for one chain did not generalize
(§5), a reminder that selected subsets calibrate rubrics but do not
estimate levels, and that single-run differences should be replicated
before being quoted as stable. Video-only phase recognition is unsolved in
all tested 2B models, and the 4B/7B family comparison under v2.1 has not
been rerun.
