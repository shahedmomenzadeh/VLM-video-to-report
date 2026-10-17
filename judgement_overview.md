# Judgement overview — video-grounded evaluation for surgical reports

Status: implemented 2026-10-17 (`evaluation/judge.py`, `RUBRIC_VERSION = "visual-1"`).

This document explains the current judgement method, how the judge prompt is
built, and the end-to-end evaluation pipeline. It replaces the earlier
text-only "evidence → claim counting" audit, which measured agreement with an
AI-written note rather than with the surgery.

---

## 1. Why the method changed

The old pipeline was: `video → AI evidence notes → text-only claim judge →
support / contradiction / unassessable rates`. Four problems motivated the
redesign:

1. **The evidence note became a substitute for the video.** A correct report
   detail (e.g. "pupil") was marked *NotEstablished* only because the note
   omitted it. In at least two pilot chunks the note itself disagreed with the
   phase timeline and both teacher versions of the same clip.
2. **Atomic claim counting overweights small, repeated details.** One wrong
   instrument could appear as three "contradicted claims", while listing
   ordinary anatomy inflated the denominator. Support rates were not stable
   measures of report quality.
3. **The temporal judge inferred what it could not observe.** It judged stale
   instruments and "progress" from report text only. Legitimate continuation
   (aspiration, emulsification) was treated as failure to progress.
4. **Framework metadata leaked into the score.** Code-supplied phase headers
   and cautious ("candidate") mentions were sometimes counted as factual
   errors.

The replacement judges **one report against the video clip itself**, with two
independent, anchored scores and a short list of concrete issues.

---

## 2. The method in one picture

```
 frozen clip (mp4)  ─┐
                     ├─→  multimodal judge (frontier VLM)  ─→  accuracy 0-3
 candidate report ───┘                                         coverage 0-3
 (clinical content                                             issues[] (0-3)
  only; blinded)                                               unassessable[]
```

- **One call per chunk report.** No claim splitting, no per-claim verdicts, no
  aggregation of counts.
- **Two independent scores.** A vague but harmless report scores well on
  accuracy and poorly on coverage; an informative report with a wrong step
  scores poorly on accuracy. Neither can compensate for the other.
- **Ordinal scales, not percentages.** Accuracy/coverage are reported as means
  of anchored 0–3 integers per video (with idle / non-idle breakdowns), never
  relabelled as "factuality %".
- **Abstention is explicit.** A detail that cannot be resolved from the clip
  goes in `unassessable`; it is neither credited nor penalized. The judge is
  never asked to guess.
- **Zero retries by policy, one transport/parse retry in practice.** Parsing
  is total: unusable responses become error rows with `accuracy = coverage =
  null` (reported as "unjudged"), never a silent neutral score.

### Inputs the judge never sees

- the teacher report (evaluation stays teacher-referenced for lexical metrics
  only, never teacher-dependent for the judge);
- model identity, setting (s1/s3), or any run metadata;
- detector outputs, confidences, phase timelines;
- previous judge scores.

Clips are referenced by neutral ids (`clip-<10 hex>`), so no phase-coded chunk
id reaches the prompt. `video_id`, `chunk_id`, `tag`, `setting` are attached
to the saved row afterwards, for joins only.

### Content normalization (before the call)

`clinical_content(report)` strips, in code:

- the code-generated `Phase: …` header line (phase recognition for s1 is
  measured separately and deterministically — see §4);
- all `MM:SS(.ss)` timestamps;
- `[class-id]` tags injected by the renderer (e.g. `[Phaco-Handpiece]`).

What remains is only the natural-language clinical content. This makes s1/s3
outputs formally comparable for the judge and keeps frameworks differences out
of the score.

---

## 3. How the prompt is built

### System prompt (verbatim; source of truth: `evaluation/judge.py`)

The prompt defines the scales, the "do not penalize" tolerance list, the
material-error list, the abstention rule, three calibration anchors, and the
exact JSON shape.

```
You are an expert cataract surgeon reviewing ONE short intraoperative video
clip together with a candidate clinical report written about that clip. Judge
how faithfully the report describes the clip. You see the clip directly;
nothing else is provided. Grade content, not style.

SCALES (integers, independent)
Accuracy - is what the report states true of the clip?
3 = Main description correct; no material factual error (minor imprecision is fine).
2 = Main description correct, but with a minor error or overstatement.
1 = Contains a material error (below), though some content is correct.
0 = Predominantly wrong, irrelevant, or no usable description.
Coverage - does the report capture what matters in the clip?
3 = Captures the main activity and important visible changes.
2 = Captures the main activity but misses some useful detail.
1 = Mostly vague, or misses the main activity; or omits a clearly visible consequential event.
0 = No meaningful account of the visible activity.
If the clip shows no meaningful activity (idle), a report saying so is
accurate and complete.

MATERIAL ERRORS (accuracy <= 1 when one dominates what the report describes)
- describes a different surgical step than the one visible;
- invents an implant, complication, or event not visible in the clip;
- confuses clearly distinguishable instruments in a way that changes the
  described action;
- misstates a clearly visible consequential adverse event (coverage <= 1 when
  a visible event is omitted, even if nothing stated is false).

DO NOT PENALIZE
- equivalent terminology ("phaco probe" = "phacoemulsification handpiece";
  "keratome" = "primary knife");
- general wording when fine instrument identity is not discernible;
- non-exhaustive tool/anatomy lists (incidental omissions lower coverage at
  most one level);
- legitimate continuation of an ongoing step;
- cautious wording ("possible", "candidate", "uncertain").

ABSTAIN
A detail you cannot resolve from the clip goes in "unassessable": neither
credited nor penalized. Never guess.

EXAMPLES
- phaco trenching; "the phaco handpiece sculpts a groove" -> accuracy 3, coverage 3.
- phaco trenching; "a keratome creates the main incision" -> accuracy 1, coverage 1, material issue.
- steady capsulorhexis; repeated tearing action -> accuracy 3 (continuation is correct).

OUTPUT - only this JSON object:
{"accuracy": <0-3>, "coverage": <0-3>,
 "issues": [{"quote": "<verbatim snippet>", "severity": "minor"|"material",
             "reason": "<=15 words"}],
 "unassessable": ["<unresolvable detail>", ...]}
```

### User message

```
CLIP: <neutral clip id>

CANDIDATE REPORT (clinical content only; formatting, timestamps and phase
metadata stripped):
<clinical content>

Return only the JSON object.
```

The clip is attached as a base64 `video_url` data part (frozen mp4 from
`output-teacher/clips/`), temperature 0.0, one call, `max_tokens=400`.

### Rationale for the prompt choices

- **Anchored integer scales beat free-text critique** for comparability across
  chunks and models; the anchors define what a "minor" vs "material" problem
  costs.
- **Accept-tolerances are explicit.** Small models phrase things oddly;
  penalizing phrasing measures style, not surgery. Tolerances are about
  wording, never about clinical truth.
- **Strictness is reserved for material errors** — wrong step, invented
  implant/complication, instrument confusion that changes the procedure,
  missed adverse event. These are the errors that would mislead a reader.
- **Independent scales avoid conciseness gaming** in both directions: brevity
  cannot buy accuracy, and verbosity cannot buy coverage.
- **Anchors and abstention keep calibration stable.** The three examples pin
  the 0-3 endpoints; `unassessable` prevents "plausible but unverifiable" from
  being scored either way.

---

## 4. What is *not* judged by an LLM

| Question | Method | Where |
|---|---|---|
| Report describes the clip correctly? | video-grounded judge (2 × 0-3) | `evaluation/judge.py` |
| s1 phase recognition | deterministic vs timeline (`stated_phase` vs chunk phase; accuracy, macro-F1, confusion) | `evaluation/aggregate.py: phase_metrics_s1` |
| Protocol adherence (empty / echo / degenerate / repetition / sections) | deterministic | `evaluation/reliability.py` |
| Lexical agreement with teacher reference | deterministic (ROUGE-L, chrF, instrument-anatomy F1) | `evaluation/reference.py` |
| Chunks complete, ordered, non-overlapping, correctly timestamped in full-video reports | deterministic span validation against expected chunk table | `evaluation/compile_self.py: validate`, CLI `run timeline` |
| Polish preserved the facts | deterministic preservation check, per block, with automatic fallback to verbatim bullets | `evaluation/compile_self.py: preservation_check` |
| Narrative coherence of the assembled full report | human review of a small subset (process step, not automated) | — |

The legacy text-only audits (evidence pass, claim counting, 5-label temporal
transitions) remain in the codebase as **optional diagnostics** only
(`--with-evidence`, `--legacy-claims`, `--legacy-temporal`), kept for
comparison; they no longer feed the default dashboard conclusions.

---

## 5. End-to-end judgement pipeline

```
teacher + candidates already frozen (per-chunk reports + memory)
        │
        ▼
[1] freeze inputs      output-teacher/clips/<video>__<chunk>.mp4  (per chunk)
        │
        ▼
[2] judged pass        batch.py (default, resume-safe, parallel):
        │                for every (tag, setting, video, chunk):
        │                  content = clinical_content(report)
        │                  hash    = sha256(content)[:12]
        │                  skip if a row with same hash, no error exists
        │                  one judge call  →  row {acc, cov, issues, unassessable}
        │                rows → output-evaluation/<run>/judgments/<video>.<tag>.<setting>.jsonl
        ▼
[3] deterministic pass run.py report:
        │                reliability + lexical vs teacher + s1 phase metrics
        │                judged rows joined by chunk_id
        │                → video_scores.jsonl, judge_video_scores.jsonl,
        │                  comparisons.json (panels, judge panels, contrasts)
        ▼
[4] dashboard          summary.md (two tables + contrasts + s1 phase)
                       viewer.html (leaderboard, per-chunk judge tab, judge strip)
        │
        ▼
[5] full-video         compile_self.py → output-readable/<tag>.<setting>.self/*.readable.md
        │                code-locked spine; optional polish with preservation check
        ▼
[6] timeline check     run.py timeline  → validates spans vs expected chunk table
                       (must be 100% ok before a full-video report is used)
[7] review             human review subset for narrative coherence;
                       technical report cites Tier 0 (deterministic) and Tier J (judge)
```

### Step details

- **[2] Judging.** `python src/vlm_report_framework/evaluation/batch.py
  --videos <ids> --tags <tags> --run-id <run>` runs the judge pass by default.
  `--chunks a,b,c` and `--limit N` select calibration subsets. Progress and
  errors are logged; errored rows are saved (visible in the dashboard) and
  retried on the next run.
- **Versioning.** Every judged row records `report_hash` (of the *judged
  clinical content*), `rubric` (`visual-1`), `judge_model`, and `clip_label`.
  A regenerated report with different content automatically gets re-judged;
  identical content is skipped. Old evidence caches can never silently feed
  the new scores.
- **Composition.** Accuracy/coverage means are computed per video over judged
  chunks only (`judged` rate is shown), with idle / non-idle breakdowns.
  Setting contrasts are paired per video on identical chunk sets with a 95%
  bootstrap CI (`s3−s1`).
- **[5] Polish.** The full-video compiler always writes the deterministic
  spine. With `--polish` (local model), each phase block paragraph is checked:
  (a) ≥ 50% of the source bullets' canonical content words must survive
  (synonym-aware), and (b) no instrument may appear that was neither in the
  bullets nor in the block's tool union. A failing block falls back to the
  verbatim bullets; fallback counts land in the `.validation.json` next to the
  report. Timestamps and headers are written by code and can never drift.
- **[6] Timeline integrity** is a separate deterministic gate; correct
  timestamps do **not** imply preserved facts — that is what the step-[2]
  judge and the polish preservation check are for.

### Output locations

| Artifact | Path |
|---|---|
| judge rows (per chain) | `output-evaluation/<run>/judgments/<video>.<tag>.<setting>.jsonl` |
| per-video judge aggregate | `output-evaluation/<run>/judge_video_scores.jsonl` |
| deterministic + judge summaries | `output-evaluation/<run>/summary.md`, `comparisons.json` |
| single-chunk audit (calibration) | `run.py judge --video … --chunk … --tag … --setting …` |
| compiled full-video reports | `output-readable/<tag>.<setting>.self/<video>.readable.md` (+ `.validation.json`) |
| interactive dashboard | `output-evaluation/<run>/viewer.html` |

---

## 6. Calibration and reporting rules

Before comparing models at scale:

1. **Audit the input side once** (clips decode, intended segment is sent).
2. **Calibrate on ~20–30 varied chunks** (clean, repetitive, incorrect,
   uncertain, eventful; idle included) using `--chunks`. Read the issues, not
   just the means.
3. **Freeze the rubric** (`RUBRIC_VERSION`) before model comparisons; never
   tune the prompt to move a specific model's score.
4. **Report accuracy and coverage as ordinal means per video** with idle /
   non-idle breakdowns, `judged` rate, material-issue rate, and the
   `unassessable` rate. Do not print them as percentages or merge them into a
   single "factuality" number.
5. **Keep deterministic metrics beside the judge** (lexical vs teacher,
   reliability, s1 phase accuracy, timeline ok) so every claim in a report can
   quote a reproducible number, not only a model's opinion.

### Worked calibration examples (pilot subset, 2026-10-17)

First calibration run: 24 chunks (8 per pilot video, chosen to include idle,
repetition-heavy, and previously suspicious chunks) × 2 models × 2 settings →
**96 judged reports, 0 errors, 0 abstentions**. These rows pin the scale
endpoints:

| Report (verbatim quote) | Clip, per judge | acc | cov | Reading |
|---|---|---|---|---|
| "Forceps are introduced into the anterior chamber near the limbus." | forceps hold external tissue at the wound edge; they never enter | 2 | 2 | Overstatement that does not change the procedure → minor; coverage loses the missed detail |
| "phacoemulsification handpiece…" | the visible active tool is a coaxial I/A handpiece being withdrawn | 1 | 2 | Instrument confusion that changes the described action → material; some content still correct |
| "None clearly visible / – No change observed." | cannula actively irrigates the anterior chamber | 0 | 0 | No usable description of the clip → both at floor, material |
| "Iris prolapse confirmed… surgery paused for correction of limbal hemorrhage." | no prolapse, no hemorrhage | 0 | 0 | Invented complication dominating the report → material |
| "…the IOL is centered." | phaco probe actively emulsifying the nucleus | 0 | 0 | Invented implant/event → material |
| anatomy section says "None clearly visible" (otherwise clean report) | iris, cornea, and IOL are clearly visible | 3 | 2 | Same words cost coverage but not accuracy: nothing stated is false, the omission is a coverage matter. Shows the two scales are judged independently |
| correct capsulorhexis continuation | steady capsulorhexis | 3 | 3 | Legitimate continuation of an ongoing step is not penalized |

Observations from the run: on this adversarially-chosen subset, material-issue
rates were high for every chain (0.46–0.83); the two dominant material errors
were (a) empty/template descriptions of active surgery (GRPO `s1`), scored 0,
and (b) invented events and implants in the hint-aided SFT `s3` chain, scored
0. The subset ordering (GRPO `s3` > `s1`; SFT `s1` > `s3`) is a rubric signal,
not an estimate — see the follow-up.

**Follow-up — full pilot (same day).** After the rubric was frozen, the
remaining 312 reports were judged: **408/408 judged, 0 errors, judged = 1.0**
on all four chains (`output-evaluation/v2slim/summary.md`). On the full trio
the GRPO ordering held (s3−s1 accuracy +0.44 [0.08, 0.91], coverage +0.28
[−0.03, 0.85]), but the SFT `s1` > `s3` gap did **not** generalize (accuracy
+0.10 [−0.17, 0.38]): this subset had over-sampled the SFT-`s3` fabrication
cases. The lesson is the one above, applied to ourselves — a calibration
subset orders the rubric; only the full run estimates levels.

Known limitations, stated honestly:

- the judge is itself a model; single-call ordinal scores have variance and
  should be read as aggregates over videos, not per-chunk truth;
- borderline "material" judgments can occur where framing is ambiguous; the
  `issues` text is kept so the underlying quote can be inspected;
- per-chunk judging does not see inter-chunk context; chronology/contradiction
  across chunks is a deterministic/structural question and a small human
  review question, not a judge question.
