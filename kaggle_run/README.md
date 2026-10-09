# Kaggle runs (one script per model)

Each script is self-contained for a fresh Kaggle GPU notebook: GPU check →
pip install (pip, not uv) → HF dataset download → repo-layout materialization
(symlinks into `videos/`, `output-instruments/`, `output-teacher/clips/`) →
model weights download → per-video `candidate_reports.py` sweep (all three
settings) → `verify_candidates` audit → zip of `output-candidates/<tag>` for download.
The zip is rebuilt after **each video** (checkpoint), so partial results are
always downloadable if a session dies; reruns resume finished chunks.

```bash
!git clone https://github.com/shahedmomenzadeh/VLM-video-to-report
%cd VLM-video-to-report
!bash kaggle_run/qwen3vl-2b-grpo.sh
```

| script | model | backend | notes |
|---|---|---|---|
| `qwen3vl-2b-instruct.sh` | Qwen/Qwen3-VL-2B-Instruct | qwen3vl | 32 frames |
| `qwen3vl-4b-instruct.sh` | Qwen/Qwen3-VL-4B-Instruct | qwen3vl | 32 frames, 4-bit (fits 8GB+) |
| `qwen3vl-2b-sft-stage2.sh` | shahedm2001/qwen3-vl-2b-cataract-sft-stage2 | qwen3vl | essentials-only download |
| `qwen3vl-2b-grpo.sh` | shahedm2001/qwen3-vl-2b-cataract-grpo | qwen3vl | essentials-only download |
| `hulumed-4b.sh` | ZJU-AI4H/Hulu-Med-4B | hulumed | **16 frames @ 224px**, temp 0.6 (32f@480px exceeds its 16k context + OOMs on 8GB) |
| `hulumed-4b.sh` | ZJU-AI4H/Hulu-Med-4B | hulumed | **16 frames @ 224px**, temp 0.6 (32f@480px exceeds its 16k context + OOMs on 8GB) |
| `hulumed-7b.sh` | ZJU-AI4H/Hulu-Med-7B | hulumed | same 16f@224px operating point (reference run.sh used 224); 16GB GPU recommended |
| `lingshu-7b.sh` | lingshu-medical-mllm/Lingshu-7B | lingshu | 16GB GPU recommended; frames halve automatically on OOM |

Env knobs: `VIDEOS` (default `"all"`, the 25-video corpus; set to the pilot
trio `PH_0001_2931_S2,PH_0043_0096_S1,PH_0057_0239_S1` for a quick pass), `MAX_FRAMES`, `OUTZIP`, `HF_TOKEN` (optional;
all repos are public). Reruns resume — finished chunks are skipped. Model
tags match local runs, so returned zips merge directly into `output-candidates/`.
