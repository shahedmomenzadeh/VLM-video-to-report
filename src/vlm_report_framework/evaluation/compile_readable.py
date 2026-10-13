"""Presentation stage: compile one readable full-video report per video.

READ-ONLY w.r.t. per-chunk reports — masters are read, never modified.
Not scored: output is for human review only, kept out of Tier 1 tables.

Usage:
  .venv-vlm/bin/python src/vlm_report_framework/evaluation/compile_readable.py \
      --tag qwen3vl-2b-grpo --setting s3 --workers 4
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from evaluation.evidence import _client  # noqa: E402
from run_state import REPO  # noqa: E402

MODEL = "ag/gemini-3.8-flash"
OUT = REPO / "output-readable"

SYSTEM = ("You are a surgical report editor. You reorganize operative notes "
          "into a readable report. You never add facts.")
CONTRACT = """Rewrite the chunk reports below as ONE readable operative report that
narrates the video from start to finish IN TEMPORAL ORDER.
Rules:
- Emit one section per video chunk, in the given order. You MAY merge
  consecutive same-phase chunks into a single section with the combined span.
- NEVER reorder or group by phase: Idle sections stay exactly where they
  occur in the video. Do not collect all Idles together.
- Exact section format, repeated for every section:
## PXX: Phase Name
[MM:SS.ss-MM:SS.ss]: narrative paragraph for that span
- Time spans MUST be copied verbatim from the chunk headers below (they are
  already in MM:SS.ss — no conversion needed). The first span starts at
  00:00.00 and the last span ends at the full video duration given above.
- Output ONLY the report sections. No preamble, no timestamp math or
  conversions, no commentary on what you did, no notes section.
- DO NOT correct any evidence or claim: preserve every factual statement
  exactly as written, even where chunks contradict each other. If chunks
  disagree, keep both phrasings and mark with "(reports differ)".
- Restructure prose and flow ONLY. Add NO new facts, tools, actions, timings,
  or interpretations. Plain markdown, no JSON."""


def fmt_mmss(t: float) -> str:
    return f"{int(t // 60):02d}:{t % 60:05.2f}"


def compile_video(tag: str, setting: str, vid: str) -> tuple[str, str]:
    mp = REPO / "output-candidates" / tag / setting / f"{vid}_reports.jsonl"
    rows = [json.loads(l) for l in open(mp)]
    rows.sort(key=lambda r: (r.get("t_start", 0), r.get("t_end", 0)))
    dur = max(r.get("t_end", 0) for r in rows)
    body = "\n\n".join(
        f"[{r.get('chunk_id')} | {r.get('phase')} | "
        f"{fmt_mmss(r.get('t_start', 0))}-{fmt_mmss(r.get('t_end', 0))}]\n"
        f"{r.get('report', '')}"
        for r in rows)
    client = _client()
    t0 = time.time()
    resp = client.chat.completions.create(
        model=MODEL, temperature=0.0, max_tokens=4096,
        messages=[{"role": "system", "content": SYSTEM},
                  {"role": "user", "content":
                   f"Video {vid}, {len(rows)} chunks in order, full duration "
                   f"{fmt_mmss(dur)}.\n\n{body}\n\n{CONTRACT}"}],
        timeout=600)
    text = resp.choices[0].message.content or ""
    header = (f"<!-- presentation only — compiled from {tag}/{setting} "
              f"({len(rows)} frozen chunk reports, untouched). Not scored. -->\n\n")
    return vid, header + text


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="qwen3vl-2b-grpo")
    ap.add_argument("--setting", default="s3")
    ap.add_argument("--videos", default="all")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    if a.videos == "all":
        vids = sorted(p.name for p in (REPO / "videos").glob("PH_*") if p.is_dir())
    else:
        vids = [v.strip() for v in a.videos.split(",") if v.strip()]
    outdir = OUT / f"{a.tag}.{a.setting}"
    outdir.mkdir(parents=True, exist_ok=True)
    todo = [v for v in vids if not (outdir / f"{v}.readable.md").exists()]
    print(f"{len(todo)} videos to compile ({len(vids) - len(todo)} cached), "
          f"workers={a.workers}", flush=True)
    lock = threading.Lock()

    def one(vid: str) -> str:
        try:
            v, text = compile_video(a.tag, a.setting, vid)
            with lock:
                (outdir / f"{v}.readable.md").write_text(text)
            return f"ok {v} ({len(text)} chars)"
        except Exception as e:  # noqa: BLE001
            return f"WARN {vid}: {e}"[:200]

    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        futs = [pool.submit(one, v) for v in todo]
        for i, fut in enumerate(as_completed(futs)):
            print(f"  {i + 1}/{len(todo)}: {fut.result()}", flush=True)
    print(f"Done -> {outdir.relative_to(REPO)}")


if __name__ == "__main__":
    main()
