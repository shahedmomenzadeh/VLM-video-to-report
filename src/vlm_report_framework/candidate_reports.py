"""Part II candidate reports: frozen C(i,j) x {s1, s3} -> R(model, setting).

v2.1: students write tagged plain text (no JSON, zero retries). The single
generate call per chunk is parsed by report_render.parse_student_tags and
rendered to the canonical 5-section wire text — byte-identical headers to
the teacher, so Tier 0/Tier 1/temporal evals work unchanged.

Frozen inputs (identical across models and settings):
  clips:  output-teacher/clips/<VID>__<chunk>.mp4  (reused, never re-cut)
  tiers:  output-instruments/<VID>_chunk_instruments.csv
  phases: videos/<VID>/<VID>.timeline.json

Memory M(i,j) is the slim rolling STATE history per (model, setting, video),
chained from the candidate's own STATE lines in temporal order; every step
snapshotted (see memory.py).

Outputs (output-candidates/, git-ignored):
  <tag>/<setting>/reports/<VID>__<chunk>.md
  <tag>/<setting>/reports/<VID>__<chunk>.json
  <tag>/<setting>/memory/<VID>_memory.jsonl
  <tag>/<setting>/<VID>_reports.jsonl          master table

Usage (from repo root, with the model family's venv python):
  .venv-vlm/bin/python src/vlm_report_framework/candidate_reports.py \\
      --model <path> --backend qwen3vl --model-tag qwen3vl-2b-grpo \\
      --settings s1,s3 --videos PH_0001_2931_S2
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from backends import get_backend  # noqa: E402
from candidate_prompt import (  # noqa: E402
    build_candidate_prompt,
    stated_phase,
)
from clean_instruments import make_chunks  # noqa: E402
from memory import fresh_memory, migrate_memory, update_memory  # noqa: E402
from report_render import parse_student_tags  # noqa: E402
from run_state import (  # noqa: E402
    REPO,
    append_master,
    estimate_frames,
    load_chunk_instruments,
    load_timeline,
    read_done_ids,
    rebuild_memory,
    resolve_videos,
    snapshot,
    split_tiers,
)

OUT = REPO / "output-candidates"
CLIPS = REPO / "output-teacher" / "clips"

SETTINGS = ("s1", "s3")
# Explicit aliases for the two inference conditions (normalized to s1/s3;
# outputs always use the canonical ids so the schema stays stable).
SETTING_ALIASES = {
    "video": "s1",                    # 1. video chunk only
    "full": "s3", "phase": "s3", "phase+instrument": "s3",
    "phase-instrument": "s3",
}                                     # 2. chunk + phase + instruments
DEFAULT_VIDEOS = ["PH_0001_2931_S2", "PH_0043_0096_S1", "PH_0057_0239_S1"]


def effective_phase(setting: str, chunk: dict, cleaned: dict) -> str:
    if setting == "s1":
        return cleaned.get("phase_id") or "P??"
    return chunk["phase"]


def seen_for(setting: str, chunk: dict, tiers: dict) -> list[str] | None:
    if setting == "s3" and chunk["phase"] != "P13":
        return tiers["observed"]
    return None


def run_combo(backend, ctx, args, video_id: str, setting: str) -> None:
    segments = load_timeline(video_id)
    chunks = make_chunks(segments)
    by_id = {c["chunk_id"]: c for c in chunks}
    if args.chunks:
        wanted = [c.strip() for c in args.chunks.split(",") if c.strip()]
    else:
        wanted = [c["chunk_id"] for c in chunks]
    for cid in wanted:
        assert cid in by_id, f"unknown chunk {cid} for {video_id}"

    ci = load_chunk_instruments(video_id)
    base = OUT / args.model_tag / setting
    rep_dir, mem_dir = base / "reports", base / "memory"
    rep_dir.mkdir(parents=True, exist_ok=True)
    mem_dir.mkdir(parents=True, exist_ok=True)
    mem_path = mem_dir / f"{video_id}_memory.jsonl"
    master_path = base / f"{video_id}_reports.jsonl"

    done = read_done_ids(master_path)
    mem = migrate_memory(rebuild_memory(mem_path, fresh_memory()))
    gen = {"max_pixels": args.max_pixels, "min_pixels": args.min_pixels,
           "max_new_tokens": args.max_new_tokens,
           "temperature": args.temperature,
           "fps": args.fps, "frame_size": args.frame_size}

    for cid in wanted:
        if cid in done:
            print(f"-- skip {video_id}/{setting}/{cid} (done)")
            continue
        ch = by_id[cid]
        total_frames = estimate_frames(ch)
        sub = ci[ci["chunk_id"] == cid]
        tiers, obs_rows, weak_rows = split_tiers(sub)
        if setting == "s1":
            mem["same_phase_continuation"] = False
        else:
            mem["same_phase_continuation"] = bool(
                mem["phase_trail"] and mem["phase_trail"][-1] == ch["phase"])
        prompt = build_candidate_prompt(ch, setting, obs_rows, weak_rows, mem,
                                        total_frames)
        clip = CLIPS / f"{video_id}__{cid}.mp4"
        assert clip.exists(), f"missing frozen clip: {clip}"

        print(f"\n=== {video_id} {setting} {cid} ({ch['phase']}, "
              f"{ch['t_end'] - ch['t_start']:.1f}s) ===", flush=True)
        t0 = time.time()
        raw, n_frames = backend.generate_text(
            ctx, str(clip), prompt, args.max_frames, gen)
        dt = time.time() - t0
        print(f"  response in {dt:.0f}s ({n_frames} frames)", flush=True)
        # Zero-retry: parsing is total — every raw string yields a report.
        cleaned, report = parse_student_tags(raw, setting, ch)
        memory_update = cleaned.get("memory_update", "")
        flags_add = cleaned.get("flags_add", [])

        (rep_dir / f"{video_id}__{cid}.md").write_text(report)
        (rep_dir / f"{video_id}__{cid}.json").write_text(json.dumps(
            {"model": args.model_tag, "setting": setting,
             "video_id": video_id, "chunk_id": cid, "chunk": ch,
             "tiers_given": tiers if setting == "s3" else None,
             "n_frames_used": n_frames, "prompt": prompt,
             "raw_response": raw, "parsed": cleaned,
             "latency_s": round(dt, 1)}, indent=2))
        mem = update_memory(
            mem, ch["chunk_id"],
            effective_phase(setting, ch, cleaned),
            report, memory_update, flags_add,
            seen_classes=seen_for(setting, ch, tiers),
            idle_phases=("P13", "P??"))
        snapshot(mem_path, cid, mem)
        append_master(master_path,
                      {"model": args.model_tag, "setting": setting,
                       "video_id": video_id, "chunk_id": cid,
                       "phase": ch["phase"], "phase_name": ch["phase_name"],
                       "stated_phase": (cleaned.get("phase_id")
                                        if setting == "s1" else ch["phase"]),
                       "t_start": ch["t_start"], "t_end": ch["t_end"],
                       "tiers_given": tiers if setting == "s3" else None,
                       "report": report, "parsed": cleaned,
                       "memory_update": memory_update,
                       "flags_add": flags_add, "n_frames_used": n_frames,
                       "temperature": args.temperature,
                       "latency_s": round(dt, 1)})
        print(f"  saved ({len(report)} chars, "
              f"tools={cleaned.get('instruments_visible')})", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="local path or HF id")
    ap.add_argument("--backend", default="qwen3vl",
                    help="model family backend: qwen3vl (.venv-qwen3vl), "
                         "hulumed (.venv-hulumed), lingshu (.venv-qwen3vl)")
    ap.add_argument("--model-tag", required=True)
    ap.add_argument("--videos", default=",".join(DEFAULT_VIDEOS))
    ap.add_argument("--settings", default="s1,s3",
                    help="subset of {s1,s3} or aliases {video,full}; "
                         "s1=video only, s3=+phase+instruments")
    ap.add_argument("--chunks", default=None)
    ap.add_argument("--max-frames", type=int, default=32,
                    help="frames sampled per chunk; short clips feed all their frames")
    ap.add_argument("--max-pixels", type=int, default=307200)
    ap.add_argument("--min-pixels", type=int, default=100352)
    ap.add_argument("--fps", type=float, default=1.0,
                    help="hulumed only: sampling fps in the video dict")
    ap.add_argument("--frame-size", type=int, default=480,
                    help="hulumed only: frame size in the video dict")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.7)
    args = ap.parse_args()

    raw_settings = [s.strip().lower() for s in args.settings.split(",") if s.strip()]
    settings = [SETTING_ALIASES.get(s, s) for s in raw_settings]
    assert set(settings) <= set(SETTINGS), settings
    videos = resolve_videos(args.videos, DEFAULT_VIDEOS)

    backend = get_backend(args.backend, args.model)
    print(f"Loading {args.model} (backend={args.backend}) ...", flush=True)
    ctx = backend.load()
    print("Model loaded.", flush=True)
    for setting in settings:
        for vid in videos:
            run_combo(backend, ctx, args, vid, setting)
    print("Done.")


if __name__ == "__main__":
    main()
