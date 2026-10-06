"""Part II candidate reports: frozen C(i,j) x {s1, s2, s3} -> R(model, setting).

Each model family implements the VideoBackend interface (backends/) and runs
under its own venv (incompatible dependency versions — see E:/VLM_Evaluation).
Only the video ingest/generation differs per family; prompts, memory,
resume handling and outputs are shared.

Frozen inputs (identical across models and settings):
  clips:  output-teacher/clips/<VID>__<chunk>.mp4  (reused, never re-cut)
  tiers:  output-instruments/<VID>_chunk_instruments.csv
  phases: videos/<VID>/<VID>.timeline.json

Memory M(i,j) is chained per (model, setting, video) from the candidate's
own outputs in temporal order; every step snapshotted (see memory.py and
candidate_prompt.py for the s1/s3 rules).

Outputs (output-candidates/, git-ignored):
  <tag>/<setting>/reports/<VID>__<chunk>.md
  <tag>/<setting>/reports/<VID>__<chunk>.json
  <tag>/<setting>/memory/<VID>_memory.jsonl
  <tag>/<setting>/<VID>_reports.jsonl          master table

Usage (from repo root, with the model family's venv python):
  /mnt/e/VLM_Evaluation/.venv-qwen3vl/bin/python \\
      src/vlm_report_framework/candidate_reports.py \\
      --model <path-or-id> --backend qwen3vl \\
      --model-tag qwen3vl-2b-instruct --settings s1 --videos PH_0057_0239_S1 --chunks P03_o01
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
from memory import fresh_memory, update_memory  # noqa: E402
from run_state import (  # noqa: E402
    REPO,
    append_master,
    coerce_report,
    estimate_frames,
    load_chunk_instruments,
    load_timeline,
    parse_json_loose,
    read_done_ids,
    rebuild_memory,
    resolve_videos,
    snapshot,
    split_tiers,
)

OUT = REPO / "output-candidates"
CLIPS = REPO / "output-teacher" / "clips"

SETTINGS = ("s1", "s2", "s3")
# Explicit aliases for the three inference conditions (normalized to s1/s2/s3;
# outputs always use the canonical ids so the schema stays stable).
SETTING_ALIASES = {
    "video": "s1",                    # 1. video chunk only
    "phase": "s2",                    # 2. chunk + phase name
    "full": "s3", "phase+instrument": "s3", "phase-instrument": "s3",
}                                     # 3. chunk + phase + instruments
DEFAULT_VIDEOS = ["PH_0001_2931_S2", "PH_0043_0096_S1", "PH_0057_0239_S1"]

REFORMAT_INSTRUCTION = (
    "Your previous response below is not valid JSON. Reformat the "
    "SAME content as one JSON object with exactly the keys "
    '"report" (plain-text string), "memory_update" and "flags_add". '
    "Respond with ONLY the JSON object, no other text.\n\n"
    "Previous response:\n")


def effective_phase(setting: str, chunk: dict, report: str) -> str:
    if setting == "s1":
        return stated_phase(report)
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
    mem = rebuild_memory(mem_path, fresh_memory())
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
        try:
            parsed = parse_json_loose(raw)
        except (ValueError, AttributeError):
            print("  not JSON — one reformat retry", flush=True)
            t1 = time.time()
            raw, _ = backend.generate_text(
                ctx, str(clip), REFORMAT_INSTRUCTION + raw[:3000],
                n_frames, gen)
            dt += time.time() - t1
            try:
                parsed = parse_json_loose(raw)
            except (ValueError, AttributeError) as e:
                print(f"  WARN: unparseable after retry ({e}); storing raw", flush=True)
                parsed = {}
        report, memory_update, flags_add = coerce_report(parsed, raw)
        retried = not parsed
        if retried:
            print("  WARN: empty parse; memory_update lost for this chunk", flush=True)

        (rep_dir / f"{video_id}__{cid}.md").write_text(report or raw)
        (rep_dir / f"{video_id}__{cid}.json").write_text(json.dumps(
            {"model": args.model_tag, "setting": setting,
             "video_id": video_id, "chunk_id": cid, "chunk": ch,
             "tiers_given": tiers if setting == "s3" else None,
             "n_frames_used": n_frames, "prompt": prompt,
             "raw_response": raw, "latency_s": round(dt, 1)}, indent=2))
        mem = update_memory(
            mem, ch["chunk_id"],
            effective_phase(setting, ch, report or raw),
            report or raw, memory_update, flags_add,
            seen_classes=seen_for(setting, ch, tiers),
            idle_phases=("P13", "P??"))
        snapshot(mem_path, cid, mem)
        append_master(master_path,
                      {"model": args.model_tag, "setting": setting,
                       "video_id": video_id, "chunk_id": cid,
                       "phase": ch["phase"], "phase_name": ch["phase_name"],
                       "stated_phase": (stated_phase(report or raw)
                                        if setting == "s1" else ch["phase"]),
                       "t_start": ch["t_start"], "t_end": ch["t_end"],
                       "tiers_given": tiers if setting == "s3" else None,
                       "report": report or raw, "memory_update": memory_update,
                       "flags_add": flags_add, "n_frames_used": n_frames,
                       "temperature": args.temperature,
                       "reformat_retried": retried,
                       "latency_s": round(dt, 1)})
        print(f"  saved ({len(report or raw)} chars)", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="local path or HF id")
    ap.add_argument("--backend", default="qwen3vl",
                    help="model family backend: qwen3vl (.venv-qwen3vl), "
                         "hulumed (.venv-hulumed), lingshu (.venv-qwen3vl)")
    ap.add_argument("--model-tag", required=True)
    ap.add_argument("--videos", default=",".join(DEFAULT_VIDEOS))
    ap.add_argument("--settings", default="s1,s2,s3",
                    help="subset of {s1,s2,s3} or aliases {video,phase,full}; "
                         "1=video only, 2=+phase name, 3=+phase+instruments")
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
