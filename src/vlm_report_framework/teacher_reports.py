"""Part I teacher reports: C(i,j) + P(i) + I(i,j) + M(i,j) -> R_teacher(i,j).

- Chunk video is cut with ffmpeg (frame-accurate re-encode) and sent as a
  single base64 MP4 via OpenAI-compatible endpoint (payload key: video_url).
- Endpoint: http://localhost:20128/v1, model ag/gemini-3.8-flash.
- Hard ceiling: no chunk longer than 240 s (4 min) is ever sent.
- Memory M(i,j) is built causally in temporal order; every step snapshotted.
  Shared memory/prompt/run-state logic lives in memory.py, prompt_base.py,
  teacher_prompt.py and run_state.py.

Outputs (output-teacher/, git-ignored except reports are small text):
  clips/<VID>__<chunk>.mp4            cached subclips
  reports/<VID>__<chunk>.md           clean reference report
  reports/<VID>__<chunk>.json         raw response + usage + prompt tiers
  memory/<VID>_memory.jsonl           M(i,j) snapshots
  <VID>_teacher_reports.jsonl         master table

Usage:
  uv run python src/vlm_report_framework/teacher_reports.py --probe
  uv run python src/vlm_report_framework/teacher_reports.py                 # smoke test (4 chunks)
  uv run python src/vlm_report_framework/teacher_reports.py --all
  uv run python src/vlm_report_framework/teacher_reports.py --chunks P03_o01,P05_o01
  uv run python src/vlm_report_framework/teacher_reports.py --videos a,b,c --workers 3
"""
from __future__ import annotations

import argparse
import base64
import json
import subprocess
import sys
import time
from pathlib import Path

from openai import OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parent))
from clean_instruments import make_chunks  # noqa: E402 (reuse chunking rule)
from memory import fresh_memory, update_memory  # noqa: E402
from run_state import (  # noqa: E402
    REPO,
    append_master,
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
from teacher_prompt import SYSTEM_PROMPT, build_user_prompt  # noqa: E402

OUT = REPO / "output-teacher"
CLIPS = OUT / "clips"
REPORTS = OUT / "reports"
MEMORY = OUT / "memory"

BASE_URL = "http://localhost:20128/v1"
MODEL = "ag/gemini-3.8-flash"
MAX_SEND_S = 240.0  # 4-min ceiling
TEMPERATURE = 0.2
MAX_TOKENS = 1024
RETRIES = 3

VIDEO_ID = "PH_0057_0239_S1"
# Smoke test: contiguous span exercising memory across a phase boundary.
SMOKE_CHUNKS = ["P03_o01", "P13_o10", "P04_o01", "P05_o01"]
DEFAULT_VIDEOS = ["PH_0001_2931_S2", "PH_0043_0096_S1", "PH_0057_0239_S1"]


def cut_clip(video_id: str, chunk: dict) -> Path:
    dur = chunk["t_end"] - chunk["t_start"]
    assert dur <= MAX_SEND_S, f"{chunk['chunk_id']} is {dur:.0f}s > {MAX_SEND_S:.0f}s — re-split first"
    CLIPS.mkdir(parents=True, exist_ok=True)
    out = CLIPS / f"{video_id}__{chunk['chunk_id']}.mp4"
    if out.exists():
        return out
    src = REPO / "videos" / video_id / f"{video_id}.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", str(src),
         "-ss", f"{chunk['t_start']:.2f}", "-to", f"{chunk['t_end']:.2f}",
         "-c:v", "libx264", "-crf", "20", "-preset", "fast", "-an", str(out)],
        check=True,
    )
    return out


def call_vlm(client: OpenAI, prompt: str, clip: Path) -> tuple[str, dict, float]:
    b64 = base64.b64encode(clip.read_bytes()).decode()
    last_err: Exception | None = None
    for attempt in range(1, RETRIES + 1):
        try:
            t0 = time.time()
            resp = client.chat.completions.create(
                model=MODEL,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                messages=[{"role": "system", "content": SYSTEM_PROMPT},
                          {"role": "user", "content": [
                              {"type": "text", "text": prompt},
                              {"type": "video_url", "video_url": {
                                  "url": f"data:video/mp4;base64,{b64}"}},
                          ]}],
                timeout=600,
            )
            dt = time.time() - t0
            msg = resp.choices[0].message.content or ""
            usage = {}
            if resp.usage:
                usage = {"prompt_tokens": resp.usage.prompt_tokens,
                         "completion_tokens": resp.usage.completion_tokens,
                         "total_tokens": resp.usage.total_tokens}
            return msg, usage, dt
        except Exception as e:  # noqa: BLE001 — retry transient gateway errors
            last_err = e
            print(f"  attempt {attempt}/{RETRIES} failed: {str(e)[:200]}", flush=True)
            time.sleep(2 ** attempt)
    raise RuntimeError(f"VLM call failed after {RETRIES} tries: {last_err}")


def run_video(video_id: str, chunk_ids: list[str] | None) -> None:
    segments = load_timeline(video_id)
    chunks = make_chunks(segments)
    by_id = {c["chunk_id"]: c for c in chunks}
    if chunk_ids is None:  # --all
        wanted = [c["chunk_id"] for c in chunks]
    else:
        wanted = chunk_ids
    for cid in wanted:
        assert cid in by_id, f"unknown chunk {cid} for {video_id}"

    ci = load_chunk_instruments(video_id)
    REPORTS.mkdir(parents=True, exist_ok=True)
    MEMORY.mkdir(parents=True, exist_ok=True)
    mem_path = MEMORY / f"{video_id}_memory.jsonl"
    master_path = OUT / f"{video_id}_teacher_reports.jsonl"
    # resume: skip chunks already in master table
    done = read_done_ids(master_path)
    # rebuild memory from snapshots if resuming mid-sequence
    mem = rebuild_memory(mem_path, fresh_memory())

    client = OpenAI(base_url=BASE_URL, api_key="not-needed")
    for cid in wanted:
        if cid in done:
            print(f"-- skip {cid} (already in master table)")
            continue
        ch = by_id[cid]
        total_frames = estimate_frames(ch)
        sub = ci[ci["chunk_id"] == cid]
        tiers, obs_rows, weak_rows = split_tiers(sub)

        mem["same_phase_continuation"] = bool(
            mem["phase_trail"] and mem["phase_trail"][-1] == ch["phase"])
        prompt = build_user_prompt(ch, obs_rows, weak_rows, mem, total_frames)
        print(f"\n=== {cid} ({ch['phase']}-{ch['phase_name']}, "
              f"{ch['t_start']:.1f}-{ch['t_end']:.1f}s) ===", flush=True)
        print(f"  tiers: observed={tiers['observed']} weak={tiers['weak']}", flush=True)

        clip = cut_clip(video_id, ch)
        print(f"  clip: {clip.name} ({clip.stat().st_size / 1e6:.1f} MB)", flush=True)
        raw, usage, dt = call_vlm(client, prompt, clip)
        print(f"  response in {dt:.0f}s, usage={usage}", flush=True)
        try:
            parsed = parse_json_loose(raw)
            report = parsed.get("report", "")
            memory_update = parsed.get("memory_update", "")
            flags_add = parsed.get("flags_add", []) or []
        except (ValueError, json.JSONDecodeError, AttributeError) as e:
            print(f"  WARN: unparseable response ({e}); storing raw", flush=True)
            report, memory_update, flags_add = raw, "", []

        (REPORTS / f"{video_id}__{cid}.md").write_text(report or raw)
        (REPORTS / f"{video_id}__{cid}.json").write_text(json.dumps(
            {"video_id": video_id, "chunk_id": cid, "chunk": ch,
             "tiers": tiers, "total_frames_est": total_frames,
             "prompt": prompt, "raw_response": raw,
             "usage": usage, "latency_s": round(dt, 1)}, indent=2))
        mem = update_memory(
            mem, ch["chunk_id"], ch["phase"], report or raw, memory_update,
            flags_add,
            seen_classes=tiers["observed"] if ch["phase"] != "P13" else None)
        snapshot(mem_path, cid, mem)
        append_master(master_path,
                      {"video_id": video_id, "chunk_id": cid,
                       "phase": ch["phase"], "phase_name": ch["phase_name"],
                       "t_start": ch["t_start"], "t_end": ch["t_end"],
                       "tiers": tiers, "report": report or raw,
                       "memory_update": memory_update, "flags_add": flags_add,
                       "usage": usage, "latency_s": round(dt, 1)})
        print(f"  saved report ({len(report or raw)} chars)", flush=True)
    print(f"\nDone. Master table: {master_path.relative_to(REPO)}")


def probe() -> None:
    """1-chunk end-to-end probe (tiny P08_o01 clip) to validate the pipeline."""
    segments = load_timeline(VIDEO_ID)
    chunks = make_chunks(segments)
    ch = next(c for c in chunks if c["chunk_id"] == "P08_o01")
    clip = cut_clip(VIDEO_ID, ch)
    ci = load_chunk_instruments(VIDEO_ID)
    sub = ci[ci["chunk_id"] == "P08_o01"]
    _, obs_rows, weak_rows = split_tiers(sub)
    prompt = build_user_prompt(ch, obs_rows, weak_rows, fresh_memory(),
                               estimate_frames(ch))
    print(prompt)
    print(f"\n--- clip: {clip} ({clip.stat().st_size / 1e6:.2f} MB) ---")
    client = OpenAI(base_url=BASE_URL, api_key="not-needed")
    raw, usage, dt = call_vlm(client, prompt, clip)
    print(f"--- response ({dt:.0f}s, {usage}) ---\n{raw}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=VIDEO_ID)
    ap.add_argument("--videos", default=None,
                    help="comma-separated video IDs (all chunks each)")
    ap.add_argument("--chunks", default=None,
                    help="comma-separated chunk ids (default: smoke test)")
    ap.add_argument("--all", action="store_true", help="run all chunks of the video")
    ap.add_argument("--probe", action="store_true", help="1-chunk end-to-end probe")
    ap.add_argument("--workers", type=int, default=1,
                    help="parallel videos, one worker per video (VLM calls are independent across videos)")
    args = ap.parse_args()
    if args.probe:
        probe()
    elif args.videos:
        vids = resolve_videos(args.videos, DEFAULT_VIDEOS)
        if args.workers > 1 and len(vids) > 1:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=args.workers) as ex:
                list(ex.map(lambda v: run_video(v, None), vids))
        else:
            for v in vids:
                run_video(v, None)
    elif args.all:
        run_video(args.video, None)
    elif args.chunks:
        run_video(args.video, [c.strip() for c in args.chunks.split(",") if c.strip()])
    else:
        run_video(args.video, SMOKE_CHUNKS)


if __name__ == "__main__":
    main()
