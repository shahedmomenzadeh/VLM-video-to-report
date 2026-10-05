"""Part II candidate reports: frozen C(i,j) x {s1, s2, s3} -> R(model, setting).

Video input pattern (local .mp4 path + sampled frames, 4-bit local HF
inference) follows E:/VLM_Evaluation/qwen3VL_inference.py; each model family
runs under its own venv (here: .venv-qwen3vl for the Qwen3-VL pilot).

Frozen inputs (identical across models and settings):
  clips:  output-teacher/clips/<VID>__<chunk>.mp4  (reused, never re-cut)
  tiers:  output-instruments/<VID>_chunk_instruments.csv
  phases: videos/<VID>/<VID>.timeline.json

Memory M(i,j) is chained per (model, setting, video) from the candidate's
own outputs in temporal order; every step snapshotted (same schema as the
teacher memory, but s1 phase_trail holds the model's stated phase and
instruments_seen is s3-only — see candidate_prompt.py).

Outputs (output-candidates/, git-ignored):
  <tag>/<setting>/reports/<VID>__<chunk>.md
  <tag>/<setting>/reports/<VID>__<chunk>.json
  <tag>/<setting>/memory/<VID>_memory.jsonl
  <tag>/<setting>/<VID>_reports.jsonl          master table

Usage (from repo root, with the model family's venv python):
  /mnt/e/VLM_Evaluation/.venv-qwen3vl/bin/python \\
      src/vlm_report_framework/candidate_reports.py \\
      --model /mnt/e/VLM_Evaluation/hf_cache/hub/models--Qwen--Qwen3-VL-2B-Instruct/snapshots/89644892e4d85e24eaac8bacfd4f463576704203 \\
      --model-tag qwen3vl-2b-instruct --settings s1 --videos PH_0057_0239_S1 --chunks P03_o01
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import pandas as pd
import torch
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
)
from transformers import Qwen3VLForConditionalGeneration as ModelClass

sys.path.insert(0, str(Path(__file__).resolve().parent))
from candidate_prompt import build_candidate_prompt, stated_phase  # noqa: E402
from clean_instruments import make_chunks  # noqa: E402
from teacher_reports import fresh_memory, parse_json_loose  # noqa: E402

REPO = Path.cwd()
OUT = REPO / "output-candidates"
CLIPS = REPO / "output-teacher" / "clips"

SETTINGS = ("s1", "s2", "s3")
DEFAULT_VIDEOS = ["PH_0001_2931_S2", "PH_0043_0096_S1", "PH_0057_0239_S1"]
FPS_GUESS = 5.0


def load_model(model_path: str, use_4bit: bool = True):
    qcfg = None
    if use_4bit:
        qcfg = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = ModelClass.from_pretrained(
        model_path, quantization_config=qcfg, device_map="auto",
        torch_dtype=torch.float16, low_cpu_mem_usage=True,
        trust_remote_code=True)
    model.eval()
    return processor, model


def build_inputs(processor, video_path: str, question: str,
                 max_frames: int, max_pixels: int, min_pixels: int):
    from qwen_vl_utils import process_vision_info

    messages = [{"role": "user", "content": [
        {"type": "video", "video": video_path, "nframes": max_frames,
         "max_pixels": max_pixels, "min_pixels": min(min_pixels, max_pixels)},
        {"type": "text", "text": question}]}]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs,
                       padding=True, return_tensors="pt")
    for k, v in inputs.items():
        if torch.is_tensor(v):
            inputs[k] = v.to("cuda" if torch.cuda.is_available() else "cpu")
            if torch.is_floating_point(inputs[k]):
                inputs[k] = inputs[k].to(torch.float16)
    return inputs, text


def clip_frame_count(video_path: str) -> int | None:
    """Total frames in the clip (decord); None if unreadable."""
    try:
        from decord import VideoReader
        return len(VideoReader(video_path))
    except Exception:
        return None


@torch.no_grad()
def generate(processor, model, video_path: str, prompt: str, args) -> tuple[str, float, int]:
    total = clip_frame_count(video_path)
    n = args.max_frames if total is None else max(2, min(args.max_frames, total))
    last_err = None
    while n >= 2:
        try:
            inputs, _ = build_inputs(processor, video_path, prompt, n,
                                     args.max_pixels, args.min_pixels)
            out = model.generate(
                **inputs, max_new_tokens=args.max_new_tokens,
                do_sample=(args.temperature > 0.0),
                temperature=max(args.temperature, 1e-6),
                top_p=0.8, top_k=20, min_p=0.0,
                repetition_penalty=1.05, use_cache=True,
                eos_token_id=[151645, 151643],
                pad_token_id=processor.tokenizer.eos_token_id)
            gen = out[0][inputs["input_ids"].shape[1]:]
            txt = processor.tokenizer.decode(gen, skip_special_tokens=True)
            import re as _re
            txt = _re.sub(r"<think>.*?</think>", "", txt, flags=_re.DOTALL).strip()
            return txt, n, 0.0
        except torch.cuda.OutOfMemoryError as e:
            last_err = e
            gc.collect()
            torch.cuda.empty_cache()
            n //= 2
        except ValueError as e:
            # e.g. short clip: requested nframes > available frames
            if "nframes" not in str(e):
                raise
            last_err = e
            n //= 2
    raise RuntimeError(f"Failed even at 2 frames: {last_err}")


@torch.no_grad()
def reformat_retry(processor, model, video_path: str, prev: str,
                   args, n_frames: int) -> str:
    """Second chance: ask the model to reformat its answer as valid JSON."""
    prompt = ("Your previous response below is not valid JSON. Reformat the "
              "SAME content as one JSON object with exactly the keys "
              '"report" (plain-text string), "memory_update" and "flags_add". '
              "Respond with ONLY the JSON object, no other text.\n\n"
              f"Previous response:\n{prev[:3000]}")
    inputs, _ = build_inputs(processor, video_path, prompt, n_frames,
                             args.max_pixels, args.min_pixels)
    out = model.generate(
        **inputs, max_new_tokens=args.max_new_tokens,
        do_sample=(args.temperature > 0.0),
        temperature=max(args.temperature, 1e-6),
        top_p=0.8, top_k=20, min_p=0.0,
        repetition_penalty=1.05, use_cache=True,
        eos_token_id=[151645, 151643],
        pad_token_id=processor.tokenizer.eos_token_id)
    gen = out[0][inputs["input_ids"].shape[1]:]
    return processor.tokenizer.decode(gen, skip_special_tokens=True).strip()


def coerce_report(parsed: dict, raw: str) -> tuple[str, str, list]:
    """Extract (report, memory_update, flags_add), coercing nested objects."""
    try:
        report = parsed.get("report", "")
        memory_update = parsed.get("memory_update", "")
        flags_add = parsed.get("flags_add", []) or []
    except (ValueError, AttributeError) as e:
        print(f"  WARN: unparseable ({e}); storing raw", flush=True)
        return raw if isinstance(raw, str) else json.dumps(raw), "", []
    if isinstance(report, dict):
        report = json.dumps(report, indent=2)
    if not isinstance(report, str):
        report = str(report)
    if isinstance(memory_update, dict):
        memory_update = json.dumps(memory_update)
    if not isinstance(memory_update, str):
        memory_update = str(memory_update)
    if not isinstance(flags_add, list):
        flags_add = [flags_add]
    if not report:
        report = raw if isinstance(raw, str) else json.dumps(raw)
    return report, memory_update, flags_add


def update_memory(mem: dict, chunk: dict, setting: str, report: str,
                  memory_update: str, flags_add: list, tiers: dict) -> dict:
    summary = (mem["running_summary"] + " " + memory_update).strip()
    if len(summary) > 1600:
        summary = summary[-1600:]
    if setting == "s1":
        phase_id = stated_phase(report)
    else:
        phase_id = chunk["phase"]
    trail = (mem["phase_trail"] + [phase_id])[-10:]
    seen = dict(mem["instruments_seen"])
    if setting == "s3" and chunk["phase"] != "P13":
        for cls in tiers["observed"]:
            seen[cls] = chunk["chunk_id"]
    flags = mem["flags"] + [f for f in flags_add if f not in mem["flags"]]
    return {"running_summary": summary, "prev_report": report,
            "phase_trail": trail, "instruments_seen": seen, "flags": flags,
            "same_phase_continuation": False}


def run_combo(processor, model, args, video_id: str, setting: str) -> None:
    segments = json.load(
        open(REPO / "videos" / video_id / f"{video_id}.timeline.json"))["segments"]
    chunks = make_chunks(segments)
    if args.chunks:
        wanted = [c.strip() for c in args.chunks.split(",") if c.strip()]
    else:
        wanted = [c["chunk_id"] for c in chunks]
    by_id = {c["chunk_id"]: c for c in chunks}
    for cid in wanted:
        assert cid in by_id, f"unknown chunk {cid} for {video_id}"

    ci = pd.read_csv(REPO / "output-instruments" / f"{video_id}_chunk_instruments.csv")
    base = OUT / args.model_tag / setting
    rep_dir, mem_dir = base / "reports", base / "memory"
    rep_dir.mkdir(parents=True, exist_ok=True)
    mem_dir.mkdir(parents=True, exist_ok=True)
    mem_path = mem_dir / f"{video_id}_memory.jsonl"
    master_path = base / f"{video_id}_reports.jsonl"

    done = set()
    if master_path.exists():
        for line in open(master_path):
            try:
                done.add(json.loads(line)["chunk_id"])
            except (json.JSONDecodeError, KeyError):
                pass
    mem = fresh_memory()
    if mem_path.exists():
        for line in open(mem_path):
            try:
                mem = json.loads(line)["memory_after"]
            except (json.JSONDecodeError, KeyError):
                pass

    for cid in wanted:
        if cid in done:
            print(f"-- skip {video_id}/{setting}/{cid} (done)")
            continue
        ch = by_id[cid]
        total_frames = int(round((ch["t_end"] - ch["t_start"]) * FPS_GUESS))
        sub = ci[ci["chunk_id"] == cid]
        tiers = {"observed": sorted(sub[sub.tier == "observed"]["class_name"].tolist()),
                 "weak": sorted(sub[sub.tier == "weak"]["class_name"].tolist())}
        obs_rows = sub[sub.tier == "observed"].to_dict("records")
        weak_rows = sub[sub.tier == "weak"].to_dict("records")
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
        raw, n_frames, _ = generate(processor, model, str(clip), prompt, args)
        dt = time.time() - t0
        print(f"  response in {dt:.0f}s ({n_frames} frames)", flush=True)
        try:
            parsed = parse_json_loose(raw)
        except (ValueError, AttributeError):
            print("  not JSON — one reformat retry", flush=True)
            t1 = time.time()
            raw = reformat_retry(processor, model, str(clip), raw, args, n_frames)
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
        mem = update_memory(mem, ch, setting, report or raw, memory_update,
                            flags_add, tiers)
        with open(mem_path, "a") as f:
            f.write(json.dumps({"chunk_id": cid, "memory_after": mem}) + "\n")
        with open(master_path, "a") as f:
            f.write(json.dumps(
                {"model": args.model_tag, "setting": setting,
                 "video_id": video_id, "chunk_id": cid,
                 "phase": ch["phase"], "phase_name": ch["phase_name"],
                 "stated_phase": stated_phase(report or raw) if setting == "s1" else ch["phase"],
                 "t_start": ch["t_start"], "t_end": ch["t_end"],
                 "tiers_given": tiers if setting == "s3" else None,
                 "report": report or raw, "memory_update": memory_update,
                 "flags_add": flags_add, "n_frames_used": n_frames,
                 "temperature": args.temperature,
                 "reformat_retried": retried,
                 "latency_s": round(dt, 1)}) + "\n")
        print(f"  saved ({len(report or raw)} chars)", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="local path or HF id")
    ap.add_argument("--model-tag", required=True)
    ap.add_argument("--videos", default=",".join(DEFAULT_VIDEOS))
    ap.add_argument("--settings", default="s1,s2,s3")
    ap.add_argument("--chunks", default=None)
    ap.add_argument("--max-frames", type=int, default=32,
                    help="frames sampled per chunk; short clips feed all their frames")
    ap.add_argument("--max-pixels", type=int, default=307200)
    ap.add_argument("--min-pixels", type=int, default=100352)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.7)
    args = ap.parse_args()

    settings = [s.strip() for s in args.settings.split(",") if s.strip()]
    assert set(settings) <= set(SETTINGS), settings
    if args.videos == "all":
        videos = sorted(p.name for p in (REPO / "videos").glob("PH_*") if p.is_dir())
    else:
        videos = [v.strip() for v in args.videos.split(",") if v.strip()]

    print(f"Loading {args.model} ...", flush=True)
    processor, model = load_model(args.model)
    print("Model loaded.", flush=True)
    for setting in settings:
        for vid in videos:
            run_combo(processor, model, args, vid, setting)
    print("Done.")


if __name__ == "__main__":
    main()
