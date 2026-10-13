"""Self-polish stage: the candidate model refines its OWN chunk reports into one
readable full-video report. No Gemini involved.

READ-ONLY w.r.t. per-chunk reports — masters are read, never modified.
Text-only (no video frames: fits the local 8GB GPU in 4-bit).

Timestamp validation (deterministic): parsed section spans must be monotonic,
non-overlapping, start ≈ 0, end ≈ video duration, and cover every chunk.
One retry with the error message; then deterministic stitched fallback.

Usage (local GPU venv, offline cache):
  HF_HOME=hf_home HF_HUB_OFFLINE=1 .venv-vlm/bin/python \\
      src/vlm_report_framework/evaluation/compile_self.py \
      --tag qwen3vl-2b-grpo --setting s3 --videos PH_0001_2931_S2
"""
from __future__ import annotations

import argparse
import gc
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from backends.qwen3vl import Qwen3VLBackend  # noqa: E402
from run_state import REPO, load_timeline  # noqa: E402

OUT = REPO / "output-readable"
START_TOL = 2.0    # first span must start within 2s of 0
END_TOL = 5.0      # last span must end within 5s of video duration
GAP_TOL = 2.0      # consecutive spans must touch within 2s
SPAN_RE = re.compile(r"(\d{1,2}:\d{2}(?:\.\d{1,2})?)\s*[-\u2013]\s*"
                     r"(\d{1,2}:\d{2}(?:\.\d{1,2})?)")

SYSTEM = ("You are the surgical reporter that wrote the chunk notes below. "
          "Reorganize them into one readable operative report. Add no facts.")


def fmt(t: float) -> str:
    return f"{int(t // 60):02d}:{t % 60:05.2f}"


def to_sec(s: str) -> float:
    m, sec = s.split(":")
    return int(m) * 60 + float(sec)


def video_duration(video_id: str) -> float:
    return max(s["end_s"] for s in load_timeline(video_id))


def build_prompt(rows: list[dict], duration: float) -> str:
    table = "\n".join(
        f"- {r['chunk_id']} | {r.get('phase')} | {fmt(r['t_start'])}-{fmt(r['t_end'])}"
        for r in rows)
    body = "\n\n".join(
        f"[{r['chunk_id']}]\n{r.get('report', '')}" for r in rows)
    return (
        f"Your own chunk reports for one cataract surgery "
        f"(full duration {fmt(duration)}), in order. Chunk boundaries:\n{table}\n\n"
        f"{body}\n\nRewrite as ONE readable operative report, grouped by phase in "
        f"temporal order. You MAY merge consecutive same-phase chunks into one "
        f"section, but every section header MUST carry an exact 'MM:SS.ss-MM:SS.ss' "
        f"time span copied from the chunk boundaries above. The first span must "
        f"start at 00:00 and the last span MUST end at {fmt(duration)} — the report "
        f"must cover the whole video. Merge duplicates, keep every factual claim, "
        f"add no new facts, tools, or timings. Plain markdown.")


def validate(text: str, rows: list[dict], duration: float) -> tuple[bool, list, list]:
    spans = [(to_sec(a), to_sec(b)) for a, b in SPAN_RE.findall(text)]
    errs: list[str] = []
    if not spans:
        return False, ["no parseable MM:SS-MM:SS spans found"], []
    if spans[0][0] > START_TOL:
        errs.append(f"starts at {spans[0][0]:.1f}s, not ≈0")
    if abs(spans[-1][1] - duration) > END_TOL:
        errs.append(f"ends at {spans[-1][1]:.1f}s, video is {duration:.1f}s")
    for (a0, b0), (a1, b1) in zip(spans, spans[1:]):
        if a1 < b0 - GAP_TOL:
            errs.append(f"overlap {a1:.1f}s < {b0:.1f}s")
        if a1 > b0 + GAP_TOL:
            errs.append(f"gap {b0:.1f}s → {a1:.1f}s")
        if b0 > duration + END_TOL or a1 > duration + END_TOL:
            errs.append("span beyond video duration")
    for r in rows:  # every chunk interval must be covered by some span
        mid = (r["t_start"] + r["t_end"]) / 2
        if not any(a - GAP_TOL <= mid <= b + GAP_TOL for a, b in spans):
            errs.append(f"chunk {r['chunk_id']} uncovered")
            break
    return (not errs), errs, [[round(a, 1), round(b, 1)] for a, b in spans]


def fallback_stitch(rows: list[dict]) -> str:
    """Deterministic phase-grouped stitch (no LLM).

    Identical adjacent Actions are MERGED into one section whose span extends
    over all merged chunks (never dropped — dropping breaks end-of-video
    coverage, the exact failure the timestamp validation guards).
    """
    sections: list[list] = []  # [phase, t_start, t_end, act]
    for r in rows:
        m = re.search(r"^Actions/events:(.*?)(?=^(?:Phase:|Observed instruments:"
                      r"|Candidate but not confirmed:|Anatomy visible:)|\Z)",
                      r.get("report", ""), re.M | re.S)
        act = (m.group(1).strip() if m else "").replace("\n", " ")
        if not act:
            act = "(no parseable Actions/events section)"
        if (sections and act == sections[-1][3]
                and r.get("phase") == sections[-1][0]):
            sections[-1][2] = r["t_end"]  # extend span, don't drop the chunk
        else:
            sections.append([r.get("phase"), r["t_start"], r["t_end"], act])
    return "\n\n".join(
        f"### {ph} {fmt(a)}-{fmt(b)}\n{act}" for ph, a, b, act in sections)


def generate_text_only(processor, model, prompt: str,
                       max_new_tokens: int = 2048,
                       temperature: float = 0.3) -> str:
    import torch
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], padding=True, return_tensors="pt")
    for k, v in inputs.items():
        if torch.is_tensor(v):
            inputs[k] = v.to("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            do_sample=(temperature > 0.0), temperature=max(temperature, 1e-6),
            top_p=0.8, top_k=20, repetition_penalty=1.05, use_cache=True,
            eos_token_id=[151645, 151643],
            pad_token_id=processor.tokenizer.eos_token_id)
    seq = out[0][inputs["input_ids"].shape[1]:]
    txt = processor.tokenizer.decode(seq, skip_special_tokens=True)
    return re.sub(r"<think>.*?</think>", "", txt, flags=re.DOTALL).strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="",
                    help="local model dir (default: hf_home cache for --tag)")
    ap.add_argument("--tag", default="qwen3vl-2b-grpo")
    ap.add_argument("--setting", default="s3")
    ap.add_argument("--videos", default="all")
    ap.add_argument("--temperature", type=float, default=0.3)
    a = ap.parse_args()
    tag_models = {
        "qwen3vl-2b-grpo": "hub/models--shahedm2001--qwen3-vl-2b-cataract-grpo",
        "qwen3vl-2b-sft-stage2": "hub/models--shahedm2001--qwen3-vl-2b-cataract-sft-stage2",
    }
    if a.model:
        model_path = a.model
    else:
        snap = next((REPO / "hf_home" / tag_models[a.tag] / "snapshots").iterdir())
        model_path = str(snap)
    vids = ([v.strip() for v in a.videos.split(",") if v.strip()]
            if a.videos != "all"
            else sorted(p.name for p in (REPO / "videos").glob("PH_*") if p.is_dir()))

    print(f"Loading {model_path} (4-bit, text-only) ...", flush=True)
    backend = Qwen3VLBackend(model_path)
    processor, model = backend.load()
    print("Model loaded.", flush=True)

    outdir = OUT / f"{a.tag}.{a.setting}.self"
    outdir.mkdir(parents=True, exist_ok=True)
    for vid in vids:
        dest = outdir / f"{vid}.readable.md"
        vrep = outdir / f"{vid}.validation.json"
        if dest.exists() and vrep.exists():
            print(f"-- skip {vid} (done)")
            continue
        mp = REPO / "output-candidates" / a.tag / a.setting / f"{vid}_reports.jsonl"
        rows = sorted((json.loads(l) for l in open(mp)),
                      key=lambda r: (r["t_start"], r["t_end"]))
        dur = video_duration(vid)
        t0 = time.time()
        text = generate_text_only(processor, model, build_prompt(rows, dur),
                                  temperature=a.temperature)
        ok, errs, spans = validate(text, rows, dur)
        if not ok:  # one retry with the concrete errors
            print(f"  retry {vid}: {errs}", flush=True)
            text = generate_text_only(
                processor, model,
                build_prompt(rows, dur) + "\n\nYour previous draft failed validation:\n"
                + "\n".join(f"- {e}" for e in errs)
                + "\nFix ONLY the time spans; keep all facts.",
                temperature=a.temperature)
            ok, errs, spans = validate(text, rows, dur)
        fallback = False
        if not ok:
            text = fallback_stitch(rows)
            ok, errs, spans = validate(text, rows, dur)
            fallback = True
            print(f"  FALLBACK {vid}: {errs}", flush=True)
        header = (f"<!-- self-polished by {a.tag}/{a.setting} "
                  f"({len(rows)} frozen chunk reports, untouched). "
                  f"Fallback={fallback}. Not scored. -->\n\n")
        dest.write_text(header + text)
        vrep.write_text(json.dumps(
            {"video_id": vid, "duration": round(dur, 1), "ok": ok,
             "fallback": fallback, "errors": errs, "spans": spans,
             "latency_s": round(time.time() - t0, 1)}, indent=1))
        print(f"  saved {vid} ok={ok} fallback={fallback} "
              f"spans={len(spans)} ({time.time() - t0:.0f}s)", flush=True)
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
    print(f"Done -> {outdir.relative_to(REPO)}")


if __name__ == "__main__":
    main()
