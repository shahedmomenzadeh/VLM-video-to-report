"""Self-polish stage v2.1: code-locked timeline spine + per-block prose polish.

READ-ONLY w.r.t. per-chunk reports — masters are read, never modified.
Text-only (no video frames: fits the local 8GB GPU in 4-bit).

v2.1 redesign: Python owns the timeline. build_spine() groups consecutive
same-phase chunks into phase blocks with exact [t_start-t_end] spans,
unions observed tools, and concatenates unique action bullets. The result
is ALWAYS a valid full-video report (deterministic stitch, zero LLM).

Optional --polish pass: the model's own text-only LLM rewrites ONLY the
bullets inside each phase block into 2-4 smooth sentences. Python writes
every `## PXX: Phase Name` header and `[MM:SS.ss-MM:SS.ss]:` span around
the polished paragraph, so timestamps can never drift, gap, or overlap.
Validation from v2 is kept as a final assert (spine always passes).

Usage (local GPU venv, offline cache):
  HF_HOME=hf_home HF_HUB_OFFLINE=1 .venv-vlm/bin/python \\
      src/vlm_report_framework/evaluation/compile_self.py \\
      --tag qwen3vl-2b-grpo --setting s3 --videos PH_0001_2931_S2
  # deterministic spine only (no GPU needed beyond chunk reports):
  .venv-vlm/bin/python src/vlm_report_framework/evaluation/compile_self.py \\
      --tag qwen3vl-2b-grpo --setting s3 --videos PH_0001_2931_S2 --no-polish
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
from prompt_base import INSTRUMENT_LABELS  # noqa: E402
from report_render import STUDENT_ANAT_SYN, STUDENT_INSTR_SYN  # noqa: E402
from run_state import REPO, load_timeline  # noqa: E402

OUT = REPO / "output-readable"
START_TOL = 2.0    # first span must start within 2s of 0
END_TOL = 5.0      # last span must end within 5s of video duration
GAP_TOL = 2.0      # consecutive spans must touch within 2s
SPAN_RE = re.compile(r"(\d{1,2}:\d{2}(?:\.\d{1,2})?)\s*[-\u2013]\s*"
                     r"(\d{1,2}:\d{2}(?:\.\d{1,2})?)")


def fmt(t: float) -> str:
    return f"{int(t // 60):02d}:{t % 60:05.2f}"


def to_sec(s: str) -> float:
    m, sec = s.split(":")
    return int(m) * 60 + float(sec)


def video_duration(video_id: str) -> float:
    return max(s["end_s"] for s in load_timeline(video_id))


def _actions_of(report: str) -> list[str]:
    m = re.search(r"^Actions/events:(.*?)(?=^(?:Phase:|Observed instruments:"
                  r"|Candidate but not confirmed:|Anatomy visible:)|\Z)",
                  report or "", re.M | re.S)
    if not m:
        return []
    return [b.strip("- *\t ") for b in m.group(1).strip().split("\n")
            if b.strip("- *\t ")]


def _tools_of(report: str) -> list[str]:
    m = re.search(r"^Observed instruments:(.*?)(?=^(?:Phase:|Candidate but "
                  r"not confirmed:|Anatomy visible:|Actions/events:)|\Z)",
                  report or "", re.M | re.S)
    if not m:
        return []
    txt = m.group(1).strip()
    if txt.lower().startswith("none"):
        return []
    return [b.strip("- *\t ") for b in txt.split("\n") if b.strip("- *\t ")]


# --- polish preservation check (facts must survive the rewrite) --------------
# Correct timestamps do NOT prove the polish kept the facts, so every polished
# block is checked against its source bullets: content words must mostly
# survive (synonym-aware), and no instrument may appear that was neither in
# the bullets nor in the block's union of tools. A failing block falls back to
# the deterministic bullets (never silently accepted).
_STOP = {"the", "a", "an", "and", "or", "with", "without", "into", "from",
         "through", "onto", "its", "it", "this", "that", "then", "than",
         "is", "are", "was", "were", "be", "been", "has", "have", "had",
         "of", "to", "in", "on", "at", "by", "as", "no", "not",
         "observed", "visible", "using", "used", "use", "performed",
         "performs", "perform", "which", "where", "while"}
_WORD_RE = re.compile(r"[a-z][a-z\-/]{2,}")
_SYN_RE = re.compile("|".join(
    re.escape(p) for p in sorted(set(STUDENT_INSTR_SYN) | set(STUDENT_ANAT_SYN),
                                 key=len, reverse=True)))


def _canon_text(text: str) -> str:
    """Lowercase + map known instrument/anatomy terms to canonical ids."""
    def sub(m):
        return (STUDENT_INSTR_SYN.get(m.group(0))
                or STUDENT_ANAT_SYN[m.group(0)]).lower()
    return _SYN_RE.sub(sub, (text or "").lower())


def _fact_tokens(text: str) -> set[str]:
    return {w for w in _WORD_RE.findall(_canon_text(text)) if w not in _STOP}


def preservation_check(actions: list[str], tools: list[str],
                       polished: str) -> dict:
    src = " ".join(actions or [])
    src_tok, pol_tok = _fact_tokens(src), _fact_tokens(polished or "")
    recall = round(len(src_tok & pol_tok) / max(1, len(src_tok)), 3)
    src_l, pol_l = _canon_text(src), _canon_text(polished or "")
    added = [cid for cid, disp in INSTRUMENT_LABELS.items()
             if cid not in tools
             and any(term in pol_l and term not in src_l
                     for term in (cid.lower(), disp.lower()))]
    return {"ok": recall >= 0.5 and not added,
            "recall": recall, "added_tools": added}


def build_spine(rows: list[dict]) -> list[dict]:
    """Group consecutive same-phase chunks into phase blocks (deterministic).

    Returns [{phase, phase_name, t_start, t_end, chunks, tools, actions}].
    Idle chunks stay in place (never merged across a non-idle boundary).
    """
    blocks: list[dict] = []
    for r in rows:
        acts = _actions_of(r.get("report", ""))
        tools = _tools_of(r.get("report", ""))
        if blocks and r.get("phase") == blocks[-1]["phase"]:
            b = blocks[-1]
            b["t_end"] = r["t_end"]
            b["chunks"].append(r["chunk_id"])
            for t in tools:
                if t not in b["tools"]:
                    b["tools"].append(t)
            for a in acts:
                if a not in b["actions"]:
                    b["actions"].append(a)
        else:
            blocks.append({"phase": r.get("phase"), "phase_name": r.get("phase_name", ""),
                           "t_start": r["t_start"], "t_end": r["t_end"],
                           "chunks": [r["chunk_id"]], "tools": list(tools),
                           "actions": list(acts)})
    return blocks


def stitch_spine(blocks: list[dict]) -> str:
    """Deterministic full-video markdown from the spine (always valid)."""
    secs = []
    for b in blocks:
        acts = " ".join(b["actions"]) if b["actions"] else "(no change observed.)"
        tools = f" Tools: {'; '.join(b['tools'])}." if b["tools"] else ""
        secs.append(f"## {b['phase']}: {b['phase_name']}\n"
                    f"[{fmt(b['t_start'])}-{fmt(b['t_end'])}]: {acts}{tools}")
    return "\n\n".join(secs)


POLISH_SYSTEM = ("You are the surgical reporter that wrote the notes below. "
                 "Rewrite them as smooth prose. Add no facts, tools, or times.")

POLISH_USER = """Rewrite ONLY the bullets below as 2-4 smooth sentences of
operative prose. Rules: keep every factual claim; add no new facts, tools,
actions, anatomy, or timings; NEVER write timestamps, headers, or phase
names; output ONLY the paragraph, no preamble.

Bullets:
{bullets}"""


def polish_block(processor, model, bullets: list[str],
                 temperature: float = 0.3) -> str:
    import torch
    prompt = POLISH_USER.format(bullets="\n".join(f"- {b}" for b in bullets))
    messages = [{"role": "system", "content": POLISH_SYSTEM},
                {"role": "user", "content": prompt}]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], padding=True, return_tensors="pt")
    for k, v in inputs.items():
        if torch.is_tensor(v):
            inputs[k] = v.to("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=512,
            do_sample=(temperature > 0.0), temperature=max(temperature, 1e-6),
            top_p=0.8, top_k=20, repetition_penalty=1.05, use_cache=True,
            eos_token_id=[151645, 151643],
            pad_token_id=processor.tokenizer.eos_token_id)
    seq = out[0][inputs["input_ids"].shape[1]:]
    txt = processor.tokenizer.decode(seq, skip_special_tokens=True)
    txt = re.sub(r"<think>.*?</think>", "", txt, flags=re.DOTALL).strip()
    # Code-lock: strip any timestamps/headers the model sneaks in.
    txt = SPAN_RE.sub("", txt)
    txt = re.sub(r"^#{1,3}\s*P\d{2}.*$", "", txt, flags=re.M).strip()
    txt = re.sub(r"\s{2,}", " ", txt).strip()
    return txt


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


def compile_video_deterministic(tag: str, setting: str, vid: str) -> tuple[str, str, list]:
    mp = REPO / "output-candidates" / tag / setting / f"{vid}_reports.jsonl"
    if not mp.exists():
        # Teacher full-video stitch uses the same spine (teacher has no tag dir).
        mp = REPO / "output-teacher" / f"{vid}_teacher_reports.jsonl"
    rows = sorted((json.loads(l) for l in open(mp)),
                  key=lambda r: (r["t_start"], r["t_end"]))
    blocks = build_spine(rows)
    return vid, stitch_spine(blocks), blocks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="",
                    help="local model dir (default: hf_home cache for --tag)")
    ap.add_argument("--tag", default="qwen3vl-2b-grpo")
    ap.add_argument("--setting", default="s3")
    ap.add_argument("--videos", default="all")
    ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument("--no-polish", action="store_true",
                    help="deterministic spine only (no LLM call)")
    a = ap.parse_args()
    tag_models = {
        "qwen3vl-2b-grpo": "hub/models--shahedm2001--qwen3-vl-2b-cataract-grpo",
        "qwen3vl-2b-sft-stage2": "hub/models--shahedm2001--qwen3-vl-2b-cataract-sft-stage2",
    }
    vids = ([v.strip() for v in a.videos.split(",") if v.strip()]
            if a.videos != "all"
            else sorted(p.name for p in (REPO / "videos").glob("PH_*") if p.is_dir()))

    processor = model = None
    if not a.no_polish:
        from backends.qwen3vl import Qwen3VLBackend
        if a.model:
            model_path = a.model
        else:
            snap = next((REPO / "hf_home" / tag_models[a.tag] / "snapshots").iterdir())
            model_path = str(snap)
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
        t0 = time.time()
        _, stitched, blocks = compile_video_deterministic(a.tag, a.setting, vid)
        dur = video_duration(vid)
        polished = False
        preservation: dict | None = None
        if processor is not None:
            try:
                secs: list[str] = []
                checks: list[dict] = []
                for b in blocks:
                    if b["actions"]:
                        para = polish_block(processor, model, b["actions"],
                                            temperature=a.temperature)
                        chk = preservation_check(b["actions"], b["tools"], para)
                        if not chk["ok"]:  # rewrite drifted -> keep facts verbatim
                            para = " ".join(b["actions"])
                        chk["fallback"] = not chk["ok"]
                        checks.append(chk)
                    else:
                        para = "(no change observed.)"
                    tools_note = (f" Tools: {'; '.join(b['tools'])}."
                                  if b["tools"] else "")
                    secs.append(f"## {b['phase']}: {b['phase_name']}\n"
                                f"[{fmt(b['t_start'])}-{fmt(b['t_end'])}]: {para}{tools_note}")
                stitched = "\n\n".join(secs)
                polished = True
                preservation = {
                    "n_blocks": len(checks),
                    "n_fallbacks": sum(c["fallback"] for c in checks),
                    "checks": checks}
            except Exception as e:  # noqa: BLE001 — spine already valid
                print(f"  polish failed {vid}: {e}; keeping deterministic spine",
                      flush=True)
            gc.collect()
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass
        rows = sorted((json.loads(l) for l in open(
            REPO / "output-candidates" / a.tag / a.setting / f"{vid}_reports.jsonl"
            if (REPO / "output-candidates" / a.tag / a.setting / f"{vid}_reports.jsonl").exists()
            else REPO / "output-teacher" / f"{vid}_teacher_reports.jsonl")),
            key=lambda r: (r["t_start"], r["t_end"]))
        ok, errs, spans = validate(stitched, rows, dur)
        fall_note = (f" Preservation-check fallbacks: {preservation['n_fallbacks']}"
                     f"/{preservation['n_blocks']} blocks." if preservation else "")
        header = (f"<!-- self-polished by {a.tag}/{a.setting} "
                  f"({len(rows)} frozen chunk reports, untouched). "
                  f"Spine=code-locked, polished={polished}.{fall_note} Not scored. -->\n\n")
        dest.write_text(header + stitched)
        vrep.write_text(json.dumps(
            {"video_id": vid, "duration": round(dur, 1), "ok": ok,
             "fallback": False, "polished": polished, "errors": errs,
             "spans": spans, "blocks": len(blocks), "preservation": preservation,
             "latency_s": round(time.time() - t0, 1)}, indent=1))
        print(f"  saved {vid} ok={ok} blocks={len(blocks)} polished={polished} "
              f"spans={len(spans)} ({time.time() - t0:.0f}s)", flush=True)
    print(f"Done -> {outdir.relative_to(REPO)}")


if __name__ == "__main__":
    main()
