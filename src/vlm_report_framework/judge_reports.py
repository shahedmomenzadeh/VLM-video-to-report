"""Eval B/C runner: video-level LLM judge (teacher-blind) + order-swapped pairwise.

Reads stitched docs (video_documents.py). Judge model MUST be a different
family from the teacher (default endpoint below served Gemini as teacher —
override --base-url/--model for GPT/Claude).

Modes:
  --mode score    one call per (video x candidate source)
  --mode pairwise focused contrasts (s1-vs-s3 within tag; s3 cross-tag),
                  each run twice with A/B swapped
Evidence: timeline + aggregated tiers as text; --video full adds the whole
  surgery mp4 as video_url (large: ~40MB base64 each — prefer text-evidence
  first, enable video only where bandwidth allows).

Outputs (output-judge/, git-ignored):
  llm_scores/<judge-tag>/scores.jsonl   (resume by video+source)
  pairwise/<judge-tag>/pairs.jsonl

Usage:
  uv run python src/vlm_report_framework/judge_reports.py --mode score \\
      --judge-tag gpt-judge-v1 --model gpt-4o --base-url https://api.openai.com/v1
"""
from __future__ import annotations

import argparse
import base64
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from judge_prompt import (  # noqa: E402
    SYSTEM_PROMPT,
    build_evidence_block,
    build_pairwise_prompt,
    build_scoring_prompt,
)
from run_state import REPO, parse_json_loose  # noqa: E402

DOCS = REPO / "output-judge" / "video-documents"
OUT_SCORE = REPO / "output-judge" / "llm_scores"
OUT_PAIR = REPO / "output-judge" / "pairwise"


def evidence_for(video_id: str) -> str:
    tl = json.load(open(REPO / "videos" / video_id / f"{video_id}.timeline.json"))
    lines = [f"{s['phase']} ({s['phase_name']}): {s['start_s']:.1f}-{s['end_s']:.1f}s"
             for s in tl["segments"]]
    import pandas as pd
    ci = pd.read_csv(REPO / "output-instruments" / f"{video_id}_chunk_instruments.csv")
    agg = (ci.groupby(["class_name", "tier"])["n_frames"].sum()
            .reset_index().sort_values("n_frames", ascending=False))
    tiers = "\n".join(f"- {r.class_name} [{r.tier}]: {r.n_frames} frames"
                      for r in agg.itertuples())
    return build_evidence_block("\n".join(lines), tiers or "(no detections)")


def call_judge(client, model: str, prompt: str, video_path: Path | None,
               temperature: float, max_tokens: int) -> tuple[str, float]:
    content: list = [{"type": "text", "text": prompt}]
    if video_path is not None:
        b64 = base64.b64encode(video_path.read_bytes()).decode()
        content.append({"type": "video_url", "video_url": {
            "url": f"data:video/mp4;base64,{b64}"}})
    t0 = time.time()
    resp = client.chat.completions.create(
        model=model, temperature=temperature, max_tokens=max_tokens,
        messages=[{"role": "system", "content": SYSTEM_PROMPT},
                  {"role": "user", "content": content}],
        timeout=600)
    return resp.choices[0].message.content or "", time.time() - t0


def done_keys(path: Path, key: tuple) -> set:
    done = set()
    if path.exists():
        for line in open(path):
            try:
                r = json.loads(line)
                done.add(tuple(r[k] for k in key))
            except (json.JSONDecodeError, KeyError):
                pass
    return done


def candidate_sources() -> list[tuple[str, str]]:
    """(doc_suffix, source_label) for every non-teacher stitched doc."""
    out = []
    for p in sorted(DOCS.glob("*.md")):
        if ".teacher." in p.name:
            continue
        stem = p.name[:-len(".md")]  # <VID>.<tag>.<setting>
        vid, rest = stem.split(".", 1)
        out.append((vid, rest, p))
    return out


def run_score(args, client) -> None:
    out_dir = OUT_SCORE / args.judge_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    master = out_dir / "scores.jsonl"
    done = done_keys(master, ("video_id", "source"))
    cands = [(v, s, p) for v, s, p in candidate_sources()
             if not args.videos or v in args.videos]
    for vid, src, p in cands:
        if (vid, src) in done:
            print(f"-- skip {vid} {src} (done)")
            continue
        prompt = build_scoring_prompt(p.read_text(), evidence_for(vid))
        vpath = (REPO / "videos" / vid / f"{vid}.mp4") if args.video_full else None
        print(f"=== score {vid} {src} ===", flush=True)
        try:
            raw, dt = call_judge(client, args.model, prompt, vpath,
                                 args.temperature, args.max_tokens)
            parsed = parse_json_loose(raw)
            row = {"video_id": vid, "source": src, "judge": args.model,
                   **{k: parsed.get(k) for k in (
                       "groundedness", "completeness",
                       "instrument_correctness", "temporal_coherence",
                       "echo_flag", "reason")},
                   "latency_s": round(dt, 1)}
        except Exception as e:  # noqa: BLE001 — store failure, keep going
            row = {"video_id": vid, "source": src, "judge": args.model,
                   "error": str(e)[:300]}
            print(f"  WARN: {e}", flush=True)
        with open(master, "a") as f:
            f.write(json.dumps(row) + "\n")
    print(f"Done. {master.relative_to(REPO)}")


def pairwise_contrasts() -> list[tuple[str, str, str, str]]:
    """Focused contrasts: (video, label_a, label_b, note). Order swapped by caller."""
    docs = {(v, s): p for v, s, p in candidate_sources()}
    out = []
    vids = sorted(set(v for v, _, _ in candidate_sources()))
    tags = sorted(set(s.rsplit(".", 1)[0] for _, s, _ in candidate_sources()))
    for vid in vids:
        for tag in tags:
            a, b = f"{tag}.s1", f"{tag}.s3"
            if (vid, a) in docs and (vid, b) in docs:
                out.append((vid, a, b, f"{tag}: s1-vs-s3"))
        if len(tags) > 1:
            a, b = f"{tags[0]}.s3", f"{tags[1]}.s3"
            if (vid, a) in docs and (vid, b) in docs:
                out.append((vid, a, b, "s3 cross-tag"))
    return out


def run_pairwise(args, client) -> None:
    out_dir = OUT_PAIR / args.judge_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    master = out_dir / "pairs.jsonl"
    done = done_keys(master, ("video_id", "label_a", "label_b", "order"))
    docs = {(v, s): p for v, s, p in candidate_sources()}
    for vid, la, lb, note in pairwise_contrasts():
        if args.videos and vid not in args.videos:
            continue
        ev = evidence_for(vid)
        for order, (xa, xb) in (("AB", (la, lb)), ("BA", (lb, la))):
            if (vid, la, lb, order) in done:
                print(f"-- skip {vid} {la}-vs-{lb} {order} (done)")
                continue
            prompt = build_pairwise_prompt(docs[(vid, xa)].read_text(),
                                           docs[(vid, xb)].read_text(), ev)
            print(f"=== pair {vid} {xa}-vs-{xb} ===", flush=True)
            try:
                raw, dt = call_judge(client, args.model, prompt, None,
                                     args.temperature, args.max_tokens)
                parsed = parse_json_loose(raw)
                winner = parsed.get("winner")
                # normalize to la/lb frame
                if order == "BA" and winner in ("A", "B"):
                    winner = "B" if winner == "A" else "A"
                row = {"video_id": vid, "label_a": la, "label_b": lb,
                       "order": order, "note": note, "judge": args.model,
                       "winner": winner, "reason": parsed.get("reason"),
                       "latency_s": round(dt, 1)}
            except Exception as e:  # noqa: BLE001
                row = {"video_id": vid, "label_a": la, "label_b": lb,
                       "order": order, "note": note, "error": str(e)[:300]}
                print(f"  WARN: {e}", flush=True)
            with open(master, "a") as f:
                f.write(json.dumps(row) + "\n")
    print(f"Done. {master.relative_to(REPO)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("score", "pairwise"), default="score")
    ap.add_argument("--model", default="gpt-4o",
                    help="judge model id (must differ from teacher family)")
    ap.add_argument("--base-url", default="http://localhost:20128/v1")
    ap.add_argument("--api-key", default="not-needed")
    ap.add_argument("--judge-tag", default="judge-v1")
    ap.add_argument("--videos", default=None,
                    help="comma-separated video IDs (default: all stitched)")
    ap.add_argument("--video-full", action="store_true",
                    help="attach full surgery mp4 as judge video evidence")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=512)
    args = ap.parse_args()
    if args.videos:
        args.videos = [v.strip() for v in args.videos.split(",") if v.strip()]
    assert DOCS.exists(), f"run video_documents.py first ({DOCS} missing)"

    from openai import OpenAI
    client = OpenAI(base_url=args.base_url, api_key=args.api_key)
    if args.mode == "score":
        run_score(args, client)
    else:
        run_pairwise(args, client)


if __name__ == "__main__":
    main()
