"""Audit output-candidates chains: structure, memory health, parse health.

For every (model, setting, video) chain it checks:
  - master rows cover exactly the timeline chunks, in temporal order
  - memory snapshots match rows 1:1 and in order
  - running_summary is present and non-decreasing (until the 1600 cap)
  - parse health: empty reports, failed reformats, missing memory_updates
  - frame stats: min/mean n_frames_used
  - s1 only: stated-phase accuracy vs ground-truth phase (incl. P?? rate)

Usage:
  uv run python src/vlm_report_framework/verify_candidates.py [--model-tag X]
Exit code is non-zero on structural problems (missing/misordered chunks or
snapshots); parse issues are warnings.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from clean_instruments import make_chunks  # noqa: E402
from run_state import REPO, load_timeline  # noqa: E402

OUT = REPO / "output-candidates"


def audit_chain(model: str, setting: str, video_id: str) -> tuple[list[str], list[str]]:
    errors, warns = [], []
    base = OUT / model / setting
    chunks = make_chunks(load_timeline(video_id))
    expected = [c["chunk_id"] for c in chunks]

    master_path = base / f"{video_id}_reports.jsonl"
    if not master_path.exists():
        return [f"{model}/{setting}/{video_id}: missing master table"], []
    rows = [json.loads(l) for l in open(master_path)]
    got = [r["chunk_id"] for r in rows]
    if got != expected[:len(got)]:
        errors.append(f"{model}/{setting}/{video_id}: order/coverage mismatch "
                      f"(got {len(got)}, expected prefix {expected[:len(got)] != got})")
    missing = [c for c in expected if c not in got]
    if missing:
        errors.append(f"{model}/{setting}/{video_id}: missing {len(missing)} chunks: {missing[:5]}")

    mem_path = base / "memory" / f"{video_id}_memory.jsonl"
    snaps = []
    if mem_path.exists():
        snaps = [json.loads(l) for l in open(mem_path)]
    if [s["chunk_id"] for s in snaps] != got:
        errors.append(f"{model}/{setting}/{video_id}: snapshots don't match master rows "
                      f"({len(snaps)} vs {len(rows)})")

    if snaps:
        lens = [len(s["memory_after"].get("running_summary", "")) for s in snaps]
        if any(b < a for a, b in zip(lens, lens[1:])):
            warns.append(f"{model}/{setting}/{video_id}: summary shrank mid-chain")
        if lens[-1] == 0:
            warns.append(f"{model}/{setting}/{video_id}: empty final summary")

    n_empty = sum(1 for r in rows if not r.get("report"))
    n_retried = sum(1 for r in rows if r.get("reformat_retried"))
    n_nomu = sum(1 for r in rows if not r.get("memory_update"))
    if n_empty:
        errors.append(f"{model}/{setting}/{video_id}: {n_empty} empty reports")
    if n_retried:
        warns.append(f"{model}/{setting}/{video_id}: {n_retried} failed parses")
    if n_nomu:
        warns.append(f"{model}/{setting}/{video_id}: {n_nomu} missing memory_updates")

    frames = [r.get("n_frames_used", -1) for r in rows]
    stat = (f"n={len(rows)} frames[min={min(frames) if frames else '-'}] "
            f"empty={n_empty} retried={n_retried} no_mu={n_nomu}")
    if setting == "s1" and rows:
        hit = sum(1 for r in rows if r.get("stated_phase") == r.get("phase"))
        unk = sum(1 for r in rows if r.get("stated_phase") == "P??")
        stat += f" phase_acc={hit}/{len(rows)} unknown={unk}"
    print(f"  {model}/{setting}/{video_id}: {stat}")
    return errors, warns


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-tag", default=None)
    args = ap.parse_args()
    models = [args.model_tag] if args.model_tag else sorted(
        p.name for p in OUT.iterdir() if p.is_dir())
    n_err, n_warn = 0, 0
    for model in models:
        for setting in ("s1", "s2", "s3"):
            masters = sorted((OUT / model / setting).glob("*_reports.jsonl"))
            if not masters:
                print(f"  {model}/{setting}: no tables")
                continue
            for mp in masters:
                vid = mp.name.replace("_reports.jsonl", "")
                errors, warns = audit_chain(model, setting, vid)
                for e in errors:
                    print("ERROR:", e)
                for w in warns:
                    print("WARN:", w)
                n_err += len(errors)
                n_warn += len(warns)
    print(f"done: {n_err} errors, {n_warn} warnings")
    return 1 if n_err else 0


if __name__ == "__main__":
    raise SystemExit(main())
