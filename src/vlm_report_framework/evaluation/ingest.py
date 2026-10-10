"""Eval 0: inventory — join expected chunks to saved rows (post-hoc, no models)."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from clean_instruments import make_chunks  # noqa: E402
from run_state import REPO, load_timeline  # noqa: E402

CAND = REPO / "output-candidates"
TEACHER = REPO / "output-teacher"
CLIPS = TEACHER / "clips"
OUT_EVAL = REPO / "output-evaluation"


def discover_tags() -> list[str]:
    if not CAND.exists():
        return []
    out = []
    for p in CAND.iterdir():
        if not p.is_dir():
            continue
        if any((p / s).is_dir() for s in ("s1", "s2", "s3")):
            out.append(p.name)
    return sorted(out)


def expected_chunks(video_id: str) -> list[dict]:
    return make_chunks(load_timeline(video_id))


def load_master(path: Path) -> tuple[list[dict], list[str]]:
    rows, dupes = [], []
    seen: set[str] = set()
    if path.exists():
        for line in open(path):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            rows.append(r)
            cid = r.get("chunk_id", "?")
            if cid in seen:
                dupes.append(cid)
            seen.add(cid)
    return rows, dupes


def load_teacher(video_id: str) -> dict[str, dict]:
    p = TEACHER / f"{video_id}_teacher_reports.jsonl"
    rows, _ = load_master(p)
    return {r["chunk_id"]: r for r in rows}


def inventory(videos: list[str] | None = None,
              tags: list[str] | None = None) -> list[dict]:
    tags = tags or discover_tags()
    if videos is None:
        videos = sorted(p.name for p in (REPO / "videos").glob("PH_*")
                        if p.is_dir())
    out = []
    for vid in videos:
        try:
            exp = [c["chunk_id"] for c in expected_chunks(vid)]
        except FileNotFoundError:
            continue
        teacher = load_teacher(vid)
        for tag in tags:
            for setting in ("s1", "s2", "s3"):
                mp = CAND / tag / setting / f"{vid}_reports.jsonl"
                rows, dupes = load_master(mp)
                got = [r.get("chunk_id") for r in rows]
                missing = [c for c in exp if c not in got]
                extra = [c for c in got if c not in exp]
                misordered = got != exp[:len(got)] and not missing[:1]
                mem_path = CAND / tag / setting / "memory" / f"{vid}_memory.jsonl"
                n_snaps = 0
                if mem_path.exists():
                    n_snaps = sum(1 for _ in open(mem_path))
                out.append({
                    "video_id": vid, "tag": tag, "setting": setting,
                    "n_expected": len(exp), "n_present": len(rows),
                    "n_missing": len(missing), "missing": missing[:5],
                    "n_extra": len(extra), "n_dupes": len(dupes),
                    "misordered": bool(misordered and got),
                    "n_snaps": n_snaps, "snaps_match": n_snaps == len(rows),
                    "teacher_chunks": len(teacher),
                    "master_exists": mp.exists(),
                })
    return out


def write_manifest(run_id: str, params: dict) -> Path:
    d = OUT_EVAL / run_id
    d.mkdir(parents=True, exist_ok=True)
    h = hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()[:12]
    (d / "manifest.json").write_text(json.dumps(
        {"run_id": run_id, "input_hash": h, "params": params}, indent=1))
    return d
