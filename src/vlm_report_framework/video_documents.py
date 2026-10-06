"""Video-level document stitching for Part III judging.

R_video = concat_i [chunk header + R_i] in temporal order, per source:
  teacher: output-teacher/<VID>_teacher_reports.jsonl
  candidate: output-candidates/<tag>/<setting>/<VID>_reports.jsonl

Idle collapsing: consecutive P13 chunks merge into one entry
  "[P13 x N | t0-t1] <first-idle report>" so Idle text (~40% of chunks)
  doesn't dominate lexical + LLM scores.

Outputs (output-judge/video-documents/, git-ignored):
  <VID>.teacher.md / .json
  <VID>.<tag>.<setting>.md / .json
  index.jsonl  (one row per doc: n_chunks, n_collapsed, chars)

Usage:
  uv run python src/vlm_report_framework/video_documents.py [--videos all]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_state import REPO, load_timeline, resolve_videos  # noqa: E402

OUT = REPO / "output-judge" / "video-documents"
TEACHER = REPO / "output-teacher"
CAND = REPO / "output-candidates"


def load_rows(path: Path) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    rows.sort(key=lambda r: (r["t_start"], r["t_end"]))
    return rows


def section_of(report: str, header: str) -> str:
    """Extract one labeled section body from a 5-section report string."""
    import re
    pat = re.compile(
        rf"^{header}:(.*?)(?=^(?:Phase|Observed instruments|Candidate but not confirmed|Anatomy visible|Actions/events):|\Z)",
        re.M | re.S)
    m = pat.search(report or "")
    return m.group(1).strip() if m else ""


def stitch(rows: list[dict]) -> tuple[str, list[dict], dict]:
    """Stitch rows into one video doc with Idle collapsing.

    Returns (markdown, entries, stats). entries keep chunk_ids for traceability.
    """
    entries: list[dict] = []
    i = 0
    while i < len(rows):
        r = rows[i]
        if r.get("phase") == "P13":
            j = i
            while j < len(rows) and rows[j].get("phase") == "P13":
                j += 1
            run = rows[i:j]
            rep = run[0].get("report", "")
            entries.append({
                "kind": "idle_run",
                "chunk_ids": [x["chunk_id"] for x in run],
                "phase": "P13",
                "t_start": run[0]["t_start"], "t_end": run[-1]["t_end"],
                "report": rep,
            })
            i = j
        else:
            entries.append({
                "kind": "chunk",
                "chunk_ids": [r["chunk_id"]],
                "phase": r.get("phase"), "phase_name": r.get("phase_name"),
                "t_start": r["t_start"], "t_end": r["t_end"],
                "report": r.get("report", ""),
            })
            i += 1
    parts = []
    for e in entries:
        if e["kind"] == "idle_run":
            n = len(e["chunk_ids"])
            parts.append(
                f"## [Idle x{n} | {e['t_start']:.1f}-{e['t_end']:.1f}s | "
                f"{'+'.join(e['chunk_ids'])}]\n{e['report']}")
        else:
            parts.append(
                f"## [{e['chunk_ids'][0]} | {e['phase']} ({e.get('phase_name','')}) | "
                f"{e['t_start']:.1f}-{e['t_end']:.1f}s]\n{e['report']}")
    md = "\n\n".join(parts)
    stats = {"n_chunks": len(rows), "n_entries": len(entries),
             "n_idle_collapsed": sum(len(e["chunk_ids"]) - 1 for e in entries
                                     if e["kind"] == "idle_run"),
             "chars": len(md)}
    return md, entries, stats


def discover_candidates() -> list[tuple[str, str, Path]]:
    out = []
    if not CAND.exists():
        return out
    for tag_dir in sorted(CAND.iterdir()):
        if not tag_dir.is_dir():
            continue
        for setting in ("s1", "s2", "s3"):
            for mp in sorted((tag_dir / setting).glob("*_reports.jsonl")):
                vid = mp.name.replace("_reports.jsonl", "")
                out.append((tag_dir.name, setting, mp, vid))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", default="all")
    args = ap.parse_args()
    vids = set(resolve_videos(args.videos, [])) if args.videos != "all" else None
    OUT.mkdir(parents=True, exist_ok=True)

    index_rows = []
    # teacher docs
    for mp in sorted(TEACHER.glob("*_teacher_reports.jsonl")):
        vid = mp.name.replace("_teacher_reports.jsonl", "")
        if vids is not None and vid not in vids:
            continue
        rows = load_rows(mp)
        md, entries, stats = stitch(rows)
        (OUT / f"{vid}.teacher.md").write_text(md)
        (OUT / f"{vid}.teacher.json").write_text(json.dumps(
            {"video_id": vid, "source": "teacher", "entries": entries,
             "stats": stats}, indent=1))
        index_rows.append({"video_id": vid, "source": "teacher", **stats})
        print(f"{vid}.teacher: {stats['n_chunks']} chunks -> "
              f"{stats['n_entries']} entries ({stats['chars']} chars)")
    # candidate docs
    for tag, setting, mp, vid in discover_candidates():
        if vids is not None and vid not in vids:
            continue
        rows = load_rows(mp)
        md, entries, stats = stitch(rows)
        (OUT / f"{vid}.{tag}.{setting}.md").write_text(md)
        (OUT / f"{vid}.{tag}.{setting}.json").write_text(json.dumps(
            {"video_id": vid, "source": f"{tag}/{setting}",
             "entries": entries, "stats": stats}, indent=1))
        index_rows.append({"video_id": vid, "source": f"{tag}/{setting}", **stats})
        print(f"{vid}.{tag}.{setting}: {stats['n_chunks']} chunks -> "
              f"{stats['n_entries']} entries")
    with open(OUT / "index.jsonl", "w") as f:
        for r in index_rows:
            f.write(json.dumps(r) + "\n")
    print(f"Done. {len(index_rows)} docs in {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
