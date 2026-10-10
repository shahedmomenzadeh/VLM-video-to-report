"""Eval 1: reliability / protocol adherence (deterministic, no LLM)."""
from __future__ import annotations

import json
import re
import sys
import zlib
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from run_state import REPO, parse_json_loose  # noqa: E402

CAND = REPO / "output-candidates"

SECTIONS = ["Phase:", "Observed instruments:",
            "Candidate but not confirmed:",
            "Anatomy visible:", "Actions/events:"]
ECHO_RE = re.compile(r"median conf|N frames|phase-plausible|frames analyzed",
                     re.IGNORECASE)
RETRY_KNOWN_FALSE = object()  # placeholder, real logic below


def repetition_score(text: str) -> float:
    """0 = diverse, 1 = fully repetitive (zlib ratio + trigram dominance)."""
    if not text or len(text) < 200:
        return 0.0
    comp = len(zlib.compress(text.encode())) / max(1, len(text.encode()))
    toks = text.lower().split()
    tris = Counter(zip(toks, toks[1:], toks[2:])) if len(toks) > 3 else Counter()
    top = max(tris.values()) / max(1, sum(tris.values())) if tris else 0.0
    # low compressibility ratio => repetitive; combine with trigram dominance
    return round(min(1.0, (1.0 - comp) * 0.7 + top * 0.7), 3)


def check_report(report: str, raw: str | None = None) -> dict:
    rep = report or ""
    secs = {s: (s in rep) for s in SECTIONS}
    # validity states: raw parses => initial ok; report non-empty but raw
    # didn't parse => retry/recoverable path (old flag unreliable, see below)
    json_ok = True
    if raw:
        try:
            parse_json_loose(raw)
        except Exception:
            json_ok = rep.strip() != ""
    return {
        "empty": not rep.strip(),
        "chars": len(rep),
        "sections_present": sum(secs.values()),
        "sections_missing": [s for s, v in secs.items() if not v],
        "echo_hit": bool(ECHO_RE.search(rep)),
        "repetition": repetition_score(rep),
        "degenerate": repetition_score(rep) > 0.75 and len(rep) > 500,
        "raw_json_ok": json_ok,
    }


def check_row(row: dict, raw: str | None) -> dict:
    c = check_report(row.get("report", ""), raw)
    c.update({
        "chunk_id": row.get("chunk_id"),
        "has_memory_update": bool((row.get("memory_update") or "").strip()),
        "reformat_retried_flag": bool(row.get("reformat_retried")),
        # old flag = "final parse empty", not "retry happened" — keep for
        # continuity but do not trust it (see candidate_reports.py:159)
        "stated_phase": row.get("stated_phase"),
    })
    return c


def load_raw(tag: str, setting: str, video_id: str, chunk_id: str) -> str | None:
    p = CAND / tag / setting / "reports" / f"{video_id}__{chunk_id}.json"
    if p.exists():
        try:
            return json.load(open(p)).get("raw_response")
        except Exception:
            return None
    return None


def audit(tag: str, setting: str, video_id: str) -> list[dict]:
    mp = CAND / tag / setting / f"{video_id}_reports.jsonl"
    if not mp.exists():
        return []
    return [check_row(r, load_raw(tag, setting, video_id, r.get("chunk_id", "")))
            for r in map(json.loads, open(mp))]
