"""Shared run state: frozen inputs, resume handling, JSON parsing.

Both pipelines (API teacher, local candidates) read the same frozen inputs,
resume by skipping finished chunks, rebuild memory from snapshots, and parse
the same {report, memory_update, flags_add} contract.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

REPO = Path.cwd()
FPS_GUESS = 5.0


def load_timeline(video_id: str) -> list[dict]:
    tl = json.load(open(REPO / "videos" / video_id / f"{video_id}.timeline.json"))
    return tl["segments"]


def load_chunk_instruments(video_id: str) -> pd.DataFrame:
    p = REPO / "output-instruments" / f"{video_id}_chunk_instruments.csv"
    return pd.read_csv(p)


def split_tiers(sub: pd.DataFrame) -> tuple[dict, list[dict], list[dict]]:
    tiers = {"observed": sorted(sub[sub.tier == "observed"]["class_name"].tolist()),
             "weak": sorted(sub[sub.tier == "weak"]["class_name"].tolist())}
    return (tiers, sub[sub.tier == "observed"].to_dict("records"),
            sub[sub.tier == "weak"].to_dict("records"))


def resolve_videos(spec: str, default: list[str]) -> list[str]:
    if spec == "all":
        return sorted(p.name for p in (REPO / "videos").glob("PH_*") if p.is_dir())
    if spec:
        return [v.strip() for v in spec.split(",") if v.strip()]
    return default


def read_done_ids(master_path: Path) -> set[str]:
    done = set()
    if master_path.exists():
        with open(master_path) as f:
            for line in f:
                try:
                    done.add(json.loads(line)["chunk_id"])
                except (json.JSONDecodeError, KeyError):
                    pass
    return done


def rebuild_memory(mem_path: Path, fresh: dict) -> dict:
    mem = fresh
    if mem_path.exists():
        with open(mem_path) as f:
            for line in f:
                try:
                    mem = json.loads(line)["memory_after"]
                except (json.JSONDecodeError, KeyError):
                    pass
    # tolerate snapshots written before a schema addition
    for k, v in fresh.items():
        mem.setdefault(k, v)
    # v2.1 slim migration: drop verbatim legacy keys (echo vector), keep
    # structural signals. Inline to avoid a circular import (memory.py is
    # imported by callers, not here).
    if "recent_states" not in mem:
        out = {k: fresh.get(k) for k in
               ("recent_states", "last_nonidle_state", "phase_trail",
                "instruments_seen", "flags", "same_phase_continuation")}
        out = {k: (list(v) if isinstance(v, list) else
                   (dict(v) if isinstance(v, dict) else v))
               for k, v in out.items() if v is not None}
        trail = mem.get("phase_trail", [])
        out["phase_trail"] = [t for t in trail if isinstance(t, str)][-10:] \
            if isinstance(trail, list) else []
        seen = mem.get("instruments_seen", {})
        out["recent_states"] = []
        out["last_nonidle_state"] = {}
        out["instruments_seen"] = {k: v for k, v in seen.items()
                                   if isinstance(k, str)} \
            if isinstance(seen, dict) else {}
        flags = mem.get("flags", [])
        out["flags"] = [f if isinstance(f, str) else str(f) for f in flags
                        if isinstance(flags, list) and (isinstance(f, str) or f)] or []
        out["same_phase_continuation"] = False
        return out
    return mem


def snapshot(mem_path: Path, chunk_id: str, mem: dict) -> None:
    with open(mem_path, "a") as f:
        f.write(json.dumps({"chunk_id": chunk_id, "memory_after": mem}) + "\n")


def append_master(master_path: Path, row: dict) -> None:
    with open(master_path, "a") as f:
        f.write(json.dumps(row) + "\n")


def estimate_frames(chunk: dict) -> int:
    return int(round((chunk["t_end"] - chunk["t_start"]) * FPS_GUESS))


def parse_json_loose(text: str) -> dict:
    import re
    # strip ```json fenced blocks (models often wrap the answer)
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        text = m.group(1)
    else:
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # try largest {...} substring
    a, b = text.find("{"), text.rfind("}")
    if 0 <= a < b:
        return json.loads(text[a:b + 1])
    raise ValueError("no JSON object in response")


def coerce_report(parsed: dict, raw: str) -> tuple[str, str, list]:
    """Extract (report, memory_update, flags_add), coercing nested objects.

    v1: {"report": "<free text>", ...} — returned as-is.
    v2: structured fields (phase_id/actions/...) — the caller renders and
      stores text; this fallback returns ("", mu, flags) so validation
      failures never silently become reports. Use coerce_report_v2 with a
      renderer when chunk timing is available.
    """
    try:
        # v2 structured response without a "report" string: do NOT stringify
        # the dict into a report; signal "needs rendering" with "".
        if isinstance(parsed, dict) and isinstance(parsed.get("actions"), list) \
                and "phase_id" in parsed:
            mu = parsed.get("memory_update", "")
            fa = parsed.get("flags_add", []) or []
            if not isinstance(mu, str):
                mu = str(mu)
            if not isinstance(fa, list):
                fa = [fa]
            return "", mu, [f if isinstance(f, str) else str(f) for f in fa if f]
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
    clean = []
    for f in flags_add:
        if f is None:
            continue
        if isinstance(f, str):
            s = f.strip()
            if s:
                clean.append(s)
        elif isinstance(f, dict):
            # Some VLMs (observed: Lingshu-7B) return flags_add as a list of
            # objects, e.g. [{"event": ...}]. Coerce to compact JSON strings
            # so downstream "; ".join() never sees a dict.
            clean.append(json.dumps(f))
        else:
            clean.append(str(f))
    flags_add = clean
    if not report:
        report = raw if isinstance(raw, str) else json.dumps(raw)
    return report, memory_update, flags_add
