"""Eval 5+6: settings contrasts + video-level aggregation (deterministic)."""
from __future__ import annotations

import json
import random
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from clean_instruments import make_chunks  # noqa: E402
from evaluation import ingest, reference, reliability  # noqa: E402
from run_state import load_timeline  # noqa: E402


def chunk_scores(tag: str, setting: str, video_id: str) -> list[dict]:
    """Cheap deterministic per-chunk scores (reliability + reference + phase)."""
    teach = ingest.load_teacher(video_id)
    mp = reliability.CAND / tag / setting / f"{video_id}_reports.jsonl"
    if not mp.exists():
        return []
    raw_cache: dict[str, str | None] = {}
    out = []
    for line in open(mp):
        r = json.loads(line)
        cid = r.get("chunk_id")
        if cid not in raw_cache:
            raw_cache[cid] = reliability.load_raw(tag, setting, video_id, cid)
        rel = reliability.check_row(r, raw_cache[cid])
        t = teach.get(cid, {})
        ref = reference.score_chunk(r.get("report", ""), t.get("report", ""))
        dur = (r.get("t_end", 0) or 0) - (r.get("t_start", 0) or 0)
        out.append({
            "video_id": video_id, "tag": tag, "setting": setting,
            "chunk_id": cid, "phase": r.get("phase"),
            "t_start": r.get("t_start"), "t_end": r.get("t_end"),
            "dur": round(dur, 1),
            "is_idle": r.get("phase") == "P13",
            "stated_phase": r.get("stated_phase"),
            "rougeL": ref.get("act_rougeL"), "chrf": ref.get("act_chrf"),
            "instr_f1": ref["instr"]["f1"], "anatomy_f1": ref["anatomy"]["f1"],
            **{f"rel_{k}": v for k, v in rel.items()
               if k in ("empty", "echo_hit", "degenerate", "repetition",
                        "has_memory_update", "sections_present")},
        })
    return out


def video_means(rows: list[dict], key: str) -> dict:
    vs = [r[key] for r in rows
          if isinstance(r.get(key), (int, float)) and not isinstance(r.get(key), bool)]
    return {"mean": round(statistics.mean(vs), 4) if vs else None, "n": len(vs)}


def aggregate_video(tag: str, setting: str, video_id: str,
                    rows: list[dict] | None = None) -> dict:
    rows = rows if rows is not None else chunk_scores(tag, setting, video_id)
    nonidle = [r for r in rows if not r["is_idle"]]
    idle = [r for r in rows if r["is_idle"]]
    by_phase: dict[str, dict] = {}
    for ph in sorted(set(r["phase"] for r in rows if r.get("phase"))):
        by_phase[ph] = {k: video_means([r for r in rows if r["phase"] == ph], k)
                        for k in ("rougeL", "instr_f1")}
    dur_w = sum(r["dur"] * (r["rougeL"] or 0) for r in rows)
    dur_tot = sum(r["dur"] for r in rows) or 1
    return {
        "video_id": video_id, "tag": tag, "setting": setting,
        "n_chunks": len(rows),
        "all": {k: video_means(rows, k) for k in ("rougeL", "instr_f1", "anatomy_f1")},
        "nonidle": {k: video_means(nonidle, k) for k in ("rougeL", "instr_f1")},
        "idle": {k: video_means(idle, k) for k in ("rougeL", "instr_f1")},
        "rougeL_dur_weighted": round(dur_w / dur_tot, 4),
        "echo_rate": round(sum(1 for r in rows if r.get("rel_echo_hit")) / max(1, len(rows)), 3),
        "empty_rate": round(sum(1 for r in rows if r.get("rel_empty")) / max(1, len(rows)), 3),
        "by_phase": by_phase,
    }


def phase_metrics_s1(tag: str, video_id: str) -> dict:
    """s1 recognition: accuracy, macro-F1, confusion, P?? rate."""
    rows = chunk_scores(tag, "s1", video_id)
    if not rows:
        return {}
    try:
        tl = {c["chunk_id"]: c["phase"] for c in make_chunks(load_timeline(video_id))}
    except FileNotFoundError:
        return {}
    y_true = [tl.get(r["chunk_id"], "?") for r in rows]
    y_pred = [r.get("stated_phase") or "P??" for r in rows]
    acc = sum(t == p for t, p in zip(y_true, y_pred)) / max(1, len(rows))
    labels = sorted(set(y_true) | set(y_pred))
    f1s = []
    for lb in labels:
        tp = sum(t == lb and p == lb for t, p in zip(y_true, y_pred))
        fp = sum(t != lb and p == lb for t, p in zip(y_true, y_pred))
        fn = sum(t == lb and p != lb for t, p in zip(y_true, y_pred))
        p = tp / max(1, tp + fp)
        rc = tp / max(1, tp + fn)
        f1s.append(2 * p * rc / max(1e-9, p + rc))
    conf: dict[str, dict[str, int]] = {}
    for t, p in zip(y_true, y_pred):
        conf.setdefault(t, {}).setdefault(p, 0)
        conf[t][p] += 1
    return {"video_id": video_id, "tag": tag, "n": len(rows),
            "accuracy": round(acc, 3),
            "macro_f1": round(statistics.mean(f1s), 3) if f1s else None,
            "unknown_rate": round(sum(p == "P??" for p in y_pred) / len(y_pred), 3),
            "confusion": conf}


def paired_delta(videos: list[str], tag: str, set_a: str, set_b: str,
                 key: str = "rougeL") -> dict:
    """Mean of per-video (B − A) on identical chunk sets + bootstrap CI."""
    diffs = []
    for vid in videos:
        ra = {r["chunk_id"]: r for r in chunk_scores(tag, set_a, vid)}
        rb = {r["chunk_id"]: r for r in chunk_scores(tag, set_b, vid)}
        common = [c for c in ra if c in rb
                  and isinstance(ra[c].get(key), (int, float))
                  and isinstance(rb[c].get(key), (int, float))]
        if not common:
            continue
        diffs.append(statistics.mean([rb[c][key] - ra[c][key] for c in common]))
    if not diffs:
        return {"n_videos": 0}
    rng = random.Random(0)
    boots = [statistics.mean([rng.choice(diffs) for _ in diffs]) for _ in range(2000)]
    boots.sort()
    return {"n_videos": len(diffs),
            "mean_delta": round(statistics.mean(diffs), 4),
            "ci95": [round(boots[50], 4), round(boots[1950], 4)]}
