"""Static HTML viewer: per-video model responses (s1/s2/s3) + eval dashboard.

Reads output-evaluation/<run-id>/ + output-candidates masters. Writes a single
self-contained viewer.html (works from file://, no CDN).

Usage:
  .venv-vlm/bin/python src/vlm_report_framework/evaluation/viewer.py \
      --run-id full25 --out output-evaluation/full25/viewer.html
"""
from __future__ import annotations

import argparse
import html
import json
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent.parent

TEACHER_DIR = REPO / "output-teacher"


def load_teacher(vid: str) -> dict:
    """Read teacher reports master directly (avoids ingest's pandas dep)."""
    p = TEACHER_DIR / f"{vid}_teacher_reports.jsonl"
    out = {}
    if p.exists():
        for line in open(p):
            try:
                r = json.loads(line)
                out[r.get("chunk_id", "")] = r
            except Exception:
                pass
    return out

CAND = REPO / "output-candidates"
TAGS = ["hulumed-4b", "hulumed-7b", "lingshu-7b", "qwen3vl-2b-grpo",
        "qwen3vl-2b-instruct", "qwen3vl-2b-sft-stage2", "qwen3vl-4b-instruct"]
SETTINGS = ("s1", "s2", "s3")


def chunk_rows(tag: str, setting: str, vid: str) -> list[dict]:
    mp = CAND / tag / setting / f"{vid}_reports.jsonl"
    if not mp.exists():
        return []
    rows = [json.loads(l) for l in open(mp)]
    rows.sort(key=lambda r: (r.get("t_start", 0), r.get("t_end", 0)))
    return rows


def load_raw(tag: str, setting: str, vid: str, cid: str) -> dict:
    p = CAND / tag / setting / "reports" / f"{vid}__{cid}.json"
    if not p.exists():
        return {}
    try:
        d = json.load(open(p))
        raw = d.get("raw_response", "")
        return {"prompt": (d.get("prompt") or "")[:1500],
                "raw": (json.dumps(raw, indent=1) if isinstance(raw, dict)
                        else str(raw or ""))[:1500]}
    except Exception:
        return {}


def load_teacher_mem(vid: str) -> dict:
    p = REPO / "hf_dataset" / "memory" / f"{vid}_memory.jsonl"
    out = {}
    if p.exists():
        for line in open(p):
            try:
                r = json.loads(line)
                mem = r.get("memory_after", {}) or {}
                out[r.get("chunk_id", "")] = mem.get("running_summary", "")
            except Exception:
                pass
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="full25")
    ap.add_argument("--out", default="")
    ap.add_argument("--with-raw", action="store_true",
                    help="embed truncated prompt/raw-response per chunk")
    a = ap.parse_args()
    ev = REPO / "output-evaluation" / a.run_id
    out = Path(a.out) if a.out else ev / "viewer.html"

    vids = sorted(p.name for p in (REPO / "videos").glob("PH_*") if p.is_dir())
    # claims_t1 lookup: (tag, setting, vid, chunk) -> rates + worst
    claims: dict = {}
    for f in (ev / "claims_t1").glob("*.jsonl") if (ev / "claims_t1").exists() else []:
        for line in open(f):
            try:
                r = json.loads(line)
                claims[(r["tag"], r["setting"], r["video_id"], r["chunk_id"])] = {
                    "sup": r.get("support_rate"), "contra": r.get("contradiction_rate"),
                    "unass": r.get("unassessable_rate"), "n": r.get("n_claims"),
                    "worst": r.get("rows", []) or []}
            except Exception:
                pass
    # temporal per chain + per-chunk incoming-transition map
    temporal: dict = {}
    tin: dict = {}
    td = ev / "transitions_t1"
    if td.exists():
        for f in td.glob("*.json"):
            try:
                d = json.load(open(f))
                tr = d.get("transitions", [])
                temporal[f"{d['tag']}.{d['setting']}.{d['video_id']}"] = {
                    "n": len(tr),
                    "contr": sum(1 for t in tr if t.get("contradiction")),
                    "contam": sum(1 for t in tr if t.get("contamination")),
                    "stale": sum(1 for t in tr if t.get("stale_instrument")),
                    "redun": sum(1 for t in tr if t.get("redundant"))}
                for t in tr:
                    tin[(d["tag"], d["setting"], d["video_id"],
                         t.get("curr", ""))] = {
                        "prev": t.get("prev"), "contr": t.get("contradiction"),
                        "contam": t.get("contamination"),
                        "stale": t.get("stale_instrument"),
                        "redun": t.get("redundant"),
                        "prog": t.get("progression"), "why": t.get("why", "")}
            except Exception:
                pass
    # video-grounded judge rows: (tag, setting, vid, chunk) -> verdict
    jrows: dict = {}
    jd_dir = ev / "judgments"
    if jd_dir.exists():
        for f in jd_dir.glob("*.jsonl"):
            for line in open(f):
                try:
                    r = json.loads(line)
                    jrows[(r.get("tag"), r.get("setting"), r.get("video_id"),
                           r.get("chunk_id"))] = r
                except Exception:
                    pass
    # dashboard means
    dash_det: dict = {}
    vs_path = ev / "video_scores.jsonl"
    if vs_path.exists():
        per_src: dict = {}
        for line in open(vs_path):
            try:
                v = json.loads(line)
                per_src.setdefault(f"{v['tag']}.{v['setting']}", []).append(v)
            except Exception:
                pass
        for src, vv in per_src.items():
            m = lambda f: round(statistics.mean([f(x) for x in vv]), 3)
            dash_det[src] = {
                "videos": len(vv),
                "rougeL": m(lambda x: (x["all"]["rougeL"]["mean"] or 0)),
                "instrF1": m(lambda x: (x["all"]["instr_f1"]["mean"] or 0)),
                "echo": m(lambda x: x["echo_rate"])}
    dash_t1: dict = {}
    by_src: dict = {}
    for (tag, s, _v, _c), c in claims.items():
        if c["sup"] is not None:
            by_src.setdefault(f"{tag}.{s}", []).append(c)
    for src, cc in by_src.items():
        dash_t1[src] = {k: round(statistics.mean([x[k2] for x in cc]), 3)
                        for k, k2 in (("sup", "sup"), ("contra", "contra"),
                                       ("unass", "unass"))}
        dash_t1[src]["n"] = len(cc)
    # comparisons.json: panels / contrasts / phase_s1 (+ judge panels)
    panels: dict = {}
    contrasts: dict = {}
    phase_s1: dict = {}
    panels_j: dict = {}
    contrasts_j: dict = {}
    cmp_path = ev / "comparisons.json"
    if cmp_path.exists():
        try:
            cj = json.load(open(cmp_path))
            panels = cj.get("panels", {})
            contrasts = cj.get("contrasts", {})
            phase_s1 = cj.get("phase_s1", {})
            panels_j = cj.get("panels_j", {})
            contrasts_j = cj.get("contrasts_j", {})
        except Exception:
            pass
    # temporal_t1.json global aggregates
    temp_agg: dict = {}
    ta_path = ev / "temporal_t1.json"
    if ta_path.exists():
        try:
            temp_agg = json.load(open(ta_path))
        except Exception:
            pass

    # per-video payload (reports + claim badges + tiers + evidence + temporal)
    videos: dict = {}
    for vid in vids:
        teach = load_teacher(vid)
        tmem = load_teacher_mem(vid)
        evmap: dict = {}
        for f in (ev / "evidence").glob(f"{vid}__*.json"):
            try:
                d = json.load(open(f))
                evmap[d.get("chunk_id", f.stem)] = {
                    k: d.get(k) for k in ("visible_instruments", "anatomy",
                                          "actions", "state_changes", "uncertain")}
            except Exception:
                pass
        entry: dict = {"teacher": {}, "models": {}}
        for cid, t in teach.items():
            entry["teacher"][cid] = {"report": t.get("report", ""),
                                     "phase": t.get("phase"),
                                     "t": [t.get("t_start"), t.get("t_end")],
                                     "mem": tmem.get(cid, "")}
        for tag in TAGS:
            entry["models"][tag] = {}
            for s in SETTINGS:
                chunks = []
                for r in chunk_rows(tag, s, vid):
                    cid = r.get("chunk_id", "?")
                    c = claims.get((tag, s, vid, cid), {})
                    j = jrows.get((tag, s, vid, cid), {})
                    raw = load_raw(tag, s, vid, cid) if a.with_raw else {}
                    chunks.append({
                        "cid": cid, "phase": r.get("phase"),
                        "phase_name": r.get("phase_name"),
                        "stated": r.get("stated_phase"),
                        "t": [r.get("t_start"), r.get("t_end")],
                        "report": r.get("report", ""),
                        "memup": r.get("memory_update", ""),
                        "tiers": r.get("tiers_given"),
                        "frames": r.get("n_frames_used"),
                        "lat": r.get("latency_s"),
                        "jacc": j.get("accuracy"), "jcov": j.get("coverage"),
                        "jissues": j.get("issues") or [],
                        "junass": j.get("unassessable") or [],
                        "jerr": j.get("error"),
                        "sup": c.get("sup"), "contra": c.get("contra"),
                        "unass": c.get("unass"), "ncl": c.get("n"),
                        "worst": (c.get("worst") or [])[:3],
                        "ev": evmap.get(cid),
                        "tin": tin.get((tag, s, vid, cid)),
                        "prompt": raw.get("prompt"),
                        "rawresp": raw.get("raw")})
                entry["models"][tag][s] = chunks
        videos[vid] = entry

    payload = {"videos": videos, "tags": TAGS, "settings": list(SETTINGS),
               "dash_det": dash_det, "dash_t1": dash_t1, "panels": panels,
               "panels_j": panels_j, "contrasts": contrasts,
               "contrasts_j": contrasts_j, "phase_s1": phase_s1,
               "temp_agg": temp_agg, "temporal": temporal, "run_id": a.run_id,
               "has_raw": a.with_raw}
    data_js = json.dumps(payload).replace("</", "<\\/")
    tpl = (Path(__file__).resolve().parent / "dashboard_template.html").read_text()
    page = tpl.replace("__DATA__", data_js).replace("__RUN__", html.escape(a.run_id))
    out.write_text(page)
    print(f"wrote {out} ({out.stat().st_size/1e6:.1f} MB)")



if __name__ == "__main__":
    main()
