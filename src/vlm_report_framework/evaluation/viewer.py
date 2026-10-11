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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from evaluation import ingest  # noqa: E402
from run_state import REPO  # noqa: E402

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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="full25")
    ap.add_argument("--out", default="")
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
    # temporal per chain: "tag.setting.vid" -> counts
    temporal: dict = {}
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
    for (tag, s, _v, _c), _ in claims.items():
        pass
    by_src: dict = {}
    for (tag, s, _v, _c), c in claims.items():
        if c["sup"] is not None:
            by_src.setdefault(f"{tag}.{s}", []).append(c)
    for src, cc in by_src.items():
        dash_t1[src] = {k: round(statistics.mean([x[k2] for x in cc]), 3)
                        for k, k2 in (("sup", "sup"), ("contra", "contra"),
                                       ("unass", "unass"))}
        dash_t1[src]["n"] = len(cc)

    # per-video payload (reports + claim badges + tiers)
    videos: dict = {}
    for vid in vids:
        teach = ingest.load_teacher(vid)
        entry: dict = {"teacher": {}, "models": {}}
        for cid, t in teach.items():
            entry["teacher"][cid] = {"report": t.get("report", ""),
                                     "phase": t.get("phase"),
                                     "t": [t.get("t_start"), t.get("t_end")]}
        for tag in TAGS:
            entry["models"][tag] = {}
            for s in SETTINGS:
                chunks = []
                for r in chunk_rows(tag, s, vid):
                    cid = r.get("chunk_id", "?")
                    c = claims.get((tag, s, vid, cid), {})
                    chunks.append({
                        "cid": cid, "phase": r.get("phase"),
                        "t": [r.get("t_start"), r.get("t_end")],
                        "report": r.get("report", ""),
                        "tiers": r.get("tiers_given"),
                        "frames": r.get("n_frames_used"),
                        "sup": c.get("sup"), "contra": c.get("contra"),
                        "worst": (c.get("worst") or [])[:3]})
                entry["models"][tag][s] = chunks
        videos[vid] = entry

    payload = {"videos": videos, "tags": TAGS, "dash_det": dash_det,
               "dash_t1": dash_t1, "temporal": temporal, "run_id": a.run_id}
    data_js = json.dumps(payload)
    h = lambda s: html.escape(s or "")
    det_rows = "".join(
        f"<tr><td>{h(s)}</td><td>{d['videos']}</td><td>{d['rougeL']}</td>"
        f"<td>{d['instrF1']}</td><td>{d['echo']}</td></tr>"
        for s, d in sorted(dash_det.items()))
    t1_rows = "".join(
        f"<tr><td>{h(s)}</td><td>{d['n']}</td><td>{d['sup']}</td>"
        f"<td>{d['contra']}</td><td>{d['unass']}</td></tr>"
        for s, d in sorted(dash_t1.items()))

    page = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>Eval viewer — {h(a.run_id)}</title>
<style>
body{{font-family:system-ui,sans-serif;margin:0;background:#111;color:#ddd}}
header{{padding:12px 20px;background:#1c1c1c;border-bottom:1px solid #333}}
main{{display:flex;min-height:90vh}}
nav{{width:270px;min-width:270px;border-right:1px solid #333;padding:12px;overflow:auto}}
section{{flex:1;padding:12px 20px;overflow:auto}}
table{{border-collapse:collapse;font-size:12px;margin-bottom:18px}}
td,th{{border:1px solid #444;padding:4px 8px;text-align:right}}
td:first-child,th:first-child{{text-align:left}}
.tabs button,.models button{{margin:2px;padding:5px 10px;background:#222;color:#ddd;border:1px solid #555;cursor:pointer}}
.tabs button.on,.models button.on{{background:#0a5;color:#fff}}
.chunk{{border:1px solid #333;margin:8px 0;padding:8px;background:#181818}}
.chunk summary{{cursor:pointer}}
pre{{white-space:pre-wrap;font-size:12px;background:#0d0d0d;padding:8px}}
.badge{{display:inline-block;padding:1px 7px;border-radius:8px;font-size:11px;margin-left:6px}}
.ok{{background:#0a5}}.mid{{background:#a80}}.bad{{background:#a33}}
select{{padding:6px;font-size:14px;width:100%}}
h2{{margin-top:26px}}small{{color:#999}}
</style></head><body>
<header><b>Video-report eval viewer</b> <small>run {h(a.run_id)} · 25 videos · 7 models × s1/s2/s3 + teacher · Tier-1 support/contradiction + temporal per chain</small></header>
<main><nav>
<h3>Video</h3><select id="vidsel" size="27" style="height:60vh"></select>
</nav><section>
<h2>Dashboard — teacher agreement (deterministic, 25 videos)</h2>
<table><tr><th>source</th><th>videos</th><th>ROUGE-L</th><th>instrF1</th><th>echo</th></tr>{det_rows}</table>
<h2>Dashboard — Tier-1 factuality (support / contradiction / unassessable)</h2>
<table><tr><th>source</th><th>chunks</th><th>sup</th><th>contra</th><th>unass</th></tr>{t1_rows}</table>
<h2>Responses</h2>
<div class="models" id="mtabs"></div>
<div class="tabs" id="stabs"></div>
<div id="chain"></div>
<div id="chunks"></div>
</section></main>
<script>const D={data_js};
const vs=Object.keys(D.videos).sort(),tags=D.tags;
const sel=document.getElementById('vidsel');
vs.forEach(v=>{{const o=document.createElement('option');o.value=v;o.textContent=v;sel.appendChild(o);}});
let curV=vs[0],curM=tags[0],curS='s3';
function badge(x){{if(x==null)return '';const c=x>=0.5?'ok':(x>=0.3?'mid':'bad');return `<span class="badge ${{c}}">${{x}}</span>`;}}
function renderModels(){{const el=document.getElementById('mtabs');el.innerHTML='';
['__teacher__'].concat(tags).forEach(t=>{{const b=document.createElement('button');
b.textContent=t==='__teacher__'?'teacher':t;if(t===curM)b.className='on';
b.onclick=()=>{{curM=t;renderModels();renderChunks();}};el.appendChild(b);}});}}
function renderSettings(){{const el=document.getElementById('stabs');el.innerHTML='';
if(curM==='__teacher__')return;
['s1','s2','s3'].forEach(s=>{{const b=document.createElement('button');b.textContent=s;
if(s===curS)b.className='on';b.onclick=()=>{{curS=s;renderSettings();renderChunks();}};el.appendChild(b);}});}}
function esc(s){{return (s||'').replace(/&/g,'&amp;').replace(/</g,'&lt;');}}
function renderChunks(){{const V=D.videos[curV],box=document.getElementById('chunks'),
ch=document.getElementById('chain');box.innerHTML='';ch.innerHTML='';
if(curM==='__teacher__'){{const ids=Object.keys(V.teacher).sort();
box.innerHTML=ids.map(id=>{{const r=V.teacher[id];
return `<div class="chunk"><b>${{id}}</b> <small>${{r.phase||''}} ${{r.t[0]}}-${{r.t[1]}}s</small><pre>${{esc(r.report)}}</pre></div>`;}}).join('');return;}}
const rows=V.models[curM][curS]||[];
const key=curM+'.'+curS+'.'+curV, T=D.temporal[curM+'.'+curS+'.'+curV];
if(T)ch.innerHTML=`<small>chain temporal: ${{T.n}} transitions · contr ${{T.contr}} · contam ${{T.contam}} · stale ${{T.stale}} · redun ${{T.redun}}</small>`;
box.innerHTML=rows.map(r=>{{const w=(r.worst||[]).map(x=>`<li>[${{x.verdict||'?'}}] ${{esc(x.claim||'')}} <small>${{esc(x.why||'')}}</small></li>`).join('');
const tiers=r.tiers?`<small>tiers given: obs=${{esc(JSON.stringify(r.tiers.observed||[]))}} weak=${{esc(JSON.stringify(r.tiers.weak||[]))}} · frames=${{r.frames}}</small>`:'';
return `<div class="chunk"><details><summary><b>${{r.cid}}</b> <small>${{r.phase||''}} ${{r.t[0]}}-${{r.t[1]}}s</small>${{badge(r.sup)}}<span class="badge bad">${{r.contra??''}}</span></summary>${{tiers}}<pre>${{esc(r.report)}}</pre>${{w?`<ul>${{w}}</ul>`:''}}</details></div>`;}}).join('');}}
sel.onchange=()=>{{curV=sel.value;renderChunks();}};
sel.value=curV;renderModels();renderSettings();renderChunks();
</script></body></html>"""
    out.write_text(page)
    print(f"wrote {out} ({out.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
