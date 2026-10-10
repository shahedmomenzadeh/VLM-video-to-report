"""CLI: python -m vlm_report_framework.evaluation.run <cmd> ..."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from evaluation import ingest, reference, reliability  # noqa: E402
from evaluation.ingest import OUT_EVAL  # noqa: E402


def cmd_inventory(a):
    rows = ingest.inventory(
        videos=a.videos.split(",") if a.videos != "all" else None,
        tags=a.tags.split(",") if a.tags else None)
    d = ingest.write_manifest(a.run_id, {"cmd": "inventory"})
    (d / "inventory.jsonl").write_text("\n".join(map(json.dumps, rows)) + "\n")
    miss = sum(r["n_missing"] for r in rows)
    print(f"{len(rows)} chains, {miss} missing chunks -> {d}")


def cmd_reliability(a):
    vid = a.video
    for tag in (a.tags.split(",") if a.tags else ingest.discover_tags()):
        for setting in ("s1", "s2", "s3"):
            rows = reliability.audit(tag, setting, vid)
            if not rows:
                continue
            n = len(rows)
            print(f"{tag}.{setting}: n={n} empty={sum(r['empty'] for r in rows)} "
                  f"echo={sum(r['echo_hit'] for r in rows)} "
                  f"degen={sum(r['degenerate'] for r in rows)} "
                  f"no_mu={sum(not r['has_memory_update'] for r in rows)} "
                  f"rep_max={max((r['repetition'] for r in rows), default=0)}")


def cmd_reference(a):
    teach = ingest.load_teacher(a.video)
    for tag in (a.tags.split(",") if a.tags else ingest.discover_tags()):
        for setting in ("s1", "s2", "s3"):
            mp = reliability.CAND / tag / setting / f"{a.video}_reports.jsonl"
            if not mp.exists():
                continue
            rs, rl, f1s = [], [], []
            n = 0
            for line in open(mp):
                r = json.loads(line)
                t = teach.get(r["chunk_id"])
                if not t:
                    continue
                d = reference.score_chunk(r.get("report", ""), t.get("report", ""))
                rs.append(d.get("act_rougeL") or 0)
                f1s.append(d["instr"]["f1"])
                n += 1
            import statistics
            m = lambda xs: round(statistics.mean(xs), 3) if xs else "-"
            print(f"{tag}.{setting}: n={n} rougeL={m(rs)} instrF1={m(f1s)}")


def cmd_evidence(a):
    from evaluation import evidence as ev
    rec = ev.evidence_for(a.video, a.chunk, a.run_id)
    print(json.dumps({k: rec.get(k) for k in
                      ("visible_instruments", "anatomy", "actions",
                       "uncertain", "error")}, indent=1)[:1500])
    if a.score:
        mp = reliability.CAND / a.tag / a.setting / f"{a.video}_reports.jsonl"
        rep = ""
        for line in open(mp):
            r = json.loads(line)
            if r["chunk_id"] == a.chunk:
                rep = r.get("report", "")
        s = ev.score_report(rep, rec)
        print(f"claims={s['n_claims']} support={s['support_rate']} "
              f"contra={s['contradiction_rate']} unass={s['unassessable_rate']}")
        for row in s["rows"][:10]:
            print(f"  [{row['verdict']}] {row['claim'][:110]}")


def cmd_temporal(a):
    from evaluation import temporal as tp
    for tag in (a.tags.split(",") if a.tags else ingest.discover_tags()):
        for setting in (a.settings.split(",") if a.settings else ("s1", "s2", "s3")):
            rows = tp.score_transitions(tag, setting, a.video, a.run_id)
            if not rows:
                continue
            n = len(rows)
            print(f"{tag}.{setting}: transitions={n} "
                  f"contr={sum(r['contradiction'] for r in rows)} "
                  f"contam={sum(r['contamination'] for r in rows)} "
                  f"stale={sum(r['stale_instrument'] for r in rows)} "
                  f"redun={sum(r['redundant'] for r in rows)}")


def cmd_phase(a):
    from evaluation import aggregate as ag
    vids = a.videos.split(",") if a.videos != "all" else [
        p.name for p in Path("videos").glob("PH_*") if p.is_dir()]
    for tag in (a.tags.split(",") if a.tags else ingest.discover_tags()):
        accs = []
        for v in vids:
            m = ag.phase_metrics_s1(tag, v)
            if m:
                accs.append(m["accuracy"])
        import statistics
        print(f"{tag}: s1 videos={len(accs)} "
              f"mean_acc={round(statistics.mean(accs), 3) if accs else '-'}")


def cmd_report(a):
    from evaluation import report as rp
    vids = a.videos.split(",") if a.videos != "all" else sorted(
        p.name for p in Path("videos").glob("PH_*") if p.is_dir())
    tags = a.tags.split(",") if a.tags else ingest.discover_tags()
    out = rp.build(a.run_id, vids, tags)
    print(f"panels={len(out['panels'])} -> {out['out']}/summary.md")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("inventory"); p.add_argument("--videos", default="all")
    p.add_argument("--tags", default=""); p.add_argument("--run-id", default="pilot")
    p = sub.add_parser("reliability"); p.add_argument("--video", required=True)
    p.add_argument("--tags", default="")
    p = sub.add_parser("reference"); p.add_argument("--video", required=True)
    p.add_argument("--tags", default="")
    p = sub.add_parser("evidence"); p.add_argument("--video", required=True)
    p.add_argument("--chunk", required=True); p.add_argument("--run-id", default="pilot")
    p.add_argument("--tag", default=""); p.add_argument("--setting", default="s3")
    p.add_argument("--score", action="store_true")
    p = sub.add_parser("temporal"); p.add_argument("--video", required=True)
    p.add_argument("--tags", default=""); p.add_argument("--settings", default="s3")
    p.add_argument("--run-id", default="pilot")
    p = sub.add_parser("phase"); p.add_argument("--videos", default="all")
    p.add_argument("--tags", default="")
    p = sub.add_parser("report"); p.add_argument("--videos", default="all")
    p.add_argument("--tags", default=""); p.add_argument("--run-id", default="pilot")
    a = ap.parse_args()
    {"inventory": cmd_inventory, "reliability": cmd_reliability,
     "reference": cmd_reference, "evidence": cmd_evidence,
     "temporal": cmd_temporal, "phase": cmd_phase,
     "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    main()
