"""Eval report: 8-panel dashboard + traceable error examples (deterministic)."""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from evaluation import aggregate, ingest  # noqa: E402
from evaluation.ingest import OUT_EVAL  # noqa: E402


def build(run_id: str, videos: list[str], tags: list[str]) -> dict:
    vids_agg = []
    for tag in tags:
        for setting in ("s1", "s2", "s3"):
            for vid in videos:
                rows = aggregate.chunk_scores(tag, setting, vid)
                if rows:
                    vids_agg.append(aggregate.aggregate_video(tag, setting, vid, rows))
    # 8-panel dashboard: mean over videos per (tag, setting)
    panels = {}
    for tag in tags:
        for setting in ("s1", "s2", "s3"):
            sub = [v for v in vids_agg if v["tag"] == tag and v["setting"] == setting]
            if not sub:
                continue
            m = lambda f: round(statistics.mean([f(v) for v in sub]), 3)
            panels[f"{tag}.{setting}"] = {
                "videos": len(sub),
                "rougeL": m(lambda v: v["all"]["rougeL"]["mean"] or 0),
                "instr_f1": m(lambda v: v["all"]["instr_f1"]["mean"] or 0),
                "echo_rate": m(lambda v: v["echo_rate"]),
                "empty_rate": m(lambda v: v["empty_rate"]),
                "nonidle_rougeL": m(lambda v: (v["nonidle"]["rougeL"]["mean"] or 0)),
            }
    # setting contrasts per tag (Eval 5)
    contrasts = {}
    for tag in tags:
        contrasts[tag] = {
            "s2-s1_rougeL": aggregate.paired_delta(videos, tag, "s1", "s2"),
            "s3-s2_rougeL": aggregate.paired_delta(videos, tag, "s2", "s3"),
            "s3-s2_instrF1": aggregate.paired_delta(videos, tag, "s2", "s3", "instr_f1"),
        }
    # s1 phase recognition per tag
    phase = {}
    for tag in tags:
        accs = [aggregate.phase_metrics_s1(tag, v).get("accuracy")
                for v in videos]
        accs = [a for a in accs if a is not None]
        phase[tag] = {"videos": len(accs),
                      "mean_accuracy": round(statistics.mean(accs), 3) if accs else None}
    d = OUT_EVAL / run_id
    d.mkdir(parents=True, exist_ok=True)
    (d / "video_scores.jsonl").write_text(
        "\n".join(map(json.dumps, vids_agg)) + "\n")
    (d / "comparisons.json").write_text(json.dumps(
        {"panels": panels, "contrasts": contrasts, "phase_s1": phase}, indent=1))
    lines = ["# Evaluation dashboard", "",
             "| source | videos | ROUGE-L | instrF1 | echo | empty | nonIdle-RougeL |",
             "|---|---|---|---|---|---|---|"]
    for src, p in sorted(panels.items()):
        lines.append(f"| {src} | {p['videos']} | {p['rougeL']} | {p['instr_f1']} | "
                     f"{p['echo_rate']} | {p['empty_rate']} | {p['nonidle_rougeL']} |")
    lines += ["", "## Setting contrasts (paired per-video, chunk-matched, 95% bootstrap CI)",
              "", "```json", json.dumps(contrasts, indent=1)[:2000], "```",
              "", "## s1 phase recognition", "", "```json",
              json.dumps(phase, indent=1), "```"]
    (d / "summary.md").write_text("\n".join(lines) + "\n")
    return {"panels": panels, "contrasts": contrasts, "phase_s1": phase,
            "out": str(d)}
