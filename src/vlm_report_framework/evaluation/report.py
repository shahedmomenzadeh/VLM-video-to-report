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
    jvids = []
    for tag in tags:
        for setting in ("s1", "s3"):
            for vid in videos:
                rows = aggregate.chunk_scores(tag, setting, vid)
                if rows:
                    vids_agg.append(aggregate.aggregate_video(tag, setting, vid, rows))
                jrows = aggregate.judge_rows(run_id, tag, setting, vid)
                if jrows and any(r["judged"] for r in jrows):
                    jvids.append(aggregate.aggregate_judge_video(
                        tag, setting, vid, jrows))
    # deterministic panel (Tier 0: lexical/reliability, teacher-referenced)
    panels = {}
    for tag in tags:
        for setting in ("s1", "s3"):
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
    # video-grounded judge panel (Tier J: ordinal 0-3, mean per video)
    panels_j = {}
    for tag in tags:
        for setting in ("s1", "s3"):
            sub = [v for v in jvids if v["tag"] == tag and v["setting"] == setting]
            if not sub:
                continue

            def mj(key, view="all"):
                vals = [v[view][key] for v in sub
                        if v[view].get(key) is not None]
                return round(statistics.mean(vals), 3) if vals else None

            panels_j[f"{tag}.{setting}"] = {
                "videos": len(sub),
                "judged_rate": round(statistics.mean(
                    [v["judged_rate"] for v in sub]), 3),
                "accuracy": mj("accuracy"),
                "coverage": mj("coverage"),
                "nonidle_accuracy": mj("accuracy", "nonidle"),
                "nonidle_coverage": mj("coverage", "nonidle"),
                "idle_accuracy": mj("accuracy", "idle"),
                "idle_coverage": mj("coverage", "idle"),
                "material_rate": mj("material_rate"),
                "unassess_rate": mj("unassess_rate"),
            }
    # setting contrasts per tag (Eval 5)
    contrasts = {}
    contrasts_j = {}
    for tag in tags:
        contrasts[tag] = {
            "s3-s1_rougeL": aggregate.paired_delta(videos, tag, "s1", "s3"),
            "s3-s1_instrF1": aggregate.paired_delta(videos, tag, "s1", "s3", "instr_f1"),
        }
        contrasts_j[tag] = {
            "s3-s1_accuracy": aggregate.paired_judge_delta(
                videos, tag, run_id, "s1", "s3", "accuracy"),
            "s3-s1_coverage": aggregate.paired_judge_delta(
                videos, tag, run_id, "s1", "s3", "coverage"),
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
    (d / "judge_video_scores.jsonl").write_text(
        "\n".join(map(json.dumps, jvids)) + ("\n" if jvids else ""))
    (d / "comparisons.json").write_text(json.dumps(
        {"panels": panels, "panels_j": panels_j,
         "contrasts": contrasts, "contrasts_j": contrasts_j,
         "phase_s1": phase}, indent=1))
    lines = ["# Evaluation dashboard", "",
             "## Tier 0 — deterministic (lexical vs teacher + protocol reliability; ↑ = higher better, ↓ = lower better)",
             "",
             "| source | videos | ROUGE-L ↑ | instrF1 ↑ | echo ↓ | empty ↓ | nonIdle-RougeL ↑ |",
             "|---|---|---|---|---|---|---|"]
    for src, p in sorted(panels.items()):
        lines.append(f"| {src} | {p['videos']} | {p['rougeL']} | {p['instr_f1']} | "
                     f"{p['echo_rate']} | {p['empty_rate']} | {p['nonidle_rougeL']} |")
    lines += ["",
              "## Tier J — video-grounded judge (mean 0-3 per video; 3 = best, 0 = worst)",
              "",
              "| source | videos | judged | accuracy ↑ | coverage ↑ | nonIdle-acc ↑ | nonIdle-cov ↑ | material ↓ | unassess |",
              "|---|---|---|---|---|---|---|---|---|"]
    for src, p in sorted(panels_j.items()):
        lines.append(f"| {src} | {p['videos']} | {p['judged_rate']} | {p['accuracy']} | "
                     f"{p['coverage']} | {p['nonidle_accuracy']} | {p['nonidle_coverage']} | "
                     f"{p['material_rate']} | {p['unassess_rate']} |")
    lines += ["",
              "Notes: Tier J accuracy/coverage are QUALITY grades — higher is better "
              "(0 = no usable description, 1 = material error, 2 = minor error, "
              "3 = correct); they are judged independently (a vague but harmless "
              "report scores well on accuracy and poorly on coverage). material = share "
              "of chunks with a material issue (lower is better); unassess = share with "
              "an abstained detail (descriptive, not scored).",
              "",
              "## Setting contrasts (s3−s1, paired per-video, chunk-matched, 95% bootstrap CI)",
              "", "```json",
              json.dumps({"deterministic": contrasts, "judge": contrasts_j}, indent=1)[:3000],
              "```",
              "", "## s1 phase recognition (deterministic, code)",
              "", "```json", json.dumps(phase, indent=1), "```"]
    (d / "summary.md").write_text("\n".join(lines) + "\n")
    return {"panels": panels, "panels_j": panels_j, "contrasts": contrasts,
            "contrasts_j": contrasts_j, "phase_s1": phase, "out": str(d)}
