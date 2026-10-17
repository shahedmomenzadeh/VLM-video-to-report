"""End-to-end batch runner over N videos (parallel, resume-safe).

Default (v3): per-chunk VIDEO-GROUNDED JUDGE (evaluation/judge.py) — one
multimodal call per report, accuracy+coverage 0-3, resume-safe by report hash.
Legacy text-only audits (evidence pass / claim counting / temporal labels) are
debugging diagnostics, off by default: --with-evidence, --legacy-claims,
--legacy-temporal.
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from evaluation import evidence as ev  # noqa: E402
from evaluation import ingest, temporal  # noqa: E402
from evaluation.ingest import OUT_EVAL  # noqa: E402
from run_state import REPO  # noqa: E402

CAND = REPO / "output-candidates"


def all_chunks(videos: list[str]) -> list[tuple[str, str]]:
    out = []
    for vid in videos:
        try:
            for c in ingest.expected_chunks(vid):
                out.append((vid, c["chunk_id"]))
        except FileNotFoundError:
            pass
    return out


def done_claims(path: Path) -> set[str]:
    done = set()
    if path.exists():
        for line in open(path):
            try:
                done.add(json.loads(line)["chunk_id"])
            except Exception:
                pass
    return done


def find_evidence(vid: str, cid: str, run_id: str) -> dict | None:
    """Shared evidence lookup: run_id cache, else older runs' (copied forward)."""
    import shutil as _sh
    cands = [OUT_EVAL / run_id / "evidence" / f"{vid}__{cid}.json"]
    cands += [OUT_EVAL / fb / "evidence" / f"{vid}__{cid}.json"
              for fb in ("v2trio", "full25", "full5", "pilot") if fb != run_id]
    for i, p in enumerate(cands):
        if p.exists():
            try:
                rec = json.load(open(p))
            except Exception:
                continue
            if i > 0:
                dest = cands[0]
                dest.parent.mkdir(parents=True, exist_ok=True)
                _sh.copyfile(p, dest)
            return rec
    return None


def run_evidence(videos: list[str], run_id: str, workers: int) -> None:
    chunks = all_chunks(videos)
    print(f"evidence: {len(chunks)} chunks, workers={workers}", flush=True)

    def one(vc):
        vid, cid = vc
        try:
            ev.evidence_for(vid, cid, run_id)
            return f"ok {vid} {cid}"
        except Exception as e:
            return f"WARN {vid} {cid}: {e}"[:200]

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, fut in enumerate(as_completed([pool.submit(one, vc) for vc in chunks])):
            if i % 25 == 0:
                print(f"  evidence {i}/{len(chunks)}: {fut.result()}", flush=True)
    print("evidence done", flush=True)


def run_claims(videos: list[str], tags: list[str], run_id: str, workers: int,
               mode: str = "tier1") -> None:
    cdir = "claims_t1" if mode == "tier1" else "claims"
    score_fn = ev.score_report_tier1 if mode == "tier1" else ev.score_report
    tasks = []
    for tag in tags:
        for setting in ("s1", "s3"):
            for vid in videos:
                mp = CAND / tag / setting / f"{vid}_reports.jsonl"
                if not mp.exists():
                    continue
                rows = [json.loads(l) for l in open(mp)]
                out = OUT_EVAL / run_id / cdir / f"{vid}.{tag}.{setting}.jsonl"
                done = done_claims(out)
                evcache: dict[str, dict] = {}
                for r in rows:
                    if r.get("chunk_id") not in done:
                        tasks.append((tag, setting, vid, r.get("chunk_id", ""),
                                      r.get("report", ""), out))
    print(f"claims ({mode}): {len(tasks)} reports to score, workers={workers}", flush=True)
    lock = threading.Lock()
    (OUT_EVAL / run_id / cdir).mkdir(parents=True, exist_ok=True)

    def one(t):
        tag, setting, vid, cid, rep, out = t
        rec = find_evidence(vid, cid, run_id)
        if rec is None:
            return f"WARN no-evidence {vid} {cid}"
        try:
            s = score_fn(rep, rec)
            row = {"video_id": vid, "tag": tag, "setting": setting,
                   "chunk_id": cid, "n_claims": s["n_claims"],
                   "support_rate": s["support_rate"],
                   "contradiction_rate": s["contradiction_rate"],
                   "unassessable_rate": s["unassessable_rate"],
                   "rows": s["rows"]}
            with lock, open(out, "a") as f:
                f.write(json.dumps(row) + "\n")
            return f"ok {vid}.{tag}.{setting} {cid} sup={s['support_rate']}"
        except Exception as e:
            return f"WARN {vid} {cid}: {e}"[:200]

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(one, t) for t in tasks]
        for i, fut in enumerate(as_completed(futs)):
            if i % 50 == 0:
                print(f"  claims {i}/{len(tasks)}: {fut.result()}", flush=True)
    print("claims done", flush=True)


def run_judgments(videos: list[str], tags: list[str], run_id: str, workers: int,
                  chunks: set[str] | None = None, limit: int = 0,
                  model: str = "") -> None:
    """One video-grounded judge call per report (resume-safe by report hash)."""
    from evaluation import judge as jd  # lazy: keeps openai out of import path
    model = model or jd.MODEL
    chunk_filter = chunks or set()
    tasks: list[tuple] = []
    n_current = 0
    for tag in tags:
        for setting in ("s1", "s3"):
            for vid in videos:
                mp = CAND / tag / setting / f"{vid}_reports.jsonl"
                if not mp.exists():
                    continue
                out = OUT_EVAL / run_id / "judgments" / f"{vid}.{tag}.{setting}.jsonl"
                existing = jd.load_latest(out)
                chain = 0
                for r in (json.loads(l) for l in open(mp)):
                    cid = r.get("chunk_id", "")
                    if chunk_filter and cid not in chunk_filter:
                        continue
                    report = r.get("report", "")
                    content = jd.clinical_content(report)
                    ex = existing.get(cid)
                    if (ex and ex.get("report_hash") == jd.report_hash(content)
                            and not ex.get("error")):
                        n_current += 1
                        continue
                    tasks.append((vid, tag, setting, cid, report, out))
                    chain += 1
                    if limit and chain >= limit:
                        break
    print(f"judgments: {len(tasks)} to score ({n_current} current), "
          f"workers={workers}", flush=True)
    results: dict[str, list[dict]] = {}
    lock = threading.Lock()

    def one(t):
        vid, tag, setting, cid, report, out = t
        try:
            row = jd.judge_one(vid, cid, report, tag, setting, model)
        except Exception as e:
            return f"WARN {vid}.{tag}.{setting} {cid}: {e}"[:200]
        with lock:
            results.setdefault(str(out), []).append(row)
        if row.get("error"):
            return f"ERROR {vid}.{tag}.{setting} {cid}: {row['error'][:90]}"
        return (f"ok {vid}.{tag}.{setting} {cid} "
                f"acc={row['accuracy']} cov={row['coverage']}")

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(one, t) for t in tasks]
        for i, fut in enumerate(as_completed(futs)):
            if i % 25 == 0 or i == len(futs) - 1:
                print(f"  judge {i + 1}/{len(tasks)}: {fut.result()}", flush=True)
    for op, rows in results.items():
        jd.merge_rows(Path(op), rows)
    n_err = sum(1 for rows in results.values() for r in rows if r.get("error"))
    print(f"judgments done: {sum(len(v) for v in results.values())} scored, "
          f"{n_err} errors", flush=True)


def run_temporal(videos: list[str], tags: list[str], run_id: str, workers: int,
                 mode: str = "tier1") -> None:
    tasks = [(t, s, v) for t in tags for s in ("s1", "s3") for v in videos]
    print(f"temporal ({mode}): {len(tasks)} chains, workers={workers}", flush=True)

    def one(t):
        tag, setting, vid = t
        try:
            if mode == "tier1":
                rows = temporal.score_chain(tag, setting, vid, run_id)
            else:
                rows = temporal.score_transitions(tag, setting, vid, run_id)
            n = len(rows)
            return (f"ok {vid}.{tag}.{setting}: tr={n} "
                    f"contr={sum(r['contradiction'] for r in rows)} "
                    f"contam={sum(r['contamination'] for r in rows)} "
                    f"stale={sum(r['stale_instrument'] for r in rows)} "
                    f"redun={sum(r['redundant'] for r in rows)}")
        except Exception as e:
            return f"WARN {t}: {e}"[:200]

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, fut in enumerate(as_completed([pool.submit(one, t) for t in tasks])):
            print(f"  temporal {i}/{len(tasks)}: {fut.result()}", flush=True)
    print("temporal done", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", required=True)
    ap.add_argument("--tags", default="")
    ap.add_argument("--run-id", default="v2slim")
    ap.add_argument("--workers", type=int, default=8,
                    help="default for both pools unless overridden")
    ap.add_argument("--judge-workers", type=int, default=None,
                    help="parallel judge calls (default: --workers)")
    ap.add_argument("--vlm-workers", type=int, default=None,
                    help="parallel VLM evidence calls, legacy (default: --workers)")
    ap.add_argument("--llm-workers", type=int, default=None,
                    help="parallel LLM legacy claim/temporal calls (default: --workers)")
    ap.add_argument("--skip-judge", action="store_true",
                    help="skip the video-grounded judge pass")
    ap.add_argument("--chunks", default="",
                    help="CSV chunk-id filter (calibration subsets)")
    ap.add_argument("--limit", type=int, default=0,
                    help="max judge calls per chain after the filter")
    ap.add_argument("--judge-model", default="",
                    help="gateway model for the judge (default: ag/gemini-3.8-flash)")
    ap.add_argument("--with-evidence", action="store_true",
                    help="legacy: candidate-blind evidence note per clip (debugging)")
    ap.add_argument("--legacy-claims", action="store_true",
                    help="legacy: text-only claim counting (research diagnostic)")
    ap.add_argument("--legacy-temporal", action="store_true",
                    help="legacy: 5-label temporal audit (research diagnostic)")
    ap.add_argument("--mode", choices=("tier1", "atomic"), default="tier1",
                    help="legacy claims mode: tier1: 1 call/report; atomic: "
                         "split+verdicts (calibration subset)")
    a = ap.parse_args()
    videos = [v.strip() for v in a.videos.split(",") if v.strip()]
    tags = [t.strip() for t in a.tags.split(",") if t.strip()] or ingest.discover_tags()
    chunks = {c.strip() for c in a.chunks.split(",") if c.strip()}
    judge_w = a.judge_workers or a.workers
    vlm_w = a.vlm_workers or a.workers
    llm_w = a.llm_workers or a.workers
    OUT_EVAL.joinpath(a.run_id).mkdir(parents=True, exist_ok=True)
    if not a.skip_judge:
        run_judgments(videos, tags, a.run_id, judge_w, chunks, a.limit,
                      a.judge_model)
    if a.with_evidence:
        run_evidence(videos, a.run_id, vlm_w)
    if a.legacy_claims:
        run_claims(videos, tags, a.run_id, llm_w, a.mode)
    if a.legacy_temporal:
        run_temporal(videos, tags, a.run_id, llm_w, a.mode)
    print(f"Done -> output-evaluation/{a.run_id}")


if __name__ == "__main__":
    main()
