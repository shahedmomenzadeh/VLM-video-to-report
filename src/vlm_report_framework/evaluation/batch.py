"""End-to-end batch runner: evidence + claims + temporal over N videos (parallel, resume-safe)."""
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
              for fb in ("full25", "full5", "pilot") if fb != run_id]
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
        for setting in ("s1", "s2", "s3"):
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


def run_temporal(videos: list[str], tags: list[str], run_id: str, workers: int,
                 mode: str = "tier1") -> None:
    tasks = [(t, s, v) for t in tags for s in ("s1", "s2", "s3") for v in videos]
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
    ap.add_argument("--run-id", default="full5")
    ap.add_argument("--workers", type=int, default=8,
                        help="default for both pools unless overridden")
    ap.add_argument("--vlm-workers", type=int, default=None,
                        help="parallel VLM evidence calls (default: --workers)")
    ap.add_argument("--llm-workers", type=int, default=None,
                        help="parallel LLM claim/temporal calls (default: --workers)")
    ap.add_argument("--skip-evidence", action="store_true")
    ap.add_argument("--skip-claims", action="store_true")
    ap.add_argument("--skip-temporal", action="store_true")
    ap.add_argument("--mode", choices=("tier1", "atomic"), default="tier1",
                        help="tier1: 1 call/report + 1 call/chain; atomic: Tier-2 "
                             "split+verdicts + per-pair (calibration subset)")
    a = ap.parse_args()
    videos = [v.strip() for v in a.videos.split(",") if v.strip()]
    tags = [t.strip() for t in a.tags.split(",") if t.strip()] or ingest.discover_tags()
    vlm_w = a.vlm_workers or a.workers
    llm_w = a.llm_workers or a.workers
    OUT_EVAL.joinpath(a.run_id).mkdir(parents=True, exist_ok=True)
    if not a.skip_evidence:
        run_evidence(videos, a.run_id, vlm_w)
    if not a.skip_claims:
        run_claims(videos, tags, a.run_id, llm_w, a.mode)
    if not a.skip_temporal:
        run_temporal(videos, tags, a.run_id, llm_w, a.mode)
    print(f"Done -> output-evaluation/{a.run_id}")


if __name__ == "__main__":
    main()
