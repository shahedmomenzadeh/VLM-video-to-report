"""Eval A: automatic reference agreement, video-level.

Candidate R_video vs teacher R_video^T (stitched docs from video_documents.py):
  primary   ROUGE-L (rouge-score), chrF (sacrebleu)
  secondary METEOR (nltk, WordNet-based — flat on surgical terms, report-only)
  report-only BLEU-4 (sacrebleu), CIDEr (pycocoevalcap if installed, else skipped)

Scoring levels (averaged per video):
  full doc + per-section (Observed instruments / Actions-events) via section_of().
  Per-section keeps the long Actions paragraph from drowning instrument errors.

Outputs (output-judge/lexical/, git-ignored):
  scores.jsonl  (one row per video x candidate source)
  summary.md    (mean per candidate source)

Usage:
  uv run python src/vlm_report_framework/score_lexical.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_state import REPO  # noqa: E402
from video_documents import section_of  # noqa: E402

DOCS = REPO / "output-judge" / "video-documents"
OUT = REPO / "output-judge" / "lexical"


def score_pair(ref: str, hyp: str) -> dict:
    s: dict = {}
    # ROUGE-L
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        s["rougeL_f"] = round(scorer.score(ref, hyp)["rougeL"].fmeasure, 4)
    except Exception as e:  # noqa: BLE001
        s["rougeL_f"] = None
        s["rougeL_err"] = str(e)[:100]
    # BLEU-4 + chrF
    try:
        import sacrebleu
        s["bleu"] = round(sacrebleu.corpus_bleu([hyp], [[ref]]).score, 2)
        s["chrf"] = round(sacrebleu.corpus_chrf([hyp], [[ref]]).score, 2)
    except Exception as e:  # noqa: BLE001
        s["bleu"] = s["chrf"] = None
        s["sacrebleu_err"] = str(e)[:100]
    # METEOR (needs wordnet; download once)
    try:
        import nltk
        try:
            nltk.data.find("corpora/wordnet")
        except LookupError:
            nltk.download("wordnet", quiet=True)
        from nltk.translate.meteor_score import meteor_score
        s["meteor"] = round(meteor_score([ref.split()], hyp.split()), 4)
    except Exception as e:  # noqa: BLE001
        s["meteor"] = None
        s["meteor_err"] = str(e)[:100]
    # CIDEr is corpus-level (TF-IDF over the corpus): single-pair calls
    # degenerate to 0. Computed per-source over all videos in main().
    return s


def cider_corpus(refs: list[str], hyps: list[str]) -> float | None:
    """Corpus-level CIDEr (TF-IDF over the corpus).

    NOTE: CIDEr is designed for short captions with a large corpus. On 3
    long video-docs every n-gram's document frequency saturates and even
    self-match scores 0.0 — so here the corpus is CHUNKS (all chunk reports
    of a source), not video-docs. Returns one number per candidate source.
    """
    try:
        from pycocoevalcap.cider.cider import Cider
        gts = {i: [r] for i, r in enumerate(refs)}
        res = {i: [h] for i, h in enumerate(hyps)}
        score, _ = Cider().compute_score(gts, res)
        return round(float(score), 4)
    except Exception:
        return None


def cider_chunk_corpus(source: str) -> tuple[float | None, int]:
    """CIDEr over the chunk corpus for one candidate source (tag.setting)."""
    import glob
    tag, setting = source.split(".")
    refs, hyps = [], []
    for tf in sorted(glob.glob(str(REPO / "output-teacher" / "*_teacher_reports.jsonl"))):
        vid = Path(tf).name.replace("_teacher_reports.jsonl", "")
        cf = REPO / "output-candidates" / tag / setting / f"{vid}_reports.jsonl"
        if not cf.exists():
            continue
        crows = {}
        for line in open(cf):
            r = json.loads(line)
            crows[r["chunk_id"]] = r.get("report", "")
        for line in open(tf):
            r = json.loads(line)
            if r["chunk_id"] in crows:
                refs.append(r.get("report", ""))
                hyps.append(crows[r["chunk_id"]])
    if not refs:
        return None, 0
    return cider_corpus(refs, hyps), len(refs)


def main() -> None:
    teacher_docs = sorted(DOCS.glob("*.teacher.md"))
    assert teacher_docs, f"no stitched docs in {DOCS} — run video_documents.py first"
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for tdoc in teacher_docs:
        vid = tdoc.name.split(".")[0]
        ref_full = tdoc.read_text()
        ref_obs = section_of(ref_full, "Observed instruments")
        ref_act = section_of(ref_full, "Actions/events")
        for cdoc in sorted(DOCS.glob(f"{vid}.*.*.md")):
            if ".teacher." in cdoc.name:
                continue
            src = cdoc.name[len(vid) + 1:-len(".md")]
            hyp_full = cdoc.read_text()
            row: dict = {"video_id": vid, "source": src}
            row.update({f"full_{k}": v for k, v in
                        score_pair(ref_full, hyp_full).items()})
            row.update({f"obs_{k}": v for k, v in
                        score_pair(ref_obs, section_of(hyp_full, "Observed instruments")).items()})
            row.update({f"act_{k}": v for k, v in
                        score_pair(ref_act, section_of(hyp_full, "Actions/events")).items()})
            rows.append(row)
            print(f"{vid} {src}: rougeL={row['full_rougeL_f']} "
                  f"chrf={row['full_chrf']} meteor={row['full_meteor']} "
                  f"bleu={row['full_bleu']}")
    with open(OUT / "scores.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    # CIDEr corpus-level per source over the CHUNK corpus (video-doc corpus
    # of 3 degenerates: shared boilerplate saturates IDF, self-match = 0.0)
    cider_by_src = {}
    for src in sorted(set(r["source"] for r in rows)):
        score, n = cider_chunk_corpus(src)
        cider_by_src[src] = score
        print(f"CIDEr chunk-corpus {src}: n={n} cider={score}")
    for r in rows:
        r["full_cider_corpus"] = cider_by_src.get(r["source"])
    with open(OUT / "scores.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    # summary: mean per source over videos
    import statistics
    sources = sorted(set(r["source"] for r in rows))
    lines = ["# Lexical agreement (mean over videos; CIDEr corpus-level per source)", "",
             "| source | videos | full_rougeL | full_chrF | full_METEOR | full_BLEU | full_CIDEr | obs_rougeL | act_rougeL |",
             "|---|---|---|---|---|---|---|---|---|"]
    for src in sources:
        sub = [r for r in rows if r["source"] == src]
        def mean(k: str):  # noqa: ANN202
            vs = [r[k] for r in sub if isinstance(r.get(k), (int, float))]
            return round(statistics.mean(vs), 3) if vs else "-"
        lines.append(f"| {src} | {len(sub)} | {mean('full_rougeL_f')} | "
                     f"{mean('full_chrf')} | {mean('full_meteor')} | {mean('full_bleu')} | "
                     f"{sub[0].get('full_cider_corpus', '-')} | "
                     f"{mean('obs_rougeL_f')} | {mean('act_rougeL_f')} |")
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")
    print(f"Done. {len(rows)} rows -> {OUT.relative_to(REPO)}/scores.jsonl")


if __name__ == "__main__":
    main()
