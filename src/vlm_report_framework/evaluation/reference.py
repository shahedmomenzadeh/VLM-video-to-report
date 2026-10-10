"""Eval 2: teacher reference agreement (deterministic, no LLM, no BERTScore)."""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from clean_instruments import EXPECTED  # noqa: E402

CLASSES = sorted({c for s in EXPECTED.values() for c in s} | {"Forceps"})
SYN = {  # only genuinely equivalent terms
    "i/a probe": "I-A-Handpiece", "i-a probe": "I-A-Handpiece",
    "irrigation": "I-A-Handpiece", "aspiration": "I-A-Handpiece",
    "capsulorhexis forceps": "Cap-Forceps", "capsule forceps": "Cap-Forceps",
    "keratome": "Primary-Knife", "primary knife": "Primary-Knife",
    "phaco probe": "Phaco-Handpiece", "phaco handpiece": "Phaco-Handpiece",
    "lens injector": "Lens-Injector", "second instrument": "Second-Instrument",
    "tissue forceps": "Forceps",
}
LABELS = ["Phase:", "Observed instruments:", "Candidate but not confirmed:",
          "Anatomy visible:", "Actions/events:"]


def section(report: str, header: str) -> str:
    """All-occurrences section extract (fixes old first-match-only bug)."""
    pat = re.compile(rf"^{re.escape(header)}(.*?)(?=^(?:Phase:|Observed instruments:|Candidate but not confirmed:|Anatomy visible:|Actions/events:)|\Z)",
                     re.M | re.S)
    return "\n".join(m.group(1).strip() for m in pat.finditer(report or "")).strip()


def actions_text(report: str) -> str:
    return section(report, "Actions/events:")


def canon_set(text: str) -> set[str]:
    t = (text or "").lower()
    found = set()
    for k, v in SYN.items():
        if k in t:
            found.add(v)
    for c in CLASSES:
        if c.lower() in t:
            found.add(c)
    return found


def prf(pred: set[str], ref: set[str]) -> dict:
    tp = len(pred & ref)
    p = tp / max(1, len(pred))
    r = tp / max(1, len(ref))
    f = 2 * p * r / max(1e-9, p + r)
    return {"p": round(p, 3), "r": round(r, 3), "f1": round(f, 3),
            "tp": tp, "n_pred": len(pred), "n_ref": len(ref)}


def lexical(ref: str, hyp: str) -> dict:
    out: dict = {}
    try:
        from rouge_score import rouge_scorer
        s = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        out["rougeL"] = round(s.score(ref, hyp)["rougeL"].fmeasure, 4)
    except Exception as e:
        out["rougeL_err"] = str(e)[:100]
    try:
        import sacrebleu
        out["chrf"] = round(sacrebleu.sentence_chrf(hyp, [ref]).score, 2)
    except Exception as e:
        out["chrf_err"] = str(e)[:100]
    return out


def score_chunk(cand_report: str, teacher_report: str) -> dict:
    ca, ta = actions_text(cand_report), actions_text(teacher_report)
    d = {f"act_{k}": v for k, v in lexical(ta, ca).items()}
    d["instr"] = prf(canon_set(section(cand_report, "Observed instruments:")),
                     canon_set(section(teacher_report, "Observed instruments:")))
    d["anatomy"] = prf(canon_set(section(cand_report, "Anatomy visible:")),
                       canon_set(section(teacher_report, "Anatomy visible:")))
    return d
