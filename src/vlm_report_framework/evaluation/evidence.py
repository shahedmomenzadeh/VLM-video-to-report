"""Eval 3: evidence-grounded factuality (VLM evidence + LLM claims).

Model: ag/gemini-3.8-flash via http://localhost:20128/v1 (user-managed gateway).
Evidence pass is candidate-blind (clip only). Claims pass is text-only.
"""
from __future__ import annotations

import base64
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from run_state import REPO, parse_json_loose  # noqa: E402

BASE_URL = "http://localhost:20128/v1"
MODEL = "ag/gemini-3.8-flash"
OUT_EVAL = REPO / "output-evaluation"
CLIPS = REPO / "output-teacher" / "clips"

EV_SYSTEM = ("You are an expert cataract-surgery video annotator. "
             "Record ONLY what is clearly visible. Do not guess.")
EV_CONTRACT = """Respond with ONLY a JSON object, no other text. Keys exactly:
- "visible_instruments": list of tool names clearly visible
- "anatomy": list of anatomical structures clearly visible
- "actions": 2-5 short factual sentences of what happens
- "state_changes": list of observable state changes (or [])
- "uncertain": list of things unclear/unassessable (or [])
Reason from pixels only; timeline/YOLO text (if any) is auxiliary."""

CLAIM_SPLIT_SYSTEM = "Split a surgical report into atomic factual claims. No judgment."
CLAIM_SPLIT_CONTRACT = """Respond with ONLY a JSON object: {"claims": ["claim1", ...]}.
One fact per claim (tool presence, anatomy, action each separate). Non-factual
formatting text (headers, 'None') is excluded."""

JUDGE_SYSTEM = ("You judge single claims against a visual evidence record. "
                "No outside knowledge.")
JUDGE_CONTRACT = """Respond with ONLY a JSON object: {"verdict": <one of
"Supported" | "Contradicted" | "NotEstablished" | "NonFactual">, "why": "≤20 words"}.
"NotEstablished" = evidence record neither confirms nor refutes; never guess."""

# --- Tier 1: one call per report (counts + worst offenses, ~10x cheaper) ---
T1_SYSTEM = ("You audit a surgical report against a visual evidence record. "
             "Count carefully; cite only real report phrases.")
T1_CONTRACT = """Respond with ONLY a JSON object, no other text. Keys exactly:
- "n_assessable": number of atomic factual claims (exclude headers/"None" text)
- "supported": of those, number clearly backed by the evidence record
- "contradicted": of those, number clearly refuted by the evidence record
- "worst": up to 3 of the most consequential contradicted/unsupported claims as
  [{"claim": "<verbatim phrase>", "verdict": "Contradicted"|"NotEstablished",
    "why": "≤20 words"}] (or [] if none)."""


def _client():
    from openai import OpenAI
    return OpenAI(base_url=BASE_URL, api_key="not-needed")


def _call(client, model, system, user_text, clip: Path | None = None,
          max_tokens: int = 512) -> tuple[str, float]:
    content: list = [{"type": "text", "text": user_text}]
    if clip is not None:
        b64 = base64.b64encode(clip.read_bytes()).decode()
        content.append({"type": "video_url", "video_url": {
            "url": f"data:video/mp4;base64,{b64}"}})
    t0 = time.time()
    r = client.chat.completions.create(
        model=model, temperature=0.0, max_tokens=max_tokens,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": content}], timeout=600)
    return r.choices[0].message.content or "", time.time() - t0


_EV_FALLBACKS = ("full25", "full5", "pilot")


def evidence_for(video_id: str, chunk_id: str, run_id: str) -> dict:
    """Candidate-blind visual annotation; cached per chunk, shared across runs.

    Primary cache is OUT_EVAL/<run_id>/evidence (namespaced so prompt/model
    changes don't collide); falls back to older run-ids' caches read-only
    (evidence is chunk/model-independent) and copies the hit forward.
    """
    out = OUT_EVAL / run_id / "evidence" / f"{video_id}__{chunk_id}.json"
    if out.exists():
        return json.load(open(out))
    for fb in _EV_FALLBACKS:
        cand = OUT_EVAL / fb / "evidence" / f"{video_id}__{chunk_id}.json"
        if fb != run_id and cand.exists():
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(cand.read_text())
            return json.load(open(out))
    clip = CLIPS / f"{video_id}__{chunk_id}.mp4"
    assert clip.exists(), f"missing frozen clip {clip}"
    raw, dt = _call(_client(), MODEL, EV_SYSTEM,
                    f"Annotate this cataract-surgery chunk ({chunk_id}).\n\n{EV_CONTRACT}",
                    clip, 512)
    rec = {"video_id": video_id, "chunk_id": chunk_id,
           "model": MODEL, "latency_s": round(dt, 1)}
    try:
        rec.update(parse_json_loose(raw))
    except Exception as e:
        rec["error"] = str(e)[:200]
        rec["raw"] = raw[:1000]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, indent=1))
    return rec


def split_claims(report: str) -> list[str]:
    raw, _ = _call(_client(), MODEL, CLAIM_SPLIT_SYSTEM,
                   f"Report:\n{report[:4000]}\n\n{CLAIM_SPLIT_CONTRACT}", None, 512)
    try:
        cs = parse_json_loose(raw).get("claims", [])
        return [c for c in cs if isinstance(c, str) and c.strip()][:30]
    except Exception:
        return []


def judge_claim(claim: str, evidence: dict) -> dict:
    ev = json.dumps({k: evidence.get(k) for k in
                     ("visible_instruments", "anatomy", "actions",
                      "state_changes", "uncertain")})[:3000]
    raw, _ = _call(_client(), MODEL, JUDGE_SYSTEM,
                   f"EVIDENCE:\n{ev}\n\nCLAIM: {claim}\n\n{JUDGE_CONTRACT}",
                   None, 256)
    try:
        p = parse_json_loose(raw)
        v = p.get("verdict")
        if v not in ("Supported", "Contradicted", "NotEstablished", "NonFactual"):
            return {"verdict": "NotEstablished", "why": f"bad-verdict:{v}"}
        return {"verdict": v, "why": str(p.get("why", ""))[:120]}
    except Exception as e:
        return {"verdict": "NotEstablished", "why": f"parse-fail:{e}"[:120]}


def score_report(report: str, evidence: dict) -> dict:
    """Tier 2 (atomic): split + per-claim verdicts. Kept for calibration subset."""
    claims = split_claims(report)
    rows = [{"claim": c, **judge_claim(c, evidence)} for c in claims]
    n = len([r for r in rows if r["verdict"] != "NonFactual"]) or 1
    s = sum(1 for r in rows if r["verdict"] == "Supported")
    c = sum(1 for r in rows if r["verdict"] == "Contradicted")
    u = sum(1 for r in rows if r["verdict"] == "NotEstablished")
    return {"n_claims": len(rows), "support_rate": round(s / n, 3),
            "contradiction_rate": round(c / n, 3),
            "unassessable_rate": round(u / max(1, len(rows)), 3),
            "rows": rows}


def score_report_tier1(report: str, evidence: dict) -> dict:
    """Tier 1: ONE text call per report (counts + worst offenses)."""
    import json as _json
    ev = _json.dumps({k: evidence.get(k) for k in
                      ("visible_instruments", "anatomy", "actions",
                       "state_changes", "uncertain")})[:3000]
    raw, _ = _call(_client(), MODEL, T1_SYSTEM,
                   f"EVIDENCE:\n{ev}\n\nREPORT:\n{report[:4000]}\n\n{T1_CONTRACT}",
                   None, 512)
    try:
        p = parse_json_loose(raw)
        n = int(p.get("n_assessable", 0)) or 1
        s = int(p.get("supported", 0))
        c = int(p.get("contradicted", 0))
        worst = p.get("worst", []) or []
        return {"n_claims": n,
                "support_rate": round(min(1.0, s / n), 3),
                "contradiction_rate": round(min(1.0, c / n), 3),
                "unassessable_rate": round(max(0.0, 1 - (s + c) / n), 3),
                "rows": [w for w in worst if isinstance(w, dict)][:3]}
    except Exception as e:
        return {"n_claims": 0, "support_rate": None,
                "contradiction_rate": None, "unassessable_rate": None,
                "rows": [], "error": str(e)[:200]}
