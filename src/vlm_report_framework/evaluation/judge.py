"""Eval J: video-grounded per-chunk judge (v3, see docs at repo root:
judgement_overview.md).

ONE multimodal call per report: the FROZEN CLIP plus the report's clinical
content (code-generated phase header, timestamps, and [class-id] tags are
stripped before sending) -> {"accuracy", "coverage", "issues", "unassessable"}
on anchored 0-3 scales. The judge never sees the teacher report, the model
identity/setting, detector outputs, or prior scores; clips are referred to by
neutral ids. Chronology/timeline checks stay in deterministic code.

Gateway: ag/gemini-3.8-flash via http://localhost:20128/v1 (user-managed).
"""
from __future__ import annotations

import base64
import hashlib
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from run_state import REPO, parse_json_loose  # noqa: E402

BASE_URL = "http://localhost:20128/v1"
MODEL = "ag/gemini-3.8-flash"
OUT_EVAL = REPO / "output-evaluation"
CLIPS = REPO / "output-teacher" / "clips"

# Bump whenever the rubric/system prompt changes so old scores can be told
# apart from new ones without re-judging ambiguity.
RUBRIC_VERSION = "visual-1"

JUDGE_SYSTEM = """You are an expert cataract surgeon reviewing ONE short intraoperative video clip together with a candidate clinical report written about that clip. Judge how faithfully the report describes the clip. You see the clip directly; nothing else is provided. Grade content, not style.

SCALES (integers, independent)
Accuracy - is what the report states true of the clip?
3 = Main description correct; no material factual error (minor imprecision is fine).
2 = Main description correct, but with a minor error or overstatement.
1 = Contains a material error (below), though some content is correct.
0 = Predominantly wrong, irrelevant, or no usable description.
Coverage - does the report capture what matters in the clip?
3 = Captures the main activity and important visible changes.
2 = Captures the main activity but misses some useful detail.
1 = Mostly vague, or misses the main activity; or omits a clearly visible consequential event.
0 = No meaningful account of the visible activity.
If the clip shows no meaningful activity (idle), a report saying so is accurate and complete.

MATERIAL ERRORS (accuracy <= 1 when one dominates what the report describes)
- describes a different surgical step than the one visible (for example incision creation during lens implantation);
- invents an implant, complication, or event that is not visible in the clip;
- confuses clearly distinguishable instruments in a way that changes the described action;
- misstates a clearly visible consequential adverse event. If such an event is visible and the report omits it, coverage <= 1 even when nothing stated is false.

DO NOT PENALIZE
- equivalent terminology ("phaco probe" = "phacoemulsification handpiece"; "keratome" = "primary knife");
- general wording when fine instrument identity is not discernible;
- non-exhaustive tool/anatomy lists (omitting incidental detail lowers coverage at most one level);
- legitimate continuation of an ongoing step (repeated aspiration or emulsification is correct, not stagnation);
- cautious wording ("possible", "candidate", "uncertain"): an unverifiable hedge is not a false claim.

ABSTAIN
A detail you cannot resolve from the clip goes in "unassessable": it is neither credited nor penalized. Never guess.

EXAMPLES
- Clip: phaco probe trenches the lens nucleus. Report: "The phaco handpiece sculpts a groove in the nucleus." -> accuracy 3, coverage 3.
- Clip: phaco probe trenches the lens nucleus. Report: "A keratome creates the main corneal incision." -> accuracy 1, coverage 1, issue severity material (wrong surgical step).
- Clip: steady capsulorhexis; report repeats the tearing action from the previous segment. Continuation is correct -> accuracy 3, no issue.

OUTPUT - respond with ONLY this JSON object, no other text:
{"accuracy": <0-3>, "coverage": <0-3>, "issues": [{"quote": "<verbatim report snippet>", "severity": "minor"|"material", "reason": "<=15 words"}], "unassessable": ["<unresolvable detail>", ...]}
issues: 0-3 items, only real problems in the report ([] if none). unassessable: [] if none."""

JUDGE_USER = """CLIP: {label}

CANDIDATE REPORT (clinical content only; formatting, timestamps and phase metadata stripped):
{content}

Return only the JSON object."""


def _client():
    from openai import OpenAI
    return OpenAI(base_url=BASE_URL, api_key="not-needed")


def _call(client, model: str, user: str, clip: Path,
          max_tokens: int = 400) -> tuple[str, float]:
    """One multimodal call: text + frozen clip as a base64 data URL."""
    b64 = base64.b64encode(clip.read_bytes()).decode()
    content = [{"type": "text", "text": user},
               {"type": "video_url",
                "video_url": {"url": f"data:video/mp4;base64,{b64}"}}]
    t0 = time.time()
    r = client.chat.completions.create(
        model=model, temperature=0.0, max_tokens=max_tokens,
        messages=[{"role": "system", "content": JUDGE_SYSTEM},
                  {"role": "user", "content": content}], timeout=600)
    return r.choices[0].message.content or "", time.time() - t0


# --- content normalization + blinding ---------------------------------------

_TS_RE = re.compile(r"\d{1,2}:\d{2}(?:\.\d+)?")
_PHASE_LINE_RE = re.compile(r"^[ \t]*Phase\s*:.*$", re.I | re.M)
_BRACKET_RE = re.compile(r"[ \t]*\[[^\[\]\n]{1,60}\][ \t]*")


def clinical_content(report: str) -> str:
    """Strip code-generated phase header, timestamps, and [class-id] tags.

    What remains is the natural-language clinical content that the judge
    should grade; phase recognition for s1 is measured separately in code.
    """
    t = _PHASE_LINE_RE.sub("", report or "")
    t = _TS_RE.sub("", t)
    t = _BRACKET_RE.sub(" ", t)
    t = re.sub(r"[ \t]{2,}", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def report_hash(content: str) -> str:
    return hashlib.sha256(content.encode()).hexdigest()[:12]


def clip_label(video_id: str, chunk_id: str) -> str:
    """Neutral, stable identifier shown to the judge (not phase-coded)."""
    h = hashlib.sha1(f"{video_id}__{chunk_id}".encode()).hexdigest()[:10]
    return f"clip-{h}"


def _clean_parsed(p: dict) -> dict:
    """Validate/clamp the judge JSON; raise when the grades are unusable."""
    def grade(x):
        try:
            v = int(x)
        except Exception:
            return None
        return v if 0 <= v <= 3 else None

    a, c = grade(p.get("accuracy")), grade(p.get("coverage"))
    if a is None or c is None:
        raise ValueError(
            f"bad grades accuracy={p.get('accuracy')!r} coverage={p.get('coverage')!r}")
    issues = []
    for it in (p.get("issues") or [])[:4]:
        if not isinstance(it, dict):
            continue
        q = str(it.get("quote", "")).strip()
        if not q:
            continue
        sev = str(it.get("severity", "minor")).strip().lower()
        sev = "material" if sev.startswith(("mat", "maj", "sev", "high")) else "minor"
        issues.append({"quote": q[:200], "severity": sev,
                       "reason": str(it.get("reason", "")).strip()[:160]})
    unass = [str(u).strip()[:160] for u in (p.get("unassessable") or [])
             if str(u).strip()][:6]
    return {"accuracy": a, "coverage": c, "issues": issues, "unassessable": unass}


def judge_one(video_id: str, chunk_id: str, report: str, tag: str = "",
              setting: str = "", model: str = "", attempts: int = 2) -> dict:
    """Judge one report against its frozen clip. Never raises for LLM/parse
    failures: the returned row carries accuracy/coverage None + "error"."""
    content = clinical_content(report)
    clip = CLIPS / f"{video_id}__{chunk_id}.mp4"
    row = {"video_id": video_id, "tag": tag, "setting": setting,
           "chunk_id": chunk_id, "clip_label": clip_label(video_id, chunk_id),
           "report_hash": report_hash(content), "rubric": RUBRIC_VERSION,
           "judge_model": model or MODEL, "accuracy": None, "coverage": None,
           "issues": [], "unassessable": [], "latency_s": None, "error": None}
    if not content:
        row["error"] = "empty clinical content"
        return row
    assert clip.exists(), f"missing frozen clip {clip}"
    last = "unknown error"
    for _ in range(max(1, attempts)):
        try:
            raw, dt = _call(_client(), row["judge_model"],
                            JUDGE_USER.format(label=row["clip_label"],
                                              content=content), clip)
            row.update(_clean_parsed(parse_json_loose(raw)))
            row["latency_s"] = round(dt, 1)
            row["error"] = None
            return row
        except Exception as e:  # noqa: BLE001 — recorded, one retry
            last = str(e)[:200]
    row["error"] = last
    return row


# --- persistence (resume-safe, report-hash versioned) ------------------------

def load_latest(path: Path) -> dict[str, dict]:
    """Last row wins per chunk_id (append/replace history stays readable)."""
    out: dict[str, dict] = {}
    if path.exists():
        for line in open(path):
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("chunk_id"):
                out[r["chunk_id"]] = r
    return out


def merge_rows(path: Path, rows: list[dict]) -> None:
    """Merge new judged rows by chunk_id (sorted; stable diffs)."""
    merged = load_latest(path)
    for r in rows:
        merged[r["chunk_id"]] = r
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(
        json.dumps(merged[c], ensure_ascii=False) for c in sorted(merged)) + "\n")
