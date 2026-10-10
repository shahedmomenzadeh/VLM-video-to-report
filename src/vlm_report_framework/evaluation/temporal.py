"""Eval 4: temporal / memory consistency (text LLM, saved outputs only)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from evaluation.evidence import BASE_URL, MODEL, OUT_EVAL, _call, _client  # noqa: E402
from run_state import REPO, parse_json_loose  # noqa: E402

CAND = REPO / "output-candidates"

SYSTEM = ("You compare two consecutive surgical-chunk reports. "
          "Flag only clear errors, not stylistic overlap.")
CONTRACT = """Respond with ONLY a JSON object, no other text. Keys exactly:
- "contradiction": true if curr contradicts prev on a persistent state (or false)
- "contamination": true if curr restates prev's past event as current observation
- "stale_instrument": true if curr carries over a tool with no current support
- "redundant": true if curr adds nothing vs prev (near-duplicate)
- "progression": 1 (restart/error), 2 (static/legit continuation), 3 (clear progress)
- "why": "≤25 words"."""


def _rows(tag: str, setting: str, video_id: str) -> list[dict]:
    mp = CAND / tag / setting / f"{video_id}_reports.jsonl"
    if not mp.exists():
        return []
    rows = [json.loads(l) for l in open(mp)]
    rows.sort(key=lambda r: (r.get("t_start", 0), r.get("t_end", 0)))
    return rows


def judge_transition(prev: dict, curr: dict) -> dict:
    txt = (f"PREV [{prev.get('chunk_id')} | {prev.get('phase')}]:\n"
           f"{(prev.get('report') or '')[:1500]}\n\n"
           f"CURR [{curr.get('chunk_id')} | {curr.get('phase')}]:\n"
           f"{(curr.get('report') or '')[:1500]}\n\n{CONTRACT}")
    try:
        raw, _ = _call(_client(), MODEL, SYSTEM, txt, None, 256)
        p = parse_json_loose(raw)
        return {
            "contradiction": bool(p.get("contradiction", False)),
            "contamination": bool(p.get("contamination", False)),
            "stale_instrument": bool(p.get("stale_instrument", False)),
            "redundant": bool(p.get("redundant", False)),
            "progression": int(p.get("progression", 2)),
            "why": str(p.get("why", ""))[:150],
        }
    except Exception as e:
        return {"contradiction": False, "contamination": False,
                "stale_instrument": False, "redundant": False,
                "progression": 2, "why": f"parse-fail:{e}"[:150]}


CHAIN_SYSTEM = ("You audit a sequence of consecutive surgical-chunk reports. "
                "Flag only clear errors, not stylistic overlap.")
CHAIN_CONTRACT = """Respond with ONLY a JSON object, no other text:
{"transitions": [{"prev": "<chunk_id>", "curr": "<chunk_id>",
"contradiction": bool, "contamination": bool, "stale_instrument": bool,
"redundant": bool, "progression": 1|2|3}]} — one entry per CONSECUTIVE pair
in order. contradiction = curr flips a persistent state; contamination = past
event restated as current; stale_instrument = tool carried with no support;
redundant = adds nothing; progression 1 = restart/error, 2 = static/legit
continuation, 3 = clear progress."""


def score_chain(tag: str, setting: str, video_id: str, run_id: str) -> list[dict]:
    """Tier 1: ONE call per (video, tag, setting) chain (~30 pairs per call)."""
    out = OUT_EVAL / run_id / "transitions_t1" / f"{video_id}.{tag}.{setting}.json"
    if out.exists():
        try:
            return json.load(open(out))["transitions"]
        except Exception:
            pass
    rows = _rows(tag, setting, video_id)
    if len(rows) < 2:
        return []
    parts = []
    for prev, curr in zip(rows, rows[1:]):
        parts.append(f"--- {prev.get('chunk_id')} -> {curr.get('chunk_id')} ---\n"
                     f"PREV [{prev.get('phase')}]: {(prev.get('report') or '')[:700]}\n"
                     f"CURR [{curr.get('phase')}]: {(curr.get('report') or '')[:700]}")
    try:
        raw, _ = _call(_client(), MODEL, CHAIN_SYSTEM,
                       "\n\n".join(parts) + f"\n\n{CHAIN_CONTRACT}", None, 2048)
        p = parse_json_loose(raw)
        trans = p.get("transitions", [])
        scored = [{"video_id": video_id, "tag": tag, "setting": setting,
                   "prev": t.get("prev"), "curr": t.get("curr"),
                   "same_phase": None,
                   "contradiction": bool(t.get("contradiction", False)),
                   "contamination": bool(t.get("contamination", False)),
                   "stale_instrument": bool(t.get("stale_instrument", False)),
                   "redundant": bool(t.get("redundant", False)),
                   "progression": int(t.get("progression", 2)),
                   "why": ""} for t in trans if isinstance(t, dict)]
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"video_id": video_id, "tag": tag,
                                   "setting": setting, "transitions": scored},
                                  indent=1))
        return scored
    except Exception as e:
        return [{"video_id": video_id, "tag": tag, "setting": setting,
                 "prev": None, "curr": None, "error": str(e)[:200]}]
    if out.exists():
        return [json.loads(l) for l in open(out)]
    rows = _rows(tag, setting, video_id)
    scored = []
    for prev, curr in zip(rows, rows[1:]):
        j = judge_transition(prev, curr)
        scored.append({"video_id": video_id, "tag": tag, "setting": setting,
                       "prev": prev.get("chunk_id"), "curr": curr.get("chunk_id"),
                       "same_phase": prev.get("phase") == curr.get("phase"),
                       **j})
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(map(json.dumps, scored)) + ("\n" if scored else ""))
    return scored
