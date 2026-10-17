"""Shared prompt building blocks for teacher and candidate prompts.

Setting-specific context (phase line, instrument tiers, phase-guess instruction)
and the output contract live in teacher_prompt.py / candidate_prompt.py;
everything here is identical across pipelines and models.
"""
from __future__ import annotations

import json

# Raw YOLO class name -> human surgical term shown to VLMs.
INSTRUMENT_LABELS: dict[str, str] = {
    "Cannula": "cannula",
    "Cap-Cystotome": "cystotome (capsulorhexis needle)",
    "Cap-Forceps": "capsulorhexis forceps",
    "Forceps": "tissue forceps",
    "I-A-Handpiece": "irrigation-aspiration handpiece",
    "Lens-Injector": "lens injector",
    "Phaco-Handpiece": "phacoemulsification handpiece",
    "Primary-Knife": "primary (keratome) knife",
    "Second-Instrument": "second instrument (chopper/manipulator)",
    "Secondary-Knife": "secondary (paracentesis) knife",
}


def label(class_name: str) -> str:
    term = INSTRUMENT_LABELS.get(class_name, class_name)
    return f"{term} [{class_name}]" if term != class_name else term


def instrument_block(rows: list[dict]) -> str:
    """Format chunk-instrument rows as 'name (n/total frames, med conf x.xx)' lines."""
    lines = []
    for r in rows:
        lines.append(
            f"- {label(r['class_name'])} "
            f"({r['n_frames']} frames, median conf {r['med_conf']:.2f})"
        )
    return "\n".join(lines)


def fmt_time(t: float) -> str:
    m, s = divmod(t, 60)
    return f"{int(m):02d}:{s:05.2f}"


# Closed anatomy vocabulary (v2 contracts). Models must use these ids exactly;
# renderer maps them to display terms. Keep minimal: only structures the
# cataract microscope view can actually show.
ANATOMY_TERMS: dict[str, str] = {
    "cornea": "cornea",
    "pupil": "pupil",
    "iris": "iris",
    "anterior_chamber": "anterior chamber",
    "lens_capsule": "lens capsule",
    "lens_nucleus": "lens nucleus",
    "conjunctiva_eyelid": "conjunctiva/eyelids",
    "incision_site": "incision site",
}

# Phase id -> name (13-phase cataract ontology + P?? fallback for s1).
PHASE_NAMES: dict[str, str] = {
    "P01": "Incision", "P02": "Viscoelastic", "P03": "Capsulorhexis",
    "P04": "Hydrodissection", "P05": "Phacoemulsification",
    "P06": "Irrigation-Aspiration", "P07": "Capsule Polishing",
    "P08": "Lens Implantation", "P09": "Lens Positioning",
    "P10": "Viscoelastic Suction", "P11": "Anterior Chamber Flushing",
    "P12": "Tonifying-Antibiotics", "P13": "Idle",
}


def candidate_block(rows: list[dict]) -> str:
    """Single merged unconfirmed-candidate list with per-item evidence hints.

    v2 replaces the two-group "Observed (strong evidence)" / "Candidate but
    not confirmed" display: nothing is ever called confirmed. Each row shows
    a strength hint the model must weigh visually, not trust.
    Rows carry a 'hint' key ('strong'/'weak'); fall back to tier-agnostic
    wording when absent.
    """
    lines = []
    for r in rows:
        hint = r.get("hint", "")
        tag = f" [{hint} hint" if hint else " [hint"
        lines.append(
            f"- {label(r['class_name'])} "
            f"({r['n_frames']} frames, median conf {r['med_conf']:.2f}{tag}: "
            f"unconfirmed proposal, verify visually)"
        )
    return "\n".join(lines)


def strip_timestamps(text: str) -> str:
    """Remove MM:SS(.ss) timestamp shapes so time never leaks via memory."""
    import re as _re
    return _re.sub(r"\d{1,2}:\d{2}(?:\.\d+)?", "[time]", text or "")


def background_block(memory: dict) -> str:
    """BACKGROUND section: slim rolling STATE history (v2.1).

    Time-blind and verbatim-free: the model sees at most the last
    RECENT_CAP one-sentence STATE updates + the last non-idle state +
    phase trail + active flags. No full reports, no timestamps, no
    instrument lists to copy from — this is the echo-hallucination fix.

    Legacy snapshots (prev_report / running_summary) are ignored: only
    the slim keys are rendered. Callers migrate via memory.migrate_memory.
    """
    # Defensive: tolerate legacy snapshots without migration.
    recent = memory.get("recent_states") or []
    trail = memory.get("phase_trail") or []
    flags = memory.get("flags") or []
    last_ni = memory.get("last_nonidle_state") or {}
    # Legacy fallback: if no slim history but a legacy summary exists,
    # carry a single truncated line (timestamp-stripped) so very old
    # resume chains are not totally blind. Verbatim reports are never used.
    lines: list[str] = []
    if trail:
        lines.append(f"Phase trail: {' -> '.join(trail)}")
    if recent:
        lines.append("Recent progression (context only, not current evidence):")
        n = len(recent)
        for i, rs in enumerate(recent[-4:]):
            try:
                ph = str(rs.get("phase", "?"))
                st = strip_timestamps(str(rs.get("state", "")))[:300]
                # Position, not chunk-id — chunk-id shapes are an echo
                # vector for small RL-tuned models (degenerate loops).
                pos = ("most recent" if i == len(recent[-4:]) - 1
                       else f"{n - len(recent[-4:]) + i + 1} chunks ago")
                lines.append(f"  * {ph} ({pos}): {st}")
            except Exception:
                continue
    elif memory.get("running_summary"):
        lines.append("Earlier summary: "
                     + strip_timestamps(str(memory["running_summary"]))[-400:])
    if last_ni and isinstance(last_ni, dict) and last_ni.get("state"):
        # Only repeat when it is not already the most recent entry.
        if not recent or last_ni.get("chunk_id") != recent[-1].get("chunk_id"):
            lines.append("Last active (non-idle) state: "
                         f"{last_ni.get('phase')}: "
                         + strip_timestamps(str(last_ni.get('state')))[:300])
    if flags:
        safe = [f if isinstance(f, str) else json.dumps(f) for f in flags]
        lines.append("Known event flags: " + "; ".join(safe))
    if not lines:
        return "BACKGROUND: none — this is the first chunk of the surgery."
    return ("BACKGROUND (earlier chunks — context only, "
            "never current visual evidence):\n" + "\n".join(lines))
