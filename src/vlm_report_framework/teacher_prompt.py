"""Prompt construction for Part I teacher reports (v2 structured contract).

C(i,j) + P(i) + I(i,j) + M(i,j) -> structured fields -> rendered R_teacher(i,j)

v2: the model returns JSON FIELDS ONLY (phase_id, instrument id lists,
anatomy ids, action bullets). Timestamps, headers, and display names are
rendered deterministically by report_render.py — the model must never write
them. All instrument/anatomy ids come from closed vocabularies; anything
else is a schema error and triggers a retry.

Shared blocks (labels, background, formatting) live in prompt_base.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prompt_base import (  # noqa: E402
    ANATOMY_TERMS,
    INSTRUMENT_LABELS,
    PHASE_NAMES,
    background_block,
    candidate_block,
    fmt_time,
)

__all__ = ["INSTRUMENT_LABELS", "SYSTEM_PROMPT", "build_user_prompt",
           "ALLOWED_ANATOMY", "PHASE_IDS"]

PHASE_IDS = sorted(PHASE_NAMES)
ALLOWED_ANATOMY = sorted(ANATOMY_TERMS)

SYSTEM_PROMPT = """You are a factual describer of cataract surgery video.
Describe ONLY what is visually supported in the CURRENT video chunk.
Do NOT invent instruments, actions, anatomy, or events that are not visible.
Do NOT explain, interpret intent, or describe mechanisms (no "to ...", no
"using ultrasound/vacuum energy"). State only what the camera shows.
The candidate instrument list comes from an imperfect auxiliary detector:
every entry is an UNCONFIRMED PROPOSAL, never a fact. Report an instrument
only if you clearly see it in the current chunk; otherwise leave it out.
Use ONLY the instrument and anatomy ids given below, copied exactly.
Never write timestamps, phase names, headers, or section titles — the
report text is assembled by code from your fields.
BACKGROUND about earlier chunks is context only: use it to judge whether an
action is starting vs. continuing, never describe it as if visible now."""

REPORT_SECTIONS = """Respond with ONLY a JSON object, no other text, with exactly these keys:
- "phase_id": the phase id given above (copy it exactly).
- "instruments_visible": list of instrument ids CLEARLY visible in the
  current chunk (subset of the candidate ids below; [] if none).
- "instruments_uncertain": list of candidate ids with weak/trace visual
  support (subset of the candidate ids below; [] if none). An id cannot
  appear in both lists.
- "anatomy": list of anatomy ids clearly visible AND relevant to this
  chunk (subset of the anatomy list below; list only what is shown, never
  enumerate defaults; [] if nothing relevant).
- "actions": list of 1-5 short factual bullets (each one observable action
  or state change, max ~20 words, no timestamps, no purpose clauses).
  If you see an irregularity or adverse event actually visible in this
  chunk (e.g. capsule tear, iris trauma, bleeding, lens instability,
  instrument slippage), describe it concretely HERE as a bullet — there is
  no separate field for it. If nothing abnormal is visible, just describe
  normal progression; do not assert absence beyond what is shown.
- "memory_update": one or two sentences of compressed state for the NEXT
  chunk (what changed / what persists at chunk end, e.g. flap state,
  fragment volume, lens position). Empty string if nothing carries over.
- "flags_add": list of persistent event tags like
  "capsule_flap_torn@P03_o01". Empty list if none.

Example (for a different chunk — follow the shape, never copy content):
{"phase_id": "P01", "instruments_visible": ["Forceps"],
"instruments_uncertain": [], "anatomy": ["cornea", "pupil"],
"actions": ["Primary knife creates a clear corneal incision.",
"Forceps stabilize the globe during entry."],
"memory_update": "Clear corneal incision made.", "flags_add": []}"""


def build_user_prompt(
    chunk: dict,
    observed: list[dict],
    weak: list[dict],
    memory: dict,
    total_frames: int,
) -> str:
    dur = chunk["t_end"] - chunk["t_start"]
    parts = [
        f"CURRENT CHUNK: duration {dur:.1f}s of a cataract surgery, "
        f"phase {chunk['phase']} ({chunk['phase_name']}). "
        "No timestamps are shown — never write times; they are added by code.",
    ]
    if memory.get("same_phase_continuation"):
        parts.append(
            "NOTE: this chunk continues the SAME phase as the previous chunk. "
            "Report only what STARTED, ENDED, or CHANGED in this chunk; "
            "one bullet for steady continuation at most."
        )
    parts.append(background_block(memory))

    for r in observed:
        r["hint"] = "strong"
    for r in weak:
        r["hint"] = "weak"
    rows = list(observed) + list(weak)
    allowed = sorted({r["class_name"] for r in rows})
    parts.append(
        "Unconfirmed instrument proposals from the auxiliary detector "
        f"in this chunk ({total_frames} frames analyzed). "
        "Every entry is UNCONFIRMED — verify visually before listing it. "
        "Your instruments_visible / instruments_uncertain must be subsets "
        "of these ids, copied exactly:"
    )
    if rows:
        parts.append(candidate_block(rows))
    else:
        parts.append("No proposals in this chunk: both lists must be [].")
    parts.append(
        "Allowed instrument ids in this chunk: "
        + (", ".join(allowed) if allowed else "(none)")
        + ". Allowed anatomy ids: " + ", ".join(ALLOWED_ANATOMY) + "."
    )
    parts.append(REPORT_SECTIONS)
    return "\n\n".join(parts)


def allowed_instruments(observed: list[dict], weak: list[dict]) -> set[str]:
    """Closed instrument vocabulary for this chunk (union of proposals)."""
    return {r["class_name"] for r in list(observed) + list(weak)}
