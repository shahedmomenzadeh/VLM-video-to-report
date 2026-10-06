"""Prompt construction for Part I teacher reports.

C(i,j) + P(i) + I(i,j) + M(i,j) -> R_teacher(i,j)

Shared blocks (labels, background, formatting) live in prompt_base.py;
this module holds the teacher's system prompt, output contract, and the
full-context assembly (phase + tiers always given).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prompt_base import (  # noqa: E402
    INSTRUMENT_LABELS,
    background_block,
    fmt_time,
    instrument_block,
)

__all__ = ["INSTRUMENT_LABELS", "SYSTEM_PROMPT", "build_user_prompt"]

SYSTEM_PROMPT = """You are a factual describer of cataract surgery video.
Describe ONLY what is visually supported in the CURRENT video chunk:
actions, instruments, anatomy, and important events.
Do NOT invent instruments, actions, or events that are not visible.
Candidate instruments come from an auxiliary perception model that is
imperfect — never assume a listed instrument is present; verify visually.
BACKGROUND about earlier chunks is context only: use it to judge whether an
action is starting vs. continuing, never describe it as if visible now."""

REPORT_SECTIONS = """Return a JSON object with exactly these keys:
- "report": the reference report, using these sections:
  Phase: <phase id and name, time window>
  Observed instruments: <instruments clearly visible, one per line;
    use "None clearly visible" if applicable>
  Candidate but not confirmed: <weak/trace detections, verified against video;
    use "None" if nothing>
  Anatomy visible: <cornea, pupil, lens/capsule, etc.>
  Actions/events: <factual visual description of what occurs>
- "memory_update": one or two sentences of compressed state for the NEXT
  chunk (what changed / what persists at chunk end, e.g. flap state,
  fragment volume, lens position). Empty string if nothing carries over.
- "flags_add": list of persistent event tags like
  "capsule_flap_torn@P03_o01". Empty list if none."""


def build_user_prompt(
    chunk: dict,
    observed: list[dict],
    weak: list[dict],
    memory: dict,
    total_frames: int,
) -> str:
    t0 = fmt_time(chunk["t_start"])
    t1 = fmt_time(chunk["t_end"])
    parts = [
        f"CURRENT CHUNK: {t0}-{t1} of a cataract surgery, "
        f"phase {chunk['phase']} ({chunk['phase_name']}).",
    ]
    if memory.get("same_phase_continuation"):
        parts.append(
            "NOTE: this chunk continues the SAME phase as the previous chunk. "
            "Emphasize what CHANGED vs. the previous chunk (progress, instrument "
            "state, anatomy state) rather than re-describing the setup."
        )
    parts.append(background_block(memory))

    parts.append(
        "Candidate instruments detected by an auxiliary perception model "
        f"in this chunk ({total_frames} frames analyzed):"
    )
    if observed:
        parts.append("Observed (strong evidence, phase-plausible):\n"
                     + instrument_block(observed))
    else:
        parts.append("Observed (strong evidence, phase-plausible): none.")
    if weak:
        parts.append("Candidate but not clearly confirmed "
                     "(weak/trace detections — verify visually, "
                     "do not assume present):\n" + instrument_block(weak))
    else:
        parts.append("Candidate but not clearly confirmed: none.")
    if not observed and not weak:
        parts.append("No instruments met the detection threshold in this chunk. "
                     "Describe anatomy and state only; do not invent tools.")
    parts.append(REPORT_SECTIONS)
    return "\n\n".join(parts)
