"""Prompt construction for Part II candidate reports (3 settings).

Settings (same frozen chunk C(i,j) in all three):
  s1: video only                    -> R_video
  s2: video + phase P(i)            -> R_phase
  s3: video + phase P(i) + instruments I(i,j) -> R_phase+instrument

Shared blocks (labels, background, formatting) live in prompt_base.py.
Memory M(i,j) is chained per (model, setting, video) from the candidate's OWN
previous outputs: running_summary + prev_report always; phase_trail uses the
given phase for s2/s3 and the model's stated phase (best-effort parse) for
s1; instruments_seen is populated only for s3, where the observed tier is
given — for s1/s2 the model's self-reported instruments are unverifiable,
so seeding memory with them would corrupt the chain.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prompt_base import (  # noqa: E402
    background_block,
    fmt_time,
    instrument_block,
)

SYSTEM_PROMPT = """You are a factual describer of cataract surgery video.
Describe ONLY what is visually supported in the CURRENT video chunk:
actions, instruments, anatomy, and important events.
Do NOT invent instruments, actions, or events that are not visible.
BACKGROUND about earlier chunks is context only: use it to judge whether an
action is starting vs. continuing, never describe it as if visible now."""

SAFEGUARD = """Candidate instruments come from an auxiliary perception model
that is imperfect — never assume a listed instrument is present; verify
visually before reporting it."""

PHASE_LIST = ("P01-Incision, P02-Viscoelastic, P03-Capsulorhexis, "
              "P04-Hydrodissection, P05-Phacoemulsification, "
              "P06-Irrigation-Aspiration, P07-Capsule Polishing, "
              "P08-Lens Implantation, P09-Lens Positioning, "
              "P10-Viscoelastic Suction, P11-Anterior Chamber Flushing, "
              "P12-Tonifying-Antibiotics, P13-Idle.")

REPORT_CONTRACT = """Respond with ONLY a JSON object, no other text. It must have
exactly these keys:
- "report": a single PLAIN-TEXT string (never a nested JSON object),
  using exactly these labeled lines in this order:
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
  "capsule_flap_torn@P03_o01". Empty list if none.

Example response (for a different chunk — follow the shape, not the content):
{
  "report": "Phase: P01 (Incision), 00:57.00-01:03.10\\nObserved instruments: tissue forceps [Forceps]\\nCandidate but not confirmed: None\\nAnatomy visible: cornea, pupil, iris\\nActions/events: The primary knife creates a clear corneal incision while tissue forceps stabilize the globe. No intraocular entry yet.",
  "memory_update": "Clear corneal incision made; globe stabilized with forceps, no intraocular entry yet.",
  "flags_add": []
}"""

_PHASE_RE = re.compile(r"Phase:\s*(P\d{2})", re.IGNORECASE)


def stated_phase(report: str) -> str:
    """Best-effort parse of the model's own Phase line (s1 memory trail)."""
    m = _PHASE_RE.search(report or "")
    return m.group(1).upper() if m else "P??"


def build_candidate_prompt(
    chunk: dict,
    setting: str,
    observed: list[dict],
    weak: list[dict],
    memory: dict,
    total_frames: int,
) -> str:
    assert setting in ("s1", "s2", "s3"), setting
    t0, t1 = fmt_time(chunk["t_start"]), fmt_time(chunk["t_end"])
    parts = [f"CURRENT CHUNK: {t0}-{t1} of a cataract surgery."]
    if setting == "s1":
        parts.append(
            "No phase or instrument information is provided. Identify the "
            "surgical phase yourself from the video and state it in the "
            "Phase line as 'Phase: PXX (Name), <time window>' using one of: "
            + PHASE_LIST
        )
    else:
        parts.append(f"Known surgical phase: {chunk['phase']} ({chunk['phase_name']}).")
        if memory.get("same_phase_continuation"):
            parts.append(
                "NOTE: this chunk continues the SAME phase as the previous chunk. "
                "Emphasize what CHANGED vs. the previous chunk (progress, instrument "
                "state, anatomy state) rather than re-describing the setup."
            )
    parts.append(background_block(memory))
    if setting == "s3":
        parts.append(SAFEGUARD)
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
    parts.append(REPORT_CONTRACT)
    return "\n\n".join(parts)
