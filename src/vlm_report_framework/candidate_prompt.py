"""Prompt construction for Part II candidate reports, v2.1 tagged-text contract.

Settings (same frozen chunk C(i,j) in both):
  s1: video only                    -> R_video
  s3: video + phase P(i) + instruments I(i,j) -> R_phase+instrument

v2.1: students write SIMPLE TAGGED PLAIN TEXT (no JSON, no timestamps).
Python (report_render.parse_student_tags) canonicalizes tool/anatomy names
and renders the canonical 5-section wire text, so eval headers stay
byte-identical. Time-blind: prompts carry duration only, never MM:SS.
Memory is the slim rolling STATE history (see memory.py).
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prompt_base import (  # noqa: E402
    ANATOMY_TERMS,
    INSTRUMENT_LABELS,
    PHASE_NAMES,
    background_block,
    candidate_block,
)

SYSTEM_PROMPT = """You are a factual describer of cataract surgery video.
Describe ONLY what is visually supported in the CURRENT video chunk.
Do NOT invent instruments, actions, anatomy, or events that are not visible.
Do NOT explain intent or mechanisms (no "to ...", no "using energy").
Write plain tagged lines (no JSON, no markdown headers).
Never write timestamps or time windows — they are added by code.
BACKGROUND about earlier chunks is context only: use it to judge whether an
action is starting vs. continuing, never describe it as if visible now."""

SAFEGUARD = """The instrument proposals below come from an imperfect auxiliary
detector: every entry is an UNCONFIRMED PROPOSAL, never a fact. List a tool
under TOOLS only if you clearly see it in the current chunk."""

PHASE_LIST = ("P01-Incision, P02-Viscoelastic, P03-Capsulorhexis, "
              "P04-Hydrodissection, P05-Phacoemulsification, "
              "P06-Irrigation-Aspiration, P07-Capsule Polishing, "
              "P08-Lens Implantation, P09-Lens Positioning, "
              "P10-Viscoelastic Suction, P11-Anterior Chamber Flushing, "
              "P12-Tonifying-Antibiotics, P13-Idle.")

ALLOWED_ANATOMY = sorted(ANATOMY_TERMS)
ALL_INSTRUMENTS = sorted(INSTRUMENT_LABELS)

STUDENT_CONTRACT_S1 = """Write your answer as plain tagged lines, exactly these tags:
PHASE: <P01..P13 id only, e.g. P05> (state the phase you see)
TOOLS: <comma-separated tools clearly visible, or none>
UNCERTAIN_TOOLS: <comma-separated tools with weak/trace support, or none>
ANATOMY: <comma-separated structures clearly visible and relevant, or none>
EVENTS:
- <1-5 short factual bullets, one observable action each, no timestamps>
- <if you see an irregularity actually visible here (capsule tear, iris
  trauma, bleeding, lens instability, slippage), describe it HERE as a bullet>
STATE: <one sentence: what changed / persists at chunk end, for next chunk>
FLAGS: <persistent event tags like capsule_flap_torn, or none>

Rules: plain instrument names are fine (code maps them); never write
timestamps, phase names, headers, or JSON; keep bullets short."""

STUDENT_CONTRACT_S3 = """Write your answer as plain tagged lines, exactly these tags
(do NOT write a PHASE line — code already knows the phase):
TOOLS: <comma-separated tools clearly visible, or none>
UNCERTAIN_TOOLS: <comma-separated tools with weak/trace support, or none>
ANATOMY: <comma-separated structures clearly visible and relevant, or none>
EVENTS:
- <1-5 short factual bullets, one observable action each, no timestamps>
- <if you see an irregularity actually visible here (capsule tear, iris
  trauma, bleeding, lens instability, slippage), describe it HERE as a bullet>
STATE: <one sentence: what changed / persists at chunk end, for next chunk>
FLAGS: <persistent event tags like capsule_flap_torn, or none>

Rules: plain instrument names are fine (code maps them); never write
timestamps, phase names, headers, or JSON; keep bullets short."""

_PHASE_RE = re.compile(r"^PHASE\s*:\s*(P\d{2})", re.I | re.M)


def stated_phase(report: str) -> str:
    """Best-effort parse of the student's own phase (s1 memory trail).

    v2.1: reads the PHASE: tag; falls back to legacy v2 "phase_id" JSON
    and v1 "Phase: PXX" lines for old rows.
    """
    m = _PHASE_RE.search(report or "")
    if m:
        return m.group(1).upper()
    m2 = re.search(r'"phase_id"\s*:\s*"(P\d{2})"', report or "", re.I)
    if m2:
        return m2.group(1).upper()
    m3 = re.search(r"Phase:\s*(P\d{2})", report or "", re.I)
    return m3.group(1).upper() if m3 else "P??"


def build_candidate_prompt(
    chunk: dict,
    setting: str,
    observed: list[dict],
    weak: list[dict],
    memory: dict,
    total_frames: int,
) -> str:
    assert setting in ("s1", "s3"), setting
    dur = chunk["t_end"] - chunk["t_start"]
    parts = [f"CURRENT CHUNK: duration {dur:.1f}s of a cataract surgery."]
    if setting == "s1":
        parts.append(
            "No phase or instrument information is provided. Identify the "
            "surgical phase yourself from the video and state it as "
            "PHASE: <id> using one of: " + PHASE_LIST
        )
    else:
        parts.append(f"Known surgical phase: {chunk['phase']} ({chunk['phase_name']}). "
                     "Do NOT write a PHASE line (code already knows it).")
        if memory.get("same_phase_continuation"):
            parts.append(
                "NOTE: this chunk continues the SAME phase as the previous chunk. "
                "Report only what STARTED, ENDED, or CHANGED in this chunk; "
                "one bullet for steady continuation at most."
            )
    parts.append(background_block(memory))
    if setting == "s3":
        parts.append(SAFEGUARD)
        for r in observed:
            r["hint"] = "strong"
        for r in weak:
            r["hint"] = "weak"
        rows = list(observed) + list(weak)
        parts.append(
            "Unconfirmed instrument proposals from the auxiliary detector "
            f"in this chunk ({total_frames} frames analyzed). Verify visually:"
        )
        if rows:
            parts.append(candidate_block(rows))
        else:
            parts.append("No proposals in this chunk: answer TOOLS: none.")
    else:
        parts.append(
            "No instrument information is provided. Name only tools you "
            "clearly see (official list: " + ", ".join(ALL_INSTRUMENTS) + ")."
        )
    parts.append(STUDENT_CONTRACT_S1 if setting == "s1" else STUDENT_CONTRACT_S3)
    return "\n\n".join(parts)


def allowed_instruments(setting: str, observed: list[dict],
                        weak: list[dict]) -> set[str]:
    """Reference vocabulary for this (setting, chunk) — informational only.

    v2.1 students are NOT validated against this (no retries); the
    normalizer in report_render maps natural names in code.
    """
    if setting == "s3":
        return {r["class_name"] for r in list(observed) + list(weak)}
    return set(INSTRUMENT_LABELS)
