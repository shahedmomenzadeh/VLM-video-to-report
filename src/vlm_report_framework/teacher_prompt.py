"""Prompt construction for Part I teacher reports.

C(i,j) + P(i) + I(i,j) + M(i,j) -> R_teacher(i,j)

YOLO class names are mapped to surgical terms for the VLM; raw names are kept
in parentheses for traceability.
"""
from __future__ import annotations

# Raw YOLO class name -> human surgical term shown to the teacher VLM.
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


def build_user_prompt(
    chunk: dict,
    observed: list[dict],
    weak: list[dict],
    memory: dict,
    total_frames: int,
) -> str:
    t0 = _fmt(chunk["t_start"])
    t1 = _fmt(chunk["t_end"])
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
    bg = []
    if memory.get("running_summary"):
        bg.append(f"Summary of earlier chunks:\n{memory['running_summary']}")
    if memory.get("prev_report"):
        bg.append(f"Report of immediately previous chunk:\n{memory['prev_report']}")
    trail = memory.get("phase_trail") or []
    seen = memory.get("instruments_seen") or {}
    flags = memory.get("flags") or []
    struct = []
    if trail:
        struct.append(f"Phase trail: {' -> '.join(trail)}")
    if seen:
        struct.append("Instruments seen in earlier chunks: "
                      + ", ".join(f"{k} (last: {v})" for k, v in seen.items()))
    if flags:
        struct.append("Known events: " + "; ".join(flags))
    if struct:
        bg.append("\n".join(struct))
    if bg:
        parts.append("BACKGROUND (context only, not current evidence):\n" + "\n\n".join(bg))
    else:
        parts.append("BACKGROUND: none — this is the first chunk of the surgery.")

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


def _fmt(t: float) -> str:
    m, s = divmod(t, 60)
    return f"{int(m):02d}:{s:05.2f}"
