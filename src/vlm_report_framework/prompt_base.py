"""Shared prompt building blocks for teacher and candidate prompts.

Setting-specific context (phase line, instrument tiers, phase-guess instruction)
and the output contract live in teacher_prompt.py / candidate_prompt.py;
everything here is identical across pipelines and models.
"""
from __future__ import annotations

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


def background_block(memory: dict) -> str:
    """BACKGROUND section: compressed full history + verbatim recent chunks.

    Verbatim coverage: the immediately previous chunk always; plus the most
    recent non-Idle chunk when it differs (i.e. previous was Idle) — the
    scan-back rule. Both are labeled so the model treats them as context
    only, never current evidence.
    """
    bg = []
    if memory.get("running_summary"):
        bg.append(f"Summary of earlier chunks:\n{memory['running_summary']}")
    if memory.get("prev_report"):
        trail = memory.get("phase_trail") or []
        prev_idle = bool(trail) and trail[-1] in ("P13", "P??")
        tag = (" (Idle — transition context only)"
               if prev_idle else " (most recent action)")
        bg.append(f"Report of immediately previous chunk{tag}:\n{memory['prev_report']}")
    if (memory.get("prev_nonidle_report")
            and memory["prev_nonidle_report"] != memory.get("prev_report")):
        bg.append("Most recent NON-IDLE chunk "
                  "(last substantive action, context only):\n"
                  f"{memory['prev_nonidle_report']}")
    struct = []
    if memory.get("phase_trail"):
        struct.append(f"Phase trail: {' -> '.join(memory['phase_trail'])}")
    if memory.get("instruments_seen"):
        struct.append("Instruments seen in earlier chunks: "
                      + ", ".join(f"{k} (last: {v})"
                                  for k, v in memory["instruments_seen"].items()))
    if memory.get("flags"):
        struct.append("Known events: " + "; ".join(memory["flags"]))
    if struct:
        bg.append("\n".join(struct))
    if not bg:
        return "BACKGROUND: none — this is the first chunk of the surgery."
    return "BACKGROUND (context only, not current evidence):\n" + "\n\n".join(bg)
