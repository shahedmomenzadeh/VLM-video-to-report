"""Shared causal memory M(i,j): creation, update rule, schema.

One memory dict per (video [, model, setting]) chain, updated once per chunk
in temporal order. The VLM contributes only `memory_update` + `flags_add`
inside its JSON reply; the merge itself is deterministic code.

Schema:
  running_summary:     compressed state, tail-capped at ~1600 chars
  prev_report:         verbatim report of the immediately previous chunk
  prev_nonidle_report: verbatim report of the most recent NON-IDLE chunk
                       (scan-back: Idle chunks carry the field forward)
  phase_trail:         last 10 phase ids (Idle included; s1 holds stated phases)
  instruments_seen:    class -> last chunk id (selective; see callers)
  flags:               append-only deduped event tags
  same_phase_continuation: transient hint flag, recomputed before each prompt
"""
from __future__ import annotations

SUMMARY_CAP = 1600
TRAIL_CAP = 10
# Anti-cascade: verbatim memory is capped so one degenerate (looping)
# response can't bloat every downstream prompt. Above the longest
# legit teacher report (~1000 chars); only truncates pathological dumps.
VERBATIM_CAP = 1500


def fresh_memory() -> dict:
    return {"running_summary": "", "prev_report": "", "prev_nonidle_report": "",
            "phase_trail": [], "instruments_seen": {}, "flags": [],
            "same_phase_continuation": False}


def update_memory(
    mem: dict,
    chunk_id: str,
    phase_id: str,
    report: str,
    memory_update: str,
    flags_add: list,
    seen_classes: list[str] | None = None,
    idle_phases: tuple[str, ...] = ("P13",),
) -> dict:
    """Advance memory by one chunk.

    phase_id: given phase (teacher / s2 / s3) or the model's stated phase (s1).
    seen_classes: instrument classes to record as seen here, or None to leave
      the record untouched. Callers pass observed-tier classes only when the
      chunk is a trustworthy source (teacher: non-Idle; candidate s3: non-Idle;
      otherwise None — never seed memory with unverifiable self-reports).
    """
    summary = (mem.get("running_summary", "") + " " + (memory_update or "")).strip()
    if len(summary) > SUMMARY_CAP:
        summary = summary[-SUMMARY_CAP:]
    trail = (mem.get("phase_trail", []) + [phase_id])[-TRAIL_CAP:]
    seen = dict(mem.get("instruments_seen", {}))
    if seen_classes:
        for cls in seen_classes:
            seen[cls] = chunk_id
    prev_nonidle = (report if phase_id not in idle_phases
                    else mem.get("prev_nonidle_report", ""))
    flags = list(mem.get("flags", []))
    for f in (flags_add or []):
        # Normalize: older snapshots / raw model outputs may hold dict flags.
        if isinstance(f, dict):
            import json as _json
            f = _json.dumps(f)
        elif not isinstance(f, str):
            f = str(f)
        if f not in flags:
            flags.append(f)
    return {"running_summary": summary, "prev_report": report[:VERBATIM_CAP],
            "prev_nonidle_report": prev_nonidle[:VERBATIM_CAP],
            "phase_trail": trail, "instruments_seen": seen, "flags": flags,
            "same_phase_continuation": False}
