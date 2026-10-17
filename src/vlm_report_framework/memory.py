"""Shared causal memory M(i,j): creation, update rule, schema.

One memory dict per (video [, model, setting]) chain, updated once per chunk
in temporal order. The VLM contributes only a short STATE sentence +
optional FLAGS; the merge itself is deterministic code.

v2.1 slim schema (time-blind, compact):
  recent_states:     rolling list (max 4) of {chunk_id, phase, state}
                     one-sentence end-state per chunk, oldest first
  last_nonidle_state: {chunk_id, phase, state} of most recent NON-IDLE chunk
                     (carried forward across Idle gaps)
  phase_trail:       last 10 phase ids (Idle included; s1 holds stated phases)
  instruments_seen:  class -> last chunk id (selective; see callers)
  flags:             append-only deduped event tags
  same_phase_continuation: transient hint flag, recomputed before each prompt

Legacy keys (running_summary, prev_report, prev_nonidle_report) are
auto-migrated on load: verbatim reports are NOT carried forward (they are
the echo-hallucination vector); only phase_trail/instruments/flags survive.
"""
from __future__ import annotations

TRAIL_CAP = 10
RECENT_CAP = 4
STATE_CAP = 300  # one STATE sentence must fit; truncate pathological dumps


def fresh_memory() -> dict:
    return {"recent_states": [], "last_nonidle_state": {},
            "phase_trail": [], "instruments_seen": {}, "flags": [],
            "same_phase_continuation": False}


def migrate_memory(mem: dict) -> dict:
    """Upgrade a legacy snapshot dict to the slim schema in place."""
    if "recent_states" in mem and "phase_trail" in mem:
        # Already slim (but ensure all keys exist for forward-compat).
        mem.setdefault("last_nonidle_state", {})
        mem.setdefault("instruments_seen", {})
        mem.setdefault("flags", [])
        mem.setdefault("same_phase_continuation", False)
        return mem
    # Legacy v1/v2: drop verbatim reports + running_summary (bloat/echo
    # vector). Keep only structural signals.
    out = fresh_memory()
    trail = mem.get("phase_trail", [])
    if isinstance(trail, list):
        out["phase_trail"] = [t for t in trail if isinstance(t, str)][-TRAIL_CAP:]
    seen = mem.get("instruments_seen", {})
    if isinstance(seen, dict):
        out["instruments_seen"] = {k: v for k, v in seen.items()
                                   if isinstance(k, str)}
    flags = mem.get("flags", [])
    if isinstance(flags, list):
        for f in flags:
            s = f if isinstance(f, str) else str(f)
            if s and s not in out["flags"]:
                out["flags"].append(s)
    return out


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

    phase_id: given phase (teacher / s3) or the model's stated phase (s1).
    report: canonical rendered text (kept for signature compat; NOT stored
      verbatim — only the short STATE sentence enters recent_states).
    memory_update: the model's STATE sentence (1-2 sentences, timestamp-free).
    seen_classes: instrument classes to record as seen here, or None to leave
      the record untouched. Callers pass observed-tier classes only when the
      chunk is a trustworthy source (teacher: non-Idle; candidate s3: non-Idle;
      otherwise None — never seed memory with unverifiable self-reports).
    """
    mem = migrate_memory(dict(mem))
    state = (memory_update or "").strip().replace("\n", " ")[:STATE_CAP]
    if not state:
        # Fallback so the chain never goes blind: derive a minimal state
        # from the phase (never from verbatim report text).
        state = "No distinct state change reported."
    recent = list(mem.get("recent_states", []))
    recent.append({"chunk_id": chunk_id, "phase": phase_id, "state": state})
    recent = recent[-RECENT_CAP:]
    last_ni = dict(mem.get("last_nonidle_state", {}) or {})
    if phase_id not in idle_phases:
        last_ni = {"chunk_id": chunk_id, "phase": phase_id, "state": state}
    trail = (mem.get("phase_trail", []) + [phase_id])[-TRAIL_CAP:]
    seen = dict(mem.get("instruments_seen", {}))
    if seen_classes:
        for cls in seen_classes:
            seen[cls] = chunk_id
    flags = list(mem.get("flags", []))
    for f in (flags_add or []):
        # Normalize: older snapshots / raw model outputs may hold dict flags.
        if isinstance(f, dict):
            import json as _json
            f = _json.dumps(f)
        elif not isinstance(f, str):
            f = str(f)
        f = f.strip()
        if f and f not in flags:
            flags.append(f)
    return {"recent_states": recent, "last_nonidle_state": last_ni,
            "phase_trail": trail, "instruments_seen": seen, "flags": flags,
            "same_phase_continuation": False}
