"""Canonical report renderer + v2 structured-response validation.

v2 contract: the model returns STRUCTURED FIELDS ONLY (no skeleton text,
no timestamps). This module validates the fields and renders the exact
5-section wire text deterministically, so timestamps/headers can never
drift. Header labels are byte-identical to the v1 free-text format so
eval parsers (evaluation/reference.py), the viewer, and readable
compilers keep working unchanged.
"""
from __future__ import annotations

import re

from prompt_base import ANATOMY_TERMS, INSTRUMENT_LABELS, PHASE_NAMES, fmt_time, label

TIMESTAMP_RE = re.compile(r"\d{1,2}:\d{2}(?:\.\d+)?")


def render_report(
    phase_id: str,
    t_start: float,
    t_end: float,
    instruments_visible: list,
    instruments_uncertain: list,
    anatomy: list,
    actions: list,
) -> str:
    """Render canonical 5-section text. All timestamps/headers from code."""
    phase_id = phase_id.upper()
    pname = PHASE_NAMES.get(phase_id, phase_id)
    lines = [f"Phase: {phase_id} ({pname}), {fmt_time(t_start)}-{fmt_time(t_end)}",
             "",
             "Observed instruments:"]
    if instruments_visible:
        lines += [f"- {label(c)}" for c in instruments_visible]
    else:
        lines.append("None clearly visible")
    lines += ["", "Candidate but not confirmed:"]
    if instruments_uncertain:
        lines += [f"- {label(c)}" for c in instruments_uncertain]
    else:
        lines.append("None")
    lines += ["", "Anatomy visible:"]
    lines.append(", ".join(ANATOMY_TERMS[a] for a in anatomy) if anatomy
                 else "None clearly visible")
    lines += ["", "Actions/events:"]
    lines += [f"- {a}" for a in actions] if actions else ["- No change observed."]
    return "\n".join(lines)


def validate_struct(
    parsed: dict,
    allowed_instr: set[str],
    require_phase: str | None = None,
) -> tuple[bool, str, dict]:
    """Validate a v2 structured response.

    Returns (ok, error, cleaned). `cleaned` holds normalized fields on
    success. `require_phase`: exact phase the model must state (teacher /
    s2 / s3); None means any P01..P13 (s1 self-guess).
    """
    if not isinstance(parsed, dict):
        return False, "response is not a JSON object", {}
    phase = parsed.get("phase_id")
    if isinstance(phase, str):
        # Normalize: accept "P01-Incision" / "p01" etc., keep the PXX code.
        m = re.match(r"\s*(P\d{2})", phase.upper())
        phase = m.group(1) if m else phase.upper()
    if not isinstance(phase, str) or phase.upper() not in PHASE_NAMES:
        return False, f"phase_id must be one of {sorted(PHASE_NAMES)}", {}
    phase = phase.upper()
    if require_phase and phase != require_phase:
        return False, f"phase_id {phase} != given phase {require_phase}", {}
    vis = parsed.get("instruments_visible", [])
    unc = parsed.get("instruments_uncertain", [])
    if not isinstance(vis, list) or not isinstance(unc, list):
        return False, "instrument fields must be lists of class ids", {}
    # Unambiguous normalization: case-insensitive exact match against the
    # allowed ids ("forceps" -> "Forceps"). True synonyms ("Sinskey hook")
    # are NOT mapped — silently resolving them would inject our judgment
    # into model outputs and bias instrument scores.
    canon = {c.lower(): c for c in allowed_instr}
    vis = [canon.get(c.lower().strip(), c) if isinstance(c, str) else c
           for c in vis]
    unc = [canon.get(c.lower().strip(), c) if isinstance(c, str) else c
           for c in unc]
    bad = [c for c in vis + unc if c not in allowed_instr]
    if bad:
        return False, (f"unknown instrument ids {bad}; allowed: "
                       f"{sorted(allowed_instr)}"), {}
    if set(vis) & set(unc):
        return False, "an instrument cannot be both visible and uncertain", {}
    anat = parsed.get("anatomy", [])
    if not isinstance(anat, list):
        return False, "anatomy must be a list of ids", {}
    bad_a = [a for a in anat if a not in ANATOMY_TERMS]
    if bad_a:
        return False, (f"unknown anatomy ids {bad_a}; allowed: "
                       f"{sorted(ANATOMY_TERMS)}"), {}
    acts = parsed.get("actions", [])
    if not isinstance(acts, list) or not acts or not all(
            isinstance(a, str) and a.strip() for a in acts):
        return False, "actions must be a non-empty list of strings", {}
    if len(acts) > 6:
        return False, f"actions has {len(acts)} bullets, max 6", {}
    for a in acts:
        if TIMESTAMP_RE.search(a):
            return False, (f"actions must not contain timestamps: {a[:60]!r}; "
                            "timestamps are added by the renderer"), {}
    mu = parsed.get("memory_update", "")
    fa = parsed.get("flags_add", []) or []
    if not isinstance(mu, str):
        mu = str(mu)
    if not isinstance(fa, list):
        fa = [fa]
    return True, "", {
        "phase_id": phase,
        "instruments_visible": list(vis),
        "instruments_uncertain": list(unc),
        "anatomy": list(anat),
        "actions": [a.strip() for a in acts],
        "memory_update": mu.strip(),
        "flags_add": [f if isinstance(f, str) else str(f) for f in fa if f],
    }


def is_v2(parsed: dict) -> bool:
    """True if a parsed response uses the v2 structured schema."""
    return isinstance(parsed, dict) and isinstance(parsed.get("actions"), list) \
        and "phase_id" in parsed


# --- v2.1 student 4-tag plain-text contract ---------------------------------
# Small VLMs cannot reliably emit escaped JSON with exact YOLO ids. They
# write simple tagged lines; THIS module canonicalizes in code (synonym map
# + case-insensitive match) and feeds the same render_report(), so eval
# headers stay byte-identical. No GPU retry is ever needed: parsing is total
# (every input yields a cleaned dict + report text).

# Natural student phrase -> canonical 10-class YOLO id.
STUDENT_INSTR_SYN: dict[str, str] = {
    "phacoemulsification handpiece": "Phaco-Handpiece",
    "phaco handpiece": "Phaco-Handpiece",
    "phaco probe": "Phaco-Handpiece",
    "phacoemulsification probe": "Phaco-Handpiece",
    "phaco": "Phaco-Handpiece",
    "irrigation-aspiration handpiece": "I-A-Handpiece",
    "irrigation aspiration handpiece": "I-A-Handpiece",
    "i/a handpiece": "I-A-Handpiece",
    "i-a handpiece": "I-A-Handpiece",
    "i/a probe": "I-A-Handpiece",
    "irrigation": "I-A-Handpiece",
    "aspiration": "I-A-Handpiece",
    "capsulorhexis forceps": "Cap-Forceps",
    "capsule forceps": "Cap-Forceps",
    "cap forceps": "Cap-Forceps",
    "cystotome": "Cap-Cystotome",
    "capsulorhexis needle": "Cap-Cystotome",
    "cystotome (capsulorhexis needle)": "Cap-Cystotome",
    "cannula": "Cannula",
    "tissue forceps": "Forceps",
    "forceps": "Forceps",
    "iris forceps": "Forceps",
    "lens injector": "Lens-Injector",
    "injector": "Lens-Injector",
    "second instrument": "Second-Instrument",
    "chopper": "Second-Instrument",
    "manipulator": "Second-Instrument",
    "sinskey hook": "Second-Instrument",
    "second instrument (chopper/manipulator)": "Second-Instrument",
    "primary knife": "Primary-Knife",
    "keratome": "Primary-Knife",
    "primary (keratome) knife": "Primary-Knife",
    "secondary knife": "Secondary-Knife",
    "paracentesis knife": "Secondary-Knife",
    "secondary (paracentesis) knife": "Secondary-Knife",
    "needle": "Cannula",
}

# Natural student phrase -> canonical anatomy id.
STUDENT_ANAT_SYN: dict[str, str] = {
    "cornea": "cornea",
    "pupil": "pupil",
    "iris": "iris",
    "anterior chamber": "anterior_chamber",
    "lens capsule": "lens_capsule",
    "capsule": "lens_capsule",
    "capsular bag": "lens_capsule",
    "lens nucleus": "lens_nucleus",
    "nucleus": "lens_nucleus",
    "lens": "lens_nucleus",
    "conjunctiva": "conjunctiva_eyelid",
    "eyelid": "conjunctiva_eyelid",
    "conjunctiva/eyelids": "conjunctiva_eyelid",
    "incision": "incision_site",
    "incision site": "incision_site",
    "wound": "incision_site",
}

_TAG_RE = {
    "phase": re.compile(r"^PHASE\s*:\s*(.+)$", re.I | re.M),
    "tools": re.compile(r"^TOOLS\s*:\s*(.+)$", re.I | re.M),
    "uncertain": re.compile(r"^UNCERTAIN_TOOLS\s*:\s*(.+)$", re.I | re.M),
    "anatomy": re.compile(r"^ANATOMY\s*:\s*(.+)$", re.I | re.M),
    "state": re.compile(r"^STATE\s*:\s*(.+)$", re.I | re.M),
    "flags": re.compile(r"^FLAGS\s*:\s*(.+)$", re.I | re.M),
}
_EVENTS_RE = re.compile(r"^EVENTS\s*:\s*(.*?)(?=^(?:PHASE|TOOLS|UNCERTAIN_TOOLS|"
                        r"ANATOMY|STATE|FLAGS)\s*:|\Z)", re.I | re.M | re.S)


def _split_list(text: str) -> list[str]:
    parts = re.split(r"[,;\n]+", text or "")
    out = []
    for p in parts:
        p = re.sub(r"^[\-\*\d\.\)\s]+", "", p.strip())
        p = p.strip().strip(".")
        if p and p.lower() not in ("none", "none clearly visible", "n/a", "-"):
            out.append(p)
    return out


def _canon_instr(term: str) -> str | None:
    t = term.strip().lower()
    if not t or t in ("none", "no instruments", "no instrument"):
        return None
    if t in STUDENT_INSTR_SYN:
        return STUDENT_INSTR_SYN[t]
    # Case-insensitive exact id match ("forceps" -> "Forceps").
    for cid in INSTRUMENT_LABELS:
        if t == cid.lower():
            return cid
    # Human display-term match ("cannula" is both; "tissue forceps" covered).
    for cid, disp in INSTRUMENT_LABELS.items():
        if t == disp.lower():
            return cid
    return None


def _canon_anat(term: str) -> str | None:
    t = term.strip().lower()
    if not t or t in ("none", "none clearly visible"):
        return None
    if t in STUDENT_ANAT_SYN:
        return STUDENT_ANAT_SYN[t]
    for aid in ANATOMY_TERMS:
        if t == aid.lower() or t == ANATOMY_TERMS[aid].lower():
            return aid
    return None


def scrub_timestamps(text: str) -> str:
    """Remove MM:SS(.ss) shapes from model prose (time lives in code)."""
    return TIMESTAMP_RE.sub("", text or "").strip()


def parse_student_tags(raw: str, setting: str, chunk: dict) -> tuple[dict, str]:
    """Parse v2.1 student tagged text -> (cleaned, report_text). Total: never raises.

    cleaned keys: phase_id, instruments_visible, instruments_uncertain,
      anatomy, actions, memory_update, flags_add, unmatched_* (transparency).
    report_text: canonical 5-section wire text via render_report().
    """
    text = raw or ""
    # --- phase ---
    if setting == "s1":
        m = _TAG_RE["phase"].search(text)
        ph = "P??"
        if m:
            pm = re.search(r"P\d{2}", m.group(1).upper())
            ph = pm.group(0) if pm else "P??"
        else:
            pm = re.search(r'"phase_id"\s*:\s*"(P\d{2})"', text, re.I)
            if pm:
                ph = pm.group(1).upper()
            else:
                pm2 = re.search(r"Phase:\s*(P\d{2})", text, re.I)
                if pm2:
                    ph = pm2.group(1).upper()
        if ph not in PHASE_NAMES:
            ph = "P??"
    else:
        ph = chunk["phase"]
    # --- tools ---
    vis, unc, unmatched_instr = [], [], []
    for key, dest in (("tools", vis), ("uncertain", unc)):
        m = _TAG_RE[key].search(text)
        if m:
            for term in _split_list(m.group(1)):
                c = _canon_instr(term)
                if c and c not in dest:
                    dest.append(c)
                elif not c:
                    unmatched_instr.append(term)
    # If the model ignored tags and wrote a JSON-ish list, salvage ids.
    if not vis and not unc:
        low = text.lower()
        for cid in INSTRUMENT_LABELS:
            if re.search(r"\b" + re.escape(cid) + r"\b", text):
                vis.append(cid)
        if not vis:
            for phrase, cid in STUDENT_INSTR_SYN.items():
                if len(phrase) >= 5 and phrase in low and cid not in vis:
                    vis.append(cid)
    vis = [c for c in vis if c not in unc]
    # --- anatomy ---
    anat, unmatched_anat = [], []
    m = _TAG_RE["anatomy"].search(text)
    if m:
        for term in _split_list(m.group(1)):
            c = _canon_anat(term)
            if c and c not in anat:
                anat.append(c)
            elif not c:
                unmatched_anat.append(term)
    # --- events ---
    actions: list[str] = []
    m = _EVENTS_RE.search(text)
    ev_block = m.group(1).strip() if m else ""
    if ev_block:
        for line in ev_block.split("\n"):
            b = re.sub(r"^[\-\*\d\.\)\s]+", "", line.strip()).strip()
            b = scrub_timestamps(b)
            b = re.sub(r"\s{2,}", " ", b).strip()
            if b and b.lower() not in ("none", "no change", "no change observed"):
                actions.append(b)
    if not actions:
        # Last-resort salvage: any non-tag prose line without timestamps.
        for line in text.split("\n"):
            if re.match(r"^(PHASE|TOOLS|UNCERTAIN_TOOLS|ANATOMY|EVENTS|STATE|FLAGS)\s*:",
                        line.strip(), re.I):
                continue
            b = scrub_timestamps(line.strip())
            if len(b) > 20 and "{" not in b and "}" not in b:
                actions.append(b[:200])
                if len(actions) >= 3:
                    break
    actions = [a[:300] for a in actions[:5]]
    # --- state / flags ---
    m = _TAG_RE["state"].search(text)
    state = scrub_timestamps(m.group(1).strip())[:300] if m else ""
    if not state:
        state = actions[0][:300] if actions else "No distinct state change reported."
    m = _TAG_RE["flags"].search(text)
    flags: list[str] = []
    if m:
        for f in _split_list(m.group(1)):
            f = f.strip()
            if f:
                flags.append(f)
    # Phase P?? has no display name; render under given/given-or-unknown.
    render_phase = ph if ph in PHASE_NAMES else chunk.get("phase", "P13")
    pname_note = "" if ph in PHASE_NAMES else ""
    report = render_report(render_phase, chunk["t_start"], chunk["t_end"],
                           vis, unc, anat, actions or ["No change observed."])
    cleaned = {"phase_id": ph, "instruments_visible": vis,
               "instruments_uncertain": unc, "anatomy": anat,
               "actions": actions, "memory_update": state, "flags_add": flags,
               "unmatched_instruments": unmatched_instr[:10],
               "unmatched_anatomy": unmatched_anat[:10],
               "render_note": pname_note}
    return cleaned, report
