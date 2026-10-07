"""Eval B prompts: evidence-grounded video-level judge (teacher-blind) + pairwise.

The judge NEVER sees the teacher report. Evidence hierarchy (in prompt):
  video (primary) > phase timeline > YOLO tiers (imperfect auxiliary).
Candidate memory text is judged as coherence signal, not evidence.

Criteria (1-5, anchors in prompt):
  groundedness, completeness, instrument_correctness, temporal_coherence.
Pairwise: winner A/B/Tie + reason, run order-swapped by the caller.
"""
from __future__ import annotations

SCORING_CONTRACT = """Respond with ONLY a JSON object, no other text. Keys exactly:
- "groundedness": 1-5 (5 = every claim visually supported; 1 = multiple inventions)
- "completeness": 1-5 (5 = all key operative steps described; 1 = major gaps)
- "instrument_correctness": 1-5 (5 = no invented tools, weak evidence hedged; 1 = invented tools)
- "temporal_coherence": 1-5 (5 = starts/continues/ends judged correctly across chunks; 1 = contradictions or restarts)
- "echo_flag": true if the report copies perception-model metadata verbatim
  (e.g. "median conf", "N frames") instead of describing what is visible, else false
- "reason": 2-4 sentences citing chunk headers (e.g. P05_o01_c2) for the lowest score.

Ignore style and length. Score facts only."""

SYSTEM_PROMPT = """You are an expert cataract-surgery video evaluator. \
Score ONLY what is visually supported. The candidate instrument list comes \
from an imperfect perception model — a candidate that lists a tool without \
visual evidence is wrong even if the tool is in the list."""

# --- prompt variants for calibration (see judge_reports.py --prompt-version) ---
#
# v1: original strict prompt above (floor-effect suspect).
# v2: calibrated scale — explicit mid-scale meanings + full-scale-use
#     instruction. Tests whether v1's floor was an anchor wording artifact.
# v3: v2 + resident-relative framing + symmetric justification (cite evidence
#     for the HIGHEST score too). Tests whether forcing positive evidence
#     retrieval lifts indiscriminate 1s.
V2_CONTRACT = """Respond with ONLY a JSON object, no other text. Keys exactly:
- "groundedness": 1-5. 1 = nearly nothing supported / empty or garbage;
  2 = isolated correct elements amid major invention; 3 = mixed, roughly
  half the claims supported; 4 = mostly supported with minor overstatements;
  5 = every substantive claim visually supported.
- "completeness": 1-5. 1 = almost all operative steps missing; 2 = a few
  steps present; 3 = about half the key steps; 4 = most steps, minor gaps;
  5 = all key operative steps described.
- "instrument_correctness": 1-5. 1 = tools invented throughout; 2 = some
  correct, several invented or unhedged weak-tier copies; 3 = mostly correct
  with lapses; 4 = correct with minor hedging lapses;
  5 = no invented tools, weak evidence hedged.
- "temporal_coherence": 1-5. 1 = contradictions or restarts throughout;
  2 = frequent progression errors; 3 = roughly coherent with clear errors;
  4 = coherent with minor slips; 5 = starts/continues/ends judged correctly.
- "echo_flag": true if the report copies perception-model metadata verbatim
  (e.g. "median conf", "N frames", "phase-plausible") instead of describing
  what is visible, else false.
- "reason": 2-4 sentences citing chunk headers (e.g. P05_o01_c2).

Use the FULL scale: ordinary imperfect reports should land at 2-4. Reserve
1 for empty/garbage/wholly unsupported reports and 5 for near-perfect ones.
Ignore style and length. Score facts only."""

V2_SYSTEM = """You are an expert cataract-surgery video evaluator. Score ONLY \
what is visually supported against the attached video and the auxiliary \
timeline/tiers. The instrument tier list comes from an imperfect perception \
model — listing a tool without visual evidence is an error even if the tool \
is in the list. Calibrate against the full 1-5 range described in the task."""

V3_EXTRA = """\nFor your HIGHEST criterion score, quote one chunk header and the \
supported claim that earned it; for your LOWEST, quote the failing header \
and claim. Judge as you would a competent surgical resident's operative \
note: credit correct observations before penalizing errors."""


def get_prompt_version(v: str) -> tuple[str, str]:
    """Return (system, contract) for a prompt version."""
    if v == "v1":
        return SYSTEM_PROMPT, SCORING_CONTRACT
    if v == "v2":
        return V2_SYSTEM, V2_CONTRACT
    if v == "v3":
        return V2_SYSTEM, V2_CONTRACT + V3_EXTRA
    raise ValueError(f"unknown prompt version: {v}")


def build_evidence_block(timeline_txt: str, tiers_txt: str) -> str:
    return ("EVIDENCE (the video file is attached separately and is primary;\n"
            "timeline and tiers below are auxiliary):\n\n"
            f"Phase timeline:\n{timeline_txt}\n\n"
            f"Aggregated instrument tiers (observed = strong, weak = verify-visually):\n{tiers_txt}")


def build_scoring_prompt(candidate_doc: str, evidence_block: str,
                         version: str = "v1") -> str:
    _, contract = get_prompt_version(version)
    return ("CANDIDATE video report (chunk headers in [brackets] for localization):\n\n"
            f"{candidate_doc}\n\n{evidence_block}\n\n{contract}")


PAIRWISE_CONTRACT = """Respond with ONLY a JSON object, no other text. Keys exactly:
- "winner": "A", "B" or "Tie" (factually better report; ignore style/length)
- "reason": 2-3 sentences citing chunk headers for the deciding difference."""


def build_pairwise_prompt(doc_a: str, doc_b: str, evidence_block: str) -> str:
    return ("Two candidate video reports for the SAME surgery.\n\n"
            f"Report A:\n{doc_a}\n\nReport B:\n{doc_b}\n\n"
            f"{evidence_block}\n\n{PAIRWISE_CONTRACT}")
