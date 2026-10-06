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


def build_evidence_block(timeline_txt: str, tiers_txt: str) -> str:
    return ("EVIDENCE (the video file is attached separately and is primary;\n"
            "timeline and tiers below are auxiliary):\n\n"
            f"Phase timeline:\n{timeline_txt}\n\n"
            f"Aggregated instrument tiers (observed = strong, weak = verify-visually):\n{tiers_txt}")


def build_scoring_prompt(candidate_doc: str, evidence_block: str) -> str:
    return ("CANDIDATE video report (chunk headers in [brackets] for localization):\n\n"
            f"{candidate_doc}\n\n{evidence_block}\n\n{SCORING_CONTRACT}")


PAIRWISE_CONTRACT = """Respond with ONLY a JSON object, no other text. Keys exactly:
- "winner": "A", "B" or "Tie" (factually better report; ignore style/length)
- "reason": 2-3 sentences citing chunk headers for the deciding difference."""


def build_pairwise_prompt(doc_a: str, doc_b: str, evidence_block: str) -> str:
    return ("Two candidate video reports for the SAME surgery.\n\n"
            f"Report A:\n{doc_a}\n\nReport B:\n{doc_b}\n\n"
            f"{evidence_block}\n\n{PAIRWISE_CONTRACT}")
