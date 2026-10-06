"""Local-VLM backend interface.

Each model family implements VideoBackend under its own venv (they need
incompatible dependency versions — see E:/VLM_Evaluation run.sh). The runner
(candidate_reports.py) only talks to this interface, so adding a family
(Hulu-Med, Lingshu, ...) means adding one module here, nothing else changes:

  every backend receives: frozen local .mp4 clip path + text prompt
  every backend returns:  (response text, frames actually used)

Heavy imports (torch, transformers, qwen_vl_utils, decord) live inside the
backend modules so the runner stays importable everywhere.
"""
from __future__ import annotations


class VideoBackend:
    name = "base"

    def __init__(self, model_path: str):
        self.model_path = model_path

    def load(self):
        """Load processor+model once. Returns an opaque ctx for generate_text."""
        raise NotImplementedError

    def generate_text(self, ctx, video_path: str, prompt: str,
                      max_frames: int, gen: dict) -> tuple[str, int]:
        """One sampled-frames generation. Returns (text, frames used)."""
        raise NotImplementedError


def get_backend(name: str, model_path: str) -> VideoBackend:
    if name == "qwen3vl":
        from .qwen3vl import Qwen3VLBackend
        return Qwen3VLBackend(model_path)
    raise ValueError(f"unknown backend: {name} (available: qwen3vl)")
