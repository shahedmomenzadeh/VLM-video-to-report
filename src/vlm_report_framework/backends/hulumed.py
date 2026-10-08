"""HuluMed backend: conversation-style video dict, fps-based sampling.

Pattern follows VLM_Surgery_Evaluation/hulumed_inference.py exactly
(processor conversation + compat patch + halving retry ladder).
Runs under .venv-hulumed (pinned transformers==4.51.2).

Known model IDs: ZJU-AI4H/Hulu-Med-7B (frame-size 224, temp 0.6 in run.sh),
ZJU-AI4H/Hulu-Med-4B.
"""
from __future__ import annotations

import gc

from . import VideoBackend


def _patch_hulumed_processor_compatibility():
    """Patches ProcessorMixin to support HulumedProcessor under transformers >= 4.49."""
    import inspect
    import transformers.processing_utils

    orig_from_pretrained = transformers.processing_utils.ProcessorMixin.from_pretrained

    @classmethod
    def patched_from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        orig_get_args = getattr(cls, "_get_arguments_from_pretrained", None)
        if orig_get_args is not None and not getattr(orig_get_args, "_patched_compat", False):
            try:
                sig = inspect.signature(orig_get_args)
                pos_params = [
                    p for p in sig.parameters.values()
                    if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
                ]
                if len(pos_params) <= 2:
                    @classmethod
                    def wrapped_get_args(subcls, path, *extra_args, **kw):
                        return orig_get_args(path, **kw)
                    wrapped_get_args._patched_compat = True
                    cls._get_arguments_from_pretrained = wrapped_get_args
            except Exception:
                pass
        return orig_from_pretrained.__func__(cls, pretrained_model_name_or_path, *args, **kwargs)

    transformers.processing_utils.ProcessorMixin.from_pretrained = patched_from_pretrained


class HuluMedBackend(VideoBackend):
    name = "hulumed"

    def load(self):
        import torch
        from transformers import (
            AutoModelForCausalLM,
            AutoProcessor,
            BitsAndBytesConfig,
        )

        _patch_hulumed_processor_compatibility()
        qcfg = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
        processor = AutoProcessor.from_pretrained(
            self.model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            self.model_path, trust_remote_code=True, device_map="auto",
            quantization_config=qcfg, attn_implementation="sdpa")
        model.eval()
        return processor, model

    def generate_text(self, ctx, video_path: str, prompt: str,
                      max_frames: int, gen: dict) -> tuple[str, int]:
        """fps/max_frames/size conversation with halving retry ladder.

        gen keys: max_new_tokens, temperature, fps (default 1.0),
        frame_size (default 480). Short clips are capped to their real
        length (decord probe) before the ladder starts.
        """
        import torch

        processor, model = ctx
        fps = gen.get("fps", 1.0)
        frame_size = gen.get("frame_size", 480)

        total = self._clip_frame_count(video_path)
        # fps-based sampling yields ~duration*fps frames: a sub-second clip
        # at fps=1.0 decodes to 0-1 frames and the processor's image resize
        # crashes with bare IndexError (seen on a 0.7s P13 chunk). Guarantee
        # ~3 sampled frames by raising fps for short clips (max_frames still
        # caps the total, so long clips are unaffected).
        duration = self._clip_duration(video_path)
        fps_eff = fps
        if duration and duration > 0:
            fps_eff = max(fps, min(8.0, 3.0 / duration))
        effective = max_frames if total is None else max(2, min(max_frames, total))
        ladder = []
        f = effective
        while f >= 2:
            ladder.append(f)
            f //= 2
        if not ladder:
            ladder = [2]

        last_err = None
        for attempt in ladder:
            conversation = [{
                "role": "user",
                "content": [
                    {"type": "video", "video": {
                        "video_path": video_path, "fps": fps_eff,
                        "max_frames": attempt, "size": frame_size}},
                    {"type": "text", "text": prompt}]}]
            try:
                inputs = processor(
                    conversation=conversation, add_system_prompt=True,
                    add_generation_prompt=True, return_tensors="pt")
                inputs = {
                    k: (v.cuda().to(torch.float16)
                        if isinstance(v, torch.Tensor) and v.is_floating_point()
                        else v.cuda() if isinstance(v, torch.Tensor) else v)
                    for k, v in inputs.items()}
                with torch.no_grad():
                    out = model.generate(
                        **inputs, max_new_tokens=gen["max_new_tokens"],
                        do_sample=(gen["temperature"] > 0.0),
                        temperature=(gen["temperature"]
                                     if gen["temperature"] > 0.0 else None),
                        use_cache=True,
                        pad_token_id=processor.tokenizer.eos_token_id)
                try:
                    txt = processor.batch_decode(
                        out, skip_special_tokens=True, use_think=False)[0].strip()
                except TypeError:  # older transformers without use_think
                    txt = processor.batch_decode(
                        out, skip_special_tokens=True)[0].strip()
                del inputs, out
                return txt, attempt
            except torch.cuda.OutOfMemoryError as e:
                last_err = e
                gc.collect()
                torch.cuda.empty_cache()
            except (ValueError, AttributeError, IndexError) as e:
                msg = str(e)
                if isinstance(e, IndexError) or "nframes" in msg \
                        or "read_video" in msg:
                    # Short-clip empty decode: sample denser and retry.
                    fps_eff = min(fps_eff * 2.0, 16.0)
                    last_err = e
                    gc.collect()
                    torch.cuda.empty_cache()
                    continue
                raise
        gc.collect()
        torch.cuda.empty_cache()
        raise RuntimeError(f"HuluMed failed even at 2 frames: {last_err}")

    @staticmethod
    def _clip_frame_count(video_path: str) -> int | None:
        try:
            from decord import VideoReader
            return len(VideoReader(video_path))
        except Exception:
            return None

    @staticmethod
    def _clip_duration(video_path: str) -> float | None:
        try:
            from decord import VideoReader
            vr = VideoReader(video_path)
            fps = vr.get_avg_fps()
            if fps and fps > 0:
                return len(vr) / float(fps)
        except Exception:
            pass
        return None
