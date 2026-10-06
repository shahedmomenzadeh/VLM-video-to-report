"""Qwen3-VL backend: local .mp4 path + sampled frames, 4-bit HF inference.

Video input pattern follows E:/VLM_Evaluation/qwen3VL_inference.py.
Runs under .venv-qwen3vl (also serves Lingshu-7B, same processor family).
"""
from __future__ import annotations

import gc
import re

from . import VideoBackend


class Qwen3VLBackend(VideoBackend):
    name = "qwen3vl"

    def load(self):
        import torch
        from transformers import AutoProcessor, BitsAndBytesConfig
        from transformers import Qwen3VLForConditionalGeneration as ModelClass

        qcfg = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
        processor = AutoProcessor.from_pretrained(
            self.model_path, trust_remote_code=True)
        model = ModelClass.from_pretrained(
            self.model_path, quantization_config=qcfg, device_map="auto",
            torch_dtype=torch.float16, low_cpu_mem_usage=True,
            trust_remote_code=True)
        model.eval()
        return processor, model

    def _build_inputs(self, processor, video_path: str, question: str,
                      max_frames: int, max_pixels: int, min_pixels: int):
        import torch
        from qwen_vl_utils import process_vision_info

        messages = [{"role": "user", "content": [
            {"type": "video", "video": video_path, "nframes": max_frames,
             "max_pixels": max_pixels, "min_pixels": min(min_pixels, max_pixels)},
            {"type": "text", "text": question}]}]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(text=[text], images=image_inputs, videos=video_inputs,
                           padding=True, return_tensors="pt")
        for k, v in inputs.items():
            if torch.is_tensor(v):
                inputs[k] = v.to("cuda" if torch.cuda.is_available() else "cpu")
                if torch.is_floating_point(inputs[k]):
                    inputs[k] = inputs[k].to(torch.float16)
        return inputs

    @staticmethod
    def _clip_frame_count(video_path: str) -> int | None:
        try:
            from decord import VideoReader
            return len(VideoReader(video_path))
        except Exception:
            return None

    def generate_text(self, ctx, video_path: str, prompt: str,
                      max_frames: int, gen: dict) -> tuple[str, int]:
        """Sampled-frames generation with clamp-to-clip + halve-on-OOM.

        gen keys: max_pixels, min_pixels, max_new_tokens, temperature.
        Short clips feed all their frames (n = min(max_frames, total)).
        """
        import torch

        processor, model = ctx
        total = self._clip_frame_count(video_path)
        n = max_frames if total is None else max(2, min(max_frames, total))
        last_err = None
        with torch.no_grad():
            while n >= 2:
                try:
                    inputs = self._build_inputs(
                        processor, video_path, prompt, n,
                        gen["max_pixels"], gen["min_pixels"])
                    out = model.generate(
                        **inputs, max_new_tokens=gen["max_new_tokens"],
                        do_sample=(gen["temperature"] > 0.0),
                        temperature=max(gen["temperature"], 1e-6),
                        top_p=0.8, top_k=20, min_p=0.0,
                        repetition_penalty=1.05, use_cache=True,
                        eos_token_id=[151645, 151643],
                        pad_token_id=processor.tokenizer.eos_token_id)
                    seq = out[0][inputs["input_ids"].shape[1]:]
                    txt = processor.tokenizer.decode(seq, skip_special_tokens=True)
                    return (re.sub(r"<think>.*?</think>", "", txt,
                                   flags=re.DOTALL).strip(), n)
                except torch.cuda.OutOfMemoryError as e:
                    last_err = e
                    gc.collect()
                    torch.cuda.empty_cache()
                    n //= 2
                except ValueError as e:
                    if "nframes" not in str(e):
                        raise
                    last_err = e
                    n //= 2
        raise RuntimeError(f"Failed even at 2 frames: {last_err}")
