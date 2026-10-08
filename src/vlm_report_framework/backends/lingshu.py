"""Lingshu backend: Qwen2.5-VL-architecture message + video_kwargs merge.

Pattern follows VLM_Surgery_Evaluation/lingshu_inference.py exactly
(nframes message, process_vision_info with return_video_kwargs, batch_decode,
think-strip, halving retry ladder).
Runs under .venv-qwen3vl (same venv as the Qwen3-VL backend).

Known model ID: lingshu-medical-mllm/Lingshu-7B.
"""
from __future__ import annotations

import gc
import re

from . import VideoBackend


class LingshuBackend(VideoBackend):
    name = "lingshu"

    def load(self):
        import torch
        from transformers import AutoProcessor, BitsAndBytesConfig
        try:
            from transformers import Qwen2_5_VLForConditionalGeneration as ModelClass
        except ImportError:
            from transformers import AutoModelForImageTextToText as ModelClass

        qcfg = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
        processor = AutoProcessor.from_pretrained(
            self.model_path, trust_remote_code=True)
        model = ModelClass.from_pretrained(
            self.model_path, quantization_config=qcfg, device_map="auto",
            torch_dtype=torch.float16, low_cpu_mem_usage=True,
            trust_remote_code=True)
        model.eval()
        return processor, model

    def _build_inputs(self, processor, primary_device, video_path: str,
                      question: str, max_frames: int,
                      max_pixels: int, min_pixels: int):
        import torch
        from qwen_vl_utils import process_vision_info

        messages = [{"role": "user", "content": [
            {"type": "video", "video": video_path,
             "min_pixels": min(min_pixels, max_pixels),
             "max_pixels": max_pixels, "nframes": max_frames},
            {"type": "text", "text": question}]}]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            messages, return_video_kwargs=True)
        kwargs = {"text": [text], "images": image_inputs, "videos": video_inputs,
                  "padding": True, "return_tensors": "pt"}
        if video_kwargs:
            for k, v in video_kwargs.items():
                if isinstance(v, list) and len(v) == 1:
                    video_kwargs[k] = v[0]
            kwargs.update(video_kwargs)
        inputs = processor(**kwargs)
        moved = {}
        for k, v in inputs.items():
            if torch.is_tensor(v):
                v = v.to(primary_device)
                if torch.is_floating_point(v):
                    v = v.to(torch.float16)
            moved[k] = v
        return moved

    def generate_text(self, ctx, video_path: str, prompt: str,
                      max_frames: int, gen: dict) -> tuple[str, int]:
        """nframes message with halving retry ladder.

        gen keys: max_pixels, min_pixels, max_new_tokens, temperature.
        Short clips feed all their frames (capped before the ladder).
        """
        import torch

        processor, model = ctx
        try:
            primary_device = next(model.parameters()).device
        except StopIteration:
            primary_device = torch.device("cuda" if torch.cuda.is_available()
                                          else "cpu")

        total = self._clip_frame_count(video_path)
        effective = max_frames if total is None else max(2, min(max_frames, total))
        if total is not None:
            # qwen-vl-utils rounds requested nframes to a multiple of 2:
            # an odd n on a tiny odd-length clip overshoots the decodable
            # count (seen: requesting 4 from a 3-frame clip). Floor to even.
            effective = max(2, (effective // 2) * 2)
        ladder = []
        f = effective
        while f >= 2:
            ladder.append(f)
            f = f // 2 if f > 3 else f - 1
        if not ladder:
            ladder = [2]

        last_err = None
        for attempt in ladder:
            try:
                inputs = self._build_inputs(
                    processor, primary_device, video_path, prompt, attempt,
                    gen["max_pixels"], gen["min_pixels"])
                input_len = inputs["input_ids"].shape[1]
                with torch.no_grad():
                    out = model.generate(
                        **inputs, max_new_tokens=gen["max_new_tokens"],
                        do_sample=(gen["temperature"] > 0.0),
                        temperature=(gen["temperature"]
                                     if gen["temperature"] > 0.0 else None),
                        top_p=0.8 if gen["temperature"] > 0.0 else None,
                        top_k=20 if gen["temperature"] > 0.0 else None,
                        repetition_penalty=1.05, use_cache=True,
                        pad_token_id=processor.tokenizer.eos_token_id)
                gen_ids = out[:, input_len:]
                txt = processor.batch_decode(
                    gen_ids, skip_special_tokens=True,
                    clean_up_tokenization_spaces=False)[0].strip()
                txt = re.sub(r"<think>.*?(?:</think>|$)", "", txt, flags=re.DOTALL)
                if "</think>" in txt:
                    txt = txt.split("</think>")[-1]
                del inputs, out, gen_ids
                return txt.strip(), attempt
            except torch.cuda.OutOfMemoryError as e:
                last_err = e
                gc.collect()
                torch.cuda.empty_cache()
            except (ValueError, AttributeError) as e:
                msg = str(e)
                if "nframes" not in msg and "frame" not in msg.lower() \
                        and "read_video" not in msg:
                    raise
                last_err = e
                gc.collect()
                torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.empty_cache()
        raise RuntimeError(f"Lingshu failed even at 2 frames: {last_err}")

    @staticmethod
    def _clip_frame_count(video_path: str) -> int | None:
        try:
            from decord import VideoReader
            return len(VideoReader(video_path))
        except Exception:
            return None
