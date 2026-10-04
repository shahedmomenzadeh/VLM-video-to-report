"""Extract per-frame instrument detections to CSV (Part I: I_{i,j} inputs).

Runs the YOLO segmentation model in detection-only mode (no annotated video)
and saves one row per instrument detection:

    video_id, frame, t_s, class_id, class_name, conf,
    x1, y1, x2, y2, w, h, area_px, area_rel

Notes:
- The model has 12 classes; Cornea (3) and Pupil (9) are tissues, not
  instruments, so they are excluded here. The teacher-VLM prompt needs
  candidate *instruments* only.
- frame = 0-based_decode frame index in stream order; t_s = frame / fps.
- Run: uv run python src/vlm_report_framework/extract_instruments.py [--videos all]
- Existing raw CSVs are skipped (delete to force re-extraction).
"""
import argparse
from pathlib import Path

import cv2
import pandas as pd
from ultralytics import YOLO

REPO = Path.cwd()
MODEL_PATH = REPO / "yolo-model" / "model.pt"
OUTPUT_DIR = REPO / "output-instruments"

DEFAULT_VIDEOS = [
    "PH_0001_2931_S2",  # S2
    "PH_0043_0096_S1",  # S1
    "PH_0057_0239_S1",  # S1
]

TISSUE_IDS = {3, 9}  # Cornea, Pupil


def video_fps(video: Path) -> float:
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return fps, n, w, h


def resolve_videos(spec: str) -> list[Path]:
    if spec == "all":
        ids = sorted(p.name for p in (REPO / "videos").glob("PH_*") if p.is_dir())
    else:
        ids = [v.strip() for v in spec.split(",") if v.strip()]
    paths = [REPO / "videos" / vid / f"{vid}.mp4" for vid in ids]
    for v in paths:
        assert v.exists(), f"Video not found: {v}"
    return paths


def main(videos: list[Path]):
    assert MODEL_PATH.exists(), f"Model not found: {MODEL_PATH}"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(MODEL_PATH))
    print(f"Task: {model.task}, classes: {model.names}")

    for video in videos:
        video_id = video.stem
        out = OUTPUT_DIR / f"{video_id}_instruments_raw.csv"
        if out.exists():
            print(f"-- skip {video_id} (raw CSV exists: {out})", flush=True)
            continue
        fps, n_frames, w, h = video_fps(video)
        frame_area = w * h
        print(f"\n=== {video_id} ({w}x{h} @ {fps:.2f} fps, {n_frames} frames) ===", flush=True)

        rows = []
        frame_idx = -1
        for r in model.predict(source=str(video), stream=True, verbose=False, device=0, save=False):
            frame_idx += 1
            boxes = r.boxes
            if boxes is None or len(boxes) == 0:
                continue
            xyxy = boxes.xyxy.cpu().numpy()
            conf = boxes.conf.cpu().numpy()
            cls = boxes.cls.cpu().numpy().astype(int)
            for (x1, y1, x2, y2), c, cf in zip(xyxy, cls, conf):
                if int(c) in TISSUE_IDS:
                    continue
                bw, bh = float(x2 - x1), float(y2 - y1)
                area = bw * bh
                rows.append({
                    "video_id": video_id,
                    "frame": frame_idx,
                    "t_s": round(frame_idx / fps, 3),
                    "class_id": int(c),
                    "class_name": model.names[int(c)],
                    "conf": round(float(cf), 4),
                    "x1": round(float(x1), 1),
                    "y1": round(float(y1), 1),
                    "x2": round(float(x2), 1),
                    "y2": round(float(y2), 1),
                    "w": round(bw, 1),
                    "h": round(bh, 1),
                    "area_px": round(area, 1),
                    "area_rel": round(area / frame_area, 6),
                })

        df = pd.DataFrame(rows, columns=[
            "video_id", "frame", "t_s", "class_id", "class_name", "conf",
            "x1", "y1", "x2", "y2", "w", "h", "area_px", "area_rel",
        ])
        out = OUTPUT_DIR / f"{video_id}_instruments_raw.csv"
        df.to_csv(out, index=False)
        print(f"Saved {len(df)} instrument detections -> {out.relative_to(REPO)}", flush=True)
        if len(df):
            print(df.groupby("class_name").agg(n=("conf", "size"),
                  conf_mean=("conf", "mean")).round(3).to_string(), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", default=",".join(DEFAULT_VIDEOS),
                    help="comma-separated video IDs or 'all'")
    args = ap.parse_args()
    main(resolve_videos(args.videos))
