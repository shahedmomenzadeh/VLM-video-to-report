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
- Run: uv run python src/vlm_report_framework/extract_instruments.py
"""
from pathlib import Path

import cv2
import pandas as pd
from ultralytics import YOLO

REPO = Path.cwd()
MODEL_PATH = REPO / "yolo-model" / "model.pt"
OUTPUT_DIR = REPO / "output-instruments"

VIDEOS = [
    REPO / "videos" / "PH_0001_2931_S2" / "PH_0001_2931_S2.mp4",  # S2
    REPO / "videos" / "PH_0043_0096_S1" / "PH_0043_0096_S1.mp4",  # S1
    REPO / "videos" / "PH_0057_0239_S1" / "PH_0057_0239_S1.mp4",  # S1
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


def main():
    assert MODEL_PATH.exists(), f"Model not found: {MODEL_PATH}"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(MODEL_PATH))
    print(f"Task: {model.task}, classes: {model.names}")

    for video in VIDEOS:
        video_id = video.stem
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
    main()
