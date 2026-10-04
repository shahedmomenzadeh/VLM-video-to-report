"""Run YOLO segmentation on a mixed S1/S2 sample of 3 videos.

Model: yolo-model/model.pt (ultralytics segment, 12 classes)
Output: annotated videos in output-segmented/
"""
from pathlib import Path
from ultralytics import YOLO

BASE = Path(__file__).resolve().parents[2] if "__file__" in globals() else Path.cwd()
# Fallback: repo root is cwd when run as script from repo root
REPO = Path.cwd()
MODEL_PATH = REPO / "yolo-model" / "model.pt"
OUTPUT_DIR = REPO / "output-segmented"

VIDEOS = [
    REPO / "videos" / "PH_0001_2931_S2" / "PH_0001_2931_S2.mp4",  # S2
    REPO / "videos" / "PH_0043_0096_S1" / "PH_0043_0096_S1.mp4",  # S1 (full 13-phase coverage)
    REPO / "videos" / "PH_0057_0239_S1" / "PH_0057_0239_S1.mp4",  # S1
]

def main():
    assert MODEL_PATH.exists(), f"Model not found: {MODEL_PATH}"
    for v in VIDEOS:
        assert v.exists(), f"Video not found: {v}"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(MODEL_PATH))
    print(f"Task: {model.task}, classes: {model.names}")

    for video in VIDEOS:
        print(f"\n=== Segmenting {video.name} ===", flush=True)
        count = 0
        # stream=True avoids accumulating all Results (with masks) in RAM,
        # which caused OOM/SIGBUS on the previous run.
        for _ in model.predict(
            source=str(video),
            save=True,            # save annotated video
            project=str(OUTPUT_DIR),
            name=video.stem,      # output-segmented/<VIDEO_ID>/
            exist_ok=True,
            device=0,             # CUDA GPU (available)
            verbose=False,
            stream=True,
        ):
            count += 1
        print(f"Done {video.name}: {count} frames processed", flush=True)

    print("\nAll videos segmented. Outputs:")
    for p in sorted(OUTPUT_DIR.rglob("*")):
        if p.is_file():
            print(f"  {p.relative_to(REPO)} ({p.stat().st_size / 1e6:.1f} MB)")

if __name__ == "__main__":
    main()
