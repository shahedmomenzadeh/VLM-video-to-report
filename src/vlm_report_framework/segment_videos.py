"""Run YOLO segmentation on surgical videos.

Model: yolo-model/model.pt (ultralytics segment, 12 classes)
Output: annotated videos in output-segmented/ (unless --no-save).

Usage:
  uv run python src/vlm_report_framework/segment_videos.py                  # default 3-video sample
  uv run python src/vlm_report_framework/segment_videos.py --videos all    # all videos/ dirs
  uv run python src/vlm_report_framework/segment_videos.py --videos PH_0001_2931_S2,PH_0043_0096_S1
  uv run python src/vlm_report_framework/segment_videos.py --no-save       # inference only, no annotated video
"""
import argparse
from pathlib import Path
from ultralytics import YOLO

BASE = Path(__file__).resolve().parents[2] if "__file__" in globals() else Path.cwd()
# Fallback: repo root is cwd when run as script from repo root
REPO = Path.cwd()
MODEL_PATH = REPO / "yolo-model" / "model.pt"
OUTPUT_DIR = REPO / "output-segmented"

DEFAULT_VIDEOS = [
    "PH_0001_2931_S2",  # S2
    "PH_0043_0096_S1",  # S1 (full 13-phase coverage)
    "PH_0057_0239_S1",  # S1
]


def discover_videos() -> list[str]:
    return sorted(p.name for p in (REPO / "videos").glob("PH_*") if p.is_dir())


def resolve_videos(spec: str) -> list[Path]:
    if spec == "all":
        ids = discover_videos()
    else:
        ids = [v.strip() for v in spec.split(",") if v.strip()]
    paths = [REPO / "videos" / vid / f"{vid}.mp4" for vid in ids]
    for v in paths:
        assert v.exists(), f"Video not found: {v}"
    return paths

def main(videos: list[Path], save: bool):
    assert MODEL_PATH.exists(), f"Model not found: {MODEL_PATH}"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(MODEL_PATH))
    print(f"Task: {model.task}, classes: {model.names}")

    for video in videos:
        if save:
            existing = OUTPUT_DIR / video.stem / f"{video.stem}.avi"
            if existing.exists():
                print(f"-- skip {video.name} (already segmented: {existing})", flush=True)
                continue
        print(f"\n=== Segmenting {video.name} (save={save}) ===", flush=True)
        count = 0
        # stream=True avoids accumulating all Results (with masks) in RAM,
        # which caused OOM/SIGBUS on the previous run.
        for _ in model.predict(
            source=str(video),
            save=save,              # save annotated video (skip with --no-save)
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", default=",".join(DEFAULT_VIDEOS),
                    help="comma-separated video IDs or 'all'")
    ap.add_argument("--no-save", action="store_true",
                    help="run inference without saving annotated video")
    args = ap.parse_args()
    main(resolve_videos(args.videos), save=not args.no_save)
