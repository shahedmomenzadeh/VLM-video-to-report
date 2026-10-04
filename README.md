# VLM-video-to-report

Generate surgical reports from video using a vision-language model (VLM) pipeline.
Current stage: YOLO segmentation of cataract surgery videos (instrument + anatomy masks).

## Setup

Requires Python ≥ 3.13 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

## Usage

### 1. Segment videos with YOLO

Place your segmentation model at `yolo-model/model.pt` and input videos under
`videos/<VIDEO_ID>/<VIDEO_ID>.mp4` —
video/model/output folders are git-ignored (large + private). The
`videos/README.md` dataset doc is not pushed (it lives next to the ignored data).

Run segmentation on the configured video list:

```bash
uv run python src/vlm_report_framework/segment_videos.py
```

Annotated videos (masks + boxes) are written to `output-segmented/<VIDEO_ID>/`
(git-ignored).

### Model

- Ultralytics YOLO **segmentation** (`yolo-model/model.pt`, git-ignored)
- Classes (12): `Cannula`, `Cap-Cystotome`, `Cap-Forceps`, `Cornea`, `Forceps`,
  `I-A-Handpiece`, `Lens-Injector`, `Phaco-Handpiece`, `Primary-Knife`, `Pupil`,
  `Second-Instrument`, `Secondary-Knife`

## Layout

```text
├── src/vlm_report_framework/
│   └── segment_videos.py   # YOLO segmentation runner (streamed inference)
├── videos/                 # input videos, git-ignored
├── yolo-model/             # model.pt, git-ignored
└── output-segmented/       # annotated outputs, git-ignored
```

## Notes

- Inference uses `stream=True` + CUDA (`device=0`) so per-frame mask results are
  not accumulated in RAM (avoids OOM/SIGBUS on long videos).
