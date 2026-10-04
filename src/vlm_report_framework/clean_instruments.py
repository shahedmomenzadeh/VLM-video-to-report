"""Clean YOLO instrument detections + aggregate per phase subchunk (Part I: I_{i,j}).

Pipeline:  output-instruments/<VID>_instruments_raw.csv (+ videos/<VID>/<VID>.timeline.json)
        -> output-instruments/<VID>_instruments_clean.csv   (one row per raw detection + status)
        -> output-instruments/<VID>_chunk_instruments.csv   (one row per chunk x instrument)

Cleaning algorithm (3 layers — intrinsic evidence first, phase prior only as a
soft flag, never a hard delete, so Idle/transition frames can't lose real data):

  L1 detection-level (drop with reason):
    - conf < 0.25                                    -> low_conf
    - box area < 0.1% of frame                       -> tiny_box
    - duplicate same-frame same-class box: keep max conf
  L2 track-level (gap-tolerant runs per class, gap <= 2 frames):
    - run length >= 3 frames (~0.6 s @ 5 fps)        -> keep (persistent)
    - run length 1-2 frames: keep only if conf >= 0.6, else -> fleeting
  L3 class-level (per video — kills hallucinated classes):
    - drop whole class if coverage < 0.5% of frames AND max run < 5
      AND median conf < 0.6                          -> weak_class

Phase prior (soft): each detection is tagged with its phase from the timeline
and whether its class is plausible there (P13 Idle allows everything, since it
holds transitions). Chunk tiers then follow exactly the teacher-prompt design:

    strong evidence + phase-plausible -> "observed"
    otherwise present                 -> "weak"  ("candidate but not confirmed")

Chunks: one per timeline segment; segments longer than ~1 min (the VLM's
effective short-video window) are split into ceil(dur/60) ~1-min subchunks.

Run: uv run python src/vlm_report_framework/clean_instruments.py
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

REPO = Path.cwd()
RAW_DIR = REPO / "output-instruments"

VIDEOS = ["PH_0001_2931_S2", "PH_0043_0096_S1", "PH_0057_0239_S1"]

# --- tunables (thresholds chosen from the raw-detection analysis, see notes) ---
CONF_MIN = 0.25
AREA_MIN = 0.001
TRACK_GAP = 2        # max missed frames inside one track
TRACK_MIN = 3        # frames for a persistent track
FLEETING_CONF = 0.6  # lone/paired detections kept only above this
CLASS_COV_MIN = 0.005
CLASS_RUN_MIN = 5
CLASS_CONF_MED = 0.6
CHUNK_SPLIT_S = 65.0  # segments longer than this are split into ~1-min subchunks

# Instruments plausibly visible in each phase (surgical prior; P13 Idle = all,
# it contains entries/exits between phases). Used ONLY to tier evidence, never
# to delete detections.
EXPECTED: dict[str, set[str]] = {
    "P01": {"Primary-Knife", "Secondary-Knife", "Forceps", "Second-Instrument", "Cannula"},
    "P02": {"Cannula", "Forceps", "Second-Instrument"},
    "P03": {"Cap-Cystotome", "Cap-Forceps", "Forceps", "Second-Instrument"},
    "P04": {"Cannula", "Forceps", "Second-Instrument"},
    "P05": {"Phaco-Handpiece", "Second-Instrument", "Forceps"},
    "P06": {"I-A-Handpiece", "Second-Instrument"},
    "P07": {"I-A-Handpiece", "Cannula", "Second-Instrument"},
    "P08": {"Lens-Injector", "Forceps", "Second-Instrument"},
    "P09": {"Second-Instrument", "Cannula", "Forceps", "I-A-Handpiece"},
    "P10": {"I-A-Handpiece", "Cannula"},
    "P11": {"Cannula", "Second-Instrument"},
    "P12": {"Cannula", "Second-Instrument"},
    "P13": set(),  # Idle: everything plausible (transitions)
}


def build_tracks(frames: list[int]) -> list[list[int]]:
    """Group sorted frame ids into runs allowing gaps <= TRACK_GAP."""
    runs, cur = [], []
    for f in sorted(frames):
        if cur and f - cur[-1] > TRACK_GAP + 1:
            runs.append(cur)
            cur = []
        cur.append(f)
    if cur:
        runs.append(cur)
    return runs


def load_timeline(video_id: str) -> tuple[list[dict], float]:
    tl = json.load(open(REPO / "videos" / video_id / f"{video_id}.timeline.json"))
    segments = tl["segments"]
    fps_guess = 5.0
    return segments, fps_guess


def make_chunks(segments: list[dict]) -> list[dict]:
    """One chunk per timeline segment; long segments split into ~1-min pieces."""
    chunks = []
    for s in segments:
        dur = s["end_s"] - s["start_s"]
        n = max(1, round(dur / 60.0)) if dur > CHUNK_SPLIT_S else 1
        for k in range(n):
            a = s["start_s"] + dur * k / n
            b = s["start_s"] + dur * (k + 1) / n
            cid = f"{s['phase']}_o{s['occurrence']:02d}" + (f"_c{k + 1}" if n > 1 else "")
            chunks.append({"chunk_id": cid, "phase": s["phase"],
                           "phase_name": s["phase_name"], "t_start": a, "t_end": b})
    return chunks


def phase_at(segments: list[dict], t: float) -> dict:
    for s in segments:
        if s["start_s"] <= t < s["end_s"] or (t == s["end_s"] == segments[-1]["end_s"]):
            return s
    return segments[-1]


def clean_video(video_id: str) -> None:
    raw = pd.read_csv(RAW_DIR / f"{video_id}_instruments_raw.csv")
    segments, _ = load_timeline(video_id)
    n_video_frames = int(raw["frame"].max()) + 1

    df = raw.copy()
    df["status"] = "keep"
    df["drop_reason"] = ""

    # L1a: duplicate same-frame same-class boxes -> keep max conf
    df = df.sort_values("conf").drop_duplicates(["frame", "class_name"], keep="last")
    # L1b/c: confidence + size floors
    m = df["conf"] < CONF_MIN
    df.loc[m, ["status", "drop_reason"]] = ("dropped", "low_conf")
    m = (df["status"] == "keep") & (df["area_rel"] < AREA_MIN)
    df.loc[m, ["status", "drop_reason"]] = ("dropped", "tiny_box")

    # L2: track-level persistence on survivors
    keep = df["status"] == "keep"
    run_len_of = {}  # (class, frame) -> run length
    max_run = {}
    for cls, gd in df[keep].groupby("class_name"):
        runs = build_tracks(gd["frame"].tolist())
        max_run[cls] = max((len(r) for r in runs), default=0)
        for r in runs:
            for f in r:
                run_len_of[(cls, f)] = len(r)
    df["run_len"] = [run_len_of.get((c, f), 0) for c, f in zip(df["class_name"], df["frame"])]
    m = (df["status"] == "keep") & (df["run_len"] < TRACK_MIN) & (df["conf"] < FLEETING_CONF)
    df.loc[m, ["status", "drop_reason"]] = ("dropped", "fleeting")

    # L3: class-level support (kills hallucinated classes like Secondary-Knife)
    kept = df["status"] == "keep"
    for cls, gd in df[kept].groupby("class_name"):
        cov = gd["frame"].nunique() / n_video_frames
        if cov < CLASS_COV_MIN and max_run.get(cls, 0) < CLASS_RUN_MIN \
                and gd["conf"].median() < CLASS_CONF_MED:
            df.loc[(df["class_name"] == cls) & (df["status"] == "keep"),
                   ["status", "drop_reason"]] = ("dropped", "weak_class")

    # Phase tag (soft prior only)
    segs_of_t = [phase_at(segments, t) for t in df["t_s"]]
    df["phase"] = [s["phase"] for s in segs_of_t]
    df["phase_name"] = [s["phase_name"] for s in segs_of_t]
    df["phase_expected"] = [
        True if ph == "P13" else (cls in EXPECTED.get(ph, set()))
        for ph, cls in zip(df["phase"], df["class_name"])
    ]

    df.to_csv(RAW_DIR / f"{video_id}_instruments_clean.csv", index=False)

    # ---- chunk aggregation: tiers for the teacher-VLM prompt ----
    chunks = make_chunks(segments)
    kept_df = df[df["status"] == "keep"]
    rows = []
    for ch in chunks:
        sub = kept_df[(kept_df["t_s"] >= ch["t_start"]) & (kept_df["t_s"] < ch["t_end"])]
        for cls, gd in sub.groupby("class_name"):
            frames = sorted(gd["frame"].unique())
            runs = build_tracks(frames)
            n = len(frames)
            maxr = max((len(r) for r in runs), default=0)
            medc = round(float(gd["conf"].median()), 3)
            exp = bool(gd["phase_expected"].all())
            strong = (n >= 8) or (n >= 5 and (maxr >= TRACK_MIN or medc >= 0.7))
            tier = "observed" if (strong and exp) else "weak"
            rows.append({
                "video_id": video_id, "chunk_id": ch["chunk_id"],
                "phase": ch["phase"], "phase_name": ch["phase_name"],
                "t_start": round(ch["t_start"], 1), "t_end": round(ch["t_end"], 1),
                "class_name": cls, "n_frames": n,
                "frac": round(n / max(1, int((ch["t_end"] - ch["t_start"]) * 5)), 3),
                "max_run": maxr, "med_conf": medc,
                "phase_expected": exp, "tier": tier,
            })
    chunks_df = pd.DataFrame(rows)
    chunks_df.to_csv(RAW_DIR / f"{video_id}_chunk_instruments.csv", index=False)

    # ---- console report ----
    n_raw, n_keep = len(df), int((df["status"] == "keep").sum())
    print(f"\n{'=' * 70}\n{video_id}: {n_raw} raw -> {n_keep} kept "
          f"({n_keep / max(1, n_raw):.1%})")
    print("dropped by reason:")
    print(df[df.status == "dropped"].groupby(["class_name", "drop_reason"]).size().to_string())
    print(f"\n{len(chunks)} chunks, {len(chunks_df)} chunk-instrument rows "
          f"({int((chunks_df.tier == 'observed').sum())} observed / "
          f"{int((chunks_df.tier == 'weak').sum())} weak)")
    print("per-class kept frames:")
    print(kept_df.groupby("class_name")["frame"].nunique().sort_values(ascending=False).to_string())


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for vid in VIDEOS:
        clean_video(vid)


if __name__ == "__main__":
    main()
