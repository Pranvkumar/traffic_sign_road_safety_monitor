"""Frame-by-frame video inference with annotated output video and per-frame JSON lines."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from .inference import Pipeline, draw, to_json

SUPPORTED_VIDEOS = {".mp4", ".avi", ".mov", ".mkv"}


def predict_video(pipeline: Pipeline, source: str | Path, output_dir: str | Path, frame_stride: int = 1,
                  max_frames: int | None = None) -> dict:
    source = Path(source)
    if not source.exists():
        raise FileNotFoundError(f"Video not found: {source}")
    if source.suffix.lower() not in SUPPORTED_VIDEOS:
        raise ValueError(f"Unsupported video type '{source.suffix}'. Supported: {sorted(SUPPORTED_VIDEOS)}")
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise OSError(f"Could not open video: {source}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 10.0
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_video = out_dir / f"{source.stem}_annotated.mp4"
    out_jsonl = out_dir / f"{source.stem}_frames.jsonl"
    writer, idx, processed, totals = None, 0, 0, []
    signs_per_frame = []
    with out_jsonl.open("w", encoding="utf-8") as fh:
        while True:
            ok, frame = cap.read()
            if not ok:                                   # end of file (or unreadable frame) -> stop cleanly
                break
            idx += 1
            if (idx - 1) % frame_stride == 0:
                res = pipeline.predict(frame)
                vis = draw(frame, res)
                if writer is None:
                    h, w = vis.shape[:2]
                    writer = cv2.VideoWriter(str(out_video), cv2.VideoWriter_fourcc(*"mp4v"), fps / frame_stride, (w, h))
                writer.write(vis)
                fh.write(json.dumps({"frame_index": idx - 1, "time_s": (idx - 1) / fps, **to_json(res)}) + "\n")
                totals.append(res["timing_ms"]["total_ms"])
                signs_per_frame.append(res["n_signs"])
                processed += 1
                if max_frames and processed >= max_frames:
                    break
    cap.release()
    if writer is not None:
        writer.release()
    return {"source": str(source), "frames_read": idx, "frames_processed": processed,
            "annotated_video": str(out_video) if processed else None, "frame_log": str(out_jsonl),
            "median_total_ms": float(np.median(totals)) if totals else None,
            "frames_with_signs": int(np.sum(np.array(signs_per_frame) > 0)) if signs_per_frame else 0}
