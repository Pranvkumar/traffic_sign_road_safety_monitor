"""Latency and memory benchmark of the full image pipeline on the current machine.

This is NOT an edge-device benchmark: results describe the machine it ran on.
"""

from __future__ import annotations

import os
import platform
import resource
import time

import numpy as np

from ..data.manifest import load_manifest
from ..utils.config import load_config, resolve
from .inference import Pipeline, read_image


def run(config_path: str = "configs/default.yaml", repeats: int = 20) -> dict:
    dcfg = load_config("configs/data.yaml")["idd_signs"]
    df = load_manifest(resolve(dcfg["manifest"]))
    rows = df[df.split == "test"].sample(min(repeats, int((df.split == "test").sum())), random_state=0)
    images = [read_image(resolve(dcfg["root"]) / r) for r in rows["relative_path"]]
    pipe = Pipeline.from_config(config_path)
    pipe.predict(images[0])                                   # warm-up (not timed)
    stages: dict[str, list[float]] = {}
    for img in images:
        t0 = time.perf_counter()
        res = pipe.predict(img)
        wall = (time.perf_counter() - t0) * 1000
        for k, v in {**res["timing_ms"], "end_to_end_wall_ms": wall}.items():
            stages.setdefault(k, []).append(v)
    peak_rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    summary = {k: {"median": float(np.median(v)), "p90": float(np.percentile(v, 90))} for k, v in stages.items()}
    return {"input": "IDD test images 1280x964, batch size 1", "repeats": len(images), "warm_up_runs": 1,
            "latency_ms": summary, "fps_end_to_end": 1000 / summary["end_to_end_wall_ms"]["median"],
            "peak_rss_mb": round(peak_rss_mb, 1),
            "hardware": {"machine": platform.machine(), "processor": platform.processor() or "unknown",
                         "cpu_count": os.cpu_count(), "gpu": "none"},
            "precision": "float32/float64 NumPy, CPU only",
            "note": "Cloud CPU container used for development; not an edge-device measurement."}
