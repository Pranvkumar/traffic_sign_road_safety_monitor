"""Bounding-box annotation reading and conversion.

Internal canonical box format: pixel ``[x1, y1, x2, y2]`` (float).
COCO uses ``[x, y, w, h]``; YOLO uses normalised ``[cx, cy, w, h]``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def read_yolo_file(path: str | Path) -> np.ndarray:
    """Return an (N, 5) array of ``class cx cy w h`` rows. Missing file -> error."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Annotation file missing: {path}")
    rows = []
    for line_no, line in enumerate(path.read_text().splitlines(), 1):
        parts = line.split()
        if not parts:
            continue
        if len(parts) != 5:
            raise ValueError(f"{path}:{line_no}: expected 5 values, got {len(parts)}")
        rows.append([float(v) for v in parts])
    return np.array(rows, dtype=np.float64).reshape(-1, 5)


def yolo_to_xyxy(yolo: np.ndarray, width: int, height: int) -> np.ndarray:
    """Normalised (cx, cy, w, h) -> pixel (x1, y1, x2, y2)."""
    yolo = np.asarray(yolo, dtype=np.float64).reshape(-1, 4)
    cx, cy, w, h = yolo[:, 0] * width, yolo[:, 1] * height, yolo[:, 2] * width, yolo[:, 3] * height
    return np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)


def xyxy_to_yolo(boxes: np.ndarray, width: int, height: int) -> np.ndarray:
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    w, h = boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]
    return np.stack([(boxes[:, 0] + w / 2) / width, (boxes[:, 1] + h / 2) / height, w / width, h / height], axis=1)


def xyxy_to_coco(boxes: np.ndarray) -> np.ndarray:
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    return np.stack([boxes[:, 0], boxes[:, 1], boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]], axis=1)


def clip_boxes(boxes: np.ndarray, width: int, height: int) -> np.ndarray:
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4).copy()
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, width)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, height)
    return boxes


def validate_yolo(rows: np.ndarray, known_classes: set[int]) -> list[str]:
    """Return a list of human-readable problems with YOLO rows (empty list == OK)."""
    problems = []
    for i, (cls, cx, cy, w, h) in enumerate(np.asarray(rows).reshape(-1, 5)):
        if int(cls) not in known_classes:
            problems.append(f"row {i}: unknown class {int(cls)}")
        if not (0 <= cx <= 1 and 0 <= cy <= 1):
            problems.append(f"row {i}: centre outside [0,1]")
        if w <= 0 or h <= 0:
            problems.append(f"row {i}: zero/negative size")
        elif cx - w / 2 < -1e-6 or cx + w / 2 > 1 + 1e-6 or cy - h / 2 < -1e-6 or cy + h / 2 > 1 + 1e-6:
            problems.append(f"row {i}: box extends outside image")
        elif max(w / h, h / w) > 6:
            problems.append(f"row {i}: implausible aspect ratio {w / h:.2f}")
    return problems
