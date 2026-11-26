"""YOLO (Ultralytics) traffic-sign detector + classifier in one stage.

Why this exists: the NumPy HOG sliding-window detector reached only AP@0.5 ~= 0.01 on the
held-out split (docs/STATUS.md), so boxes were mostly wrong and the crop classifier then
received background patches and answered "unknown". A one-stage detector trained on the
European Traffic Sign Collection (ETSC, 171 classes, YOLO format) localises AND names the
sign, so the pipeline no longer depends on a separate crop classifier.

Optional dependency: `ultralytics` (+ torch). Import errors are raised with a clear message.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


class YoloSignDetector:
    """Wraps an Ultralytics checkpoint and returns boxes, scores and per-box labels."""

    def __init__(self, checkpoint: str | Path, score_threshold: float = 0.25, nms_iou: float = 0.5,
                 max_detections: int = 50, imgsz: int = 1280, device: str | None = None,
                 unknown_threshold: float = 0.4):
        checkpoint = Path(checkpoint)
        if not checkpoint.exists():
            raise FileNotFoundError(f"YOLO checkpoint not found: {checkpoint} "
                                    f"(train it with scripts/train_yolo.py)")
        try:
            from ultralytics import YOLO
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise ImportError("The YOLO detector needs `pip install ultralytics torch`.") from exc
        self.model = YOLO(str(checkpoint))
        self.names = {int(k): str(v).strip() for k, v in self.model.names.items()}
        self.score_threshold = score_threshold
        self.nms_iou = nms_iou
        self.max_detections = max_detections
        self.imgsz = imgsz
        self.device = device
        self.unknown_threshold = unknown_threshold

    def detect_and_classify(self, img_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[dict]]:
        """Return (boxes xyxy [N,4], scores [N], labels). Ultralytics expects BGR numpy input."""
        r = self.model.predict(img_bgr, conf=self.score_threshold, iou=self.nms_iou, agnostic_nms=True, imgsz=self.imgsz,
                               max_det=self.max_detections, device=self.device, verbose=False)[0]
        if r.boxes is None or len(r.boxes) == 0:
            return np.zeros((0, 4), np.float32), np.zeros(0, np.float32), []
        boxes = r.boxes.xyxy.cpu().numpy().astype(np.float32)
        scores = r.boxes.conf.cpu().numpy().astype(np.float32)
        cls = r.boxes.cls.cpu().numpy().astype(int)
        order = np.argsort(-scores)
        boxes, scores, cls = boxes[order], scores[order], cls[order]
        labels = []
        for k, s in zip(cls, scores):
            name = self.names.get(int(k), str(k))
            known = float(s) >= self.unknown_threshold
            labels.append({"class_id": int(k) if known else -1,
                           "class_name": name if known else "unknown",
                           "class_confidence": float(s), "top_guess": name})
        return boxes, scores, labels
