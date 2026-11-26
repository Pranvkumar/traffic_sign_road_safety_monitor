"""End-to-end image inference: sign detection -> crop classification -> road mask -> overlay + JSON.

Missing checkpoints raise clear errors; the pipeline never invents predictions.
A component can be disabled explicitly (e.g. segmentation) and is then reported as
"not loaded" — distinct from "loaded, nothing found".
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .. import DISCLAIMER
from ..data.transforms import preprocess_crop
from ..models.detector import SignDetector
from ..models.features import sign_features
from ..models.mlp_numpy import MLP, Standardizer
from ..models.segmenter import RoadSegmenter
from ..utils.config import load_config, resolve

SUPPORTED_IMAGES = {".jpg", ".jpeg", ".png", ".bmp", ".ppm", ".webp"}


def read_image(path: str | Path) -> np.ndarray:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input not found: {path}")
    if path.suffix.lower() not in SUPPORTED_IMAGES:
        raise ValueError(f"Unsupported image type '{path.suffix}'. Supported: {sorted(SUPPORTED_IMAGES)}")
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise OSError(f"Could not decode image: {path}")
    return img


class SignClassifier:
    """HOG-MLP crop classifier with an explicit 'unknown' outcome."""

    def __init__(self, checkpoint: str | Path, unknown_threshold: float = 0.6):
        self.model, extras, self.meta = MLP.load(checkpoint)
        self.std = Standardizer(extras["mean"], extras["std"])
        self.names = {int(k): v for k, v in self.meta.get("class_names", {}).items()}
        self.size = int(self.meta.get("preprocessing", {}).get("crop_size", 32))
        self.unknown_threshold = unknown_threshold

    def classify(self, crops_bgr: list[np.ndarray]) -> list[dict]:
        if not crops_bgr:
            return []
        x = np.stack([preprocess_crop(c, self.size) for c in crops_bgr])
        proba = self.model.predict_proba(self.std.transform(sign_features(x)))
        out = []
        for p in proba:
            k = int(p.argmax())
            known = float(p[k]) >= self.unknown_threshold
            out.append({"class_id": k if known else -1, "class_name": self.names.get(k, str(k)) if known else "unknown",
                        "class_confidence": float(p[k]), "top_guess": self.names.get(k, str(k))})
        return out


@dataclass
class Pipeline:
    detector: object | None          # SignDetector (NumPy baseline) or YoloSignDetector
    classifier: SignClassifier | None
    segmenter: RoadSegmenter | None
    config: dict = field(default_factory=dict)
    hazard: object | None = None     # YOLO pothole detector (HRP4K); None = not configured / checkpoint missing
    hazard_note: str = "not available (no hazard model configured)"

    def _attach_hazard(self) -> "Pipeline":
        hcfg = self.config.get("hazard") or {}
        if not hcfg.get("checkpoint"):
            return self
        ckpt = resolve(hcfg["checkpoint"])
        if not Path(ckpt).exists():
            self.hazard_note = f"not available (checkpoint missing: {hcfg['checkpoint']}; see scripts/train_hazard.py)"
            return self
        from ..models.yolo_detector import YoloSignDetector
        self.hazard = YoloSignDetector(ckpt, score_threshold=float(hcfg.get("score_threshold", 0.3)),
                                       nms_iou=float(hcfg.get("nms_iou", 0.5)),
                                       max_detections=int(hcfg.get("max_detections", 30)),
                                       imgsz=int(hcfg.get("imgsz", 960)), device=hcfg.get("device"),
                                       unknown_threshold=0.0)
        self.hazard_note = "loaded (YOLO, potholes - HRP4K)"
        return self

    @classmethod
    def from_config(cls, config_path: str = "configs/default.yaml", use_segmenter: bool = True) -> "Pipeline":
        cfg = load_config(config_path)
        dcfg, ccfg = cfg["detector"], cfg.get("classifier", {})
        seg = None
        if use_segmenter and cfg.get("segmenter", {}).get("checkpoint"):
            seg = RoadSegmenter.load(resolve(cfg["segmenter"]["checkpoint"]))
        if dcfg.get("type", "numpy") == "yolo":
            # One-stage detector that also names the sign; no crop classifier needed.
            from ..models.yolo_detector import YoloSignDetector
            det = YoloSignDetector(resolve(dcfg["checkpoint"]),
                                   score_threshold=float(dcfg.get("score_threshold") or 0.25),
                                   nms_iou=float(dcfg.get("nms_iou", 0.5)),
                                   max_detections=int(dcfg.get("max_detections", 50)),
                                   imgsz=int(dcfg.get("imgsz", 1280)), device=dcfg.get("device"),
                                   unknown_threshold=float(dcfg.get("unknown_threshold", 0.4)))
            return cls(det, None, seg, cfg)._attach_hazard()
        det = SignDetector.load(resolve(dcfg["checkpoint"]))
        if dcfg.get("score_threshold") is not None:
            det.score_threshold = float(dcfg["score_threshold"])
        det.nms_iou = float(dcfg.get("nms_iou", det.nms_iou))
        det.max_detections = int(dcfg.get("max_detections", det.max_detections))
        clf = SignClassifier(resolve(ccfg["checkpoint"]), float(ccfg.get("unknown_threshold", 0.6)))
        return cls(det, clf, seg, cfg)._attach_hazard()

    @property
    def is_yolo(self) -> bool:
        return hasattr(self.detector, "detect_and_classify")

    def status(self) -> dict:
        det = "not loaded" if self.detector is None else ("loaded (YOLO, ETSC 171 classes)" if self.is_yolo
                                                          else "loaded (NumPy HOG baseline)")
        clf = ("built into YOLO detector" if self.is_yolo else
               "loaded" if self.classifier else "not loaded")
        return {"detector": det,
                "classifier": clf,
                "segmenter": "loaded" if self.segmenter else "not loaded",
                "hazard_detector": self.hazard_note}

    def predict(self, img_bgr: np.ndarray) -> dict:
        if img_bgr is None or img_bgr.size == 0:
            raise ValueError("Empty image")
        if self.detector is None:
            raise RuntimeError("Detector not loaded — cannot run sign detection.")
        t = {}
        t0 = time.perf_counter()
        if self.is_yolo:
            boxes, scores, labels = self.detector.detect_and_classify(img_bgr)
            t["detection_ms"] = (time.perf_counter() - t0) * 1000
            t["classification_ms"] = 0.0
        else:
            boxes, scores = self.detector.detect(img_bgr)
            t["detection_ms"] = (time.perf_counter() - t0) * 1000
            crops = []
            h, w = img_bgr.shape[:2]
            for b in boxes:
                x1, y1, x2, y2 = np.clip(b, 0, [w, h, w, h]).astype(int)
                crops.append(img_bgr[y1:max(y2, y1 + 1), x1:max(x2, x1 + 1)])
            t0 = time.perf_counter()
            labels = self.classifier.classify(crops) if self.classifier else [{} for _ in crops]
            t["classification_ms"] = (time.perf_counter() - t0) * 1000
        road = None
        if self.segmenter is not None:
            t0 = time.perf_counter()
            mask = self.segmenter.predict_mask(img_bgr)
            t["segmentation_ms"] = (time.perf_counter() - t0) * 1000
            road = {"mask": mask, "road_coverage": float(mask.mean())}
        dets = [{"box_xyxy": [round(float(v), 1) for v in b], "detection_score": float(s), **lab}
                for b, s, lab in zip(boxes, scores, labels)]
        hazards = None                                     # None = not checked; [] = checked, nothing found
        if self.hazard is not None:
            t0 = time.perf_counter()
            hb, hs, hl = self.hazard.detect_and_classify(img_bgr)
            t["hazard_ms"] = (time.perf_counter() - t0) * 1000
            hazards = [{"box_xyxy": [round(float(v), 1) for v in b], "score": float(s),
                        "type": lab.get("top_guess", "hazard")} for b, s, lab in zip(hb, hs, hl)]
        t["total_ms"] = sum(t.values())
        return {"detections": dets, "n_signs": len(dets), "road": road,
                "hazards": hazards, "n_hazards": None if hazards is None else len(hazards), "timing_ms": t,
                "status": self.status(), "disclaimer": DISCLAIMER}


def draw(img_bgr: np.ndarray, result: dict) -> np.ndarray:
    vis = img_bgr.copy()
    if result.get("road") is not None:
        m = result["road"]["mask"].astype(bool)
        vis[m] = (0.6 * vis[m] + 0.4 * np.array([255, 0, 255])).astype(np.uint8)
    s = max(vis.shape[:2]) / 1000.0                       # scale text/lines to the image size
    fs, th = max(0.4, 0.5 * s), max(1, int(round(1.5 * s)))
    for i, d in enumerate(result["detections"], 1):
        x1, y1, x2, y2 = map(int, d["box_xyxy"])
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), max(2, th))
        label = f"#{i} {d.get('class_name', 'sign')} {d.get('class_confidence', d['detection_score']):.2f}"
        (tw, tht), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, th)
        ty = y1 - 4 if y1 - tht - 6 > 0 else y2 + tht + 4
        cv2.rectangle(vis, (x1, ty - tht - 4), (x1 + tw + 4, ty + 4), (0, 0, 255), -1)
        cv2.putText(vis, label, (x1 + 2, ty), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), th, cv2.LINE_AA)
    for d in result.get("hazards") or []:                 # potholes: orange boxes
        x1, y1, x2, y2 = map(int, d["box_xyxy"])
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 140, 255), max(2, th))
        label = f"{d['type']} {d['score']:.2f}"
        (tw, tht), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, th)
        ty = y2 + tht + 4 if y2 + tht + 8 < vis.shape[0] else y1 - 4
        cv2.rectangle(vis, (x1, ty - tht - 4), (x1 + tw + 4, ty + 4), (0, 140, 255), -1)
        cv2.putText(vis, label, (x1 + 2, ty), cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), th, cv2.LINE_AA)
    hz = "" if result.get("n_hazards") is None else f"  hazards: {result['n_hazards']}"
    msg = f"signs: {result['n_signs']}{hz}  {result['timing_ms']['total_ms']:.0f} ms  | RESEARCH DEMO - NOT SAFETY CERTIFIED"
    cv2.putText(vis, msg, (10, int(25 * max(1, s))), cv2.FONT_HERSHEY_SIMPLEX, 0.6 * max(1, s), (255, 255, 255),
                th + 1, cv2.LINE_AA)
    return vis


def to_json(result: dict) -> dict:
    """Strip the mask array so the result can be written as JSON."""
    out = {k: v for k, v in result.items() if k != "road"}
    out["road"] = None if result.get("road") is None else {"road_coverage": result["road"]["road_coverage"]}
    return out
