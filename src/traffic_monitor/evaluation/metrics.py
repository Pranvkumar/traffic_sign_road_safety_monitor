"""Evaluation metrics written out explicitly so they can be checked by hand in tests."""

from __future__ import annotations

import numpy as np


# ---------------- classification ----------------
def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(cm, (np.asarray(y_true), np.asarray(y_pred)), 1)
    return cm


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> dict:
    cm = confusion_matrix(y_true, y_pred, n_classes)
    tp = np.diag(cm).astype(float)
    support = cm.sum(1).astype(float)
    pred_count = cm.sum(0).astype(float)
    precision = np.divide(tp, pred_count, out=np.zeros_like(tp), where=pred_count > 0)
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(tp), where=(precision + recall) > 0)
    present = support > 0
    w = support / support.sum()
    return {
        "accuracy": float(tp.sum() / cm.sum()),
        "macro_precision": float(precision[present].mean()),
        "macro_recall": float(recall[present].mean()),
        "macro_f1": float(f1[present].mean()),
        "weighted_f1": float((f1 * w).sum()),
        "per_class": {int(c): {"precision": float(precision[c]), "recall": float(recall[c]),
                               "f1": float(f1[c]), "support": int(support[c])} for c in range(n_classes)},
        "confusion_matrix": cm.tolist(),
    }


def top_k_accuracy(proba: np.ndarray, y_true: np.ndarray, k: int = 5) -> float:
    topk = np.argsort(-proba, axis=1)[:, :k]
    return float(np.mean([y in row for y, row in zip(y_true, topk)]))


def expected_calibration_error(proba: np.ndarray, y_true: np.ndarray, n_bins: int = 10) -> float:
    conf, pred = proba.max(1), proba.argmax(1)
    correct = (pred == y_true).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def binary_metrics(y_true: np.ndarray, score: np.ndarray, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true).astype(int)
    pred = (np.asarray(score) >= threshold).astype(int)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {"precision": p, "recall": r, "f1": 2 * p * r / (p + r) if p + r else 0.0,
            "brier": float(np.mean((np.asarray(score) - y_true) ** 2))}


# ---------------- detection ----------------
def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between (N,4) and (M,4) xyxy boxes -> (N,M)."""
    a, b = np.asarray(a, float).reshape(-1, 4), np.asarray(b, float).reshape(-1, 4)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


def match_detections(pred_boxes: np.ndarray, scores: np.ndarray, gt_boxes: np.ndarray,
                     iou_thr: float = 0.5) -> np.ndarray:
    """Greedy one-to-one matching in descending score order. Returns a bool TP flag per prediction."""
    order = np.argsort(-np.asarray(scores))
    tp = np.zeros(len(order), dtype=bool)
    if len(gt_boxes) == 0 or len(order) == 0:
        return tp
    ious = box_iou(np.asarray(pred_boxes)[order], gt_boxes)
    used = np.zeros(len(gt_boxes), dtype=bool)
    for i in range(len(order)):
        cand = np.where(~used & (ious[i] >= iou_thr))[0]
        if len(cand):
            j = cand[np.argmax(ious[i, cand])]
            used[j] = True
            tp[order[i]] = True
    return tp


def average_precision(per_image: list[tuple[np.ndarray, np.ndarray, np.ndarray]], iou_thr: float = 0.5) -> dict:
    """VOC-style all-point interpolated AP for one class.

    per_image: list of (pred_boxes, scores, gt_boxes) tuples.
    """
    all_scores, all_tp, n_gt = [], [], 0
    for boxes, scores, gts in per_image:
        n_gt += len(gts)
        all_tp.append(match_detections(boxes, scores, gts, iou_thr))
        all_scores.append(np.asarray(scores, float))
    if n_gt == 0:
        return {"ap": float("nan"), "n_gt": 0}
    scores = np.concatenate(all_scores) if all_scores else np.zeros(0)
    tps = np.concatenate(all_tp) if all_tp else np.zeros(0, bool)
    order = np.argsort(-scores)
    tps = tps[order]
    ctp, cfp = np.cumsum(tps), np.cumsum(~tps)
    recall = ctp / n_gt
    precision = ctp / np.maximum(ctp + cfp, 1)
    mrec = np.concatenate([[0], recall, [recall[-1] if len(recall) else 0]])
    mpre = np.concatenate([[1], precision, [0]])
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))
    return {"ap": ap, "n_gt": int(n_gt), "max_recall": float(recall[-1]) if len(recall) else 0.0}


def detection_at_threshold(per_image, score_thr: float, iou_thr: float = 0.5) -> dict:
    """Missed-sign rate FN/(TP+FN), precision and false positives per image at a fixed threshold."""
    tp = fp = fn = 0
    for boxes, scores, gts in per_image:
        keep = np.asarray(scores) >= score_thr
        flags = match_detections(np.asarray(boxes)[keep], np.asarray(scores)[keep], gts, iou_thr)
        tp += int(flags.sum())
        fp += int((~flags).sum())
        fn += int(len(gts) - flags.sum())
    return {"score_threshold": score_thr, "iou_threshold": iou_thr, "tp": tp, "fp": fp, "fn": fn,
            "precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / (tp + fn) if tp + fn else 0.0,
            "missed_sign_rate": fn / (tp + fn) if tp + fn else float("nan"),
            "false_positives_per_image": fp / max(len(per_image), 1)}


# ---------------- segmentation ----------------
def mask_metrics(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int, ignore_index: int | None = 255) -> dict:
    yt, yp = np.asarray(y_true).ravel(), np.asarray(y_pred).ravel()
    if ignore_index is not None:
        keep = yt != ignore_index
        yt, yp = yt[keep], yp[keep]
    cm = confusion_matrix(yt, yp, n_classes)
    tp = np.diag(cm).astype(float)
    fp, fn = cm.sum(0) - tp, cm.sum(1) - tp
    iou = np.divide(tp, tp + fp + fn, out=np.full_like(tp, np.nan), where=(tp + fp + fn) > 0)
    dice = np.divide(2 * tp, 2 * tp + fp + fn, out=np.full_like(tp, np.nan), where=(2 * tp + fp + fn) > 0)
    return {"pixel_accuracy": float(tp.sum() / max(cm.sum(), 1)), "mean_iou": float(np.nanmean(iou)),
            "per_class_iou": iou.tolist(), "per_class_dice": dice.tolist(), "confusion_matrix": cm.tolist()}
