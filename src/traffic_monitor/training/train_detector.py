"""Train/evaluate the HOG sliding-window sign detector with two scorers (logistic vs MLP)."""

from __future__ import annotations

import shutil
import time

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402

from ..data.annotations import clip_boxes  # noqa: E402
from ..data.datasets import load_detection_sample  # noqa: E402
from ..data.manifest import load_manifest  # noqa: E402
from ..evaluation.metrics import average_precision, box_iou, detection_at_threshold  # noqa: E402
from ..models.detector import N_FEATURES, SignDetector, WindowScorer, crop_square, dense_windows, nms, window_features  # noqa: E402
from ..models.mlp_numpy import MLP, Standardizer, make_optimizer, train  # noqa: E402
from ..utils.config import load_config, resolve  # noqa: E402
from ..utils.logging import get_logger  # noqa: E402
from ..utils.reproducibility import file_sha256, run_metadata, save_json, set_seed  # noqa: E402
from .common import log_experiment  # noqa: E402

log = get_logger(__name__)


def _load(root, row):
    img, gt = load_detection_sample(root, row)
    return img, clip_boxes(gt, img.shape[1], img.shape[0])


def sample_patches(rows, root, cfg, rng):
    pos, neg = [], []
    for _, row in rows.iterrows():
        img, gt = _load(root, row)
        h, w = img.shape[:2]
        for box in gt:
            side = max(box[2] - box[0], box[3] - box[1])
            if side < cfg["min_train_box_side"]:
                continue
            for j in range(cfg["jitter_per_box"] + 1):
                b = box.copy()
                if j:
                    shift = rng.uniform(-0.1, 0.1, 2) * side
                    b = b + np.r_[shift, shift]
                p = crop_square(img, b, 1.0 if j == 0 else rng.uniform(0.9, 1.1))
                if p is not None:
                    pos.append(p)
                    if cfg.get("flip_positives", False):
                        pos.append(p[:, ::-1].copy())   # mirrored sign is still "a sign" (class-agnostic)
        added = 0
        for _ in range(cfg["negatives_per_image"] * 5):      # bounded number of attempts
            if added >= cfg["negatives_per_image"]:
                break
            side = float(np.exp(rng.uniform(np.log(16), np.log(120))))
            x, y = rng.uniform(0, w - side), rng.uniform(0, h - side)
            cand = np.array([[x, y, x + side, y + side]])
            if len(gt) and box_iou(cand, gt).max() >= 0.1:
                continue
            p = crop_square(img, cand[0])
            if p is not None:
                neg.append(p)
                added += 1
    return np.stack(pos), np.stack(neg)


def fit_logreg(x, y, C, seed):
    std = Standardizer().fit(x)
    lr = LogisticRegression(C=C, max_iter=2000, class_weight="balanced", random_state=seed)
    lr.fit(std.transform(x), y)
    return WindowScorer("logreg", std, lr.coef_.ravel().astype(np.float32), float(lr.intercept_[0]))


def fit_mlp(x, y, mc, seed):
    std = Standardizer().fit(x)
    rng = np.random.default_rng(seed)
    pos = np.where(y == 1)[0]
    reps = max(1, int((y == 0).sum() / max(len(pos), 1)))
    idx = np.concatenate([np.where(y == 0)[0], np.tile(pos, reps)])   # balance by oversampling positives
    idx = rng.permutation(idx)
    mlp = MLP([x.shape[1], *mc["hidden"], 2], "relu", "he", weight_decay=mc["weight_decay"], seed=seed)
    train(mlp, std.transform(x[idx]), y[idx], make_optimizer("momentum", mc["lr"]), mc["epochs"],
          mc["batch_size"], seed=seed, keep_best=False)
    return WindowScorer("mlp", std, mlp=mlp)


def hard_negatives(scorer, rows, root, cfg):
    feats = []
    for _, row in rows.iterrows():
        img, gt = _load(root, row)
        cand_f, cand_b, cand_s = [], [], []
        for f, b in dense_windows(img, cfg["sign_sizes"], cfg["stride"]):
            s = scorer.score(f)
            k = s > 0.5
            cand_f.append(f[k]), cand_b.append(b[k]), cand_s.append(s[k])
        f, b, s = np.concatenate(cand_f), np.concatenate(cand_b), np.concatenate(cand_s)
        if len(b) == 0:
            continue
        if len(gt):
            far = box_iou(b, gt).max(1) < 0.1
            f, b, s = f[far], b[far], s[far]
        keep = nms(b, s, 0.3)[: cfg["hard_negatives_per_image"]]
        feats.append(f[keep])
    return np.concatenate(feats) if feats else np.zeros((0, N_FEATURES), np.float32)


def evaluate_split(scorers: dict, rows, root, cfg, label=""):
    """HOG is computed once per image; each scorer then scores the same windows.

    For the AP curve, windows with score >= eval_min_score are kept, capped to the
    eval_max_candidates highest before NMS (documented approximation of the low-score tail).
    Latency is measured separately with the real `SignDetector.detect` at the chosen threshold.
    """
    per = {k: [] for k in scorers}
    min_s, cap = cfg["eval_min_score"], cfg["eval_max_candidates"]
    for n, (_, row) in enumerate(rows.iterrows()):
        img, gt = _load(root, row)
        windows = list(dense_windows(img, cfg["sign_sizes"], cfg["stride"]))
        for name, sc in scorers.items():
            bs, ss = [], []
            for f, b in windows:
                s = sc.score(f)
                k = s >= min_s
                bs.append(b[k]), ss.append(s[k])
            boxes, scores = np.concatenate(bs), np.concatenate(ss)
            top = np.argsort(-scores)[:cap]
            boxes, scores = boxes[top], scores[top]
            keep = nms(boxes, scores, cfg["nms_iou"])[:100] if len(scores) else np.zeros(0, int)
            per[name].append((boxes[keep], scores[keep], gt))
        if n % 25 == 0:
            log.info("%s eval %d/%d", label, n, len(rows))
    return per


def measure_latency(det: SignDetector, rows, root, n: int = 20) -> dict:
    """End-to-end detector latency at its operating threshold (1 warm-up run not timed)."""
    imgs = [_load(root, r)[0] for _, r in rows.head(n + 1).iterrows()]
    det.detect(imgs[0])
    t = []
    for img in imgs[1:]:
        t0 = time.perf_counter()
        det.detect(img)
        t.append((time.perf_counter() - t0) * 1000)
    return {"median": float(np.median(t)), "p90": float(np.percentile(t, 90)), "n": len(t),
            "note": "1280x964 input, CPU, batch 1, HOG + scoring + NMS at the chosen threshold"}


def best_f1_threshold(per_image, iou):
    best = (0.0, 0.5)
    for thr in np.round(np.arange(0.1, 0.99, 0.05), 2):
        r = detection_at_threshold(per_image, thr, iou)
        f1 = 2 * r["precision"] * r["recall"] / (r["precision"] + r["recall"]) if r["precision"] + r["recall"] else 0
        if f1 > best[0]:
            best = (f1, float(thr))
    return best[1], best[0]


def size_breakdown(per_image, thr, iou):
    """Recall by ground-truth sign width bucket (small/distant signs are the hard case)."""
    buckets = {"<16px": (0, 16), "16-32px": (16, 32), "32-64px": (32, 64), ">=64px": (64, 1e9)}
    out = {}
    for name, (lo, hi) in buckets.items():
        sub = []
        for boxes, scores, gt in per_image:
            wgt = gt[:, 2] - gt[:, 0] if len(gt) else np.zeros(0)
            sel = gt[(wgt >= lo) & (wgt < hi)] if len(gt) else gt
            sub.append((boxes, scores, sel))
        r = detection_at_threshold(sub, thr, iou)
        out[name] = {"n_gt": r["tp"] + r["fn"], "recall": r["recall"]}
    return out


def draw_examples(per_image, rows, root, thr, path, n=6):
    rows = rows.reset_index(drop=True)
    idx = np.linspace(0, len(rows) - 1, n).astype(int)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, i in zip(axes.ravel(), idx):
        img, _ = _load(root, rows.iloc[i])
        boxes, scores, gt = per_image[i]
        vis = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).copy()
        for b in gt.astype(int):
            cv2.rectangle(vis, tuple(b[:2]), tuple(b[2:]), (0, 255, 0), 3)
        for b, s in zip(boxes, scores):
            if s >= thr:
                b = b.astype(int)
                cv2.rectangle(vis, tuple(b[:2]), tuple(b[2:]), (255, 0, 0), 3)
                cv2.putText(vis, f"{s:.2f}", (b[0], max(b[1] - 4, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        ax.imshow(vis)
        ax.set_title(rows.iloc[i]["sample_id"][-28:], fontsize=8)
        ax.axis("off")
    fig.suptitle(f"Test examples (evenly spaced, not cherry-picked). Green = ground truth, red = prediction ≥ {thr}")
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)


def run(config_path: str = "configs/detector.yaml") -> dict:
    cfg = load_config(config_path)
    seed = cfg["seed"]
    rng = set_seed(seed)
    dcfg = load_config(cfg["data_config"])["idd_signs"]
    root, mpath = resolve(dcfg["root"]), resolve(dcfg["manifest"])
    df = load_manifest(mpath)
    tr, va, te = (df[df.split == s] for s in ("train", "val", "test"))
    out = resolve(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    pos, neg = sample_patches(tr, root, cfg, rng)
    x = np.vstack([window_features(pos), window_features(neg)])
    y = np.r_[np.ones(len(pos), int), np.zeros(len(neg), int)]
    log.info("Initial patches: %d pos, %d neg", len(pos), len(neg))
    # Hard-negative mining over several rounds, each on a different random slice of training images
    order = tr.sample(frac=1.0, random_state=seed)
    per_round = int(np.ceil(len(order) / cfg["hard_negative_rounds"]))
    x2, y2, hn_total = x, y, 0
    for r in range(cfg["hard_negative_rounds"]):
        base = fit_logreg(x2, y2, cfg["scorers"]["logreg"]["C"], seed)
        hn = hard_negatives(base, order.iloc[r * per_round:(r + 1) * per_round], root, cfg)
        hn_total += len(hn)
        log.info("Round %d: hard negatives mined %d", r, len(hn))
        x2, y2 = np.vstack([x2, hn]), np.r_[y2, np.zeros(len(hn), int)]
    scorers = {"logreg": fit_logreg(x2, y2, cfg["scorers"]["logreg"]["C"], seed),
               "mlp": fit_mlp(x2, y2, cfg["scorers"]["mlp"], seed)}
    train_seconds = time.time() - t0

    iou = cfg["eval_iou"]
    log.info("Scorers trained in %.0fs; evaluating on validation", train_seconds)
    val_per = evaluate_split(scorers, va, root, cfg, "val")
    results = {}
    for name in scorers:
        thr, f1 = best_f1_threshold(val_per[name], iou)
        results[name] = {"val": {"ap50": average_precision(val_per[name], iou),
                                 "ap30": average_precision(val_per[name], 0.3),
                                 "chosen_threshold": thr, "f1_at_threshold": f1,
                                 "at_threshold": detection_at_threshold(val_per[name], thr, iou)}}
    selected = max(results, key=lambda n: results[n]["val"]["ap50"]["ap"])

    # ---- test: evaluated once, thresholds frozen from validation ----
    test_per = evaluate_split(scorers, te, root, cfg, "test")
    for name in scorers:
        thr = results[name]["val"]["chosen_threshold"]
        det = SignDetector(scorers[name], thr, cfg["nms_iou"], cfg["max_detections"], tuple(cfg["sign_sizes"]), cfg["stride"])
        results[name]["test"] = {
            "ap50": average_precision(test_per[name], iou), "ap30": average_precision(test_per[name], 0.3),
            "at_val_threshold": detection_at_threshold(test_per[name], thr, iou),
            "recall_by_size": size_breakdown(test_per[name], thr, iou),
            "latency_ms_per_image": measure_latency(det, te, root),
        }
        ckpt = det.save(out / f"detector_{name}.npz", {"trained_on": "idd_signs frontFar train split",
                                                        "manifest_sha256": file_sha256(mpath)})
        results[name]["checkpoint"] = str(ckpt)
        draw_examples(test_per[name], te, root, thr, resolve(f"outputs/figures/detector_examples_{name}.png"))
        log_experiment({"run_id": f"det_{name}_{time.strftime('%Y%m%d%H%M%S')}", "task": "detection",
                        "model": f"hog_sliding_window_{name}", "dataset": "idd_signs_frontFar",
                        "manifest_sha256": file_sha256(mpath)[:12], "seed": seed,
                        "key_settings": str(cfg["scorers"][name]), "best_epoch": "",
                        "val_metric": "AP@0.5", "val_value": round(results[name]["val"]["ap50"]["ap"], 4),
                        "test_value": round(results[name]["test"]["ap50"]["ap"], 4), "checkpoint": str(ckpt),
                        "notes": f"thr={thr} chosen on val"})
    shutil.copy(results[selected]["checkpoint"], resolve("models/checkpoints/detector.npz"))
    summary = {"selected_by_val_ap50": selected, "train_seconds": round(train_seconds, 1),
               "n_pos_patches": int(len(pos)), "n_neg_patches": int(len(neg)), "n_hard_negatives": int(hn_total),
               "images": {"train": len(tr), "val": len(va), "test": len(te)}}
    save_json({"meta": run_metadata(cfg, seed), "summary": summary, "results": results}, out / "metrics.json")
    return {"summary": summary, **{n: {"val_ap50": r["val"]["ap50"]["ap"], "test_ap50": r["test"]["ap50"]["ap"],
                                       "test_missed_sign_rate": r["test"]["at_val_threshold"]["missed_sign_rate"],
                                       "test_fp_per_image": r["test"]["at_val_threshold"]["false_positives_per_image"]}
                                   for n, r in results.items()}}


if __name__ == "__main__":
    import json

    print(json.dumps(run(), indent=2, default=str))
