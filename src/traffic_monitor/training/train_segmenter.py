"""Train/evaluate the road-surface segmenter (logistic baseline vs NumPy MLP) on comma10k."""

from __future__ import annotations

import shutil
import time

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402

from ..data.manifest import load_manifest  # noqa: E402
from ..evaluation.metrics import mask_metrics  # noqa: E402
from ..models.mlp_numpy import MLP, Standardizer, make_optimizer, train  # noqa: E402
from ..models.segmenter import DOWNSCALE, IGNORE, RoadSegmenter, colour_mask_to_ids, pixel_features  # noqa: E402
from ..utils.config import load_config, resolve  # noqa: E402
from ..utils.logging import get_logger  # noqa: E402
from ..utils.reproducibility import file_sha256, run_metadata, save_json, set_seed  # noqa: E402
from .common import log_experiment  # noqa: E402

log = get_logger(__name__)


def load_pair(root, row, colour_map):
    img = cv2.imread(str(root / row["relative_path"]))
    m = cv2.imread(str(root / row["annotation_path"]))
    if img is None or m is None:
        raise OSError(f"Could not read {row['sample_id']}")
    return img, colour_mask_to_ids(cv2.cvtColor(m, cv2.COLOR_BGR2RGB), colour_map)


def evaluate(seg, rows, root, cmap):
    yt, yp, times = [], [], []
    for _, row in rows.iterrows():
        img, ids = load_pair(root, row, cmap)
        t0 = time.perf_counter()
        pred = seg.predict_mask(img)
        times.append(time.perf_counter() - t0)
        # evaluate on a 1/2-resolution nearest-neighbour grid to bound memory (documented)
        yt.append(ids[::2, ::2])
        yp.append(pred[::2, ::2])
    m = mask_metrics(np.concatenate([a.ravel() for a in yt]), np.concatenate([a.ravel() for a in yp]), 2, IGNORE)
    m["latency_ms_median"] = float(np.median(times) * 1000)
    return m


def run(config_path: str = "configs/segmenter.yaml") -> dict:
    cfg = load_config(config_path)
    seed = cfg["seed"]
    rng = set_seed(seed)
    dcfg = load_config(cfg["data_config"])["comma10k"]
    root, mpath, cmap = resolve(dcfg["root"]), resolve(dcfg["manifest"]), dcfg["colour_map"]
    df = load_manifest(mpath)
    tr, va, te = (df[df.split == s] for s in ("train", "val", "test"))
    out = resolve(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    xs, ys = [], []
    for _, row in tr.iterrows():
        img, ids = load_pair(root, row, cmap)
        f, (sh, sw) = pixel_features(img)
        lab = cv2.resize(ids, (sw, sh), interpolation=cv2.INTER_NEAREST).ravel()
        ok = np.where(lab != IGNORE)[0]
        pick = rng.choice(ok, min(cfg["pixels_per_image"], len(ok)), replace=False)
        xs.append(f[pick])
        ys.append(lab[pick])
    x, y = np.vstack(xs), np.concatenate(ys).astype(int)
    std = Standardizer().fit(x)
    log.info("training pixels: %d (road share %.2f)", len(y), y.mean())

    lr = LogisticRegression(C=cfg["logreg"]["C"], max_iter=2000, random_state=seed).fit(std.transform(x), y)
    mc = cfg["mlp"]
    mlp = MLP([x.shape[1], *mc["hidden"], 2], weight_decay=mc["weight_decay"], seed=seed)
    train(mlp, std.transform(x), y, make_optimizer("momentum", mc["lr"]), mc["epochs"], mc["batch_size"],
          seed=seed, keep_best=False)
    models = {"logreg": RoadSegmenter("logreg", std, lr.coef_.ravel(), float(lr.intercept_[0])),
              "mlp": RoadSegmenter("mlp", std, mlp=mlp)}
    train_seconds = time.time() - t0

    results = {}
    for name, seg in models.items():
        results[name] = {"val": evaluate(seg, va, root, cmap)}
    selected = max(results, key=lambda n: results[n]["val"]["mean_iou"])
    for name, seg in models.items():
        results[name]["test"] = evaluate(seg, te, root, cmap)
        ckpt = seg.save(out / f"segmenter_{name}.npz", {"trained_on": "comma10k subset train split",
                                                        "manifest_sha256": file_sha256(mpath)})
        results[name]["checkpoint"] = str(ckpt)
        log_experiment({"run_id": f"seg_{name}_{time.strftime('%Y%m%d%H%M%S')}", "task": "segmentation",
                        "model": f"pixel_{name}", "dataset": "comma10k_subset", "manifest_sha256": file_sha256(mpath)[:12],
                        "seed": seed, "key_settings": str(cfg.get(name)), "best_epoch": "", "val_metric": "mIoU",
                        "val_value": round(results[name]["val"]["mean_iou"], 4),
                        "test_value": round(results[name]["test"]["mean_iou"], 4), "checkpoint": str(ckpt), "notes": ""})
    shutil.copy(results[selected]["checkpoint"], resolve("models/checkpoints/segmenter.npz"))

    # figure: evenly spaced test images, image | ground truth | prediction | overlay
    seg = models[selected]
    rows = te.reset_index(drop=True).iloc[np.linspace(0, len(te) - 1, 4).astype(int)]
    fig, axes = plt.subplots(4, 4, figsize=(14, 10))
    for r, (_, row) in enumerate(rows.iterrows()):
        img, ids = load_pair(root, row, cmap)
        pred = seg.predict_mask(img)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        gt_vis = np.where(ids == IGNORE, 0.5, ids.astype(float))
        overlay = rgb.copy()
        overlay[pred == 1] = (0.5 * overlay[pred == 1] + 0.5 * np.array([255, 0, 255])).astype(np.uint8)
        for c, (im, title) in enumerate([(rgb, "image"), (gt_vis, "ground truth (grey = ignored)"),
                                         (pred, f"prediction ({selected})"), (overlay, "overlay")]):
            axes[r, c].imshow(im, cmap="gray" if c in (1, 2) else None)
            axes[r, c].set_title(title if r == 0 else "", fontsize=9)
            axes[r, c].axis("off")
    fig.suptitle("Road-surface segmentation, comma10k test split (evenly spaced, not cherry-picked)")
    fig.tight_layout()
    fig.savefig(resolve("outputs/figures/segmentation_examples.png"), dpi=80)
    plt.close(fig)

    summary = {"selected_by_val_miou": selected, "train_pixels": int(len(y)), "train_seconds": round(train_seconds, 1),
               "images": {"train": len(tr), "val": len(va), "test": len(te)},
               "eval_note": f"features at 1/{DOWNSCALE} res, prediction upsampled; metrics on 1/2-res grid; my_car ignored",
               **{n: {"val_miou": r["val"]["mean_iou"], "test_miou": r["test"]["mean_iou"],
                      "test_road_iou": r["test"]["per_class_iou"][1], "test_road_dice": r["test"]["per_class_dice"][1],
                      "test_pixel_acc": r["test"]["pixel_accuracy"], "latency_ms": r["test"]["latency_ms_median"]}
                  for n, r in results.items()}}
    save_json({"meta": run_metadata(cfg, seed), "summary": summary, "results": results}, out / "metrics.json")
    return summary


if __name__ == "__main__":
    import json

    print(json.dumps(run(), indent=2))
