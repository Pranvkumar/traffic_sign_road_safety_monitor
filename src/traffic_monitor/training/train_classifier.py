"""Train and compare three GTSRB sign classifiers: pixel-MLP baseline, HOG-MLP and a NumPy CNN.

Selection uses the validation split (grouped by physical-sign track). The official
GTSRB test set is evaluated once per model after training is finished.
"""

from __future__ import annotations

import shutil
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from ..data.transforms import augment_crop, to_gray  # noqa: E402
from ..evaluation.metrics import classification_metrics, expected_calibration_error, top_k_accuracy  # noqa: E402
from ..models.cnn_numpy import SmallCNN  # noqa: E402
from ..models.features import sign_features  # noqa: E402
from ..models.mlp_numpy import MLP, Standardizer, make_optimizer, train  # noqa: E402
from ..utils.config import load_config, resolve  # noqa: E402
from ..utils.logging import get_logger  # noqa: E402
from ..utils.reproducibility import run_metadata, save_json, set_seed  # noqa: E402
from .common import as_float, class_names, gtsrb_arrays, log_experiment  # noqa: E402

log = get_logger(__name__)


def _augment_batch(xb: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return np.stack([augment_crop(x, rng) for x in xb])


def train_cnn(model: SmallCNN, x, y, xv, yv, opt, epochs, batch_size, seed, augment) -> dict:
    rng = np.random.default_rng(seed)
    hist, best = {"train_loss": [], "val_acc": []}, (-1, None, -1)
    for epoch in range(epochs):
        order, losses, t0 = rng.permutation(len(x)), [], time.time()
        for i in range(0, len(x), batch_size):
            idx = order[i:i + batch_size]
            xb = _augment_batch(x[idx], rng) if augment else x[idx]
            loss, grads = model.loss_and_grads(xb, y[idx])
            opt.step(model.params, grads)
            losses.append(loss)
        acc = float(np.mean(model.predict(xv) == yv))
        hist["train_loss"].append(float(np.mean(losses)))
        hist["val_acc"].append(acc)
        log.info("cnn epoch %d loss %.4f val_acc %.4f (%.0fs)", epoch, np.mean(losses), acc, time.time() - t0)
        if acc > best[0]:
            best = (acc, {k: v.copy() for k, v in model.params.items()}, epoch)
    model.params = best[1]
    hist["best_epoch"], hist["best_val_acc"] = best[2], best[0]
    return hist


def evaluate(proba: np.ndarray, y: np.ndarray, n_classes: int) -> dict:
    m = classification_metrics(y, proba.argmax(1), n_classes)
    m["top5_accuracy"] = top_k_accuracy(proba, y, 5)
    m["ece"] = expected_calibration_error(proba, y)
    return m


def plot_confusion(cm: np.ndarray, title: str, path) -> None:
    cm = np.asarray(cm, float)
    norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xlabel("Predicted class ID")
    ax.set_ylabel("True class ID")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, label="Row-normalised rate")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def run(config_path: str = "configs/classifier.yaml", only: list[str] | None = None) -> dict:
    cfg = load_config(config_path)
    seed = cfg["seed"]
    set_seed(seed)
    out_dir = resolve(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = resolve("outputs/figures")
    fig_dir.mkdir(parents=True, exist_ok=True)
    nc = cfg["n_classes"]
    data = gtsrb_arrays(cfg["data_config"], cfg["image_size"])
    xtr, xva, xte = as_float(data["X_train"]), as_float(data["X_val"]), as_float(data["X_test"])
    ytr, yva, yte = data["y_train"], data["y_val"], data["y_test"]
    sha = str(data["manifest_sha256"])
    results, names = {}, class_names()

    for name, mc in cfg["models"].items():
        if only and name not in only:
            continue
        t0 = time.time()
        log.info("Training %s", name)
        if name in ("pixel_mlp", "hog_mlp"):
            if name == "pixel_mlp":
                feat = lambda x: to_gray(x).reshape(len(x), -1)  # noqa: E731
            else:
                feat = sign_features
            ftr, fva, fte = feat(xtr), feat(xva), feat(xte)
            std = Standardizer().fit(ftr)
            model = MLP([ftr.shape[1], *mc["hidden"], nc], "relu", "he", weight_decay=mc["weight_decay"], seed=seed)
            hist = train(model, std.transform(ftr), ytr, make_optimizer(mc["optimizer"], mc["lr"]), mc["epochs"],
                         mc["batch_size"], std.transform(fva), yva, seed=seed)
            pva, pte = model.predict_proba(std.transform(fva)), model.predict_proba(std.transform(fte))
            preprocessing = {"crop_size": cfg["image_size"], "colour": "RGB [0,1] via preprocess_crop",
                             "features": "gray pixels" if name == "pixel_mlp" else "HOG(32,16,8,8,9)+RGB 4x4"}
            ckpt = model.save(out_dir / f"{name}.npz", {"mean": std.mean, "std": std.std},
                              {"task": "gtsrb_classification", "preprocessing": preprocessing, "feature_set": name,
                               "class_names": names, "manifest_sha256": sha})
        else:
            model = SmallCNN(nc, seed=seed)
            opt = make_optimizer(mc["optimizer"], mc["lr"])
            hist = train_cnn(model, xtr, ytr, xva, yva, opt, mc["epochs"], mc["batch_size"], seed, mc["augment"])
            pva, pte = model.predict_proba(xva), model.predict_proba(xte)
            ckpt = model.save(out_dir / f"{name}.npz", {"task": "gtsrb_classification", "class_names": names,
                                                         "preprocessing": {"crop_size": cfg["image_size"],
                                                                           "colour": "RGB [0,1] via preprocess_crop"},
                                                         "manifest_sha256": sha})
        val, test = evaluate(pva, yva, nc), evaluate(pte, yte, nc)
        results[name] = {"n_parameters": model.n_parameters(), "train_seconds": round(time.time() - t0, 1),
                         "history": hist, "val": val, "test": test, "checkpoint": str(ckpt)}
        plot_confusion(test["confusion_matrix"], f"GTSRB test confusion — {name}", fig_dir / f"confusion_{name}.png")
        log_experiment({"run_id": f"clf_{name}_{time.strftime('%Y%m%d%H%M%S')}", "task": "classification",
                        "model": name, "dataset": "gtsrb", "manifest_sha256": sha[:12], "seed": seed,
                        "key_settings": str(mc), "best_epoch": hist.get("best_epoch"),
                        "val_metric": "macro_f1", "val_value": round(val["macro_f1"], 4),
                        "test_value": round(test["macro_f1"], 4), "checkpoint": str(ckpt), "notes": ""})
        log.info("%s val acc %.4f test acc %.4f", name, val["accuracy"], test["accuracy"])

    summary = {n: {"val_accuracy": r["val"]["accuracy"], "val_macro_f1": r["val"]["macro_f1"],
                   "test_accuracy": r["test"]["accuracy"], "test_macro_f1": r["test"]["macro_f1"],
                   "test_top5": r["test"]["top5_accuracy"], "test_ece": r["test"]["ece"],
                   "n_parameters": r["n_parameters"], "train_seconds": r["train_seconds"]}
               for n, r in results.items()}
    if summary:
        summary["selected_by_val_macro_f1"] = max(summary, key=lambda n: summary[n]["val_macro_f1"])
    save_json({"meta": run_metadata(cfg, seed), "summary": summary, "results": results},
              out_dir / f"metrics{'_' + '_'.join(only) if only else ''}.json")
    deploy = cfg.get("deploy_model")
    if deploy in results:
        dst = resolve("models/checkpoints") / f"classifier_{deploy}.npz"
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(results[deploy]["checkpoint"], dst)
    return summary


if __name__ == "__main__":
    import json
    import sys

    print(json.dumps(run(only=sys.argv[1:] or None), indent=2))
