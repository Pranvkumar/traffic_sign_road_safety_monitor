"""LSTM vs non-recurrent baselines on GTSRB physical-sign tracks (see docs/TEMPORAL_TARGET.md)."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from ..data.splits import check_group_disjoint, group_split
from ..evaluation.metrics import classification_metrics
from ..models.features import sign_features
from ..models.lstm_numpy import LSTMClassifier
from ..models.mlp_numpy import MLP, Adam, Standardizer, make_optimizer, train
from ..utils.config import load_config, resolve
from ..utils.logging import get_logger
from ..utils.reproducibility import run_metadata, save_json, set_seed
from .common import as_float, gtsrb_arrays, log_experiment

log = get_logger(__name__)


def build_tracks(data) -> pd.DataFrame:
    """Official GTSRB training images (our train+val) as ordered tracks."""
    frames = []
    for split in ("train", "val"):
        frames.append(pd.DataFrame({"group": data[f"group_{split}"], "frame": data[f"frame_{split}"],
                                    "label": data[f"y_{split}"], "src": split, "idx": np.arange(len(data[f"y_{split}"]))}))
    return pd.concat(frames, ignore_index=True)


def make_sequences(df_tracks, feats, length, starts):
    xs, ys, gs = [], [], []
    for g, part in df_tracks.groupby("group"):
        part = part.sort_values("frame")
        f = feats[part["row"].to_numpy()]
        for s in starts:
            if s + length <= len(f):
                xs.append(f[s:s + length])
                ys.append(int(part["label"].iloc[0]))
                gs.append(g)
    return np.stack(xs), np.array(ys), np.array(gs)


def train_lstm(model, x, y, xv, yv, lc, seed):
    rng = np.random.default_rng(seed)
    opt = Adam(lc["lr"])
    best = (-1, None, -1)
    hist = []
    for epoch in range(lc["epochs"]):
        order = rng.permutation(len(x))
        for i in range(0, len(x), lc["batch_size"]):
            idx = order[i:i + lc["batch_size"]]
            _, g = model.loss_and_grads(x[idx], y[idx])
            for k in g:
                np.clip(g[k], -5, 5, out=g[k])          # gradient clipping for stability
            opt.step(model.params, g)
        acc = float(np.mean(model.predict(xv) == yv))
        hist.append(acc)
        if acc > best[0]:
            best = (acc, {k: v.copy() for k, v in model.params.items()}, epoch)
    model.params = best[1]
    return {"val_acc": hist, "best_epoch": best[2], "best_val_acc": best[0]}


def run(config_path: str = "configs/temporal.yaml") -> dict:
    cfg = load_config(config_path)
    seed = cfg["seed"]
    set_seed(seed)
    data = gtsrb_arrays(cfg["data_config"])
    tracks = build_tracks(data)
    imgs = np.concatenate([data["X_train"], data["X_val"]])
    tracks["row"] = np.where(tracks["src"] == "train", tracks["idx"], tracks["idx"] + len(data["y_train"]))
    # --- leakage-safe split by physical sign track, stratified by class ---
    tracks["split"] = "unassigned"
    tracks = group_split(tracks, "group", cfg["track_split"], seed=seed, stratify_col="label")
    assert not check_group_disjoint(tracks, "group")["leaked_groups"]

    t0 = time.time()
    feats = sign_features(as_float(imgs))
    tr_rows = tracks.loc[tracks.split == "train", "row"].to_numpy()
    std = Standardizer().fit(feats[tr_rows])
    z = std.transform(feats)
    # PCA fitted on training-track frames only
    _, _, vt = np.linalg.svd(z[tr_rows] - z[tr_rows].mean(0), full_matrices=False)
    comps, mu = vt[: cfg["pca_dims"]], z[tr_rows].mean(0)
    pf = (z - mu) @ comps.T
    pstd = Standardizer().fit(pf[tr_rows])
    pf = pstd.transform(pf)

    L = cfg["sequence_length"]
    xtr, ytr, _ = make_sequences(tracks[tracks.split == "train"], pf, L, range(0, 30 - L + 1, cfg["train_window_stride"]))
    xva, yva, _ = make_sequences(tracks[tracks.split == "val"], pf, L, [0])
    xte, yte, gte = make_sequences(tracks[tracks.split == "test"], pf, L, [0])
    log.info("sequences train %d val %d test %d", len(xtr), len(xva), len(xte))
    rng = np.random.default_rng(seed)
    test_mask = (rng.random(xte.shape[:2]) >= cfg["missing_frame_rate_eval"]).astype(float)
    test_mask[:, 0] = 1.0

    results = {}
    # baseline 1: last observed frame only
    for name, fn in [("last_frame_mlp", lambda x: x[:, -1]), ("mean_pool_mlp", lambda x: x.mean(1))]:
        bc = cfg["baselines"][name]
        m = MLP([pf.shape[1], *bc["hidden"], 43], seed=seed)
        h = train(m, fn(xtr), ytr, make_optimizer("momentum", bc["lr"]), bc["epochs"], bc["batch_size"],
                  fn(xva), yva, seed=seed)
        # robustness: with missing frames, baselines see the last *available* frame / mean of available frames
        last_avail = xte[np.arange(len(xte)), (test_mask * np.arange(L)).argmax(1)]
        mean_avail = (xte * test_mask[..., None]).sum(1) / test_mask.sum(1, keepdims=True)
        x_missing = last_avail if name == "last_frame_mlp" else mean_avail
        results[name] = {"n_parameters": m.n_parameters(), "best_epoch": h["best_epoch"],
                         "val_accuracy": h["best_val_acc"],
                         "test": classification_metrics(yte, m.predict(fn(xte)), 43),
                         "test_with_missing_frames": classification_metrics(yte, m.predict(x_missing), 43)}
    lc = cfg["lstm"]
    lstm = LSTMClassifier(pf.shape[1], lc["hidden"], 43, seed=seed, weight_decay=lc["weight_decay"])
    h = train_lstm(lstm, xtr, ytr, xva, yva, lc, seed)
    results["lstm"] = {"n_parameters": lstm.n_parameters(), "best_epoch": h["best_epoch"],
                       "val_accuracy": h["best_val_acc"], "val_curve": h["val_acc"],
                       "test": classification_metrics(yte, lstm.predict(xte), 43),
                       "test_with_missing_frames": classification_metrics(yte, lstm.predict(xte, test_mask), 43)}
    out = resolve(cfg["output_dir"])
    ckpt = lstm.save(out / "lstm.npz", {"target": "GTSRB track class from first frames (educational)",
                                       "feature_schema": f"HOG(324)+RGB4x4(48) -> standardise -> PCA{cfg['pca_dims']} -> standardise",
                                       "sequence_length": L},
                     {"feat_mean": std.mean, "feat_std": std.std, "pca_components": comps, "pca_mean": mu,
                      "pca_out_mean": pstd.mean, "pca_out_std": pstd.std})
    for name, r in results.items():
        for key in ("test", "test_with_missing_frames"):
            r[key] = {k: v for k, v in r[key].items() if k not in ("per_class", "confusion_matrix")}
        log_experiment({"run_id": f"temporal_{name}_{time.strftime('%Y%m%d%H%M%S')}", "task": "temporal",
                        "model": name, "dataset": "gtsrb_tracks", "manifest_sha256": str(data["manifest_sha256"])[:12],
                        "seed": seed, "key_settings": f"L={L}", "best_epoch": r["best_epoch"],
                        "val_metric": "accuracy", "val_value": round(r["val_accuracy"], 4),
                        "test_value": round(r["test"]["accuracy"], 4),
                        "checkpoint": str(ckpt) if name == "lstm" else "", "notes": "sequence-level"})
    summary = {n: {"val_acc": r["val_accuracy"], "test_acc": r["test"]["accuracy"], "test_macro_f1": r["test"]["macro_f1"],
                   "test_acc_missing_frames": r["test_with_missing_frames"]["accuracy"]} for n, r in results.items()}
    summary["tracks"] = {s: int(tracks[tracks.split == s].group.nunique()) for s in ("train", "val", "test")}
    summary["sequences"] = {"train_windows": int(len(xtr)), "val": int(len(xva)), "test": int(len(xte))}
    summary["seconds"] = round(time.time() - t0, 1)
    save_json({"meta": run_metadata(cfg, seed), "summary": summary, "results": results}, out / "metrics.json")
    return summary


if __name__ == "__main__":
    import json

    print(json.dumps(run(), indent=2))
