"""Train the CVAE on real training crops, visualise it, and run the augmentation ablation.

Synthetic images are only ever added to the TRAINING set. Validation is used for
checkpoint selection; the official real GTSRB test set is used once for the ablation.
"""

from __future__ import annotations

import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from ..data.transforms import to_gray  # noqa: E402
from ..evaluation.metrics import classification_metrics  # noqa: E402
from ..models.features import hog32  # noqa: E402
from ..models.mlp_numpy import MLP, Adam, Standardizer, make_optimizer, train  # noqa: E402
from ..models.vae_numpy import CVAE  # noqa: E402
from ..utils.config import load_config, resolve  # noqa: E402
from ..utils.logging import get_logger  # noqa: E402
from ..utils.reproducibility import run_metadata, save_json, set_seed  # noqa: E402
from .common import as_float, class_names, gtsrb_arrays, log_experiment  # noqa: E402

log = get_logger(__name__)


def gray_to_rgb(g: np.ndarray) -> np.ndarray:
    return np.repeat(g.reshape(-1, 32, 32, 1), 3, axis=3).astype(np.float32)


def train_vae(model, xtr, ytr, xva, yva, cfg, seed):
    rng = np.random.default_rng(seed)
    opt = Adam(cfg["lr"])
    hist, best = {"train_loss": [], "val_loss": [], "val_recon": [], "val_kl": []}, (np.inf, None, -1)
    for epoch in range(cfg["epochs"]):
        order, losses = rng.permutation(len(xtr)), []
        for i in range(0, len(xtr), cfg["batch_size"]):
            idx = order[i:i + cfg["batch_size"]]
            loss, _, _, g = model.loss_and_grads(xtr[idx], ytr[idx], rng.normal(size=(len(idx), model.cfg["latent"])))
            opt.step(model.params, g)
            losses.append(loss)
        vl, vr, vk = model.loss_and_grads(xva, yva, np.random.default_rng(0).normal(size=(len(xva), model.cfg["latent"])))[:3]
        for k, v in zip(hist, (np.mean(losses), vl, vr, vk)):
            hist[k].append(float(v))
        if vl < best[0]:
            best = (vl, {k: v.copy() for k, v in model.params.items()}, epoch)
        if epoch % 5 == 0:
            log.info("vae epoch %d train %.1f val %.1f (recon %.1f kl %.1f)", epoch, np.mean(losses), vl, vr, vk)
    model.params = best[1]
    hist["best_epoch"] = best[2]
    return hist


def figures(model, xva, yva, rare, names, fig_dir, rng):
    # samples for rare classes
    fig, axes = plt.subplots(len(rare), 9, figsize=(10, 1.25 * len(rare)))
    for r, c in enumerate(rare):
        real = xva[yva == c]
        axes[r, 0].imshow(real[0].reshape(32, 32) if len(real) else np.zeros((32, 32)), cmap="gray")
        axes[r, 0].set_ylabel(f"{c}", rotation=0, labelpad=12)
        s = model.sample(np.full(8, c), rng)
        for j in range(8):
            axes[r, j + 1].imshow(s[j].reshape(32, 32), cmap="gray", vmin=0, vmax=1)
        for ax in axes[r]:
            ax.set_xticks([]), ax.set_yticks([])
    axes[0, 0].set_title("real (val)", fontsize=8)
    axes[0, 4].set_title("synthetic CVAE samples (not real observations)", fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "vae_samples_rare_classes.png", dpi=110)
    plt.close(fig)
    # reconstructions
    idx = rng.choice(len(xva), 10, replace=False)
    mu, _, _ = model.encode(xva[idx], yva[idx])
    rec = model.decode(mu, yva[idx])[0]
    fig, axes = plt.subplots(2, 10, figsize=(10, 2.4))
    for j in range(10):
        axes[0, j].imshow(xva[idx[j]].reshape(32, 32), cmap="gray")
        axes[1, j].imshow(rec[j].reshape(32, 32), cmap="gray")
        axes[0, j].axis("off"), axes[1, j].axis("off")
    axes[0, 0].set_title("val input", fontsize=8, loc="left")
    axes[1, 0].set_title("reconstruction", fontsize=8, loc="left")
    fig.tight_layout()
    fig.savefig(fig_dir / "vae_reconstructions.png", dpi=110)
    plt.close(fig)
    # latent space (PCA projection of posterior means)
    sel = np.isin(yva, [1, 2, 12, 13, 38, 25])
    mu, _, _ = model.encode(xva[sel], yva[sel])
    mc = mu - mu.mean(0)
    _, sv, vt = np.linalg.svd(mc, full_matrices=False)
    p = mc @ vt[:2].T
    bright = xva[sel].mean(1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for c in [1, 2, 12, 13, 38, 25]:
        m = yva[sel] == c
        axes[0].scatter(p[m, 0], p[m, 1], s=5, alpha=0.6, label=f"{c} {names[c]}")
    axes[0].legend(fontsize=7, markerscale=3)
    axes[0].set_title("Latent means coloured by class")
    sc = axes[1].scatter(p[:, 0], p[:, 1], s=5, c=bright, cmap="viridis")
    fig.colorbar(sc, ax=axes[1], label="mean pixel brightness")
    axes[1].set_title("Same points coloured by image brightness")
    var = (sv[:2] ** 2 / (sv ** 2).sum()).sum()
    for ax in axes:
        ax.set_xlabel("latent PC1")
        ax.set_ylabel("latent PC2")
    fig.suptitle(f"CVAE latent space, 2-D PCA projection ({var:.0%} of variance; approximate view). "
                 "Class is given to the decoder, so the latent mostly encodes style, not class.", fontsize=9)
    fig.tight_layout()
    fig.savefig(fig_dir / "vae_latent_space.png", dpi=110)
    plt.close(fig)
    return float(var)


def run(config_path: str = "configs/vae.yaml") -> dict:
    cfg = load_config(config_path)
    seed = cfg["seed"]
    rng = set_seed(seed)
    out, fig_dir = resolve(cfg["output_dir"]), resolve("outputs/figures")
    out.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    data = gtsrb_arrays(cfg["data_config"])
    gtr = to_gray(as_float(data["X_train"])).reshape(-1, 1024)
    gva = to_gray(as_float(data["X_val"])).reshape(-1, 1024)
    gte = to_gray(as_float(data["X_test"])).reshape(-1, 1024)
    ytr, yva, yte = data["y_train"], data["y_val"], data["y_test"]
    names = class_names()

    t0 = time.time()
    vae = CVAE(43, 1024, cfg["hidden"], cfg["latent"], cfg["beta"], seed)
    hist = train_vae(vae, gtr, ytr, gva, yva, cfg, seed)
    vae.save(out / "cvae.npz", {"trained_on": "GTSRB train split (grayscale 32x32)", "best_epoch": hist["best_epoch"]})
    vae_seconds = time.time() - t0

    counts = np.bincount(ytr, minlength=43)
    rare = [int(c) for c in np.argsort(counts)[: cfg["rare_classes"]]]
    latent_var = figures(vae, gva, yva, rare, names, fig_dir, rng)

    # synthetic set for rare classes (training only)
    syn_x, syn_y = [], []
    for c in rare:
        k = max(0, cfg["target_count_per_rare_class"] - int(counts[c]))
        syn_x.append(vae.sample(np.full(k, c), rng))
        syn_y.append(np.full(k, c))
    syn_x, syn_y = np.vstack(syn_x), np.concatenate(syn_y)

    ac = cfg["ablation"]["classifier"]
    f_tr, f_va, f_te = hog32(gray_to_rgb(gtr)), hog32(gray_to_rgb(gva)), hog32(gray_to_rgb(gte))
    f_syn = hog32(gray_to_rgb(syn_x))
    arms = {"real_only": (f_tr, ytr), "real_plus_synthetic": (np.vstack([f_tr, f_syn]), np.r_[ytr, syn_y])}
    ablation = {arm: [] for arm in arms}
    label_consistency = None
    for s in cfg["ablation"]["seeds"]:
        for arm, (fx, fy) in arms.items():
            std = Standardizer().fit(f_tr)          # same real-only normalisation in both arms
            m = MLP([fx.shape[1], *ac["hidden"], 43], weight_decay=ac["weight_decay"], seed=s)
            train(m, std.transform(fx), fy, make_optimizer("momentum", ac["lr"]), ac["epochs"], ac["batch_size"],
                  std.transform(f_va), yva, seed=s)
            met = classification_metrics(yte, m.predict(std.transform(f_te)), 43)
            ablation[arm].append({"seed": s, "test_accuracy": met["accuracy"], "test_macro_f1": met["macro_f1"],
                                  "rare_class_mean_recall": float(np.mean([met["per_class"][c]["recall"] for c in rare]))})
            if arm == "real_only" and label_consistency is None:
                label_consistency = float(np.mean(m.predict(std.transform(f_syn)) == syn_y))
    agg = {arm: {k: {"mean": float(np.mean([r[k] for r in runs])), "std": float(np.std([r[k] for r in runs]))}
                 for k in ("test_accuracy", "test_macro_f1", "rare_class_mean_recall")} for arm, runs in ablation.items()}
    for arm in arms:
        log_experiment({"run_id": f"vae_ablation_{arm}_{time.strftime('%Y%m%d%H%M%S')}", "task": "augmentation_ablation",
                        "model": "hog_gray_mlp", "dataset": "gtsrb", "manifest_sha256": str(data["manifest_sha256"])[:12],
                        "seed": str(cfg["ablation"]["seeds"]), "key_settings": arm, "best_epoch": "",
                        "val_metric": "", "val_value": "", "test_value": round(agg[arm]["test_macro_f1"]["mean"], 4),
                        "checkpoint": str(out / "cvae.npz"), "notes": "mean over seeds"})
    summary = {"vae_best_epoch": hist["best_epoch"], "vae_best_val_loss": float(min(hist["val_loss"])),
               "vae_train_seconds": round(vae_seconds, 1), "rare_classes": {c: names[c] for c in rare},
               "rare_class_train_counts": {c: int(counts[c]) for c in rare}, "n_synthetic": int(len(syn_y)),
               "synthetic_label_consistency(real-only classifier agrees)": label_consistency,
               "latent_pca_2d_variance_explained": latent_var, "ablation_mean_std_over_seeds": agg}
    save_json({"meta": run_metadata(cfg, seed), "summary": summary, "history": hist, "ablation_runs": ablation},
              out / "metrics.json")
    return summary


if __name__ == "__main__":
    import json

    print(json.dumps(run(), indent=2))
