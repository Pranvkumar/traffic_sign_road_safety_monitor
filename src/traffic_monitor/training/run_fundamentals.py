"""Run every neural-network fundamentals demonstration and save figures + JSON.

Outputs go to outputs/figures/fundamentals_* and experiments/runs/fundamentals/results.json.
Uses a small, fixed GTSRB subset (train/val splits only — never the test set).
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from ..data.transforms import augment_crop, to_gray  # noqa: E402
from ..fundamentals import conv, learning_rules, neurons, tensors  # noqa: E402
from ..models.cnn_numpy import SmallCNN  # noqa: E402
from ..models.features import sign_features  # noqa: E402
from ..models.mlp_numpy import MLP, Standardizer, gradient_check, make_optimizer, train  # noqa: E402
from ..utils.config import resolve  # noqa: E402
from ..utils.logging import get_logger  # noqa: E402
from ..utils.reproducibility import save_json, set_seed  # noqa: E402
from .common import as_float, gtsrb_arrays  # noqa: E402

log = get_logger(__name__)
FIG = resolve("outputs/figures")


def _save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG / f"fundamentals_{name}.png", dpi=110)
    plt.close(fig)


def _subset(x, y, per_class, rng):
    idx = np.concatenate([rng.choice(np.where(y == c)[0], min(per_class, (y == c).sum()), replace=False)
                          for c in np.unique(y)])
    return x[idx], y[idx]


def activations_demo() -> dict:
    z = np.linspace(-5, 5, 400)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for name, f, g in [("step", neurons.step, None), ("sigmoid", neurons.sigmoid, neurons.sigmoid_grad),
                       ("tanh", neurons.tanh, neurons.tanh_grad)]:
        axes[0].plot(z, f(z), label=name)
        if g is not None:
            axes[1].plot(z, g(z), label=f"d{name}/dz")
    axes[1].axhline(0, color="grey", lw=0.5)
    axes[0].set_title("Activation functions")
    axes[1].set_title("Derivatives (step: 0 everywhere except z=0 → no gradient signal)")
    for ax in axes:
        ax.set_xlabel("z (weighted input)")
        ax.legend()
    _save(fig, "activations")
    return {"logic_gates": neurons.logic_gate_demo(),
            "sigmoid_neuron_example": {k: float(v) for k, v in
                                       neurons.sigmoid_neuron([1.0, 0.5], [0.8, -0.4], 0.1).items()}}


def svd_demo(sign_gray: np.ndarray) -> dict:
    res = tensors.svd_low_rank(sign_gray)
    fig, axes = plt.subplots(1, 6, figsize=(14, 2.8))
    axes[0].imshow(sign_gray, cmap="gray")
    axes[0].set_title("original 32x32")
    for ax, (k, im) in zip(axes[1:5], res["reconstructions"].items()):
        ax.imshow(im, cmap="gray")
        ax.set_title(f"rank {k}, err {res['relative_error'][k]:.2f}")
    for ax in axes[:5]:
        ax.axis("off")
    axes[5].semilogy(res["singular_values"], "o-", ms=3)
    axes[5].set_title("singular values")
    axes[5].set_xlabel("index")
    _save(fig, "svd")
    return {"relative_error_by_rank": res["relative_error"]}


def gradient_check_demo(feats, y) -> dict:
    f = lambda v: float(np.sum(np.sin(v) * v ** 2))  # noqa: E731
    v = np.array([0.3, -1.2, 2.0])
    analytic = np.cos(v) * v ** 2 + 2 * v * np.sin(v)
    num = tensors.numerical_gradient(f, v.copy())
    x64 = feats[:40].astype(np.float64)
    out = {"toy_function_relative_error": tensors.relative_error(num, analytic)}
    for act in ("relu", "sigmoid", "tanh"):
        m = MLP([x64.shape[1], 16, 43], act, weight_decay=1e-3, seed=0)
        out[f"mlp_{act}_max_relative_error"] = gradient_check(m, x64, y[:40], n_checks=40)
    out["tolerance"] = 1e-5
    out["passed"] = all(v < 1e-5 for k, v in out.items() if k.endswith("error"))
    return out


def optimizer_demo(xtr, ytr, xva, yva, seed) -> dict:
    settings = {"batch GD (full batch)": ("gd", 0.5, None), "SGD (bs 64)": ("sgd", 0.01, 64),
                "Momentum (bs 64)": ("momentum", 0.01, 64), "Nesterov (bs 64)": ("nesterov", 0.01, 64)}
    hists = {}
    for name, (opt, lr, bs) in settings.items():
        m = MLP([xtr.shape[1], 128, 43], "relu", "he", seed=seed)
        hists[name] = train(m, xtr, ytr, make_optimizer(opt, lr), 30, bs, xva, yva, seed=seed, keep_best=False)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for name, h in hists.items():
        axes[0].plot(h["train_loss"], label=name)
        axes[1].plot(h["val_acc"], label=name)
    axes[0].set_yscale("log")
    axes[0].set_title("Training loss per epoch")
    axes[1].set_title("Validation accuracy per epoch")
    for ax in axes:
        ax.set_xlabel("epoch")
        ax.legend(fontsize=8)
    _save(fig, "optimizers")
    return {name: {"lr": settings[name][1], "batch_size": settings[name][2] or "full",
                   "final_train_loss": h["train_loss"][-1], "final_val_acc": h["val_acc"][-1],
                   "best_val_acc": max(h["val_acc"])} for name, h in hists.items()}


def ablation_demo(img_tr, ytr, ftr, fva, yva, seed) -> dict:
    """Change exactly one factor at a time relative to the baseline configuration."""
    rng = np.random.default_rng(seed)
    base = {"init": "he", "standardise": True, "weight_decay": 1e-4, "augment": False}
    variants = {"baseline": {}, "init=small(0.01)": {"init": "small"}, "no standardisation": {"standardise": False},
                "weight_decay=0": {"weight_decay": 0.0}, "weight_decay=1e-2": {"weight_decay": 1e-2},
                "augmentation x2 copies": {"augment": True}}
    aug_feats = np.vstack([sign_features(np.stack([augment_crop(im, rng) for im in img_tr])) for _ in range(2)])
    results = {}
    for name, change in variants.items():
        s = {**base, **change}
        xt, yt = (np.vstack([ftr, aug_feats]), np.r_[ytr, ytr, ytr]) if s["augment"] else (ftr, ytr)
        std = Standardizer().fit(xt) if s["standardise"] else Standardizer()
        m = MLP([xt.shape[1], 128, 43], "relu", s["init"], weight_decay=s["weight_decay"], seed=seed)
        h = train(m, std.transform(xt), yt, make_optimizer("momentum", 0.01), 20, 64,
                  std.transform(fva), yva, seed=seed, keep_best=False)
        tr_acc = float(np.mean(m.predict(std.transform(ftr)) == ytr))
        results[name] = {"settings": s, "train_acc": tr_acc, "val_acc": h["val_acc"][-1],
                         "generalisation_gap": tr_acc - h["val_acc"][-1]}
    fig, ax = plt.subplots(figsize=(8, 3.6))
    names = list(results)
    ax.barh(names, [results[n]["val_acc"] for n in names], color="#4c72b0", label="validation")
    ax.scatter([results[n]["train_acc"] for n in names], names, color="#dd8452", zorder=3, label="train")
    ax.set_xlabel("accuracy (HOG-MLP, 150 imgs/class subset)")
    ax.set_xlim(0.5, 1.0)
    ax.legend(loc="lower right")
    ax.set_title("One-factor ablations")
    _save(fig, "ablations")
    return results


def learning_rules_demo(seed) -> dict:
    rng = np.random.default_rng(seed)
    cov = np.array([[3.0, 1.2], [1.2, 1.0]])
    x = rng.multivariate_normal([0, 0], cov, 300)
    hebb_norms = learning_rules.hebbian(x / 3, lr=0.01, epochs=10)
    w_oja = learning_rules.oja(x, lr=0.005, epochs=30)
    pc1 = np.linalg.eigh(np.cov(x.T))[1][:, -1]
    clusters = np.vstack([rng.normal(c, 0.3, (100, 2)) for c in ([0, 0], [3, 3], [0, 3])])
    protos = learning_rules.competitive(clusters, 3, lr=0.2, epochs=15)
    t = x @ np.array([1.5, -0.7]) + 0.3 + rng.normal(0, 0.05, len(x))
    w_delta, mse = learning_rules.delta_rule(x, t, lr=0.01, epochs=30)
    # Hopfield on three 8x8 glyphs (+1/-1): ring, cross, triangle
    yy, xx = np.mgrid[:8, :8]
    ring = np.where(np.abs(np.hypot(yy - 3.5, xx - 3.5) - 3) < 0.9, 1, -1)
    cross = np.where((np.abs(yy - xx) < 1) | (np.abs(yy + xx - 7) < 1), 1, -1)
    tri = np.where((yy >= 7 - 2 * np.minimum(xx, 7 - xx)) & (yy <= 7), 1, -1)
    pats = np.stack([ring, cross, tri]).reshape(3, -1)
    hop = learning_rules.Hopfield(64)
    hop.store(pats)
    noisy = pats[0].copy()
    flip = rng.choice(64, 10, replace=False)
    noisy[flip] *= -1
    recalled, energies = hop.recall(noisy, steps=5)

    fig, axes = plt.subplots(1, 5, figsize=(17, 3.3))
    axes[0].plot(hebb_norms, "o-")
    axes[0].set_title("Plain Hebb: |w| grows unbounded")
    axes[0].set_xlabel("epoch")
    axes[1].scatter(*x.T, s=4, alpha=0.4)
    for vec, lab, c in [(pc1 * 3, "PCA PC1", "k"), (w_oja / np.linalg.norm(w_oja) * 3 * np.sign(w_oja @ pc1), "Oja w", "r")]:
        axes[1].arrow(0, 0, *vec, color=c, width=0.05, label=lab)
    axes[1].legend()
    axes[1].set_title("Modified Hebb (Oja) → PC1")
    axes[1].axis("equal")
    axes[2].scatter(*clusters.T, s=4, alpha=0.4)
    axes[2].scatter(*protos.T, c="r", marker="x", s=120)
    axes[2].set_title("Competitive learning prototypes")
    axes[3].plot(mse)
    axes[3].set_yscale("log")
    axes[3].set_title("Error-correction (delta rule) MSE")
    axes[3].set_xlabel("epoch")
    axes[4].imshow(np.hstack([noisy.reshape(8, 8), np.ones((8, 1)), recalled.reshape(8, 8)]), cmap="gray")
    axes[4].set_title(f"Hopfield: noisy → recalled\nenergy {energies[0]:.1f} → {energies[-1]:.1f}")
    axes[4].axis("off")
    _save(fig, "learning_rules")
    return {"hebbian_weight_norm_first_last": [float(hebb_norms[0]), float(hebb_norms[-1])],
            "oja_cosine_with_pc1": float(abs(w_oja @ pc1) / np.linalg.norm(w_oja)),
            "oja_weight_norm": float(np.linalg.norm(w_oja)),
            "competitive_prototypes": protos.round(2).tolist(),
            "delta_rule_weights(w1,w2,bias)": w_delta.round(3).tolist(), "delta_rule_true": [1.5, -0.7, 0.3],
            "hopfield_energy_trace": [round(e, 2) for e in energies],
            "hopfield_recalled_exactly": bool(np.array_equal(recalled, pats[0])),
            "note": "Syllabus demonstrations only; not used by the production pipeline."}


def convolution_demo(sign_rgb: np.ndarray, track: np.ndarray) -> dict:
    gray = to_gray(sign_rgb)
    gx, gy = conv.conv2d(gray, conv.SOBEL_X, padding=1), conv.conv2d(gray, conv.SOBEL_Y, padding=1)
    chw = sign_rgb.transpose(2, 0, 1)
    rng = np.random.default_rng(0)
    kernels = rng.normal(0, 1, (4, 3, 3, 3))
    stacked = conv.conv2d_multichannel(chw, kernels, padding=1)
    vol = to_gray(track)                                  # (T, 32, 32)
    k3 = np.zeros((2, 3, 3))
    k3[0], k3[1] = -1 / 9, 1 / 9                          # temporal difference of 3x3 local means
    motion = conv.conv3d(vol, k3)

    fig, axes = plt.subplots(2, 5, figsize=(14, 5.6))
    axes[0, 0].imshow(sign_rgb)
    axes[0, 0].set_title("input 32x32x3")
    axes[0, 1].imshow(gx, cmap="RdBu")
    axes[0, 1].set_title("2D conv: Sobel-x\n(3x3, pad 1, stride 1 → 32x32)")
    axes[0, 2].imshow(gy, cmap="RdBu")
    axes[0, 2].set_title("2D conv: Sobel-y")
    axes[0, 3].imshow(stacked[0], cmap="viridis")
    axes[0, 3].set_title("multi-channel filter 1 of 4\n(3x3x3 kernel sums 3 channels)")
    axes[0, 4].imshow(stacked[1], cmap="viridis")
    axes[0, 4].set_title("filter 2 of 4 → stack 4x32x32")
    for t in range(3):
        axes[1, t].imshow(vol[t], cmap="gray")
        axes[1, t].set_title(f"track frame t={t}")
    axes[1, 3].imshow(motion[0], cmap="RdBu")
    axes[1, 3].set_title("3D conv (2x3x3 temporal diff)\nframes 0→1, out 30x30")
    axes[1, 4].imshow(motion[1], cmap="RdBu")
    axes[1, 4].set_title("3D conv frames 1→2")
    for ax in axes.ravel():
        ax.axis("off")
    fig.suptitle("Educational convolutions on real GTSRB crops (the production CNN uses 2D convolutions only)")
    _save(fig, "convolutions")
    shapes = [{"input": n, "kernel": k, "stride": s, "padding": p, "output": conv.conv_output_size(n, k, s, p)}
              for n, k, s, p in [(32, 3, 1, 1), (32, 5, 1, 2), (32, 3, 2, 1), (32, 5, 1, 0), (16, 3, 1, 1)]]
    return {"shape_calculations(out=floor((n+2p-k)/s)+1)": shapes, "multichannel_output_shape": list(stacked.shape),
            "conv3d_input_shape": list(vol.shape), "conv3d_output_shape": list(motion.shape)}


def cnn_feature_maps(sign_rgb: np.ndarray) -> dict:
    path = resolve("experiments/runs/classifier/small_cnn.npz")
    if not path.exists():
        return {"skipped": "small_cnn checkpoint not trained yet"}
    model, _, _ = SmallCNN.load(path)
    maps = model.feature_maps(sign_rgb[None])[0]
    k = model.params["K1"].reshape(3, 5, 5, -1)
    fig, axes = plt.subplots(2, 8, figsize=(14, 3.8))
    for i in range(8):
        kern = k[..., i].transpose(1, 2, 0)
        axes[0, i].imshow((kern - kern.min()) / (np.ptp(kern) + 1e-9))
        axes[0, i].set_title(f"kernel {i}", fontsize=8)
        axes[1, i].imshow(maps[i], cmap="magma")
    for ax in axes.ravel():
        ax.axis("off")
    fig.suptitle("Trained NumPy CNN: first-layer 5x5x3 kernels (top) and their ReLU feature maps (bottom)")
    _save(fig, "cnn_feature_maps")
    return {"conv1_kernels": int(k.shape[-1])}


def run(seed: int = 42) -> dict:
    rng = set_seed(seed)
    data = gtsrb_arrays()
    xtr_u8, ytr = _subset(data["X_train"], data["y_train"], 150, rng)
    xva_u8, yva = _subset(data["X_val"], data["y_val"], 40, rng)
    img_tr, img_va = as_float(xtr_u8), as_float(xva_u8)
    ftr, fva = sign_features(img_tr), sign_features(img_va)
    std = Standardizer().fit(ftr)
    sign = as_float(data["X_train"][data["y_train"] == 14][15])          # a 'stop' sign crop
    groups = data["group_train"]
    g0 = groups[data["y_train"] == 25][0]                                  # one road-works track
    track_idx = np.where(groups == g0)[0]
    track = as_float(data["X_train"][track_idx[np.argsort(data["frame_train"][track_idx])][-3:]])

    results = {"tensor_basics": tensors.tensor_basics(), "activations_and_neurons": activations_demo(),
               "svd": svd_demo(to_gray(sign)), "gradient_check": gradient_check_demo(std.transform(ftr), ytr)}
    log.info("optimizers ...")
    results["optimizers"] = optimizer_demo(std.transform(ftr), ytr, std.transform(fva), yva, seed)
    log.info("ablations ...")
    results["ablations"] = ablation_demo(img_tr, ytr, ftr, fva, yva, seed)
    results["learning_rules"] = learning_rules_demo(seed)
    results["convolution"] = convolution_demo(sign, track)
    results["cnn_feature_maps"] = cnn_feature_maps(sign)
    m = MLP([ftr.shape[1], 256, 43])
    results["parameters_vs_hyperparameters"] = {
        "trainable_parameters(HOG-MLP 372-256-43)": m.n_parameters(),
        "trainable_parameters(SmallCNN)": SmallCNN(43).n_parameters(),
        "hyperparameters_examples": {"hidden_sizes": [256], "learning_rate": 0.02, "batch_size": 128,
                                     "epochs": 15, "weight_decay": 1e-4, "init": "he", "optimizer": "momentum"},
        "note": "Parameters are learned by gradient descent; hyperparameters are chosen by the experimenter "
                "using the validation split.",
    }
    results["data_subset"] = {"train_images": int(len(ytr)), "val_images": int(len(yva)), "per_class_train": 150}
    save_json(results, resolve("experiments/runs/fundamentals/results.json"))
    return results


if __name__ == "__main__":
    import json

    r = run()
    print(json.dumps({k: r[k] for k in ("gradient_check", "optimizers", "ablations")}, indent=1, default=str))
