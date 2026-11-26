"""Shape tests, gradient checks and save/load round trips on tiny inputs."""

import cv2
import numpy as np
import pytest

from traffic_monitor.fundamentals import conv, learning_rules, neurons, tensors
from traffic_monitor.models.cnn_numpy import SmallCNN
from traffic_monitor.models.detector import N_FEATURES, SignDetector, WindowScorer, crop_square, dense_windows, nms, window_features
from traffic_monitor.models.lstm_numpy import LSTMClassifier
from traffic_monitor.models.mlp_numpy import MLP, Momentum, Nesterov, Standardizer, gradient_check, train
from traffic_monitor.models.vae_numpy import CVAE

RNG = np.random.default_rng(0)


def _numeric_check(model, loss_fn, n=4):
    _, grads = loss_fn()[0], loss_fn()[-1]
    worst = 0.0
    for k, p in model.params.items():
        for _ in range(n):
            idx = tuple(RNG.integers(s) for s in p.shape)
            old = p[idx]
            p[idx] = old + 1e-5
            lp = loss_fn()[0]
            p[idx] = old - 1e-5
            lm = loss_fn()[0]
            p[idx] = old
            num = (lp - lm) / 2e-5
            worst = max(worst, abs(num - grads[k][idx]) / max(abs(num) + abs(grads[k][idx]), 1e-12))
    return worst


@pytest.mark.parametrize("act", ["relu", "sigmoid", "tanh"])
def test_mlp_gradient_check(act):
    x, y = RNG.normal(size=(10, 4)), RNG.integers(0, 3, 10)
    assert gradient_check(MLP([4, 6, 3], act, weight_decay=0.01), x, y) < 1e-5


def test_mlp_loss_decreases_on_xor_and_shapes():
    x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], float)
    y = np.array([0, 1, 1, 0])
    m = MLP([2, 8, 2], "tanh", "xavier", seed=1)
    hist = train(m, x, y, Nesterov(0.1), epochs=300, batch_size=None)
    assert hist["train_loss"][-1] < hist["train_loss"][0] * 0.2
    assert (m.predict(x) == y).all()
    assert m.predict_proba(x).shape == (4, 2)


def test_mlp_save_load_and_missing(tmp_path):
    m = MLP([3, 4, 2])
    std = Standardizer().fit(RNG.normal(size=(5, 3)))
    m.save(tmp_path / "m.npz", {"mean": std.mean, "std": std.std}, {"note": "x"})
    m2, extras, meta = MLP.load(tmp_path / "m.npz")
    x = RNG.normal(size=(2, 3))
    np.testing.assert_allclose(m.predict_proba(x), m2.predict_proba(x))
    assert meta["note"] == "x" and "mean" in extras
    with pytest.raises(FileNotFoundError):
        MLP.load(tmp_path / "missing.npz")


def test_mlp_invalid_config():
    with pytest.raises(ValueError):
        MLP([3])
    with pytest.raises(ValueError):
        MLP([3, 2], activation="swish")


def test_cnn_shapes_and_gradients():
    m = SmallCNN(3, size=8, c1=2, c2=3, hidden=5, dtype=np.float64)
    x, y = RNG.random((2, 8, 8, 3)), np.array([0, 2])
    assert m.predict_proba(x).shape == (2, 3)
    assert m.feature_maps(x).shape == (2, 2, 8, 8)
    assert _numeric_check(m, lambda: m.loss_and_grads(x, y), n=2) < 1e-5
    with pytest.raises(ValueError):
        SmallCNN(3, size=10)


def test_lstm_shapes_mask_and_gradients():
    m = LSTMClassifier(3, 4, 2, weight_decay=0.01)
    x, y = RNG.normal(size=(3, 5, 3)), np.array([0, 1, 1])
    mask = np.ones((3, 5))
    mask[0, 3:] = 0
    assert m.predict_proba(x, mask).shape == (3, 2)
    assert _numeric_check(m, lambda: m.loss_and_grads(x, y, mask)) < 1e-5
    # masked trailing frames must not change the output
    x2 = x.copy()
    x2[0, 3:] = 99.0
    np.testing.assert_allclose(m.predict_proba(x, mask)[0], m.predict_proba(x2, mask)[0])


def test_vae_shapes_and_gradients():
    m = CVAE(3, 12, 6, 2)
    x, y, eps = RNG.random((4, 12)), np.array([0, 1, 2, 0]), RNG.normal(size=(4, 2))
    loss, recon, kl, g = m.loss_and_grads(x, y, eps)
    assert loss == pytest.approx(recon + kl)
    worst = 0.0
    for k, p in m.params.items():
        idx = tuple(RNG.integers(s) for s in p.shape)
        old = p[idx]
        p[idx] = old + 1e-6
        lp = m.loss_and_grads(x, y, eps)[0]
        p[idx] = old - 1e-6
        lm = m.loss_and_grads(x, y, eps)[0]
        p[idx] = old
        num = (lp - lm) / 2e-6
        worst = max(worst, abs(num - g[k][idx]) / max(abs(num) + abs(g[k][idx]), 1e-12))
    assert worst < 1e-4
    assert m.sample([0, 1], RNG).shape == (2, 12)


def test_detector_empty_image_and_no_detections(tmp_path):
    std = Standardizer(np.zeros(N_FEATURES), np.ones(N_FEATURES))
    det = SignDetector(WindowScorer("logreg", std, np.zeros(N_FEATURES, np.float32), -20.0), score_threshold=0.5)
    boxes, scores = det.detect(np.zeros((60, 80, 3), np.uint8))
    assert boxes.shape == (0, 4) and scores.shape == (0,)
    with pytest.raises(ValueError):
        det.detect(np.zeros((0, 0, 3), np.uint8))
    det.save(tmp_path / "d.npz")
    assert SignDetector.load(tmp_path / "d.npz").score_threshold == 0.5
    with pytest.raises(FileNotFoundError):
        SignDetector.load(tmp_path / "missing.npz")


def test_nms_keeps_highest_and_separate():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 10, 10], [50, 50, 60, 60]], float)
    keep = nms(boxes, np.array([0.9, 0.8, 0.7]), 0.3)
    assert keep.tolist() == [0, 2]


# ---------- fundamentals ----------
def test_mcculloch_pitts_gates_and_activations():
    g = neurons.logic_gate_demo()
    assert g["AND(w=[1,1], θ=2)"] == [0, 0, 0, 1]
    assert g["OR(w=[1,1], θ=1)"] == [0, 1, 1, 1]
    assert g["NOT(w=[-1], θ=0)"] == [1, 0]
    assert neurons.sigmoid(np.array([0.0]))[0] == pytest.approx(0.5)
    assert np.isfinite(neurons.sigmoid(np.array([-1000.0, 1000.0]))).all()


def test_numerical_gradient_and_svd():
    f = lambda v: float(np.sum(v ** 3))  # noqa: E731
    v = np.array([1.0, -2.0])
    np.testing.assert_allclose(tensors.numerical_gradient(f, v.copy()), 3 * v ** 2, rtol=1e-6)
    img = RNG.random((8, 8))
    res = tensors.svd_low_rank(img, (8,))
    assert res["relative_error"][8] < 1e-10


def test_conv_shapes_and_values():
    x = np.arange(16, dtype=float).reshape(4, 4)
    k = np.array([[1.0, 0], [0, -1]])
    out = conv.conv2d(x, k)
    assert out.shape == (3, 3) and out[0, 0] == pytest.approx(0 - 5)
    assert conv.conv_output_size(32, 5, 1, 2) == 32
    assert conv.conv2d_multichannel(RNG.random((3, 6, 6)), RNG.random((4, 3, 3, 3)), padding=1).shape == (4, 6, 6)
    assert conv.conv3d(RNG.random((5, 6, 6)), RNG.random((2, 3, 3))).shape == (4, 4, 4)
    with pytest.raises(ValueError):
        conv.conv_output_size(2, 5)


def test_learning_rules_behave():
    x = RNG.multivariate_normal([0, 0], [[3, 1], [1, 1]], 200)
    w = learning_rules.oja(x, lr=0.005, epochs=20)
    pc1 = np.linalg.eigh(np.cov(x.T))[1][:, -1]
    assert abs(w @ pc1) / np.linalg.norm(w) > 0.98
    norms = learning_rules.hebbian(x / 3, epochs=5)
    assert norms[-1] > norms[0]
    t = x @ np.array([1.0, -1.0]) + 0.5
    wd, mse = learning_rules.delta_rule(x, t, lr=0.01, epochs=30)
    assert mse[-1] < 1e-3
    hop = learning_rules.Hopfield(16)
    pat = np.where(RNG.random(16) > 0.5, 1, -1)
    hop.store(pat[None])
    noisy = pat.copy()
    noisy[:3] *= -1
    rec, energies = hop.recall(noisy)
    assert np.array_equal(rec, pat) and all(b <= a + 1e-9 for a, b in zip(energies, energies[1:]))


def test_momentum_optimizer_updates():
    params = {"w": np.array([1.0])}
    opt = Momentum(0.1, 0.9)
    opt.step(params, {"w": np.array([1.0])})
    opt.step(params, {"w": np.array([1.0])})
    assert params["w"][0] == pytest.approx(1 - 0.1 - (0.09 + 0.1))


def test_dense_window_features_match_crop_features():
    """Training crops and sliding windows must produce (nearly) the same features."""
    img = np.zeros((96, 96, 3), np.uint8)
    cv2.circle(img, (48, 48), 10, (0, 0, 255), 3)                # a red ring on black
    feats, boxes = next(dense_windows(img, (24,), stride=6))
    i = int(np.argmin(np.abs(boxes[:, :2] - [36, 36]).sum(1)))    # window at (36,36)-(60,60)
    crop = window_features(crop_square(img, boxes[i])[None])[0]
    assert feats.shape[1] == N_FEATURES
    np.testing.assert_allclose(feats[i, 324:], crop[324:], atol=1e-5)   # colour part identical
    assert feats[i, 324] > 0.05                                         # redness detected
