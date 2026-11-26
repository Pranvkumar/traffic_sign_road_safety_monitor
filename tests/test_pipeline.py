"""Pipeline, CLI and dashboard-helper tests using tiny generated checkpoints."""

import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from traffic_monitor.cli import main
from traffic_monitor.models.detector import N_FEATURES, SignDetector, WindowScorer
from traffic_monitor.models.mlp_numpy import MLP, Standardizer
from traffic_monitor.pipeline.inference import Pipeline, read_image
from traffic_monitor.pipeline.video import predict_video

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def tiny_config(tmp_path):
    """Detector that never fires + an untrained 43-class classifier, wired into a config file."""
    std = Standardizer(np.zeros(N_FEATURES), np.ones(N_FEATURES))
    SignDetector(WindowScorer("logreg", std, np.zeros(N_FEATURES, np.float32), -20.0), 0.5).save(tmp_path / "det.npz")
    clf = MLP([372, 8, 43])
    clf.save(tmp_path / "clf.npz", {"mean": np.zeros(372), "std": np.ones(372)},
             {"class_names": {str(i): f"c{i}" for i in range(43)}, "preprocessing": {"crop_size": 32}})
    cfg = {"detector": {"checkpoint": str(tmp_path / "det.npz"), "score_threshold": None},
           "classifier": {"checkpoint": str(tmp_path / "clf.npz"), "unknown_threshold": 0.6}}
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path


def test_missing_checkpoint_raises(tmp_path):
    cfg = tmp_path / "c.yaml"
    cfg.write_text(yaml.safe_dump({"detector": {"checkpoint": str(tmp_path / "none.npz")}, "classifier": {"checkpoint": "x"}}))
    with pytest.raises(FileNotFoundError):
        Pipeline.from_config(str(cfg))


def test_read_image_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_image(tmp_path / "missing.jpg")
    (tmp_path / "a.txt").write_text("x")
    with pytest.raises(ValueError):
        read_image(tmp_path / "a.txt")
    (tmp_path / "broken.jpg").write_bytes(b"not an image")
    with pytest.raises(OSError):
        read_image(tmp_path / "broken.jpg")


def test_empty_detection_output(tiny_config):
    pipe = Pipeline.from_config(str(tiny_config), use_segmenter=False)
    res = pipe.predict(np.zeros((60, 80, 3), np.uint8))
    assert res["n_signs"] == 0 and res["detections"] == []
    assert res["status"]["segmenter"] == "not loaded"
    assert "not safety-certified" in res["disclaimer"].lower() or "not safety" in res["disclaimer"].lower()


def test_classifier_unknown_policy(tiny_config):
    pipe = Pipeline.from_config(str(tiny_config), use_segmenter=False)
    out = pipe.classifier.classify([np.zeros((20, 20, 3), np.uint8)])
    assert out[0]["class_name"] == "unknown" and out[0]["class_id"] == -1   # near-uniform 43-way output


def test_video_end_of_file(tiny_config, tmp_path):
    path = tmp_path / "v.avi"
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 5, (80, 60))
    for i in range(3):
        w.write(np.full((60, 80, 3), i * 40, np.uint8))
    w.release()
    pipe = Pipeline.from_config(str(tiny_config), use_segmenter=False)
    out = predict_video(pipe, path, tmp_path / "out")
    assert out["frames_read"] == 3 and out["frames_processed"] == 3 and out["frames_with_signs"] == 0
    capped = predict_video(pipe, path, tmp_path / "out2", frame_stride=2, max_frames=1)
    assert capped["frames_read"] == 1 and capped["frames_processed"] == 1
    assert len(Path(out["frame_log"]).read_text().splitlines()) == 3
    with pytest.raises(ValueError):
        predict_video(pipe, tiny_config, tmp_path)


def test_cli_help_invalid_path_and_success(tiny_config, tmp_path, capsys):
    with pytest.raises(SystemExit) as e:
        main(["--help"])
    assert e.value.code == 0
    assert main(["predict", "--source", str(tmp_path / "nope.jpg"), "--config", str(tiny_config)]) == 2
    img = tmp_path / "frame.png"
    cv2.imwrite(str(img), np.zeros((60, 80, 3), np.uint8))
    capsys.readouterr()
    assert main(["predict", "--source", str(img), "--config", str(tiny_config), "--no-segmentation",
                 "--output", str(tmp_path / "pred")]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["n_signs"] == 0 and (tmp_path / "pred" / "frame.json").exists()


def test_dashboard_helpers_distinguish_states(tmp_path):
    spec = importlib.util.spec_from_file_location("dashboard", ROOT / "app" / "dashboard.py")
    dash = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dash)
    cfg = tmp_path / "c.yaml"
    cfg.write_text(yaml.safe_dump({"detector": {"checkpoint": str(tmp_path / "none.npz")}, "classifier": {"checkpoint": "x"}}))
    pipe, msg = dash.load_pipeline_safely(str(cfg))
    assert pipe is None and msg.startswith("Model not loaded")
    assert dash.describe_result(None, False).startswith("Model not loaded")
    assert "no traffic sign detected" in dash.describe_result({"n_signs": 0, "detections": []}, True)


def test_dashboard_plain_language_summary():
    spec = importlib.util.spec_from_file_location("dashboard", ROOT / "app" / "dashboard.py")
    dash = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dash)
    res = {"n_signs": 2, "road": {"road_coverage": 0.31},
           "timing_ms": {"total_ms": 1500, "detection_ms": 1400, "classification_ms": 1, "segmentation_ms": 20},
           "detections": [{"box_xyxy": [10, 20, 40, 52], "detection_score": 0.93, "class_id": 25,
                           "class_name": "road_works", "class_confidence": 0.95, "top_guess": "road_works"},
                          {"box_xyxy": [100, 20, 120, 40], "detection_score": 0.91, "class_id": -1,
                           "class_name": "unknown", "class_confidence": 0.36, "top_guess": "bumpy_road"}]}
    rows = dash.detections_table(res)
    assert rows[0]["Sign"] == "Road works" and rows[0]["Size"] == "30×32 px"
    assert rows[1]["Sign"] == "Unknown sign" and "Bumpy road" in rows[1]["Comment"]
    text = " ".join(dash.human_summary(res))
    assert "Road works" in text and "31%" in text and "1.5 s" in text
    assert dash.describe_result(res, True) == "Found 2 possible traffic signs."


def test_hazard_missing_checkpoint_is_reported(tiny_config, tmp_path):
    """A configured but absent hazard model must be reported, never faked."""
    import yaml
    cfg = yaml.safe_load(open(tiny_config, encoding="utf-8"))
    cfg["hazard"] = {"checkpoint": str(tmp_path / "nope.pt")}
    p = tmp_path / "cfg_h.yaml"
    p.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    pipe = Pipeline.from_config(str(p), use_segmenter=False)
    res = pipe.predict(np.zeros((60, 80, 3), np.uint8))
    assert res["hazards"] is None and res["n_hazards"] is None
    assert "checkpoint missing" in res["status"]["hazard_detector"]
