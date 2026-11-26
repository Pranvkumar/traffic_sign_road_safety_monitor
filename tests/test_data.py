import numpy as np
import pandas as pd
import pytest

from traffic_monitor.data.annotations import (clip_boxes, read_yolo_file, validate_yolo, xyxy_to_coco, xyxy_to_yolo,
                                              yolo_to_xyxy)
from traffic_monitor.data.manifest import build_gtsrb_manifest, build_yolo_manifest, load_manifest, save_manifest
from traffic_monitor.data.splits import check_group_disjoint, group_split
from traffic_monitor.data.validators import audit_manifest


# ---------- manifests ----------
def test_gtsrb_manifest_ids_tracks_and_roundtrip(gtsrb_folder, tmp_path):
    df = build_gtsrb_manifest(gtsrb_folder)
    assert len(df) == 3 * 6 + 3 * 2
    assert df["sample_id"].is_unique
    train = df[df.official_split == "train"]
    assert set(train["split"]) == {"unassigned"} and set(df[df.official_split == "test"]["split"]) == {"test"}
    assert train["physical_sign_id"].nunique() == 6            # 3 classes x 2 tracks
    assert (df["sha256"].str.len() == 64).all()
    path = save_manifest(df, tmp_path / "m.csv")
    back = load_manifest(path)
    pd.testing.assert_series_equal(back["sample_id"], df["sample_id"])


def test_yolo_manifest_sequences(yolo_folder):
    df = build_yolo_manifest(yolo_folder)
    assert len(df) == 4 and df["sequence_id"].nunique() == 2
    assert (df["annotation_path"] != "").all()


def test_missing_manifest_and_folder_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_manifest(tmp_path / "nope.csv")
    with pytest.raises(FileNotFoundError):
        build_gtsrb_manifest(tmp_path / "empty")


# ---------- splits ----------
def _toy(n_groups=20, per=5):
    return pd.DataFrame({"g": np.repeat([f"g{i}" for i in range(n_groups)], per),
                         "label": np.repeat(np.arange(n_groups) % 2, per), "split": "unassigned"})


def test_group_split_disjoint_deterministic_and_keeps_existing():
    df = _toy()
    df.loc[:4, "split"] = "test"                                     # pre-assigned (official) rows
    a = group_split(df, "g", {"train": 0.7, "val": 0.3}, seed=1)
    b = group_split(df, "g", {"train": 0.7, "val": 0.3}, seed=1)
    assert a["split"].tolist() == b["split"].tolist()
    assert check_group_disjoint(a, "g")["leaked_groups"] == []
    assert (a.loc[:4, "split"] == "test").all()
    assert set(a["split"]) == {"train", "val", "test"}


def test_group_split_stratified_covers_every_class():
    out = group_split(_toy(), "g", {"train": 0.7, "val": 0.3}, seed=0, stratify_col="label")
    assert out.groupby("split")["label"].nunique().min() == 2


def test_group_split_rejects_bad_input():
    df = _toy()
    with pytest.raises(ValueError):
        group_split(df, "g", {"train": 0.5, "val": 0.3})
    with pytest.raises(KeyError):
        group_split(df, "missing", {"train": 1.0})
    df.loc[0, "g"] = ""
    with pytest.raises(ValueError):
        group_split(df, "g", {"train": 1.0})


# ---------- annotations ----------
def test_yolo_xyxy_roundtrip_and_coco():
    yolo = np.array([[0.5, 0.5, 0.2, 0.4]])
    xyxy = yolo_to_xyxy(yolo, 100, 50)
    np.testing.assert_allclose(xyxy, [[40, 15, 60, 35]])
    np.testing.assert_allclose(xyxy_to_yolo(xyxy, 100, 50), yolo)
    np.testing.assert_allclose(xyxy_to_coco(xyxy), [[40, 15, 20, 20]])
    np.testing.assert_allclose(clip_boxes([[-5, -5, 120, 60]], 100, 50), [[0, 0, 100, 50]])


def test_validate_yolo_flags_problems():
    rows = np.array([[0, 0.5, 0.5, 0.1, 0.1], [3, 0.5, 0.5, 0.1, 0.1], [0, 0.99, 0.5, 0.1, 0.1], [0, 0.5, 0.5, 0, 0.1]])
    problems = validate_yolo(rows, {0})
    assert len(problems) == 3
    assert any("unknown class" in p for p in problems)


def test_read_yolo_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_yolo_file(tmp_path / "x.txt")
    bad = tmp_path / "bad.txt"
    bad.write_text("0 0.5 0.5\n")
    with pytest.raises(ValueError):
        read_yolo_file(bad)
    empty = tmp_path / "e.txt"
    empty.write_text("")
    assert read_yolo_file(empty).shape == (0, 5)


# ---------- audit ----------
def test_audit_reports_counts_and_leakage(gtsrb_folder):
    df = build_gtsrb_manifest(gtsrb_folder)
    df = group_split(df, "physical_sign_id", {"train": 0.5, "val": 0.5}, seed=0)
    rep = audit_manifest(df, gtsrb_folder, "physical_sign_id")
    assert rep["n_missing_files"] == 0 and rep["n_unreadable_files"] == 0
    assert rep["n_classes"] == 3 and rep["n_leaked_groups"] == 0
    df.loc[0, "split"], df.loc[1, "split"] = "train", "val"   # force leakage inside one track
    assert audit_manifest(df, gtsrb_folder, "physical_sign_id", decode=False)["n_leaked_groups"] == 1


def test_audit_detection_boxes(yolo_folder):
    rep = audit_manifest(build_yolo_manifest(yolo_folder), yolo_folder, "sequence_id", {0})
    assert rep["detection"]["n_boxes"] == 3 and rep["detection"]["n_box_problems"] == 0
