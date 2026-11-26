"""Dataset audit: decode checks, duplicates, annotation sanity, class balance and leakage."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .annotations import read_yolo_file, validate_yolo


def audit_manifest(df: pd.DataFrame, root: str | Path, group_col: str = "sequence_id",
                   known_classes: set[int] | None = None, decode: bool = True) -> dict:
    """Run all checks and return a JSON-serialisable report. Nothing is deleted."""
    root = Path(root)
    report: dict = {"n_samples": int(len(df)), "source_datasets": sorted(df["source_dataset"].unique().tolist())}

    # 1. files exist / decode
    missing, unreadable = [], []
    for rel in df["relative_path"]:
        p = root / rel
        if not p.exists():
            missing.append(rel)
        elif decode and cv2.imread(str(p), cv2.IMREAD_UNCHANGED) is None:
            unreadable.append(rel)
    report["missing_files"] = missing[:50]
    report["n_missing_files"] = len(missing)
    report["unreadable_files"] = unreadable[:50]
    report["n_unreadable_files"] = len(unreadable)

    # 2. exact and near duplicates
    if df["sha256"].astype(str).str.len().gt(0).all():
        dup = df[df["sha256"].duplicated(keep=False)]
        report["n_exact_duplicate_files"] = int(len(dup))
        cross = dup.groupby("sha256")["split"].nunique()
        report["n_exact_duplicates_across_splits"] = int((cross > 1).sum())
    if df["ahash"].astype(str).str.len().gt(0).all():
        near = df[df["ahash"].duplicated(keep=False)]
        report["n_near_duplicate_files(ahash)"] = int(len(near))
        cross = near.groupby("ahash").agg(splits=("split", "nunique"), groups=(group_col, "nunique"))
        report["n_near_duplicate_hashes_across_splits"] = int((cross["splits"] > 1).sum())
        report["n_near_duplicate_hashes_across_groups"] = int((cross["groups"] > 1).sum())

    # 3. image properties
    report["image_size"] = {
        "width_min": int(df["width"].min()), "width_max": int(df["width"].max()),
        "height_min": int(df["height"].min()), "height_max": int(df["height"].max()),
        "formats": dict(Counter(df["format"])), "channels": dict(Counter(df["channels"].astype(int))),
    }

    # 4. class balance (classification datasets)
    labelled = df[df["label_id"].astype(int) >= 0]
    if not labelled.empty:
        counts = labelled["label_id"].astype(int).value_counts().sort_index()
        report["class_counts"] = {int(k): int(v) for k, v in counts.items()}
        report["n_classes"] = int(len(counts))
        report["imbalance_ratio_max_over_min"] = float(counts.max() / counts.min())

    # 5. detection annotations
    det = df[df["annotation_format"] == "yolo_txt"]
    if not det.empty:
        n_boxes, problems, no_ann, sizes = 0, [], 0, []
        known = known_classes if known_classes is not None else {0}
        for _, row in det.iterrows():
            if not row["annotation_path"]:
                no_ann += 1
                continue
            rows = read_yolo_file(root / row["annotation_path"])
            n_boxes += len(rows)
            for msg in validate_yolo(rows, known):
                problems.append(f"{row['sample_id']}: {msg}")
            sizes.extend((rows[:, 3] * row["width"]).tolist())
        report["detection"] = {
            "images_without_annotation": no_ann, "n_boxes": n_boxes,
            "box_problems": problems[:50], "n_box_problems": len(problems),
            "box_width_px": {"min": float(np.min(sizes)), "median": float(np.median(sizes)),
                             "max": float(np.max(sizes))} if sizes else {},
        }

    # 6. leakage across splits
    if "split" in df.columns and df["split"].nunique() > 1:
        per_group = df[df["split"] != "unassigned"].groupby(group_col)["split"].nunique()
        report["leaked_groups"] = per_group[per_group > 1].index.tolist()[:50]
        report["n_leaked_groups"] = int((per_group > 1).sum())
        report["split_counts"] = {str(k): int(v) for k, v in df["split"].value_counts().items()}
    return report


def report_to_markdown(report: dict, title: str) -> str:
    lines = [f"# Data audit — {title}", ""]
    for key, value in report.items():
        if isinstance(value, dict) and len(value) > 12:
            value = f"{len(value)} entries (see JSON)"
        lines.append(f"- **{key}**: {value}")
    return "\n".join(lines) + "\n"
