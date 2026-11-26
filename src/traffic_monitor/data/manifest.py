"""Build traceable sample manifests (one row per image) for the supported datasets.

Two source layouts are supported:

* ``gtsrb_folder`` - GTSRB crops arranged as ``<root>/{train,test}/<class_id:05d>/<file>.png``.
  Training files are named ``<track>_<frame>.png`` (one track = one physical sign).
* ``yolo_detection`` - full road-scene images in ``<root>/JPEGImages/<camera>/<sequence>/<name>.jpg``
  with YOLO labels in ``<root>/labels/<camera>/<sequence>/<name>.txt``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from PIL import Image

from ..utils.reproducibility import file_sha256

MANIFEST_COLUMNS = [
    "sample_id", "source_dataset", "source_release", "source_license", "source_url",
    "relative_path", "file_name", "sha256", "ahash", "media_type", "width", "height",
    "channels", "format", "sequence_id", "frame_index", "physical_sign_id", "label_id",
    "annotation_path", "annotation_format", "label_schema_version", "official_split",
    "split", "quality_status", "notes",
]

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".ppm", ".bmp"}


def average_hash(path: str | Path, size: int = 8) -> str:
    """Perceptual average-hash (64 bits as hex) for near-duplicate detection."""
    with Image.open(path) as im:
        small = np.asarray(im.convert("L").resize((size, size), Image.BILINEAR), dtype=np.float32)
    bits = (small > small.mean()).flatten()
    return f"{int(''.join('1' if b else '0' for b in bits), 2):0{size * size // 4}x}"


def _image_info(path: Path) -> tuple[int, int, int, str]:
    with Image.open(path) as im:
        w, h = im.size
        channels = len(im.getbands())
        fmt = im.format or path.suffix.lstrip(".").upper()
    return w, h, channels, fmt


def _base_row(path: Path, root: Path, meta: dict, with_hash: bool) -> dict:
    w, h, c, fmt = _image_info(path)
    return {
        "source_dataset": meta.get("source_dataset", ""),
        "source_release": meta.get("source_release", ""),
        "source_license": meta.get("source_license", ""),
        "source_url": meta.get("source_url", ""),
        "relative_path": str(path.relative_to(root)),
        "file_name": path.name,
        "sha256": file_sha256(path) if with_hash else "",
        "ahash": average_hash(path) if with_hash else "",
        "width": w, "height": h, "channels": c, "format": fmt,
        "label_schema_version": meta.get("label_schema_version", "v1"),
        "quality_status": "unchecked",
        "notes": "",
    }


def build_gtsrb_manifest(root: str | Path, meta: dict | None = None, with_hash: bool = True) -> pd.DataFrame:
    """Manifest for GTSRB crops. Official train -> split 'unassigned', official test -> 'test'."""
    root = Path(root)
    meta = {"source_dataset": "gtsrb", **(meta or {})}
    rows = []
    for official in ("train", "test"):
        split_dir = root / official
        if not split_dir.is_dir():
            raise FileNotFoundError(f"Expected GTSRB folder {split_dir} (layout <root>/train|test/<class>/)")
        for class_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            label = int(class_dir.name)
            for path in sorted(class_dir.iterdir()):
                if path.suffix.lower() not in IMAGE_EXTS:
                    continue
                row = _base_row(path, root, meta, with_hash)
                stem = path.stem
                if official == "train" and "_" in stem:
                    track, frame = stem.split("_", 1)
                    seq = f"gtsrb_c{label:02d}_t{track}"
                    row.update(sequence_id=seq, physical_sign_id=seq, frame_index=int(frame))
                else:  # official test images are independent single images
                    row.update(sequence_id=f"gtsrb_test_{stem}", physical_sign_id=f"gtsrb_test_{stem}",
                               frame_index=0)
                row.update(sample_id=f"gtsrb_{official}_{label:02d}_{stem}", media_type="sign_crop",
                           label_id=label, annotation_path="", annotation_format="folder_class_id",
                           official_split=official, split="test" if official == "test" else "unassigned")
                rows.append(row)
    return _finalise(rows)


def build_yolo_manifest(root: str | Path, meta: dict | None = None, cameras: Iterable[str] | None = None,
                        with_hash: bool = True) -> pd.DataFrame:
    """Manifest for YOLO-format full-scene images grouped by camera/sequence folders."""
    root = Path(root)
    meta = {"source_dataset": "idd_signs", **(meta or {})}
    img_root = root / "JPEGImages"
    if not img_root.is_dir():
        raise FileNotFoundError(f"Expected images under {img_root}")
    rows = []
    for path in sorted(img_root.rglob("*")):
        if path.suffix.lower() not in IMAGE_EXTS:
            continue
        rel = path.relative_to(img_root)
        camera, sequence = rel.parts[0], rel.parts[1] if len(rel.parts) > 2 else rel.parts[0]
        if cameras and camera not in set(cameras):
            continue
        label_path = root / "labels" / rel.with_suffix(".txt")
        digits = "".join(ch for ch in path.stem if ch.isdigit())
        row = _base_row(path, root, meta, with_hash)
        row.update(
            sample_id=f"idd_{sequence}_{path.stem}", media_type="video_frame",
            sequence_id=sequence, frame_index=int(digits[:6]) if digits else -1,
            physical_sign_id="", label_id=-1,
            annotation_path=str(label_path.relative_to(root)) if label_path.exists() else "",
            annotation_format="yolo_txt", official_split="", split="unassigned",
        )
        rows.append(row)
    return _finalise(rows)


def build_mask_manifest(root: str | Path, meta: dict | None = None, img_dir: str = "imgs", mask_dir: str = "masks",
                        with_hash: bool = True) -> pd.DataFrame:
    """Manifest for image/mask pairs named ``<id>_<route>_<date>_<seg>_<frame>.png`` (comma10k layout).
    The route hash is used as the group ID so one drive never spans two splits."""
    root = Path(root)
    meta = {"source_dataset": "comma10k", **(meta or {})}
    if not (root / img_dir).is_dir():
        raise FileNotFoundError(f"Expected images under {root / img_dir}")
    rows = []
    for path in sorted((root / img_dir).glob("*.png")):
        mask = root / mask_dir / path.name
        parts = path.stem.split("_")
        row = _base_row(path, root, meta, with_hash)
        row.update(sample_id=f"c10k_{path.stem}", media_type="image",
                   sequence_id=parts[1] if len(parts) > 1 else path.stem, frame_index=-1, physical_sign_id="",
                   label_id=-1, annotation_path=str(mask.relative_to(root)) if mask.exists() else "",
                   annotation_format="rgb_colour_mask", official_split="", split="unassigned")
        rows.append(row)
    return _finalise(rows)


def _finalise(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    for col in MANIFEST_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    df = df[MANIFEST_COLUMNS]
    if df["sample_id"].duplicated().any():
        dups = df.loc[df["sample_id"].duplicated(), "sample_id"].tolist()[:5]
        raise ValueError(f"Duplicate sample_id values, e.g. {dups}")
    return df.reset_index(drop=True)


def save_manifest(df: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def load_manifest(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}. Run `build-manifest` first.")
    df = pd.read_csv(path, dtype={"sha256": str, "ahash": str, "annotation_path": str}, keep_default_na=False)
    missing = [c for c in ("sample_id", "relative_path", "split") if c not in df.columns]
    if missing:
        raise ValueError(f"Manifest {path} is missing columns {missing}")
    return df
