"""Build YOLO label files for ETSC from the per-split metadata.jsonl files.

The ETSC repo ships labels/ too, but rebuilding them from metadata.jsonl (absolute xyxy boxes)
saves ~22k extra downloads and lets us validate every box against the real image size.

Usage: python scripts/prepare_etsc.py   (after images are in data/raw/etsc/etsc_yolo/images/)
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1] / "data" / "raw" / "etsc" / "etsc_yolo"
# metadata.jsonl class ids are 1-based (1..171); etsc_yolo.yaml names and the repo's own labels/
# are 0-based (0..170). Verified 2026-11-21 against labels/train/0/1002.txt (112 -> 111, 171 -> 170).
META_OFFSET = 1


def class_ids(rec: dict) -> list[int]:
    return [int(c) - META_OFFSET for c in rec["labels"]]


# Mirror-image sign pairs (0-based ids). Add horizontally flipped copies of train images whose signs
# are ALL in this set (swapping labels) to balance left/right bend and curve classes.
# Signs with text/digits are never flipped (mirrored "70" is not a real sign).
MIRROR = {0: 1, 1: 0, 2: 3, 3: 2, 160: 161, 161: 160}


def add_mirrored_train(stats: Counter) -> None:
    meta = ROOT / "images" / "train" / "metadata.jsonl"
    for line in meta.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        ids = class_ids(rec)
        if not ids or not all(c in MIRROR for c in ids):
            continue
        src = ROOT / "images" / "train" / rec["file_name"]
        if not src.exists():
            continue
        name = "flip_" + Path(rec["file_name"]).name
        dst_img = ROOT / "images" / "train" / "flip" / name
        dst_img.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(src) as im:
            w, h = im.size
            if not dst_img.exists():
                im.transpose(Image.FLIP_LEFT_RIGHT).save(dst_img, quality=95)
        rows = []
        for (x1, y1, x2, y2), c in zip(rec["bboxes"], ids):
            fx1, fx2 = w - max(x1, x2), w - min(x1, x2)
            y1, y2 = max(0, min(y1, y2)), min(h, max(y1, y2))
            rows.append(f"{MIRROR[c]} {(fx1 + fx2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} "
                        f"{(fx2 - fx1) / w:.6f} {(y2 - y1) / h:.6f}")
        out = ROOT / "labels" / "train" / "flip" / Path(name).with_suffix(".txt")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join(rows) + "\n", encoding="utf-8")
        stats["train_mirrored_images"] += 1


def main() -> None:
    stats = Counter()
    add_mirrored_train(stats)
    for split in ("train", "val", "test"):
        meta = ROOT / "images" / split / "metadata.jsonl"
        for line in meta.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            img = ROOT / "images" / split / rec["file_name"]
            if not img.exists():
                stats[f"{split}_missing_image"] += 1
                continue
            with Image.open(img) as im:
                w, h = im.size
            rows = []
            for (x1, y1, x2, y2), c in zip(rec["bboxes"], class_ids(rec)):
                x1, x2 = max(0, min(x1, x2)), min(w, max(x1, x2))
                y1, y2 = max(0, min(y1, y2)), min(h, max(y1, y2))
                if x2 - x1 < 2 or y2 - y1 < 2:
                    stats["dropped_degenerate_box"] += 1
                    continue
                rows.append(f"{int(c)} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} "
                            f"{(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}")
            out = ROOT / "labels" / split / Path(rec["file_name"]).with_suffix(".txt")
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
            stats[f"{split}_images"] += 1
            stats[f"{split}_boxes"] += len(rows)
    print(dict(stats))


if __name__ == "__main__":
    main()
