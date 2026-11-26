"""Train the YOLO traffic-sign detector on the European Traffic Sign Collection (ETSC).

Data: https://huggingface.co/datasets/dsilbermann/european_traffic_signs (CC BY-NC-SA 4.0,
non-commercial use only) downloaded to data/raw/etsc/. ETSC ships its own train/val/test
split; we keep it as-is so test images are never seen during training.

Usage (from the project root, venv active):
    python scripts/train_yolo.py --epochs 40 --imgsz 960 --batch 8
Result: models/checkpoints/yolo_signs.pt (+ run logs under experiments/runs/yolo/).
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "raw" / "etsc" / "etsc_yolo"


def write_data_yaml() -> Path:
    """ETSC's yaml uses a relative `path:`; rewrite it with an absolute path for Ultralytics."""
    src = (DATA_DIR / "etsc_yolo.yaml").read_text(encoding="utf-8").splitlines()
    out = [f"path: {DATA_DIR.as_posix()}" if ln.startswith("path:") else ln for ln in src]
    dst = DATA_DIR / "etsc_abs.yaml"
    dst.write_text("\n".join(out) + "\n", encoding="utf-8")
    return dst


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolo11n.pt", help="COCO-pretrained starting weights")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--imgsz", type=int, default=960, help="signs are small; >640 helps recall")
    ap.add_argument("--batch", type=int, default=8, help="8 fits an 8 GB RTX 4060 at 960px")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--fraction", type=float, default=1.0, help="use part of train set for quick runs")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    from ultralytics import YOLO

    # Windows + torch 2.11/cu128: the pin-memory thread crashed with "CUDA error: resource already mapped"
    # at the first validation (2026-11-21). Pinned memory only saves a little copy time, so turn it off.
    from ultralytics.data import build as _build
    _orig_init = _build.InfiniteDataLoader.__init__

    def _no_pin(self, *a, **kw):
        kw["pin_memory"] = False
        _orig_init(self, *a, **kw)

    _build.InfiniteDataLoader.__init__ = _no_pin

    data = write_data_yaml()
    project = ROOT / "experiments" / "runs" / "yolo"
    last = project / "etsc" / "weights" / "last.pt"
    model = YOLO(str(last) if args.resume and last.exists() else args.model)
    model.train(data=str(data), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, workers=args.workers,
                project=str(project), name="etsc", exist_ok=True, seed=42, deterministic=False,
                fliplr=0.0,            # mirroring turns "left bend" into "right bend" -> wrong labels
                mosaic=1.0, close_mosaic=5, fraction=args.fraction, patience=15, plots=True,
                cache=False, amp=True, resume=args.resume and last.exists())
    best = project / "etsc" / "weights" / "best.pt"
    dst = ROOT / "models" / "checkpoints" / "yolo_signs.pt"
    shutil.copy2(best, dst)
    metrics = YOLO(str(dst)).val(data=str(data), split="test", imgsz=args.imgsz, batch=args.batch,
                                 project=str(project), name="etsc_test", exist_ok=True)
    print(f"saved {dst}\ntest mAP50={metrics.box.map50:.4f} mAP50-95={metrics.box.map:.4f}")


if __name__ == "__main__":
    main()
