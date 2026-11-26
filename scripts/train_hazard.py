"""Train the pothole (road-hazard) detector: YOLO11n fine-tuned on an HRP4K subset (CC BY 4.0).

    python scripts/prepare_hrp4k.py          # builds data/raw/hrp4k_yolo (~2,800 frames, ~1 GB)
    python scripts/train_hazard.py --epochs 30

Result: models/checkpoints/yolo_hazard.pt (+ run logs under experiments/runs/yolo/hazard*).
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "raw" / "hrp4k_yolo" / "hrp4k_yolo.yaml"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(ROOT / "models" / "pretrained" / "yolo11n.pt"))
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--batch", type=int, default=8, help="8 fits an 8 GB RTX 4060 at 960px")
    ap.add_argument("--workers", type=int, default=2,
                    help="more workers crashed on Windows with error 1455 (page file too small) on 1280px frames")
    args = ap.parse_args()
    if not DATA.exists():
        raise SystemExit(f"{DATA} missing: run scripts/prepare_hrp4k.py first")

    from ultralytics import YOLO
    from ultralytics.data import build as _build

    # Same Windows pin-memory crash workaround as scripts/train_yolo.py.
    _orig_init = _build.InfiniteDataLoader.__init__

    def _no_pin(self, *a, **kw):
        kw["pin_memory"] = False
        _orig_init(self, *a, **kw)

    _build.InfiniteDataLoader.__init__ = _no_pin

    project = ROOT / "experiments" / "runs" / "yolo"
    model = YOLO(args.model)
    model.train(data=str(DATA), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, workers=args.workers,
                project=str(project), name="hazard", exist_ok=True, seed=42, deterministic=False,
                fliplr=0.5,  # potholes have no left/right meaning, so mirroring is safe here
                mosaic=1.0, close_mosaic=5, patience=10, plots=True, cache=False, amp=True)
    dst = ROOT / "models" / "checkpoints" / "yolo_hazard.pt"
    shutil.copy2(project / "hazard" / "weights" / "best.pt", dst)
    m = YOLO(str(dst)).val(data=str(DATA), split="test", imgsz=args.imgsz, batch=args.batch,
                           project=str(project), name="hazard_test", exist_ok=True)
    print(f"saved {dst}\ntest mAP50={m.box.map50:.4f} mAP50-95={m.box.map:.4f} "
          f"P={m.box.mp:.4f} R={m.box.mr:.4f}")


if __name__ == "__main__":
    main()
