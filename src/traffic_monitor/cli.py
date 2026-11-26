"""Command-line interface. Run `python -m traffic_monitor.cli --help`."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import DISCLAIMER, __version__


def _print(obj) -> None:
    print(json.dumps(obj, indent=2, default=str))


def cmd_init_data(args) -> int:
    from .utils.config import resolve

    for sub in ("data/raw", "data/interim", "data/processed/manifests", "models/checkpoints", "outputs/predictions",
                "outputs/figures", "outputs/reports", "experiments/runs"):
        resolve(sub).mkdir(parents=True, exist_ok=True)
    print("Created data/model/output folders. Put datasets under data/raw/ (see docs/DATASET_CARD.md).")
    return 0


def cmd_build_manifest(args) -> int:
    from .data.manifest import build_gtsrb_manifest, build_mask_manifest, build_yolo_manifest, save_manifest
    from .utils.config import load_config, resolve

    cfg = load_config(args.data_config)
    if args.dataset not in cfg:
        raise SystemExit(f"Unknown dataset '{args.dataset}'. Configured: {[k for k in cfg if k != 'seed']}")
    c = cfg[args.dataset]
    meta = {k: c.get(k, "") for k in ("source_release", "source_license", "source_url")}
    builders = {"gtsrb": lambda: build_gtsrb_manifest(resolve(c["root"]), meta),
                "idd_signs": lambda: build_yolo_manifest(resolve(c["root"]), meta, c.get("cameras")),
                "comma10k": lambda: build_mask_manifest(resolve(c["root"]), meta)}
    df = builders[args.dataset]()
    path = save_manifest(df, args.output or resolve(c["manifest"]))
    print(f"Wrote {len(df)} rows to {path}")
    return 0


def cmd_split(args) -> int:
    from .data.manifest import load_manifest, save_manifest
    from .data.splits import check_group_disjoint, group_split, split_summary
    from .utils.config import load_config, resolve

    cfg = load_config(args.data_config)
    c = cfg[args.dataset]
    path = Path(args.manifest) if args.manifest else resolve(c["manifest"])
    df = load_manifest(path)
    df.loc[df["official_split"] != "test", "split"] = "unassigned"   # official test sets are kept
    group = args.group_column or c["group_column"]
    df = group_split(df, group, c["split_fractions"], seed=cfg["seed"], stratify_col=c.get("stratify_column"))
    leaked = check_group_disjoint(df, group)["leaked_groups"]
    if leaked:
        raise SystemExit(f"Leakage detected in groups: {leaked[:5]}")
    save_manifest(df, path)
    _print({"manifest": str(path), "group_column": group, "seed": cfg["seed"],
            "splits": split_summary(df, group, "label_id")})
    return 0


def cmd_audit(args) -> int:
    from .data.manifest import load_manifest
    from .data.validators import audit_manifest, report_to_markdown
    from .utils.config import load_config, resolve
    from .utils.reproducibility import save_json

    c = load_config(args.data_config)[args.dataset]
    df = load_manifest(args.manifest or resolve(c["manifest"]))
    rep = audit_manifest(df, resolve(c["root"]), c["group_column"], set(c.get("known_classes", [])) or None)
    out = resolve(f"outputs/reports/audit_{args.dataset}")
    save_json(rep, out.with_suffix(".json"))
    out.with_suffix(".md").write_text(report_to_markdown(rep, args.dataset))
    _print({k: v for k, v in rep.items() if k != "class_counts"})
    return 0


def cmd_train(module: str):
    def _run(args) -> int:
        import importlib

        mod = importlib.import_module(f".training.{module}", __package__)
        result = mod.run(args.config) if args.config else mod.run()
        _print(result)
        return 0
    return _run


def cmd_fundamentals(args) -> int:
    from .training.run_fundamentals import run

    r = run()
    _print({"gradient_check": r["gradient_check"], "figures": "outputs/figures/fundamentals_*.png"})
    return 0


def cmd_predict(args) -> int:
    import cv2

    from .pipeline.inference import Pipeline, draw, read_image, to_json
    from .utils.config import resolve
    from .utils.reproducibility import save_json

    img = read_image(args.source)                  # validates path/type before loading models
    pipe = Pipeline.from_config(args.config, use_segmenter=not args.no_segmentation)
    res = pipe.predict(img)
    out_dir = Path(args.output) if args.output else resolve("outputs/predictions")
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.source).stem
    cv2.imwrite(str(out_dir / f"{stem}_annotated.jpg"), draw(img, res))
    save_json(to_json(res), out_dir / f"{stem}.json")
    summary = to_json(res)
    summary["outputs"] = [str(out_dir / f"{stem}_annotated.jpg"), str(out_dir / f"{stem}.json")]
    _print(summary)
    return 0


def cmd_predict_video(args) -> int:
    from .pipeline.inference import Pipeline
    from .pipeline.video import predict_video
    from .utils.config import resolve

    if not Path(args.source).exists():
        raise FileNotFoundError(f"Video not found: {args.source}")
    pipe = Pipeline.from_config(args.config, use_segmenter=not args.no_segmentation)
    _print(predict_video(pipe, args.source, args.output or resolve("outputs/predictions"), args.frame_stride,
                         args.max_frames))
    return 0


def cmd_benchmark(args) -> int:
    from .pipeline.benchmark import run

    _print(run(args.config, args.repeats))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="traffic_monitor", description=f"Traffic sign recognition & road monitor "
                                f"v{__version__}. {DISCLAIMER}")
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("init-data", help="create the data/output folder layout").set_defaults(func=cmd_init_data)

    for name, fn, hlp in [("build-manifest", cmd_build_manifest, "scan a dataset folder into a manifest CSV"),
                          ("split-data", cmd_split, "group-aware train/val/test split (leakage-safe)"),
                          ("audit-data", cmd_audit, "check files, duplicates, boxes, balance, leakage")]:
        sp = sub.add_parser(name, help=hlp)
        sp.add_argument("--dataset", required=True, choices=["gtsrb", "idd_signs", "comma10k"])
        sp.add_argument("--data-config", default="configs/data.yaml")
        if name == "build-manifest":
            sp.add_argument("--output")
        else:
            sp.add_argument("--manifest")
        if name == "split-data":
            sp.add_argument("--group-column")
        sp.set_defaults(func=fn)

    for name, module, cfg in [("train-classifier", "train_classifier", "configs/classifier.yaml"),
                              ("train-detector", "train_detector", "configs/detector.yaml"),
                              ("train-segmenter", "train_segmenter", "configs/segmenter.yaml"),
                              ("train-temporal", "train_temporal", "configs/temporal.yaml"),
                              ("train-vae", "train_vae", "configs/vae.yaml")]:
        sp = sub.add_parser(name, help=f"train + evaluate ({cfg})")
        sp.add_argument("--config", default=cfg)
        sp.set_defaults(func=cmd_train(module))
    sub.add_parser("fundamentals", help="run NumPy fundamentals demos and save figures").set_defaults(func=cmd_fundamentals)

    sp = sub.add_parser("predict", help="run the pipeline on one image")
    sp.add_argument("--source", required=True)
    sp.add_argument("--config", default="configs/yolo.yaml")
    sp.add_argument("--output")
    sp.add_argument("--no-segmentation", action="store_true")
    sp.set_defaults(func=cmd_predict)

    sp = sub.add_parser("predict-video", help="run the pipeline frame by frame on a video")
    sp.add_argument("--source", required=True)
    sp.add_argument("--config", default="configs/yolo.yaml")
    sp.add_argument("--output")
    sp.add_argument("--frame-stride", type=int, default=1)
    sp.add_argument("--max-frames", type=int, default=300)
    sp.add_argument("--no-segmentation", action="store_true")
    sp.set_defaults(func=cmd_predict_video)

    sp = sub.add_parser("benchmark", help="latency/memory benchmark of the full pipeline on this machine")
    sp.add_argument("--config", default="configs/yolo.yaml")
    sp.add_argument("--repeats", type=int, default=20)
    sp.set_defaults(func=cmd_benchmark)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (FileNotFoundError, ValueError, OSError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
