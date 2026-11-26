"""Generate the markdown results tables from experiments/runs/*/metrics.json.

Usage: python scripts/summarise_results.py > outputs/reports/results_tables.md
Docs quote these tables so every number traces back to a metrics file.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "experiments" / "runs"


def load(name: str) -> dict:
    return json.loads((RUNS / name / "metrics.json").read_text())


def pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def main() -> None:
    out = []
    c = load("classifier")["summary"]
    out += ["### Sign classification — GTSRB official test set (12,630 crops, 43 classes)", "",
            "| Model | Params | Val acc | Val macro-F1 | **Test acc** | Test macro-F1 | Test top-5 | Test ECE | Train time |",
            "|---|---|---|---|---|---|---|---|---|"]
    for name in ("pixel_mlp", "hog_mlp", "small_cnn"):
        r = c[name]
        sel = " ✓ selected" if c["selected_by_val_macro_f1"] == name else ""
        out.append(f"| {name}{sel} | {r['n_parameters']:,} | {pct(r['val_accuracy'])} | {r['val_macro_f1']:.3f} | "
                   f"**{pct(r['test_accuracy'])}** | {r['test_macro_f1']:.3f} | {pct(r['test_top5'])} | {r['test_ece']:.3f} | "
                   f"{r['train_seconds']:.0f} s |")

    d = load("detector")
    out += ["", "### Traffic-sign detection — IDD-derived frontFar test split (139 images, 2 drives, 164 signs)", "",
            "| Scorer | Val AP@0.5 | **Test AP@0.5** | Test AP@0.3 | Threshold (from val) | Test recall | Missed-sign rate | FP / image | Latency (median) |",
            "|---|---|---|---|---|---|---|---|---|"]
    for name, r in d["results"].items():
        t, at = r["test"], r["test"]["at_val_threshold"]
        sel = " ✓ selected" if d["summary"]["selected_by_val_ap50"] == name else ""
        out.append(f"| HOG+colour {name}{sel} | {r['val']['ap50']['ap']:.3f} | **{t['ap50']['ap']:.3f}** | {t['ap30']['ap']:.3f} | "
                   f"{r['val']['chosen_threshold']} | {pct(at['recall'])} | {pct(at['missed_sign_rate'])} | "
                   f"{at['false_positives_per_image']:.2f} | {t['latency_ms_per_image']['median']:.0f} ms |")
    v1 = json.loads((RUNS / "detector" / "metrics_v1_hog_gray_only_FAILED.json").read_text())["results"]
    out.append(f"| *v1 (grayscale HOG only, 7.5k hard negatives) — superseded* | {v1['logreg']['val']['ap50']['ap']:.3f} / "
               f"{v1['mlp']['val']['ap50']['ap']:.3f} | {v1['logreg']['test']['ap50']['ap']:.3f} / {v1['mlp']['test']['ap50']['ap']:.3f} | | | | | | |")
    sel = d["results"][d["summary"]["selected_by_val_ap50"]]["test"]["recall_by_size"]
    out += ["", "Recall of the selected detector by sign width (test): " +
            ", ".join(f"{k}: {pct(v['recall'])} of {v['n_gt']}" for k, v in sel.items())]

    s = load("segmenter")["summary"]
    out += ["", "### Road-surface segmentation — comma10k test split (45 images, 27 routes)", "",
            "| Model | Val mIoU | **Test mIoU** | Test road IoU | Test road Dice | Test pixel acc | Latency |",
            "|---|---|---|---|---|---|---|"]
    for name in ("logreg", "mlp"):
        r = s[name]
        mark = " ✓ selected" if s["selected_by_val_miou"] == name else ""
        out.append(f"| pixel {name}{mark} | {r['val_miou']:.3f} | **{r['test_miou']:.3f}** | {r['test_road_iou']:.3f} | "
                   f"{r['test_road_dice']:.3f} | {pct(r['test_pixel_acc'])} | {r['latency_ms']:.0f} ms |")

    t = load("temporal")["summary"]
    out += ["", f"### Temporal sequence model — GTSRB tracks, first 8 frames ({t['tracks']['test']} test tracks)", "",
            "| Model | Val acc | **Test acc** | Test macro-F1 | Test acc, 25% frames missing |", "|---|---|---|---|---|"]
    for name in ("last_frame_mlp", "mean_pool_mlp", "lstm"):
        r = t[name]
        out.append(f"| {name} | {pct(r['val_acc'])} | **{pct(r['test_acc'])}** | {r['test_macro_f1']:.3f} | {pct(r['test_acc_missing_frames'])} |")

    v = load("vae")["summary"]
    a = v["ablation_mean_std_over_seeds"]
    out += ["", "### VAE augmentation ablation — GTSRB official test, mean ± std over 3 seeds", "",
            "| Training data | Test acc | Test macro-F1 | Rare-class mean recall (8 classes) |", "|---|---|---|---|"]
    for arm in ("real_only", "real_plus_synthetic"):
        r = a[arm]
        out.append(f"| {arm} | {pct(r['test_accuracy']['mean'])} ± {100 * r['test_accuracy']['std']:.1f} | "
                   f"{r['test_macro_f1']['mean']:.3f} ± {r['test_macro_f1']['std']:.3f} | "
                   f"{pct(r['rare_class_mean_recall']['mean'])} ± {100 * r['rare_class_mean_recall']['std']:.1f} |")
    out.append(f"\n{v['n_synthetic']} synthetic crops added to the training split only; "
               f"{pct(v['synthetic_label_consistency(real-only classifier agrees)'])} of them are recognised as their "
               f"intended class by the real-only classifier.")

    b = json.loads((ROOT / "outputs" / "benchmark" / "benchmark.json").read_text())
    out += ["", "### Runtime — full pipeline, 1280×964 IDD test images, batch 1", "",
            f"End-to-end median {b['latency_ms']['end_to_end_wall_ms']['median']:.0f} ms "
            f"(p90 {b['latency_ms']['end_to_end_wall_ms']['p90']:.0f} ms) ≈ {b['fps_end_to_end']:.1f} FPS; detection "
            f"{b['latency_ms']['detection_ms']['median']:.0f} ms, segmentation {b['latency_ms']['segmentation_ms']['median']:.0f} ms, "
            f"classification {b['latency_ms']['classification_ms']['median']:.1f} ms; peak RAM {b['peak_rss_mb']:.0f} MB. "
            f"Hardware: {b['hardware']['cpu_count']}-core x86_64 cloud CPU, no GPU — **not an edge-device measurement**."]
    print("\n".join(out))


if __name__ == "__main__":
    main()
