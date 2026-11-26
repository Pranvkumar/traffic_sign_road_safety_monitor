### Sign classification — GTSRB official test set (12,630 crops, 43 classes)

| Model | Params | Val acc | Val macro-F1 | **Test acc** | Test macro-F1 | Test top-5 | Test ECE | Train time |
|---|---|---|---|---|---|---|---|---|
| pixel_mlp | 273,451 | 84.0% | 0.789 | **81.5%** | 0.759 | 94.9% | 0.067 | 25 s |
| hog_mlp ✓ selected | 106,539 | 91.4% | 0.871 | **90.0%** | 0.866 | 98.4% | 0.034 | 12 s |
| small_cnn | 273,675 | 91.1% | 0.850 | **91.3%** | 0.871 | 99.0% | 0.010 | 358 s |

### Traffic-sign detection — IDD-derived frontFar test split (139 images, 2 drives, 164 signs)

| Scorer | Val AP@0.5 | **Test AP@0.5** | Test AP@0.3 | Threshold (from val) | Test recall | Missed-sign rate | FP / image | Latency (median) |
|---|---|---|---|---|---|---|---|---|
| HOG+colour logreg ✓ selected | 0.131 | **0.011** | 0.023 | 0.9 | 4.9% | 95.1% | 0.52 | 575 ms |
| HOG+colour mlp | 0.099 | **0.018** | 0.058 | 0.95 | 24.4% | 75.6% | 97.36 | 965 ms |
| *v1 (grayscale HOG only, 7.5k hard negatives) — superseded* | 0.010 / 0.078 | 0.001 / 0.004 | | | | | | |

Recall of the selected detector by sign width (test): <16px: 0.0% of 37, 16-32px: 8.2% of 49, 32-64px: 4.0% of 50, >=64px: 7.1% of 28

### Road-surface segmentation — comma10k test split (45 images, 27 routes)

| Model | Val mIoU | **Test mIoU** | Test road IoU | Test road Dice | Test pixel acc | Latency |
|---|---|---|---|---|---|---|
| pixel logreg | 0.820 | **0.852** | 0.790 | 0.883 | 93.5% | 6 ms |
| pixel mlp ✓ selected | 0.860 | **0.877** | 0.826 | 0.905 | 94.7% | 14 ms |

### Temporal sequence model — GTSRB tracks, first 8 frames (201 test tracks)

| Model | Val acc | **Test acc** | Test macro-F1 | Test acc, 25% frames missing |
|---|---|---|---|---|
| last_frame_mlp | 69.4% | **61.7%** | 0.491 | 61.7% |
| mean_pool_mlp | 73.2% | **68.2%** | 0.531 | 69.7% |
| lstm | 74.6% | **73.1%** | 0.618 | 71.6% |

### VAE augmentation ablation — GTSRB official test, mean ± std over 3 seeds

| Training data | Test acc | Test macro-F1 | Rare-class mean recall (8 classes) |
|---|---|---|---|
| real_only | 90.3% ± 0.1 | 0.872 ± 0.003 | 70.2% ± 1.4 |
| real_plus_synthetic | 90.4% ± 0.1 | 0.874 ± 0.002 | 77.2% ± 1.5 |

3330 synthetic crops added to the training split only; 57.9% of them are recognised as their intended class by the real-only classifier.

### Runtime — full pipeline, 1280×964 IDD test images, batch 1

End-to-end median 644 ms (p90 1143 ms) ≈ 1.6 FPS; detection 624 ms, segmentation 14 ms, classification 0.3 ms; peak RAM 811 MB. Hardware: 2-core x86_64 cloud CPU, no GPU — **not an edge-device measurement**.
