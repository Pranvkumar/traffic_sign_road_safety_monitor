# Model card — Traffic Sign Monitor v0.1.0

## Intended use
Offline analysis of road images and recorded dashcam video for **coursework, research and demonstration**. Outputs are suggestions for a human reviewer.

**Out of scope / not allowed:** controlling a vehicle, driver assistance, replacing road signs or a driver's judgement, any safety-critical or real-time decision, claims about accident risk. The system is not safety-certified and its hazard detector finds potholes only, with modest accuracy.

## Components (checkpoints in `models/checkpoints/`, configs in `configs/`)
| Component | Model | Trained on | Preprocessing saved with checkpoint |
|---|---|---|---|
| Sign detector | Multi-scale sliding window (24 px window, sign sizes 16–120 px, stride 6) → HOG(324) + colour-opponent 3×3 cell means(27) → logistic regression → NMS (IoU 0.3), threshold 0.9 from validation | IDD-derived frontFar train split (617 images, 13 drives) | HOG parameters, sizes, stride, feature mean/std, threshold |
| Sign classifier | NumPy MLP 372-256-43 on HOG(32×32 gray) + 4×4 RGB thumbnail; `unknown` if top probability < 0.6 | GTSRB train split (32,939 crops) | crop size, colour order, feature mean/std, class names |
| Road segmenter | NumPy MLP 18-64-32-2 per pixel at 1/8 resolution (colour, HSV, position, texture, context), upsampled | comma10k subset train split (210 images) | downscale factor, feature list, mean/std, threshold 0.5 |
| LSTM (offline experiment, not in live pipeline) | 1-layer LSTM, hidden 64, on 8-frame sequences of PCA-32 features | GTSRB tracks | feature schema, standardiser, PCA |
| Hazard (pothole) detector | YOLO11n, 1 class, 960 px, 30 epochs | HRP4K subset (2,000 train frames, CC BY 4.0) | letterbox to 960 px (Ultralytics, stored in checkpoint) |
| CVAE (training-time only) | conditional VAE, latent 16 | GTSRB train split, grayscale | — |

## Metrics (held-out test sets — full tables in `docs/EXPERIMENTS.md`)
- Classification (GTSRB official test, 43 classes): accuracy 90.0 %, macro-F1 0.866, top-5 98.4 %, ECE 0.034.
- Detection (IDD-derived test, 164 signs, IoU 0.5, threshold 0.9): AP@0.5 0.011, recall 4.9 % (missed-sign rate 95.1 %), 0.52 false positives / image.
- Segmentation (comma10k test): road IoU 0.826, Dice 0.905, mIoU 0.877, pixel accuracy 94.7 %.
- Runtime (2-core cloud CPU, 1280×964): 644 ms median per frame end-to-end, 811 MB peak RAM.

## Known failure modes
- **Missed signs (dominant failure).** The detector misses ~95 % of signs on the test drives, worst for signs < 16 px (0 % recall). It cannot be relied on to find signs.
- **Overconfident wrong classes on Indian signs.** The classifier knows only 43 German classes. In the demo image an Indian sign was labelled `speed_limit_50` with 99.6 % confidence; the `unknown` threshold catches only low-confidence cases.
- **Speed-limit digit confusion** (80↔30, 30↔50, 100↔120) on GTSRB itself.
- **Segmentation domain shift.** Bright concrete Indian roads are partly missed and dry verges marked as road; the ego-vehicle bonnet is painted as road.
- **Hazards are potholes only**: test mAP@0.5 **0.358** (mAP@0.5:0.95 0.199, precision 0.47, recall 0.38) on 400 held-out HRP4K frames. It misses many small or shallow potholes and sometimes boxes patches/stains. **No temporal alerting** in the live pipeline.

## Data and fairness / domain notes
German (GTSRB) and US (comma10k) data dominate training. Only the detector saw Indian roads (Bangalore, 16 drives, daytime). No night, rain or fog evaluation was possible. Results must not be read as Indian-road performance for classification or segmentation.

## Deployment recommendation (plan §14)
**No-go for operational use; research/demo only.** Evidence: detection recall 4.9 % at 0.52 FP/image on the test drives; classification taxonomy does not cover Indian signs; segmentation trained on another country; latency 644 ms/frame on a CPU with no edge-device measurement; dataset redistribution rights for two sources unverified. The classifier, segmenter, LSTM and VAE components are sound as course demonstrations.
