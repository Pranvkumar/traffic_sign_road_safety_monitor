# Temporal target definition (LSTM)

**This is an educational sequence-classification experiment. It is not a risk score, not an alert likelihood, and not an accident probability.**

## Why this target
No obtained dataset contains event, hazard or risk labels over time. Inventing a "risk" target from the model's own detections would be circular, so the plan's fallback was used: a narrowly defined, non-safety target that is grounded in real human labels.

## Definition
- **Input:** the first *L = 8* frames (the most distant, smallest views) of one GTSRB physical-sign track, in frame order. Each frame → HOG (324) + 4×4 RGB thumbnail (48) → standardised → PCA to 32 dims → standardised. Standardiser and PCA are fitted on training tracks only.
- **Target:** the sign class (0–42) of that track — a real GTSRB label.
- **Unit of evaluation:** one sequence per track (frames 0–7). Training additionally uses windows starting at frames 0, 2, 4, … of training tracks.
- **Split:** by physical-sign track, stratified by class, seed 42: 897 / 209 / 201 tracks (train/val/test), built from the official GTSRB training set only.
- **Missing frames:** a robustness test drops 25 % of frames at random (never frame 0) using the LSTM's mask input; baselines use the last available frame / mean of available frames.
- **Baselines:** last-frame MLP (single image) and mean-pooled MLP (order-free).
- **Metrics:** sequence-level accuracy and macro-F1 on 201 held-out tracks.

## What it would take to build a real temporal alert
Video with time-stamped, human-annotated events (e.g. IDD-MTSVD context intervals for missing signs), a written event definition and horizon, splits by drive, and probability calibration (Brier/ECE) before any output is called a likelihood.
