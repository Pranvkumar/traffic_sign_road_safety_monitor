# Architecture

```text
image / video frame (BGR)
        │
        ├──► Sign detector (models/detector.py)
        │       multi-scale 24px windows → HOG + colour-opponent cells → logistic scorer → NMS
        │       └─► boxes + detection scores
        │               │ crop each box
        │               ▼
        │       Sign classifier (pipeline/inference.py: SignClassifier)
        │       preprocess_crop 32×32 → HOG + RGB thumbnail → NumPy MLP → GTSRB class | "unknown"
        │
        ├──► Road segmenter (models/segmenter.py)
        │       1/8-res per-pixel features → NumPy MLP → upsampled road_surface mask
        │
        └──► Hazard detector: YOLO11n potholes (HRP4K) → orange boxes; status shows "not available" if the checkpoint is missing
                        │
                        ▼
   draw() overlay + JSON (boxes, classes, road coverage, timings, status, disclaimer)
   → CLI `predict` / `predict-video` (per-frame JSONL + annotated MP4) / Streamlit dashboard

Offline experiments (not in the live path):
   LSTM (models/lstm_numpy.py)  — sign-track sequence classification, docs/TEMPORAL_TARGET.md
   CVAE (models/vae_numpy.py)   — rare-class training augmentation only
   fundamentals/                — tensors, neurons, learning rules, convolutions
```

**Data flow for training:** dataset folder → `build-manifest` (sha256, groups) → `split-data` (group-aware, seed 42) → `audit-data` → `train-*` (validation for selection, test once) → `experiments/runs/<task>/metrics.json` + checkpoint (with preprocessing metadata) + row in `experiments/experiment_log.csv` → `scripts/summarise_results.py` → docs tables.

**Design rules followed:** identical preprocessing functions for training and inference (`data/transforms.py`, `models/features.py`, `models/detector.py`); every threshold and path in `configs/`; missing checkpoints raise `FileNotFoundError` (CLI exit code 2); library code logs, the CLI prints.
