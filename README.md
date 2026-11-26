# Traffic Sign Recognition and Road Safety Monitor (Project 03)

> **Research/education demo only. Not safety-certified. Must not be used to make driving decisions.**

A Python application that analyses road images and dashcam video to **detect traffic signs, classify them, and segment the road surface**. It also includes the course's **neural-network fundamentals**, an **LSTM** sequence model and a **conditional VAE** for rare-sign augmentation. Every model is implemented in NumPy (plus scikit-learn logistic regression baselines) and trains on a laptop CPU.

**Language:** Python (instructor-approved exception to the C/C++/Java default in the PBL brief).
**Implementation note:** PyTorch/YOLO could not be installed in the build environment, so the brief's ResNet/YOLO/DeepLab models are replaced by from-scratch NumPy models (MLP, 2-layer CNN, LSTM, CVAE) and a classical HOG sliding-window detector. See `docs/STATUS.md` for what this means and how to upgrade.

## Run it on your own laptop (Windows, easiest)
1. Install **Python 3.13** from python.org and tick **"Add python.exe to PATH"** in the installer.
2. Unzip this folder somewhere with ~4 GB free, then double-click **`SETUP.bat`**. It creates `.venv`, installs
   everything (CPU PyTorch + YOLO, 1-3 GB download, 5-15 min), runs the tests and a demo prediction.
3. Use it: drag any road photo onto **`run_on_image.bat`**, or double-click **`run_dashboard.bat`**
   (opens http://localhost:8501).

The sign detector is now **YOLO11n** trained on European signs (`models/checkpoints/yolo_signs.pt`,
`configs/yolo.yaml`, results in [`docs/YOLO_RESULTS.md`](docs/YOLO_RESULTS.md)). It runs on CPU too (~0.1-0.3 s per image).
NVIDIA GPU (optional, faster): after setup run
`.venv\Scripts\pip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu128`.
macOS/Linux: same steps by hand: `python3 -m venv .venv && . .venv/bin/activate && pip install torch torchvision -r requirements.txt ultralytics streamlit`,
then `PYTHONPATH=src python -m traffic_monitor.cli predict --source test_images/rural_signs.webp`.

**Licence:** the YOLO weights were trained on ETSC (CC BY-NC-SA 4.0), so they are for **non-commercial** use only.
The pothole model (`yolo_hazard.pt`) was trained on HRP4K (Zenodo 21941931, CC BY 4.0).
To retrain it: `python scripts/prepare_hrp4k.py` (downloads ~1 GB of frames and writes the dataset YAML), then `python scripts/train_hazard.py` (about 1.5 h on an RTX 4060).

## Quick start (developers)
```bash
pip install -r requirements.txt && pip install -e .
python -m pytest -q                                     # 43 tests, no data needed
python -m traffic_monitor.cli predict --source path/to/road.jpg
```
Full instructions: [`docs/USER_GUIDE.md`](docs/USER_GUIDE.md).

## Results (held-out test sets, measured on this run)
See [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md) for full tables, settings and caveats; raw numbers in `experiments/runs/*/metrics.json`.

| Component | Data (held-out test) | Selected model | Headline result |
|---|---|---|---|
| Sign classification | GTSRB official test, 12,630 crops | HOG + NumPy MLP | **90.0 %** accuracy, macro-F1 0.866 (NumPy CNN: 91.3 %) |
| Sign detection | IDD-derived Indian scenes, 139 images | HOG + colour sliding window | **AP@0.5 = 0.011**, missed-sign rate 95 % — *not fit for purpose* |
| Road segmentation | comma10k, 45 images | pixel MLP | road IoU **0.826**, mIoU 0.877 |
| Temporal (LSTM) | 201 GTSRB sign tracks | NumPy LSTM | **73.1 %** vs 68.2 % (mean-pool) and 61.7 % (single frame) |
| VAE augmentation | GTSRB test, 3 seeds | conditional VAE | rare-class recall 70.2 % → **77.2 %** |
| Runtime | 1280×964, 2-core CPU | full pipeline | 644 ms / frame (1.6 FPS) — not an edge device |
| Hazard detection (potholes) | HRP4K, 400 frames | YOLO11n | mAP@0.5 **0.358**, precision 0.47, recall 0.38 |

## Repository map
| Path | What |
|---|---|
| `src/traffic_monitor/data/` | manifests (sha256, groups), group-aware splits, audit, YOLO/COCO conversion |
| `src/traffic_monitor/fundamentals/` | tensors/SVD/grad-check, neurons, learning rules, 2D/3D convolution |
| `src/traffic_monitor/models/` | NumPy MLP (+GD/SGD/Momentum/Nesterov/Adam), CNN, LSTM, CVAE; HOG detector; road segmenter |
| `src/traffic_monitor/training/` | one script per experiment; each logs to `experiments/experiment_log.csv` |
| `src/traffic_monitor/pipeline/` | image/video inference, benchmark |
| `src/traffic_monitor/cli.py`, `app/dashboard.py` | CLI and optional Streamlit dashboard |
| `configs/` | every path, hyperparameter and threshold |
| `docs/` | dataset card, model card, taxonomy, temporal target, experiments, status, user guide |
| `outputs/figures/`, `outputs/reports/` | figures and data-audit reports |

## Data
GTSRB (cropped German signs), an IDD-derived set of Indian road scenes with sign boxes, and a comma10k subset with road masks — all obtained with `git`. Exact versions, licences and measured counts: [`docs/DATASET_CARD.md`](docs/DATASET_CARD.md). Raw data is not in this repository.

## Responsible use
Signs are classified with the German GTSRB taxonomy, so many Indian signs will be `unknown` or wrong. The road segmenter was trained on US roads. The hazard detector finds **potholes only** and misses many of them (recall 0.38) and the LSTM predicts a sign class from a short track, **not** a risk or accident probability. Measured latency is from a cloud CPU, not an edge device.
