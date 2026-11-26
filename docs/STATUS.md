# Project status against AI_PROJECT_PLAN.md (21 Nov 2026)

Legend: ✅ done and evaluated · 🟡 done with a documented limitation · ❌ not done (reason given)

## Environment constraints that shaped the build
- No `pip` installs and no PyTorch → every neural network is implemented from scratch in NumPy (MLP, CNN, LSTM, CVAE) and verified with gradient checks. YOLO / Faster R-CNN / ResNet / GoogLeNet / DeepLab / U-Net could not be trained.
- Official dataset portals (IDD, GTSRB site, Kaggle, Hugging Face) were blocked; data came from **GitHub via git** only.

## Plan checklist (§13)
| Item | Status | Notes |
|---|---|---|
| Package installs/imports | ✅ | `pip install -e .`; 41 tests pass |
| Configs complete and validated | ✅ | `configs/*.yaml`; missing keys/files raise clear errors |
| Data preparation / annotation conversion | ✅ | manifests with sha256 + perceptual hash; YOLO↔xyxy↔COCO converters |
| Dataset audit + leakage-safe split | ✅ | `audit-data`, `split-data`; 0 leaked groups in all 3 datasets |
| NumPy fundamentals / MLP | ✅ | tensors, SVD, gradient check, neurons, learning rules, optimizers, ablations, 2D/3D conv |
| Sign classifier exported | ✅ | HOG-MLP + NumPy CNN; confusion matrices |
| Sign detector exported | 🟡 | works end to end but **test AP@0.5 = 0.011** — not fit for purpose |
| Second detector architecture | 🟡 | two scorers on the same HOG windows; a real second architecture needs PyTorch |
| Road segmentation exported | 🟡 | road IoU 0.826 on comma10k (US); qualitative only on Indian roads |
| Hazard component | 🟡 | potholes only: YOLO11n on HRP4K (CC BY 4.0), test mAP@0.5 0.358; other hazard types still unlabelled |
| LSTM + target definition | 🟡 | educational sign-track target (`docs/TEMPORAL_TARGET.md`); not a risk model; not in live pipeline |
| VAE/GAN + latent figure | ✅ | CVAE, samples, reconstructions, latent PCA, 3-seed ablation |
| CLI image + video | ✅ | `predict`, `predict-video`, `benchmark` |
| Dashboard without fake outputs | 🟡 | `app/dashboard.py` written; helpers tested; Streamlit itself could not be installed to run it here |
| Tests pass | ✅ | `python -m pytest -q` → 41 passed |
| Dataset card | ✅ | licence status of two GitHub copies marked UNVERIFIED |
| Label taxonomy | ✅ | |
| Split manifests + seed | ✅ | `data/processed/manifests/*.csv`, seed 42 |
| Model card | ✅ | |
| Experiment log incl. failed runs | ✅ | `experiments/experiment_log.csv`; failed detector v1 kept |
| Ablations + confusion matrices | ✅ | |
| Latency/memory naming hardware | ✅ | cloud CPU; no edge device tested |
| Technical report, demo video, presentation, contribution record | ❌ | student deliverables; `docs/EXPERIMENTS.md` + figures are the source material |
| Go/no-go recommendation | ✅ | **No-go; research/demo only** (`docs/MODEL_CARD.md`) |

## Highest-value next steps
1. **Replace the detector** with YOLOv8n or Faster R-CNN (PyTorch) on a machine with internet, fine-tuned on the same IDD-derived split (the manifests and evaluation code are ready). Add GTSDB from <https://benchmark.ini.rub.de/> as a second benchmark.
2. **Get real Indian sign-type labels** (e.g. IDD-MTSVD after accepting its agreement) so the classifier taxonomy matches Indian signs; then the `unknown` policy can be tuned properly.
3. **Re-download GTSRB from the official site** and **IDD through its portal** to settle licence questions before any public release.
4. Train the segmenter on IDD Segmentation (Indian roads) and replace it with a U-Net/DeepLab once PyTorch is available.
5. For a real temporal alert: obtain time-stamped event labels and define the target before training.
