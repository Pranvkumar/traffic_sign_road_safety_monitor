# User guide

> Research/education demo only. Not safety-certified. Never use it for driving decisions.

## 1. Install
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .                                       # makes `traffic_monitor` importable
pip install streamlit                                  # optional, for the dashboard
```
No GPU or PyTorch is needed: every model is NumPy / scikit-learn and runs on a laptop CPU.

## 2. Get the data (git only)
```bash
mkdir -p ~/datasets && cd ~/datasets
# GTSRB (≈400 MB of PNGs)
git clone --depth 1 --filter=blob:none --no-checkout https://github.com/AsuosOnurb/traffic-sign-recognition gtsrb_src
cd gtsrb_src && git sparse-checkout set --no-cone 'data/' && git checkout && cd ..
# IDD-derived sign boxes, frontFar camera only (≈360 MB)
git clone --depth 1 --filter=blob:none --no-checkout https://github.com/AbhayVAshokan/Traffic-Sign-Detection-Dataset idd_signs
cd idd_signs && git sparse-checkout set --no-cone 'JPEGImages/frontFar/' 'labels/frontFar/' && git checkout && cd ..
# comma10k (full repo is large; this takes everything — use sparse-checkout for a subset)
git clone --depth 1 --filter=blob:none --no-checkout https://github.com/commaai/comma10k comma10k
```
For comma10k, the exact 300-file list used is reproducible from `data/processed/manifests/comma10k_all.csv`
(`relative_path` + `annotation_path` columns) — pass those paths to `git sparse-checkout set --no-cone --stdin`.

Link them into the project (paths are configurable in `configs/data.yaml`):
```bash
ln -s ~/datasets/gtsrb_src/data   data/raw/gtsrb
ln -s ~/datasets/idd_signs        data/raw/idd_signs
ln -s ~/datasets/comma10k         data/raw/comma10k
```

## 3. Data pipeline
```bash
export PYTHONPATH=src            # or `pip install -e .`
python -m traffic_monitor.cli init-data
python -m traffic_monitor.cli build-manifest --dataset gtsrb      # also: idd_signs, comma10k
python -m traffic_monitor.cli split-data     --dataset gtsrb      # group-aware, seed from configs/data.yaml
python -m traffic_monitor.cli audit-data     --dataset gtsrb      # -> outputs/reports/audit_gtsrb.{json,md}
```
The manifests used for the reported results are already in `data/processed/manifests/`; rebuilding them is only needed for new data.

## 4. Train and evaluate (each writes metrics JSON + figures + a row in experiments/experiment_log.csv)
| Command | Time on 2 CPU cores |
|---|---|
| `python -m traffic_monitor.cli fundamentals` | ~30 s |
| `python -m traffic_monitor.cli train-classifier` | ~7 min (CNN is most of it) |
| `python -m traffic_monitor.cli train-detector` | ~15–25 min |
| `python -m traffic_monitor.cli train-segmenter` | ~2 min |
| `python -m traffic_monitor.cli train-temporal` | ~30 s |
| `python -m traffic_monitor.cli train-vae` | ~3 min |

Each command takes `--config configs/<file>.yaml`; every hyperparameter lives in those files.

## 5. Run the application
```bash
python -m traffic_monitor.cli predict --source path/to/road.jpg            # -> outputs/predictions/<name>_annotated.jpg + .json
python -m traffic_monitor.cli predict-video --source path/to/clip.mp4 --max-frames 100
python -m traffic_monitor.cli benchmark                                    # latency + memory on this machine
streamlit run app/dashboard.py                                             # optional UI
```
Output JSON per image: each detection's box, detection score, GTSRB class (or `unknown` below 0.6 confidence), road-mask coverage, per-stage timings, model status and the disclaimer.

## 6. Tests
```bash
python -m compileall src scripts app
python -m pytest -q
```
Tests use tiny generated fixtures and need neither the datasets nor trained checkpoints.

## 7. Reading the outputs
- Missing checkpoint → a clear `error: ... checkpoint not found` (exit code 2). The app never shows fake predictions.
- "no traffic sign detected" means models ran and found nothing; "Model not loaded" means they did not run.
- Sign names come from the German GTSRB taxonomy. On Indian roads many signs will be `unknown` or wrong — see `docs/MODEL_CARD.md`.
