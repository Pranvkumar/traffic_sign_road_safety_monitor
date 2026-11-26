# Dataset card

**Date accessed:** 21 November 2026 · **Access route:** `git clone` from GitHub (the only route the build environment allowed; official dataset portals and Kaggle/HuggingFace were blocked by the network proxy).

All counts below were **measured** from the downloaded files by `audit-data`, not copied from catalogues. Full reports: `outputs/reports/audit_*.json|md`.

## Dataset register

| Dataset | Exact source / version | Licence / terms | Role in project |
|---|---|---|---|
| **GTSRB** (German Traffic Sign Recognition Benchmark) | PNG copy of the official *Final_Training* + labelled *Final_Test* sets in `github.com/AsuosOnurb/traffic-sign-recognition`, commit `dd4b65f` (2022-06-28) | Original benchmark: free for research, cite Stallkamp et al. (2012). **Redistribution rights of the GitHub copy: UNVERIFIED** — do not redistribute; re-download from <https://benchmark.ini.rub.de/> for the final submission if possible | Cropped-sign classification, VAE augmentation, temporal (track) experiment |
| **IDD-derived sign boxes** | `github.com/AbhayVAshokan/Traffic-Sign-Detection-Dataset`, commit `2a0f01d` (2021-01-03), **`frontFar` camera subset only** | Derived from IDD (IIIT Hyderabad), whose licence requires accepting terms on <https://idd.insaan.iiit.ac.in/>. **UNVERIFIED that the derived copy may be redistributed.** Academic coursework use only; do not publish images | Full-scene traffic-sign *localisation* on Indian roads |
| **comma10k** | `github.com/commaai/comma10k`, commit `6c205fe`; seeded random 300-image subset of `imgs/` (seed 42) | **MIT** (© 2020 Comma.ai) | Road-surface segmentation |

Not obtained (blocked or login-only): IDD Detection/Segmentation official releases, IDD-MTSVD, MissingTSMini, GTSDB official archive, BDD100K, LISA, Mapillary. These remain the recommended upgrades (see `docs/STATUS.md`).

## GTSRB

- **Unit:** cropped sign image (PNG, RGB, 25–266 px wide). 43 classes (IDs 0–42 kept unchanged; names in `configs/label_taxonomy.yaml`).
- **Measured counts:** 39,209 official-training + 12,630 official-test = **51,839** images (matches the published release).
- **Groups:** training files are `<track>_<frame>.png`; one track = 30 (29–30) frames of **one physical sign**, ordered far → near (median width 30 px at frame 0 → 94 px at frame 29).
- **Split (seed 42):** official test kept untouched as `test`. Official training split into `train`/`val` **by physical-sign track**, stratified by class: train 32,939 imgs / 1,098 tracks, val 6,270 imgs / 209 tracks; all 43 classes in every split. Leaked groups: **0**.
- **Class imbalance:** max/min count ratio 11.1 (smallest classes have 180 training crops).
- **Audit findings:** no missing or undecodable files. 16 byte-identical files (8 pairs span splits — official-train vs official-test duplicates inherited from the source); kept and documented, not deleted. The 8×8 average hash is too coarse for 32-px signs (12,628 files share a hash with another), so it is not used for de-duplication decisions on GTSRB.
- **Not a detection dataset** — used for crop classification only.

## IDD-derived traffic-sign boxes (frontFar)

- **Unit:** full road-scene video frame, JPEG 1280×964 RGB, Bangalore drives (2018). Single class `0 = traffic_sign` (no sign *type* labels). YOLO txt labels.
- **Measured counts:** 928 images in **16 drive sequences**, 1,257 boxes; every image has ≥ 1 box. Box width min 5 px, median 31 px, max 717 px.
- **Split (seed 42, by drive sequence, largest-first greedy):** train 617 imgs / 13 seqs, val 172 / 1 seq, test 139 / 2 seqs (66/19/15 %). With only 16 sequences the val/test sets are small and come from few drives — metrics carry large uncertainty.
- **Audit findings:** 7 box problems (4 boxes extend past the image border → clipped at load time; 3 implausible aspect ratios > 6, likely long rectangular boards) — kept and reported. Files named `NNNNNN_r.jpg` and `NNNNNN0.jpg` are *different* frames (mean abs. pixel difference 23), not duplicates. 10 coarse-hash collisions cross splits; manual check showed they are different scenes with similar sky/road layout. Sequence-level splitting prevents near-frame leakage.
- **Limitation:** frames were pre-filtered to contain signs, so this set cannot measure false alarms on sign-free road, and frames are ~6 frames apart (not continuous video).

## comma10k subset

- **Unit:** dashcam image 1164×874 PNG + RGB colour mask. Source classes: road `#402020`, lane markings `#ff0000`, undrivable `#808060`, movable `#00ff66`, my car `#cc00ff`.
- **Project mapping:** `road_surface` = road ∪ lane markings (1); `not_road` = undrivable ∪ movable (0); `my_car` = ignore (255); unknown colours = ignore.
- **Measured counts:** 300 images from **181 routes**. Split by route hash: train 210 / 128 routes, val 45 / 26, test 45 / 27. Leaked groups 0. Road pixels ≈ 27 % of labelled pixels.
- **Limitation:** mostly US highway/suburban driving — different road appearance from Indian roads. Applied to IDD images only qualitatively.

## Privacy

No faces or number plates were annotated or published. Example figures show public-dataset frames for coursework only; blur identifiable details before any public release.

## HRP4K (potholes, hazard model)
Source: https://zenodo.org/records/21941931 (CC BY 4.0). Not downloaded whole (12.8 GB): `scripts/prepare_hrp4k.py` range-reads 2,800 frames (2,000 train / 400 val / 400 test, some with no pothole), shrinks them to 1280 px wide and copies the YOLO labels unchanged into `data/raw/hrp4k_yolo/` with its `hrp4k_yolo.yaml`.
