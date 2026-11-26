# Label taxonomy (v1)

Machine-readable version: `configs/label_taxonomy.yaml`.

## Detection
| Project ID | Name | Source |
|---|---|---|
| 0 | `traffic_sign` | IDD-derived boxes, class 0 (unchanged) |

The detector is **class-agnostic**: it localises "a sign" only. Sign type comes from the crop classifier.

## Classification
GTSRB class IDs 0–42 are used **unchanged** (German signs, e.g. 14 = stop, 25 = road works). No mapping to Indian (IRC) sign codes has been made, because no labelled Indian sign-type data was available.

**Unknown policy:** if the classifier's top probability is below 0.6 (`configs/default.yaml → classifier.unknown_threshold`), the sign is reported as `unknown` with its top guess shown separately. Indian signs that look different from German ones (text boards, regional language, different pictograms) are expected to fall here or be misclassified — this is a known domain-shift limitation.

## Segmentation
| Project ID | Name | comma10k source classes |
|---|---|---|
| 0 | `not_road` | undrivable, movable |
| 1 | `road_surface` | road, lane markings |
| 255 | ignore | my car (ego bonnet/mount), any unrecognised colour |

`road_surface` is **not** the same as "drivable area" in BDD100K (which distinguishes direct/alternative lanes).

## Hazards — potholes only
One hazard class is supported: **pothole** (class 0), from HRP4K (Zenodo record 21941931, CC BY 4.0), which labels potholes with boxes on 4K dashcam frames. `scripts/prepare_hrp4k.py` builds a 2,000/400/400 train/val/test subset (frames shrunk to 1280 px) and `scripts/train_hazard.py` fine-tunes YOLO11n on it → `models/checkpoints/yolo_hazard.pt`; test mAP@0.5 **0.358** (mAP@0.5:0.95 0.199, precision 0.47, recall 0.38) on 400 held-out HRP4K frames.

Other hazards (debris, fallen objects, stalled vehicles) are still **not** detected: no obtained dataset labels them. comma10k's `movable` class (vehicles, people, animals) is not a hazard label, since a vehicle or pedestrian is not a hazard without context. Adding them needs a dataset with real labels or an annotation guide (definition per class, two annotators, adjudication, inter-annotator agreement) on authorised footage.
