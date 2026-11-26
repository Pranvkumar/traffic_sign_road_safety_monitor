# YOLO11n sign detector: results (2026-11-21)

## What changed
- New detector: YOLO11n (COCO-pretrained), fine-tuned on ETSC `dsilbermann/european_traffic_signs`
  (171 European sign classes, CC BY-NC-SA 4.0, so **non-commercial use only**). It detects and classifies
  in one step. The NumPy HOG detector + MLP stay as the baseline (`configs/default.yaml`).
- `configs/yolo.yaml` is now the default for the CLI (`predict`, `predict-video`, `benchmark`) and for `app/dashboard.py`.
- Weights: `models/checkpoints/yolo_signs.pt` (5.7 MB). Training run: `experiments/runs/yolo/etsc/`,
  test evaluation: `experiments/runs/yolo/etsc_test/`, log: `experiments/train_yolo.log`.

## Fixes made along the way
1. **Label offset:** ETSC `metadata.jsonl` class ids are 1-based, while the yaml and shipped labels are 0-based.
   `scripts/prepare_etsc.py` subtracts 1 (`META_OFFSET`). Checked against the repo's own `labels/train/0/1002.txt`.
   Without this fix, every sign would have trained under its neighbour's name.
2. **No "Right bend" in train:** 421 mirrored copies were added, but only of images whose signs are all mirror-safe pairs.
   Signs with digits or text are never flipped. Ultralytics `fliplr` is set to 0 for the same reason.
3. **Windows crash "CUDA error: resource already mapped"** in the pin-memory thread: pinned memory is switched off in `train_yolo.py`.
4. **"LLVM ERROR: out of memory":** orphaned DataLoader workers from the crashed run held about 20 GB of memory.
   Training now uses `--workers 2`.
5. **Class-agnostic NMS:** without it, one sign got two boxes ("50" and "70"). Now each sign gets one box and one label.

## Training
`python scripts/train_yolo.py --model models/pretrained/yolo11n.pt --epochs 25 --batch 8 --imgsz 960 --workers 2`
on an RTX 4060 8 GB: about 10 min per epoch, about 4.3 h total.

| Split | mAP50 | mAP50-95 |
|---|---|---|
| val (epoch 25) | 0.634 | 0.560 |
| **test (held out)** | **0.643** | **0.567** |

Inference: about 5 ms per image on the GPU (plus about 6 ms pre- and post-processing).

## `rural_signs.webp` (the image the old detector failed on)
| Sign | Old HOG detector | YOLO |
|---|---|---|
| Double bend | one box, `unknown` | **Double bend (right) 0.90**, correct |
| 8% grade | missed | **Dangerous descent 0.68**: right box and right sign family (gradient warning) |
| 70 km/h | missed | box found, labelled **Maximum speed limit 50 (0.66)**, which is wrong; "70" scored 0.45 on the same box |

All 3 signs are found, and 2 of the 3 labels are acceptable. Telling 50 from 70 is the weak spot.

## 20 random online photos (Openverse, CC-licensed, `sources.json` lists the licences)
The images came from random pages of searches such as "road sign", "curve warning", "speed limit" and "India road sign".
- **No false alarms on non-signs:** posters, a Hindi shop sign, landscapes and interpretive boards all had 0 boxes.
  The baseline put `unknown` boxes on most of them.
- **Plausible detections:** chevron curve markers labelled "Curve left" (4 boxes), and overhead guide signs labelled
  "Direction to place".
- **Wrong or missed:** US/Asian yellow diamond warnings (bicycle, curve, "caution steep", traffic light) and the US
  rectangular "60" limit were missed. A diamond arrow sign was labelled "Level-crossings"; a "No stopping" sign was
  labelled "Direction to place"; red signs in Spain came out `unknown`.

**Conclusion:** the model works on European-style signs (circle and triangle shapes) and stays quiet on non-signs.
It does not transfer to US-style diamond or rectangular signs, which are absent from ETSC. Indian signs follow the
same Vienna-convention shapes as European ones, so they are closer, but **no labelled Indian test set exists yet**.
A small hand-labelled Indian set would be the next step, both to measure this and to fine-tune.

## Pothole (hazard) model
`models/checkpoints/yolo_hazard.pt`: YOLO11n, 1 class (`pothole`), 30 epochs at 960 px on 2,000 HRP4K frames (CC BY 4.0). test mAP@0.5 **0.358** (mAP@0.5:0.95 0.199, precision 0.47, recall 0.38) on 400 held-out HRP4K frames.
It is a usable demo, not a reliable detector: it catches clear, dark potholes and misses small or shallow ones. On `test_images/rural_signs.webp` it reports 0 potholes (correct), so it does not disturb sign results. More training frames and 1280 px input are the obvious next steps.
