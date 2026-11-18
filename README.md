# CS117 Final Project — Team 7

**Question:** What mistakes does YOLO11n make when detecting people and the ball in
broadcast soccer footage, how common is each type of error, and does increasing the
input resolution reduce the most common type?

Members: Aditya Jain, Michelle Makhija, Rushil Dileep, Varun Mehul Nanavati

## Setup

```bash
pip install -r requirements.txt
```

## Data

Public dataset: Roboflow Universe `roboflow-jvuqo/football-players-detection-3zvbc`
(YOLOv8 format). A free Roboflow API key is required to download it.

```bash
ROBOFLOW_API_KEY=your_key python scripts/download_data.py
```

This writes to `datasets/football-players-detection/` (git-ignored).

## Baseline (YOLO11n @ imgsz 640, conf 0.25)

Runs pretrained `yolo11n.pt` and measures recall + false positives for two merged
classes — `person` (player/goalkeeper/referee, COCO class 0) and `ball`
(COCO class 32). A label matches a detection at IoU >= 0.5 (person) / >= 0.3 (ball);
unmatched labels are misses, unmatched detections are false positives.

```bash
python src/evaluate_baseline.py --split valid        # baseline, imgsz 640
python src/evaluate_baseline.py --split valid --imgsz 1280 --out results/baseline_valid_1280.json
```

Measured numbers are written to `results/` (e.g. `results/baseline_valid.json`).

## Results

Measured on the `valid` (dev) split, 49 images, `yolo11n.pt`, conf 0.25.
Full numbers: [`results/baseline_valid.json`](results/baseline_valid.json).

| Run | Split | Images | imgsz | person recall | ball recall |
|-----|-------|--------|-------|---------------|-------------|
| Baseline | valid | 49 | 640 | 56.9% | 0.0% |

Person: 642/1129 matched, 487 misses, 166 false positives.
Ball: 0/45 matched — at imgsz 640 the model's highest `sports ball` confidence on
any valid image is 0.123, below the 0.25 threshold, so no ball passes. This is the
project's "tiny ball" failure, measured. The imgsz 1280 re-run is the planned next step.
