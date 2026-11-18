# CS117 Final Project — Team 7

**Question:** What mistakes does YOLO11n make when detecting people and the ball in
soccer footage, how common is each type of error, and does increasing the input
resolution reduce the most common type?

Members: Aditya Jain, Michelle Makhija, Rushil Dileep, Varun Mehul Nanavati

**Archetype:** 5 — failure taxonomy (idea-bank #22). We run a pretrained detector
as-is (no training), measure where it fails, and group the failures by type.

## Setup

```bash
pip install -r requirements.txt
```

Use `python3` (the framework Python). On our machine `python` points at an
anaconda install that does not have `ultralytics`/`roboflow`.

## Data

Public dataset: Roboflow Universe
`new-workspace-4dtyu/football-players-ball-ojdzs-s4i2s` (v2, YOLOv8 format).
A free Roboflow API key is required to download it.

```bash
ROBOFLOW_API_KEY=your_key python3 scripts/download_data.py
```

This writes to `datasets/football-players-ball-ojdzs/` (git-ignored).

| Split | Images |
|-------|--------|
| train | 1190 |
| valid (dev) | 149 |
| test (held out) | 148 |

Classes: `ball`, `player`, `referee`. We merge `player`/`referee` (and `keeper`,
if present in a dataset) into **person** and keep `ball` separate.

## Baseline (YOLO11n @ imgsz 640, conf 0.25)

Runs pretrained `yolo11n.pt` and measures recall + false positives for two merged
classes — `person` (COCO class 0) and `ball` (COCO class 32, "sports ball"). A
label matches a detection at IoU >= 0.5 (person) / >= 0.3 (ball); unmatched labels
are misses, unmatched detections are false positives.

```bash
python3 src/evaluate_baseline.py --split valid --out results/baseline_valid_ojdzs.json
```

Measured numbers are written to `results/`.

## Results

Measured on the `valid` (dev) split, 149 images, `yolo11n.pt`, imgsz 640, conf 0.25.
Full numbers: [`results/baseline_valid_ojdzs.json`](results/baseline_valid_ojdzs.json).

| Run | Split | Images | imgsz | person recall | ball recall |
|-----|-------|--------|-------|---------------|-------------|
| Baseline | valid | 149 | 640 | **85.7%** | **20.0%** |

- Person: 1599/1867 matched, 268 misses, 404 false positives.
- Ball: 24/120 matched, 96 misses, 3 false positives — the "tiny ball" failure,
  measured: even on closer-range footage the pretrained model finds only 1 in 5
  balls.

## Design decisions

- **Metric — per-class recall (plus miss/false-positive counts).** The project
  asks *what the detector misses and why*, so recall is the natural headline, and
  we also log false positives for the error taxonomy. (We do not report mAP: a
  failure taxonomy cares about which objects are missed, not a precision-recall
  ranking.)
- **Two merged classes (person, ball).** The datasets label players, referees and
  keepers separately, but the COCO model only has a single `person` class, so we
  merge all person-type labels into `person` and map the dataset `ball` to COCO
  `sports ball`. The mapping is keyword-based, so it is robust to different class
  names across dataset versions.
- **IoU thresholds 0.5 (person) / 0.3 (ball).** People are large and benefit from a
  strict overlap test; the ball is tiny, where a strict IoU would reject detections
  that are essentially correct, so we relax it to 0.3.
- **Greedy one-to-one matching, highest IoU first.** Each ground-truth box is
  matched to at most one detection, which answers "was this object found at all."
- **Baseline settings fixed at imgsz 640, conf 0.25.** These are the defaults from
  our proposal and the operating point we evaluate against.
- **Dev/test discipline.** We develop (define categories, check the matching
  script) on the `valid` split only. The `test` split is held out and evaluated
  once, after the categories and thresholds are frozen.

### Why we switched datasets

Our proposal used `roboflow-jvuqo/football-players-detection-3zvbc`, a set of
wide, zoomed-out broadcast frames. On that data the same pipeline scored
**56.9% person / 0.0% ball** at imgsz 640 — the ball's single highest `sports ball`
confidence on any dev image was 0.123, below the 0.25 threshold, so **no ball ever
passed**. We verified this was not a bug (the class mapping is correct; the model
simply never clears the threshold). The problem was **image scale**: players and the
ball occupy too few pixels in that footage to study error *types* in a meaningful
way — almost everything collapses into one "object too small" bucket.

We switched to `new-workspace-4dtyu/football-players-ball-ojdzs-s4i2s`, which is
closer-range soccer footage. The identical pipeline now scores **85.7% person /
20.0% ball**, landing inside our proposal's expected range (75–85% person,
15–30% ball). Because only the images changed — same model, same settings, same
code — the jump confirms the original low scores came from footage scale, not a
pipeline error, and it gives us footage where distinct failure types (occlusion,
ball-near-feet, frame-edge, distant player, tiny ball) are actually separable.
Changing the dataset is permitted; it is disclosed in our check-in. The original
0% ball result on the old footage is what motivated the switch.

## Repo layout

```
src/evaluate_baseline.py           # inference + IoU matching + recall, writes results JSON
scripts/download_data.py           # pulls the Roboflow dataset (env-overridable)
results/baseline_valid_ojdzs.json  # current measured baseline (ojdzs, imgsz 640)
requirements.txt .gitignore
datasets/                          # downloaded data (git-ignored)
```
