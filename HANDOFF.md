# CS117 Project — Handoff / Status

_Last updated: 2026-11-17 (night of the check-in)_
_Repo: https://github.com/rushil-dil/117-Final-Project — local: `~/Documents/117-Final-Project`_

## TL;DR — where things stand

The **required check-in deliverable is measured**: a real YOLO11n baseline on the
`ojdzs` dataset, saved to `results/baseline_valid_ojdzs.json`. Still to do tonight:
**commit + push** the new baseline and code edits (not yet pushed), then write the
slide answers (Items 1, 3, 4, 5, 6). No more code needed for the check-in.

## Dataset change (2026-11-17) — switched to the `ojdzs` dataset

We switched the project dataset from the original `roboflow-jvuqo/football-players-detection-3zvbc`
(v20, 298/49/25) to **`new-workspace-4dtyu/football-players-ball-ojdzs-s4i2s`**
(v2, 1190/149/148). Reason: the original broadcast frames were too zoomed-out —
players and the ball occupy very few pixels, so the same pretrained model scored
far lower (ball at 0%) because of image scale, not something we could study well.
The `ojdzs` footage is closer-range and lands in the proposal's expected range.
Changing the dataset is allowed; **disclose it in check-in Item 1c** (reason:
original footage too low-resolution per object to support a useful error taxonomy).

Classes: `ball, player, referee` (3). Mapping unchanged — player/referee -> person
(COCO 0), ball -> sports ball (COCO 32); the keyword mapping already handles it, so
no code change was needed.

## Measured baseline (Item 2)

YOLO11n (`yolo11n.pt`, COCO-pretrained), imgsz 640, conf 0.25, on the **valid/dev
split of the `ojdzs` dataset (149 images)**. Metric = recall at IoU ≥ 0.5 (person)
/ ≥ 0.3 (ball).

| Class | Recall | Detail |
|-------|--------|--------|
| person | **85.7%** | 1599/1867 matched · 268 misses · 404 false positives |
| ball | **20.0%** | 24/120 matched · 96 misses · 3 false positives |

- Numbers saved to `results/baseline_valid_ojdzs.json`.
- vs. proposal estimate (75–85% person, 15–30% ball): measured is **within both**.
- For context, the old dataset scored 0% ball at 640 — see "Dataset change" above
  for why we switched (old-dataset result files have been removed).

(The imgsz-640-vs-1280 resolution comparison is the final presentation's own-method
result — not part of the check-in. Don't report 1280 numbers here.)

## How to reproduce

```bash
cd ~/Documents/117-Final-Project
pip install -r requirements.txt            # uses python3 (framework), not anaconda

# 1) download the ojdzs dataset (needs a free Roboflow API key)
RF_WORKSPACE=new-workspace-4dtyu RF_PROJECT=football-players-ball-ojdzs-s4i2s \
  RF_OUTDIR=datasets/football-players-ball-ojdzs \
  ROBOFLOW_API_KEY=your_key python3 scripts/download_data.py

# 2) run the baseline (imgsz 640)
python3 src/evaluate_baseline.py --split valid --out results/baseline_valid_ojdzs.json
```

Notes:
- Use **`python3`** on this machine — `python` points at anaconda, which does not
  have ultralytics/roboflow installed. The framework `python3` does.
- `datasets/` and `*.pt` weights are git-ignored (not committed).
- `download_data.py` and `evaluate_baseline.py` both default to the `ojdzs`
  dataset now; `download_data.py` still accepts `RF_WORKSPACE` / `RF_PROJECT` /
  `RF_OUTDIR` env overrides for other projects.
- Dataset: Roboflow `new-workspace-4dtyu/football-players-ball-ojdzs-s4i2s` v2
  (1190 train / 149 valid / 148 test). Classes: ball, player, referee.
  Mapped to: player/referee -> person (COCO 0), ball -> sports ball (COCO 32).
  (Switched from the original `-3zvbc` set — see "Dataset change" above.)

## Repo layout

```
src/evaluate_baseline.py          # inference + IoU matching + recall, writes results JSON
scripts/download_data.py          # pulls the Roboflow dataset (defaults to ojdzs)
results/baseline_valid_ojdzs.json # current measured baseline (ojdzs, imgsz 640)
requirements.txt .gitignore README.md
datasets/                         # downloaded data (git-ignored)
```

## What's left for the check-in (due tonight, 11:58pm PT)

Download `checkin_template.pptx` from Canvas, fill every lettered box, export PDF,
one person submits to "Final Project - Check-in".

- **Item 1** — question + metric (recall @ IoU 0.5/0.3); on the main plan (error
  taxonomy); note the 3 proposal-feedback changes (non-overlapping categories,
  report errors-introduced at 1280 too, soften causal claims).
- **Item 2** — DONE. Paste numbers above + command + `results/baseline_valid_ojdzs.json`.
- **Item 3** — sketch final Table 1 (error categories) and/or recall-vs-object-size
  plot, labelled axes, baseline row filled, own-method (1280) row blank with an
  owner + date before Nov 30/Dec 2.
- **Item 4** — blockers (e.g. overlapping category definitions; CPU slow for 1280),
  each with an action + named owner.
- **Item 5** — a row per member with a file/commit as evidence; repo URL.
- **Item 6** — AI tools used + one specific wrong output caught and how.

## Next steps after the check-in (for the final presentation)

- Run the **imgsz 1280** comparison as the own-method result: `python3
  src/evaluate_baseline.py --split valid --imgsz 1280 --out
  results/baseline_valid_ojdzs_1280.json`. Report errors *fixed* AND *introduced*,
  plus overall recall/FP (per the proposal feedback). This is a FINAL deliverable,
  not part of the check-in.
- Build the error taxonomy (≥100 dev errors), with a defined sampling procedure and
  a priority rule so categories don't overlap.
- Check whether the `ojdzs` train/valid/test split is separated by match clip; if it
  is a random split, re-split by clip to avoid near-duplicate frames leaking between
  dev and test.
- Freeze categories + thresholds, THEN evaluate the held-out **test split (148 imgs)** once.
- Start `CONTRIBUTIONS.md` (from the Item 5 table) and keep borrowed code in `third_party/`
  for the Dec 7 final code submission.
