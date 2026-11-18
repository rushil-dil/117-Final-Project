"""Baseline evaluation for CS117 final project (Team 7).

Runs the pretrained YOLO11n (COCO) detector on the project's soccer dataset
(football-players-ball-ojdzs; classes ball/player/referee) and measures recall +
false positives for two merged classes:

    person  <- dataset labels: player, referee (and keeper, if present)  (COCO class 0)
    ball    <- dataset label:  ball                          (COCO class 32, "sports ball")

Matching rules (from the proposal):
    - detection <-> label match requires IoU >= 0.5 for people, >= 0.3 for ball
    - an unmatched label  = a miss (hurts recall)
    - an unmatched prediction = a false positive
    - each label is matched to at most one prediction (greedy, highest IoU first)

Default run: imgsz 640, conf 0.25, on the validation ("dev") split.

Usage:
    python3 src/evaluate_baseline.py --split valid        # baseline, imgsz 640
    python3 src/evaluate_baseline.py --split valid --out results/baseline_valid_ojdzs.json

Output: a JSON file in results/ with the measured numbers, plus a printed summary.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml
from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = REPO_ROOT / "datasets" / "football-players-ball-ojdzs"

# COCO class ids produced by yolo11n.pt that we care about.
COCO_PERSON = 0
COCO_SPORTS_BALL = 32

# IoU thresholds for calling a detection a correct match.
IOU_PERSON = 0.5
IOU_BALL = 0.3

# Folder names Roboflow's yolov8 export can use for the validation split.
VAL_ALIASES = ["valid", "val"]


def load_class_names(data_yaml: Path) -> dict[int, str]:
    with open(data_yaml) as f:
        data = yaml.safe_load(f)
    names = data["names"]
    if isinstance(names, list):
        names = {i: n for i, n in enumerate(names)}
    return {int(k): str(v).lower() for k, v in names.items()}


def label_to_group(class_name: str) -> str | None:
    """Map a dataset label class name to our evaluation group.

    Handles both dataset variants:
      - original: ball / goalkeeper / player / referee
      - 9k fork:  Keeper / another-player / ball / player-white / referee
    All non-ball person-type labels (any variant of player/keeper/referee)
    merge into 'person'; 'ball' stays 'ball'.
    """
    name = class_name.lower()
    if name == "ball":
        return "ball"
    person_keywords = ("player", "keeper", "goalkeeper", "referee")
    if any(kw in name for kw in person_keywords):
        return "person"
    return None


def resolve_split_dir(data_root: Path, split: str) -> Path:
    """Find the images/ dir for a split, tolerating valid vs val naming."""
    candidates = [split]
    if split in VAL_ALIASES:
        candidates = VAL_ALIASES
    for name in candidates:
        img_dir = data_root / name / "images"
        if img_dir.is_dir():
            return data_root / name
    raise FileNotFoundError(
        f"Could not find split '{split}' under {data_root} "
        f"(looked for {', '.join(candidates)}/images)"
    )


def read_yolo_labels(label_path: Path, img_w: int, img_h: int,
                     names: dict[int, str]) -> dict[str, list[list[float]]]:
    """Read a YOLO .txt label file -> {'person': [[x1,y1,x2,y2], ...], 'ball': [...]}."""
    boxes: dict[str, list[list[float]]] = {"person": [], "ball": []}
    if not label_path.exists():
        return boxes
    with open(label_path) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 5:
                continue
            cls_id = int(float(parts[0]))
            cx, cy, w, h = (float(v) for v in parts[1:5])
            group = label_to_group(names.get(cls_id, ""))
            if group is None:
                continue
            x1 = (cx - w / 2) * img_w
            y1 = (cy - h / 2) * img_h
            x2 = (cx + w / 2) * img_w
            y2 = (cy + h / 2) * img_h
            boxes[group].append([x1, y1, x2, y2])
    return boxes


def iou(box_a: list[float], box_b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def match_group(gt_boxes: list[list[float]], pred_boxes: list[list[float]],
                iou_thr: float) -> tuple[int, int, int]:
    """Greedy one-to-one matching. Returns (matched, misses, false_positives)."""
    pairs = []
    for gi, gt in enumerate(gt_boxes):
        for pi, pr in enumerate(pred_boxes):
            score = iou(gt, pr)
            if score >= iou_thr:
                pairs.append((score, gi, pi))
    pairs.sort(reverse=True)  # highest IoU first
    used_gt: set[int] = set()
    used_pred: set[int] = set()
    matched = 0
    for score, gi, pi in pairs:
        if gi in used_gt or pi in used_pred:
            continue
        used_gt.add(gi)
        used_pred.add(pi)
        matched += 1
    misses = len(gt_boxes) - matched
    false_positives = len(pred_boxes) - matched
    return matched, misses, false_positives


def main() -> None:
    parser = argparse.ArgumentParser(description="YOLO11n soccer baseline evaluation")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA,
                        help="dataset root (contains data.yaml and split folders)")
    parser.add_argument("--split", default="valid",
                        help="split to evaluate: train / valid / test")
    parser.add_argument("--weights", default="yolo11n.pt")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--out", type=Path, default=None,
                        help="output JSON path (default: results/baseline_<split>.json)")
    args = parser.parse_args()

    data_root = args.data
    names = load_class_names(data_root / "data.yaml")
    split_dir = resolve_split_dir(data_root, args.split)
    img_dir = split_dir / "images"
    lbl_dir = split_dir / "labels"

    image_paths = sorted(
        p for p in img_dir.iterdir()
        if p.suffix.lower() in (".jpg", ".jpeg", ".png")
    )
    if not image_paths:
        raise FileNotFoundError(f"No images found in {img_dir}")

    model = YOLO(args.weights)

    totals = {
        "person": {"gt": 0, "matched": 0, "misses": 0, "fp": 0, "pred": 0},
        "ball": {"gt": 0, "matched": 0, "misses": 0, "fp": 0, "pred": 0},
    }

    print(f"Evaluating {len(image_paths)} images from split "
          f"'{split_dir.name}' at imgsz={args.imgsz}, conf={args.conf} ...")

    for idx, img_path in enumerate(image_paths, 1):
        result = model.predict(
            source=str(img_path), imgsz=args.imgsz, conf=args.conf, verbose=False
        )[0]
        img_h, img_w = result.orig_shape

        preds = {"person": [], "ball": []}
        for box in result.boxes:
            cls_id = int(box.cls.item())
            xyxy = box.xyxy[0].tolist()
            if cls_id == COCO_PERSON:
                preds["person"].append(xyxy)
            elif cls_id == COCO_SPORTS_BALL:
                preds["ball"].append(xyxy)

        label_path = lbl_dir / (img_path.stem + ".txt")
        gts = read_yolo_labels(label_path, img_w, img_h, names)

        for group, iou_thr in (("person", IOU_PERSON), ("ball", IOU_BALL)):
            matched, misses, fp = match_group(gts[group], preds[group], iou_thr)
            t = totals[group]
            t["gt"] += len(gts[group])
            t["pred"] += len(preds[group])
            t["matched"] += matched
            t["misses"] += misses
            t["fp"] += fp

        if idx % 25 == 0 or idx == len(image_paths):
            print(f"  {idx}/{len(image_paths)} images done")

    def recall(group: str) -> float:
        g = totals[group]
        return g["matched"] / g["gt"] if g["gt"] else 0.0

    summary = {
        "model": args.weights,
        "dataset": str(data_root.name),
        "split": split_dir.name,
        "num_images": len(image_paths),
        "imgsz": args.imgsz,
        "conf": args.conf,
        "iou_thresholds": {"person": IOU_PERSON, "ball": IOU_BALL},
        "person": {**totals["person"], "recall": round(recall("person"), 4)},
        "ball": {**totals["ball"], "recall": round(recall("ball"), 4)},
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }

    out_path = (args.out or (REPO_ROOT / "results" / f"baseline_{split_dir.name}.json")).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n==================== BASELINE RESULT ====================")
    print(f"  split: {summary['split']}  |  images: {summary['num_images']}  "
          f"|  imgsz: {summary['imgsz']}  |  conf: {summary['conf']}")
    for group in ("person", "ball"):
        g = summary[group]
        print(f"  {group:6s}  recall={g['recall']*100:5.1f}%  "
              f"(matched {g['matched']}/{g['gt']}, misses {g['misses']}, "
              f"false positives {g['fp']})")
    try:
        shown = out_path.relative_to(REPO_ROOT)
    except ValueError:
        shown = out_path
    print(f"  saved -> {shown}")
    print("=========================================================")


if __name__ == "__main__":
    main()
