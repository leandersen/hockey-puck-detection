import argparse
import csv
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
from ultralytics import YOLO

from utils import parse_yolo_annotation, CLASS_NAMES

PUCK_CLASS = 5

def compute_iou(box1:tuple, box2: tuple) -> float:
    """
    Intersection over Union between two boxes in (x1, y1, x2, y2) format
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    return intersection/union if union > 0 else 0.0

def yolo_box_to_pixel(box, img_w: int, img_h: int) -> tuple:
    """Convert normalized YOLO (cx,cy,w,h) to pixel (x1,y1,x2,y2)."""
    cx = box["cx"] * img_w
    cy = box["cy"] * img_h
    w  = box["w"]  * img_w
    h  = box["h"]  * img_h
    return (cx - w/2, cy - h/2, cx + w/2, cy + h/2)

def load_situation_labels(csv_path:str) -> dict:
    """
    load situation labels from CSV file.
    CSV format: image_name, situation
    """
    labels = {}
    if csv_path is None:
        return labels
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            labels[row['image_name']] = row['situation']
    return labels

def evaluate(
        model_path: str,
        img_dir: str,
        ann_dir: str,
        situation_csv: str = None,
        iou_threshold: float = 0.5,
        conf_threshold: float = 0.25,
        output_dir: str = "results/metrics"
):
    """
    Evaluate the trained YOLO modle on a set of images. Computes per-situation precision, recall, and F1 for puck detection.
    Args:
        model_path: path to trained weights,
        img_dir: path to validation images,
        ann_dir: path to validation anotations,
        situation_csv: path to CSV mapping images for game situations,
        iou_threshold: IoU threshold to count a detection as correct,
        conf_threshold: minimum confidence to consider a prediction,
        output_dir: save results to path
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok = True)

    model = YOLO(model_path)
    situation_labels = load_situation_labels(situation_csv)

    img_files = sorted(Path(img_dir).glob("*.jpg"))

    # track per-image results
    # each entry: {situation, tp, fp, fn}

    results = []

    print(f"Evaluating {len(img_files)} images...")

    for img_path in img_files:
        ann_path = Path(ann_dir) / img_path.with_suffix(".txt").name

        if not ann_path.exists():
            continue

        truth_boxes_all = parse_yolo_annotation(str(ann_path))
        truth_pucks = [b for b in truth_boxes_all if b['class_id'] == PUCK_CLASS]

        preds = model.predict(str(img_path), conf=conf_threshold, verbose = False)
        result = preds[0]
        img_h, img_w = result.orig_shape

        pred_pucks = []
        if result.boxes is not None:
            for box in result.boxes:
                if int(box.cls.item()) == PUCK_CLASS:
                    pred_pucks.append(box.xyxy[0].to_list())
        
        # convert truth to pixel coords
        truth_pixel = [yolo_box_to_pixel(b, img_w, img_h) for b in truth_pucks]

        # match predictions to truth using IoU
        matched_truth = set()
        tp = 0
        fp = 0

        for pred_box in pred_pucks:
            best_iou = 0
            best_idx = -1
            for i, truth_box in enumerate(truth_pixel):
                if i in matched_truth:
                    continue
                iou = compute_iou(pred_box, truth_box)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = i

            if best_iou >= iou_threshold and best_idx >= 0:
                tp += 1
                matched_truth.add(best_idx)
            else:
                fp += 1
        fn = len(truth_pucks) - len(matched_truth)

        situation = situation_labels.get(img_path.name, "unlabeled")

        results.append({
            "image": img_path.name,
            "situation": situation,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            'truth_pucks': len(truth_pucks),
            'pred_pucks': len(pred_pucks)
        })
    
    # compute per-situation metrics
    situation_stats = defaultdict(lambda: {"tp": 0, "fp":0, "fn":0})

    for r in results:
        s = r['situation']
        situation_stats[s]['tp'] += r['tp']
        situation_stats[s]['fp'] += r['fp']
        situation_stats[s]['fn'] += r['fn']

    #always inlude overall
    situation_stats["overall"]["tp"] = sum(r["tp"] for r in results)
    situation_stats["overall"]["fp"] = sum(r["fp"] for r in results)
    situation_stats["overall"]["fn"] = sum(r["fn"] for r in results)

    print("-" * 55)
    print("\nPuck Detection Results:\n")
    print(f"{'Situation':<15} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    print("-" * 55)

    summary_rows = []
    for situation, stats in sorted(situation_stats.items()):
        tp = stats['tp']
        fp = stats['fp']
        fn = stats['fn']

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        support = tp + fn

        print(f"{situation:<15} {precision:>10.3f} {recall:>10.3f} {f1:>10.3f} {support:>10}")
        
        summary_rows.append({
            "situation": situation,
            "precision": round(precision, 4),
            "recall":    round(recall, 4),
            "f1":        round(f1, 4),
            "support":   support,
            "tp": tp, "fp": fp, "fn": fn
        })

    # save to csv
    summary_path = output_dir / "situation_metrics.csv"
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
        writer.writeheader()
        writer.writerows(summary_rows)

    per_image_path = output_dir / "per_image_results.csv"
    with open(per_image_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"\nSaved: {summary_path}")
    print(f"Saved: {per_image_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate puck detection by game situation")
    parser.add_argument("--model",     required=True, help="Path to trained weights (.pt)")
    parser.add_argument("--img-dir",   required=True, help="Validation images folder")
    parser.add_argument("--ann-dir",   required=True, help="Validation annotations folder")
    parser.add_argument("--situations",default=None,  help="CSV mapping images to situations")
    parser.add_argument("--iou",       type=float, default=0.5)
    parser.add_argument("--conf",      type=float, default=0.25)
    parser.add_argument("--out-dir",   default="results/metrics")
    args = parser.parse_args()

    evaluate(
        model_path=args.model,
        img_dir=args.img_dir,
        ann_dir=args.ann_dir,
        situation_csv=args.situations,
        iou_threshold=args.iou,
        conf_threshold=args.conf,
        output_dir=args.out_dir
    )
