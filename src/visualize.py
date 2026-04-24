import argparse
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

from utils import (
    parse_yolo_annotation,
    draw_boxes,
    load_image,
    yolo_to_pixel,
    CLASS_NAMES,
    CLASS_COLORS
)

def predictions_to_boxes(results) -> list[dict]:
    """
    Convert predictions results into our standard box dict format so we can pass them to draw_boxes()
    Args: results - output from model.predict()
    Returns: list of box dicts matching utils.py format
    """
    boxes = []
    result = results[0]

    if result.boxes is None:
        return boxes
    
    img_h, img_w = result.orig_shape

    for box in result.boxes:
        cls = int(box.cls.item())
        conf = float(box.conf.item())
        x1, y1, x2, y2 = box.xyxy[0].tolist()

        # convert pixel coords back to normalized YOLO for draw_box
        cx = ((x1 + x2)/ 2) / img_w
        cy = ((y1 + y2)/2 / img_h)
        w = (x2 - x1)/img_w
        h = (y2 - y1) / img_h

        boxes.append({
            "class_id": cls,
            "class_name": f"{CLASS_NAMES.get(cls, 'Unknown')} {conf:.2f}",
            "cx": cx,
            "cy": cy,
            "w": w,
            "h": h
        })
    return boxes

def visualize_single(
        img_path:str,
        ann_path:str,
        model_path:str,
        output_path:str=None,
        conf_threshold:float=0.25
):
    """
    For single image, show ground truth boxes and model predictions side by side.

    Args:
        img_path: path to image file
        ann_path: path to the ground truth annotation
        model_path: path to trained YOLO weights
        output_path: output of result
        conf_threshold: minimum confidence to show a prediction
    """

    # Load image and annotations
    img = load_image(img_path)
    truth_boxes = parse_yolo_annotation(ann_path)

    # Draw ground truth on left panel
    left_panel = draw_boxes(img.copy(), truth_boxes)
    cv2.putText(left_panel, "Truth", (20,40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
    
    # run model
    model = YOLO(model_path)
    results = model.predict(img_path, conf = conf_threshold, verbose = False)
    pred_boxes = predictions_to_boxes(results)

    # draw predictions on right panel
    right_panel = draw_boxes(img.copy(), pred_boxes)
    cv2.putText(right_panel, "Predictions", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,255), 2)
    
    # stack side by side
    combined = np.concatenate([left_panel, right panel], axis = 1)

    # save or display
    if output_path:
        cv2.imwrite(output_path, combined)
        print(f'Saved to: {output_path}')
    else:
        cv2.imshow("Ground Truth vs Predictions", combined)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def visualize_failures(
        ann_dir: str,
        img_dir: str,
        model_path: str,
        output_dir: str,
        n: int = 20,
        conf_threshold: float = 0.25
):
    """
    Find images where the model missed the puck and save visualizations of those failure cases.

    args:
        ann_dir: path to annotations folder
        img_dir: path to images folder
        model_path: path to trained YOLO weights
        output_dir: path to save failure case images
        n: number of failure cases to save
        conf_threshold: minimum confidence threshold
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(model_path)
    ann_files = list(Path(ann_dir).glob("*.txt"))

    failures = []

    for ann_path in ann_files:
        img_path = Path(img_dir) / ann_path.with_suffix(".jpg").name

        if not img_path.exists():
            continue

        # check if ground truth has puck
        truth_boxes = parse_yolo_annotation(str(ann_path))
        has_puck_truth = any(b['class_id'] == 5 for b in truth_boxes)

        if not has_puck_truth:
            continue

        # run inference
        results = model.predict(str(img_path), conf = conf_threshold, verbose = False)
        pred_boxes = predictions_to_boxes(results)
        has_puck_pred = any(b['class_id'] == 5 for b in pred_boxes)

        # missed puck = ground truth has puck but model didn't find it
        if not has_puck_pred:
            failures.append((img_path, ann_path))
        
    print(f'Found {len(failures)} images where puck was missed')

    for i, (img_path, ann_path) in enumerate(failures[:n]):
        out_path = str(output_dir / f"failure_{i:03d}_{img_path.name}")
        visualize_single(
            img_path = str(img_path),
            ann_path=str(ann_path),
            model_path = model_path,
            output_path = out_path,
            conf_threshold=conf_threshold
        )
    print(f"Saved {min(n, len(failures))} failure cases to: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize detections and failures")
    subparsers = parser.add_subparsers(dest='command')

    # single image viz
    single = subparsers.add_parser("single", help = "Visualize one image")
    single.add_argument("--img", required=True, help="Path to image")
    single.add_argument("--ann", required=True, help="Path to annotation")
    single.add_argument("--model", required=True, help="Path to model weights")
    single.add_argument("--out", default=None, help="Output path (optional)")

    # Failure case visualization
    failures = subparsers.add_parser("failures", help="Find and visualize failure cases")
    failures.add_argument("--ann-dir", required=True, help="Annotations folder")
    failures.add_argument("--img-dir", required=True, help="Images folder")
    failures.add_argument("--model", required=True, help="Path to model weights")
    failures.add_argument("--out-dir",required=True, help="Output folder")
    failures.add_argument("--n", type=int, default=20)

    args = parser.parse_args()

    if args.command == 'single':
        visualize_single(args.img, args.ann, args.model, args.out)
    elif args.command == 'failures':
        visualize_failures(args.ann_dir,args.img_dir, args.model, args.out_dir, args.n)