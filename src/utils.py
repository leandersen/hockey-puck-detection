import cv2
import numpy as np
from pathlib import Path

CLASS_NAMES = {
    0: "Center Ice",
    1: "Faceoff Dots",
    2: "Goal Frame",
    3: "Goaltender",
    4: "Players",
    5: "Puck",
    6: "Referee",
}

### BGR
CLASS_COLORS = {
    0:  (255, 255, 0),
    1:  (0, 255, 255),
    2:  (255, 128, 0),
    3:  (0, 0, 255),
    4:  (0, 255, 0),
    5:  (255, 0, 255),
    6:  (128, 0, 255),
}

def parse_yolo_annotation(ann_path:str) -> list[dict]:
    """
    Parse a YOLO .txt file into a list of boxes
    Each box is a dict with keys class_id, class_namem, cx, cy, w, h
    where cx, cy, w, h, are normalized
    """
    boxes = []
    with open(ann_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            cls = int(parts[0])
            cx, cy, w, h = map(float, parts[1:5])
            boxes.append({
                "class_id":cls,
                'class_name': CLASS_NAMES.get(cls, "Unknown"),
                'cx':cx,
                'cy': cy,
                'w': w,
                'h': h,
            })
    return boxes

def yolo_to_pixel(box: dict, img_w:int, img_h:int) -> tuple:
    """
    Convert normalized YOLO box coordinates to pixel coordinates
    Args:
        box: output of parse_yolo_annotation()
        img_w: img width in pixels
        img_h: img height in pixels

    Returns: pixel coordinates of the box corners
    """
    cx = box['cx'] * img_w
    cy = box['cy'] * img_h
    w = box['w'] * img_w
    h = box['h'] * img_h

    x1 = int(cx - w/2)
    y1 = int(cy - h/2)
    x2 = int(cx + w/2)
    y2 = int(cy + h/2)

    return x1, y1, x2, y2

def draw_boxes(img:np.ndarray, boxes: list[dict]) -> np.ndarray:
    """
    Draw the boxes
    Args: Image as nparray (H,W,C) and boxes from parse_yolo_annotation()
    return: Image with boxes drawn on it 
    """
    img_h, img_w = img.shape[:2]
    img_out = img.copy()

    for box in boxes:
        x1, y1, x2, y2 = yolo_to_pixel(box, img_w, img_h)
        color = CLASS_COLORS.get(box['class_id'], (255,255,255))
        label = box['class_name']

        # Draw rectangle
        cv2.rectangle(img_out, (x1, y1), (x2, y2), color, thickness = 2)

        # Draw label background
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img_out, (x1, y1 - text_h - 4), (x1+text_w, y1), color, -1)

        # Draw label text
        cv2.putText(img_out, label, (x1, y1-2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), thickness = 1)
    return img_out

def load_image(img_path:str) -> np.ndarray:
    """
    Load an image from disk in BGR format (OpenCV default).

    Args: path to image file

    returns: image as numpy array
    """
    img=cv2.imread(str(img_path))
    if img is None:
        raise FileNotFoundError(f"Could not load image: {img_path}")
    return img