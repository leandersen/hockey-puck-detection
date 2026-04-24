import os
import shutil
import random
from pathlib import Path
import yaml

def prepare_hockeyai(
        src_frames_dir:str,
        src_annots_dir:str,
        dst_dir:str,
        val_split:float=0.2,
        seed:int=42
):
    """
    Reorganizes the HockeyAI dataset into YOLO-compatible folder structure and creates a data.yaml config file.

    Args:
        src_frames_dir: path to SHL/frames/
        src_annots_dir: path to SHL/annotations/
        dst_dir: path to data/hockeyai/yolo/
        val_split: fraction of data to use for validation
        seed: random seed for reproducibility
    """
    random.seed(seed)

    src_frames = Path(src_frames_dir)
    src_annots = Path(src_annots_dir)
    dst = Path(dst_dir)

    ## Create destination folders
    for split in ['train','val']:
        (dst / 'images' / split).mkdir(parents = True, exist_ok = True)
        (dst / 'labels' / split).mkdir(parents = True, exist_ok = True)
    
    ## Find all images that have a matching annotation file
    all_images = list(src_frames.glob("*.jpg"))
    paired = [img for img in all_images
              if (src_annots / img.stem).with_suffix(".txt").exists()]
    
    print(f'Total Images: {len(all_images)}')
    print(f'Images with labels: {len(paired)}')
    print(f'Images without labels: {len(all_images) - len(paired)}')

    ## Shuffle and split
    random.shuffle(paired)
    n_val = int(len(paired) * val_split)
    val_imgs = paired[:n_val]
    train_imgs = paired[n_val:]

    print(f"\nTrain: {len(train_imgs)} | Val: {len(val_imgs)}")

    ## Copy files into the YOLO folder structure
    for split, imgs in [('train', train_imgs), ('val',val_imgs)]:
        for img_path in imgs:
            ann_path = (src_annots / img_path.stem).with_suffix('.txt')
            shutil.copy(img_path, dst / 'images' / split / img_path.name)
            shutil.copy(ann_path, dst / 'labels' / split / ann_path.name)

    print("Files copied successfully.")

    # write data.yaml for YOLO
    data_yaml = {
        "path": str(dst.resolve()),
        "train": "images/train",
        "val": "images/val",
        'nc': 7,
        'names': [
            "Center Ice",
            "Faceoff Dots",
            "Goal Frame",
            "Goaltender",
            "Players",
            "Puck",
            "Referee"
        ]
    }

    yaml_path = dst / 'data.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(data_yaml, f, default_flow_style = False, sort_keys = False)
    
    print(f"\ndata.yaml written to: {yaml_path}")
    return str(yaml_path)

if __name__ == "__main__":
    prepare_hockeyai(
        src_frames_dir = "data/hockeyai/SHL/frames",
        src_annots_dir= "data/hockeyai/SHL/annotations",
        dst_dir = "data/hockeyai/yolo"
    )