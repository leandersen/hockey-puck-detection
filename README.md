# Hockey Puck Detection

Object detection project focused on puck detection in Swedish Hockey League (SHL) broadcast footage. Built on the [HockeyAI dataset](https://huggingface.co/SimulaMet-HOST/HockeyAI) using YOLOv8.

My goal is to train a detector, then understand where and why* it fails across different game situations, and whether a model trained on SHL footage generalizes to other leagues.

---

## Project Structure

```
hockey-puck-detection/
├── data/
│   └── hockeyai/ 
├── src/
│   ├── prepare_data.py
│   ├── train.py 
│   ├── evaluate.py 
│   ├── visualize.py
│   └── utils.py
├── notebooks/
│   └── DataExploration.ipynb
├── results/
│   ├── figures/
│   ├── metrics/
│   └── logs/
├── configs/
│   └── train_config.yaml
├── train_hpc.slurm
└── requirements.txt
```

---

## Setup

```bash
git clone https://github.com/leandersen/hockey-puck-detection
cd hockey-puck-detection
pip install -r requirements.txt
```

---

## Data

The dataset is not included in this repo. To download and prepare it:

```bash
python src/prepare_data.py
```

This will create `data/hockeyai/yolo/` with the correct train/val split and a `data.yaml` config file.

**Dataset stats:**
- 2,101 frames from SHL broadcasts
- 7 classes: Center Ice, Faceoff Dots, Goal Frame, Goaltender, Players, Puck, Referee
- Players dominates the dataset, Puck appears in every image but is small (0.03% Avg. Box Area)

---

## Training

All settings are controlled through `configs/train_config.yaml`. To train locally:

```bash
python src/train.py --config configs/train_config.yaml
```

The script auto-detects your device - no changes needed.

Results are saved to `results/hockeyai_baseline/`.

---

## Model

This project fine-tunes [YOLOv8 medium](https://github.com/ultralytics/ultralytics) (pretrained on COCO) on the HockeyAI dataset. The detector is not built from scratch. My contribution here is the training setup, the per-situation evaluation methodology, and the analysis of where and why the model fails.

YOLOv8 is a single-stage object detector developed by Ultralytics. It processes the full image in one pass and predicts bounding boxes and class labels directly, making it fast enough for real-time video applications. The medium variant (`yolov8m`) balances accuracy and speed reasonably well for this use case.

---

## Visualization

After training, visualize predictions vs. truth on a single image:

```bash
python src/visualize.py single \
  --img data/hockeyai/yolo/images/val/IMAGE.jpg \
  --ann data/hockeyai/yolo/labels/val/IMAGE.txt \
  --model results/hockeyai_baseline/weights/best.pt
```

Find all frames where the model missed the puck:

```bash
python src/visualize.py failures \
  --ann-dir data/hockeyai/yolo/labels/val \
  --img-dir data/hockeyai/yolo/images/val \
  --model results/hockeyai_baseline/weights/best.pt \
  --out-dir results/figures/failure_cases
```

---

## Next Steps

- Per-situation evaluation (open ice, board play, crease, face-off, scramble)
- Cross-domain evaluation (SHL → NHL footage)
- Architecture comparison