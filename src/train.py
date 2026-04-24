import argparse
import yaml
import torch
from pathlib import Path
from ultralytics import YOLO

def get_device():
    if torch.cuda.is_available():
        device = "cuda"
        print(f'Using GPU: {torch.cuda.get_device_name(0)}')
    elif torch.backends.mps.is_available():
        device = 'mps'
        print("Using MPS")
    else:
        device = "cpu"
        print("No GPU found. Using CPU")
    return device

def load_config(config_path:str) -> dict:
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config

def train(config_path: str):
    config = load_config(config_path)
    device = get_device()

    model = YOLO(config['model'])
    print(f"\nModel loaded: {config['model']}")
    print(f"Pretrained: {config['pretrained']}")

    print('\nStarting training...')
    results = model.train(
        data=config['data'],
        epochs=config['epochs'],
        patience=config['patience'],
        batch=config['batch'],
        imgsz=config['imgsz'],
        workers = config['workers'],
        optimizer = config['optimizer'],
        lr0 = config['lr0'],
        weight_decay=config["weight_decay"],
        project=config["project"],
        name=config["name"],
        save_period=config["save_period"],
        device=device,
        verbose=True,
    )

    print("\nTraining complete.")
    print(f'Results saved to: {config["project"]}/{config["name"]}')
    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = "Train YOLOv8 on HockeyAI Dataset")
    parser.add_argument(
        "--config",
        type = str,
        default = "configs/train_config.yaml",
        help="Path to training config file"
    )
    args = parser.parse_args()
    train(args.config)