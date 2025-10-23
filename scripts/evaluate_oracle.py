#!/usr/bin/env python3
"""
evaluate_oracle.py

Load a saved model checkpoint (PyTorch .pth) and evaluate on the processed holdout (val.pt).
Outputs overall accuracy, per-class accuracy, and confusion matrix CSV.

Usage:
  python scripts/evaluate_oracle.py --model-path outputs/.../checkpoints/model_before.pth \
    --dataset CIFAR10 --processed-root data/processed --out outputs/.../oracle_eval.csv
"""

import argparse
from pathlib import Path
import json
import csv

import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score

# Import your model builder; here we try to import from src.models.model_utils
# The project should provide model builder functions in src/models.
try:
    from src.models.model_utils import build_model_from_config
except Exception:
    # provide a simple fallback builder if project utils not available
    build_model_from_config = None


def load_processed_val(dataset_name: str, processed_root: Path):
    p = processed_root / dataset_name / "val.pt"
    if not p.exists():
        raise FileNotFoundError(f"Processed val.pt not found at {p}; run preprocess_dataset.py first.")
    data = torch.load(p)
    images = data["images"]  # tensor N,C,H,W
    labels = data["labels"]  # tensor N
    return images, labels


def simple_build_model(model_name: str, num_classes: int, input_channels: int = 3):
    # minimal architectures for sanity if project builder not available
    import torch.nn as nn
    from torchvision.models import resnet18

    if model_name.lower().startswith("resnet"):
        # use resnet18 as fallback but adjust final layer
        m = resnet18(pretrained=False)
        if input_channels != 3:
            # simple conv to expand channels -> 3 (not perfect but works for evaluation)
            first_conv = m.conv1
            m.conv1 = nn.Conv2d(input_channels, first_conv.out_channels,
                                kernel_size=first_conv.kernel_size, stride=first_conv.stride,
                                padding=first_conv.padding, bias=False)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m
    elif model_name.lower().startswith("lenet"):
        class LeNetSimple(nn.Module):
            def __init__(self, nc=1, nclass=10):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Conv2d(nc, 6, 5),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Conv2d(6, 16, 5),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Flatten(),
                    nn.Linear(16 * 4 * 4 if nc == 1 else 16 * 5 * 5, 120),
                    nn.ReLU(),
                    nn.Linear(120, 84),
                    nn.ReLU(),
                    nn.Linear(84, nclass),
                )

            def forward(self, x):
                return self.net(x)

        return LeNetSimple(nc=input_channels, nclass=num_classes)
    else:
        raise ValueError(f"No fallback builder for model {model_name}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", type=str, required=True, help="Path to model .pth")
    p.add_argument("--dataset", type=str, required=True, help="Dataset name (folder in data/processed)")
    p.add_argument("--processed-root", type=str, default="data/processed", help="Processed data root")
    p.add_argument("--model-name", type=str, default="resnet20", help="Model name for builder/fallback")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out-csv", type=str, default=None, help="Optional path to save per-class metrics / confusion matrix")
    return p.parse_args()


def evaluate(model: torch.nn.Module, images: torch.Tensor, labels: torch.Tensor, device: str, batch_size: int = 256):
    model.to(device)
    model.eval()
    n = labels.shape[0]
    preds = []
    with torch.no_grad():
        for i in range(0, n, batch_size):
            batch = images[i : i + batch_size].to(device)
            out = model(batch)
            if isinstance(out, (list, tuple)):
                out = out[0]
            probs = F.softmax(out, dim=1)
            preds.append(probs.argmax(dim=1).cpu().numpy())
    preds = np.concatenate(preds, axis=0)
    acc = accuracy_score(labels.numpy(), preds)
    cm = confusion_matrix(labels.numpy(), preds)
    per_class_acc = cm.diagonal() / np.maximum(cm.sum(axis=1), 1)
    return acc, per_class_acc, cm, preds


def main():
    args = parse_args()
    proc_root = Path(args.processed_root)
    images, labels = load_processed_val(args.dataset, proc_root)
    num_classes = int(labels.max().item()) + 1
    input_channels = int(images.shape[1])

    # build model
    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    # load checkpoint
    ckpt = torch.load(str(model_path), map_location="cpu")
    # guess model state dict key
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    elif isinstance(ckpt, dict) and "model" in ckpt:
        state_dict = ckpt["model"]
    elif isinstance(ckpt, dict) and any(k.startswith("module.") or "weight" in k for k in ckpt.keys()):
        # assume it's a state_dict
        state_dict = ckpt
    else:
        # maybe raw tensor saved
        state_dict = ckpt

    # build model: try project builder first
    model = None
    if build_model_from_config is not None:
        try:
            model = build_model_from_config({"name": args.model_name, "num_classes": num_classes, "input_channels": input_channels})
        except Exception:
            model = None

    if model is None:
        model = simple_build_model(args.model_name, num_classes=num_classes, input_channels=input_channels)

    # load weights
    try:
        model.load_state_dict(state_dict)
    except Exception:
        # try to adapt keys with 'module.' prefix
        new_state = {}
        for k, v in state_dict.items():
            nk = k.replace("module.", "") if k.startswith("module.") else k
            new_state[nk] = v
        model.load_state_dict(new_state, strict=False)

    acc, per_class_acc, cm, preds = evaluate(model, images, labels, device=args.device, batch_size=args.batch_size)

    print(f"[+] Evaluation on {args.dataset} holdout: overall acc = {acc * 100:.2f}%")
    for i, a in enumerate(per_class_acc):
        print(f"  class {i:03d}: acc = {a * 100:.2f}%")

    if args.out_csv:
        outp = Path(args.out_csv)
        outp.parent.mkdir(parents=True, exist_ok=True)
        # save per-class accuracy CSV
        with open(outp, "w", newline="", encoding="utf8") as f:
            writer = csv.writer(f)
            writer.writerow(["class_id", "per_class_acc"])
            for i, a in enumerate(per_class_acc):
                writer.writerow([i, float(a)])
        # also save confusion matrix as npy next to csv
        npy_path = outp.parent / (outp.stem + "_confusion.npy")
        np.save(str(npy_path), cm)
        print(f"[+] Saved per-class CSV -> {outp}, confusion matrix -> {npy_path}")


if __name__ == "__main__":
    main()
