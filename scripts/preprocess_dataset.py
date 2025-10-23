#!/usr/bin/env python3
"""
preprocess_dataset.py

- Loads raw datasets from torchvision (or local raw path)
- Performs deterministic transforms (to Tensor + normalization)
- Creates:
    data/processed/<DATASET>/train.pt   (tensor dataset for clients)
    data/processed/<DATASET>/val.pt     (holdout test set - 10% per class, stratified)
    data/processed/<DATASET>/meta.json  (mapping: holdout indices, client assignments)
- Supports IID or Dirichlet non-IID partitioning among clients.

Usage examples:
  # IID partition among 20 clients
  python scripts/preprocess_dataset.py --dataset CIFAR10 --out data/processed --num-clients 20

  # Non-IID (Dirichlet alpha=0.1)
  python scripts/preprocess_dataset.py --dataset CIFAR10 --out data/processed --num-clients 20 --non-iid --alpha 0.1
"""

import argparse
from pathlib import Path
import json
import math
import random
from collections import defaultdict
from typing import Dict, List

import numpy as np
import torch
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100, MNIST, FashionMNIST
from sklearn.model_selection import train_test_split


DATASET_CLS = {
    "CIFAR10": CIFAR10,
    "CIFAR100": CIFAR100,
    "MNIST": MNIST,
    "FashionMNIST": FashionMNIST,
}


DEFAULT_NORMALIZE = {
    "CIFAR10": {"mean": [0.4914, 0.4822, 0.4465], "std": [0.2470, 0.2435, 0.2616]},
    "CIFAR100": {"mean": [0.5071, 0.4865, 0.4409], "std": [0.2673, 0.2564, 0.2762]},
    "MNIST": {"mean": [0.1307], "std": [0.3081]},
    "FashionMNIST": {"mean": [0.2860], "std": [0.3530]},
}


def load_full_dataset(name: str, raw_root: Path):
    """Load train split (we will combine train+maybe use its labels)."""
    if name not in DATASET_CLS:
        raise ValueError(f"Unsupported dataset: {name}")
    cls = DATASET_CLS[name]
    # use torchvision default train split (train=True)
    dataset = cls(root=str(raw_root / name), train=True, download=True)
    # dataset.data (numpy) and dataset.targets/list
    if hasattr(dataset, "data"):
        data = np.array(dataset.data)
    else:
        # fallback: convert images via PIL to numpy
        data = np.stack([np.array(img) for img, _ in dataset])
    targets = np.array(dataset.targets)
    return data, targets


def to_tensor_and_normalize(data: np.ndarray, dataset_name: str):
    """Convert numpy image array to torch tensor with normalization [C,H,W]."""
    norm = DEFAULT_NORMALIZE[dataset_name]
    mean = np.array(norm["mean"])
    std = np.array(norm["std"])

    # Determine shape: for MNIST/FashionMNIST data shape is (N, H, W)
    if data.ndim == 3:
        # add channel axis
        data = data[:, None, ...]  # (N, 1, H, W)
    elif data.ndim == 4:
        # (N, H, W, C) -> (N, C, H, W)
        data = data.transpose(0, 3, 1, 2)
    else:
        raise ValueError("Unexpected data shape")

    data = data.astype(np.float32) / 255.0
    # broadcast mean/std to shape (C,1,1)
    mean = mean.reshape(-1, 1, 1)
    std = std.reshape(-1, 1, 1)
    data = (data - mean) / std
    return torch.from_numpy(data)  # float32 tensor


def per_class_holdout(targets: np.ndarray, test_ratio: float, seed: int = 42) -> List[int]:
    """Return an array of holdout indices that contains ~test_ratio fraction per class
    and ensures each class has at least 1 sample in holdout (if possible)."""
    rng = np.random.RandomState(seed)
    n = len(targets)
    holdout_idx = []
    labels = np.unique(targets)
    for lab in labels:
        idxs = np.where(targets == lab)[0]
        k = max(1, math.ceil(len(idxs) * test_ratio))
        chosen = rng.choice(idxs, size=k, replace=False).tolist()
        holdout_idx.extend(chosen)
    holdout_idx = sorted(list(set(holdout_idx)))
    return holdout_idx


def dirichlet_partition(targets: np.ndarray, num_clients: int, alpha: float, min_size=1, seed=42):
    """Partition data indices into num_clients using Dirichlet distribution over classes.
    Returns a dict client_id -> list(indices)
    """
    np.random.seed(seed)
    num_classes = int(np.max(targets)) + 1
    class_idx = [np.where(targets == i)[0].tolist() for i in range(num_classes)]

    client_indices = [[] for _ in range(num_clients)]
    for c, idxs in enumerate(class_idx):
        if len(idxs) == 0:
            continue
        # draw proportions for this class across clients
        proportions = np.random.dirichlet(alpha=[alpha] * num_clients)
        # scale proportions to allocate counts
        counts = (proportions * len(idxs)).astype(int)
        # adjust rounding error: ensure sum equals len(idxs)
        while counts.sum() < len(idxs):
            counts[np.random.randint(0, num_clients)] += 1
        # shuffle idxs
        np.random.shuffle(idxs)
        ptr = 0
        for j in range(num_clients):
            cnt = counts[j]
            if cnt > 0:
                client_indices[j].extend(idxs[ptr : ptr + cnt])
                ptr += cnt
    # ensure min_size
    for i in range(num_clients):
        if len(client_indices[i]) < min_size:
            # steal from the largest
            largest = max(range(num_clients), key=lambda x: len(client_indices[x]))
            if len(client_indices[largest]) > 1:
                client_indices[i].append(client_indices[largest].pop())
    return {i: sorted(list(set(client_indices[i]))) for i in range(num_clients)}


def iid_partition(indices: List[int], num_clients: int, seed=42):
    rng = np.random.RandomState(seed)
    indices = list(indices)
    rng.shuffle(indices)
    res = {i: [] for i in range(num_clients)}
    for i, idx in enumerate(indices):
        res[i % num_clients].append(idx)
    return res


def save_processed(out_dir: Path, dataset_name: str, images_tensor: torch.Tensor, labels_tensor: torch.Tensor,
                   train_idx: List[int], val_idx: List[int], client_assignment: Dict[int, List[int]]):
    out_dir = out_dir / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)
    # Create train and val tensors (train contains indices assigned to clients)
    train_images = images_tensor[train_idx]
    train_labels = labels_tensor[train_idx]
    val_images = images_tensor[val_idx]
    val_labels = labels_tensor[val_idx]

    torch.save({"images": train_images, "labels": train_labels}, out_dir / "train.pt")
    torch.save({"images": val_images, "labels": val_labels}, out_dir / "val.pt")

    meta = {
        "dataset": dataset_name,
        "num_samples": images_tensor.shape[0],
        "train_size": int(train_images.shape[0]),
        "val_size": int(val_images.shape[0]),
        "holdout_indices": val_idx,
        "client_assignment": {str(k): v for k, v in client_assignment.items()},
    }
    with open(out_dir / "meta.json", "w", encoding="utf8") as f:
        json.dump(meta, f, indent=2)
    print(f"[+] Saved processed dataset to {out_dir} (train {train_images.shape[0]}, val {val_images.shape[0]})")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=list(DATASET_CLS.keys()), help="Dataset name")
    p.add_argument("--raw-root", type=str, default="data/raw", help="Raw datasets root (downloaded)")
    p.add_argument("--out", type=str, default="data/processed", help="Processed data root")
    p.add_argument("--test-pct", type=float, default=0.1, help="Holdout pct per class (default 0.1)")
    p.add_argument("--num-clients", type=int, default=20, help="Number of federated clients to partition training data")
    p.add_argument("--non-iid", action="store_true", help="Use Dirichlet non-IID partition")
    p.add_argument("--alpha", type=float, default=0.5, help="Dirichlet alpha (if non-iid)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    raw_root = Path(args.raw_root)
    out_root = Path(args.out)
    dataset = args.dataset
    print(f"[+] Preprocessing {dataset}")

    data, targets = load_full_dataset(dataset, raw_root)
    n = len(targets)
    print(f"[+] Loaded {n} samples")

    # compute holdout per class
    holdout_idx = per_class_holdout(targets, test_ratio=args.test_pct, seed=args.seed)
    holdout_set = set(holdout_idx)
    train_indices = [i for i in range(n) if i not in holdout_set]
    print(f"[+] Holdout size: {len(holdout_idx)}  Train size: {len(train_indices)}")

    # partition train_indices to clients
    if args.non_iid:
        print(f"[+] Partitioning train data into {args.num_clients} non-IID clients (Dirichlet alpha={args.alpha})")
        client_assignment = dirichlet_partition(targets[train_indices], num_clients=args.num_clients,
                                                alpha=args.alpha, seed=args.seed)
        # dirichlet_partition returned indices local to train_indices; convert back to global indices
        # but our implementation used class-wise indices from full targets; for simplicity use alternative:
        # Build mapping by labels: for each client, replace local index with actual global index chosen from train_indices
        # To avoid complexity, use a simpler approach: collect indices per class from train_indices, then allocate.
        # We'll re-run a dirichlet on global train set (safer)
        client_assignment = dirichlet_partition(targets, num_clients=args.num_clients, alpha=args.alpha, seed=args.seed)
        # remove holdout indices from assignments (if any)
        for k in client_assignment:
            client_assignment[k] = [int(x) for x in client_assignment[k] if int(x) not in holdout_set]
    else:
        print(f"[+] Partitioning train data into {args.num_clients} IID clients")
        client_assignment = iid_partition(train_indices, num_clients=args.num_clients, seed=args.seed)

    # Convert data -> normalized torch tensors
    images_tensor = to_tensor_and_normalize(data, dataset)
    labels_tensor = torch.from_numpy(targets).long()

    save_processed(out_root, dataset, images_tensor, labels_tensor, train_idx=train_indices, val_idx=holdout_idx,
                   client_assignment=client_assignment)
    print("[+] Preprocessing finished.")


if __name__ == "__main__":
    main()
