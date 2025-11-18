"""Utilities for loading torchvision datasets and partitioning them for federated learning."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import os

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets
from torchvision.datasets import ImageFolder

from .dataset_factory import CustomTensorDataset
from .transforms import get_eval_transforms, get_train_transforms
from src.models.model_utils import infer_num_classes


_DATASET_MAP = {
    "cifar10": (datasets.CIFAR10, {"image_size": 32, "normalize": {"mean": [0.4914, 0.4822, 0.4465], "std": [0.247, 0.243, 0.261]}}),
    "cifar100": (datasets.CIFAR100, {"image_size": 32, "normalize": {"mean": [0.5071, 0.4867, 0.4408], "std": [0.2675, 0.2565, 0.2761]}}),
    "mnist": (datasets.MNIST, {"image_size": 28, "normalize": {"mean": [0.1307], "std": [0.3081]}}),
    # FEMNIST approximated by EMNIST (ByClass split)
    "femnist": (datasets.EMNIST, {"split": "byclass", "image_size": 28, "normalize": {"mean": [0.1307], "std": [0.3081]}}),
    "svhn": (datasets.SVHN, {"image_size": 32, "normalize": {"mean": [0.4377, 0.4438, 0.4728], "std": [0.1980, 0.2010, 0.1970]}, "split": "train"}),
    # FLAIR is handled through ImageFolder; expected structure: <root>/FLAIR/train, <root>/FLAIR/test
    "flair": (ImageFolder, {"image_size": 224, "normalize": {"mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]}, "relative_root": "FLAIR"}),
}


@dataclass
class PartitionResult:
    """Structure describing how data is partitioned across clients and server."""

    client_train_indices: Dict[int, List[int]]
    client_test_indices: Dict[int, List[int]]
    server_test_indices: List[int]


def _dataset_to_tensors(dataset: Dataset) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert a torchvision dataset into tensors (N, C, H, W) and labels (N)."""

    if hasattr(dataset, "data"):
        images = dataset.data
    elif hasattr(dataset, "images"):
        images = dataset.images
    else:
        imgs = []
        labels = []
        for x, y in dataset:
            imgs.append(torch.tensor(x))
            labels.append(int(y))
        images = torch.stack(imgs, dim=0)
        return images.float(), torch.tensor(labels, dtype=torch.long)

    if isinstance(images, torch.Tensor):
        tensor = images.clone()
    else:
        tensor = torch.tensor(images)

    if tensor.ndim == 4 and tensor.shape[-1] in (1, 3):
        tensor = tensor.permute(0, 3, 1, 2)
    elif tensor.ndim == 3:
        tensor = tensor.unsqueeze(1)

    if tensor.dtype == torch.uint8:
        tensor = tensor.float() / 255.0
    else:
        tensor = tensor.float()

    if hasattr(dataset, "targets"):
        labels = dataset.targets
    elif hasattr(dataset, "labels"):
        labels = dataset.labels
    else:
        labels = [dataset[i][1] for i in range(len(dataset))]

    if isinstance(labels, torch.Tensor):
        label_tensor = labels.clone().long()
    else:
        label_tensor = torch.tensor(labels, dtype=torch.long)

    return tensor, label_tensor


def _load_flair(root: str, relative_root: str) -> Tuple[Dataset, Dataset]:
    train_path = os.path.join(root, relative_root, "train")
    test_path = os.path.join(root, relative_root, "test")
    if not (os.path.isdir(train_path) and os.path.isdir(test_path)):
        raise FileNotFoundError(
            f"FLAIR dataset not found. Expected train/test folders under {os.path.join(root, relative_root)}."
        )
    train_ds = ImageFolder(train_path)
    test_ds = ImageFolder(test_path)
    return train_ds, test_ds


def load_dataset_tensors(
    dataset_name: str, root: str = "./data/raw"
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, Dict]:
    """Load a torchvision dataset and return tensors for train/test splits."""

    name = dataset_name.lower()
    if name not in _DATASET_MAP and name != "coco":
        raise ValueError(f"Unsupported dataset '{dataset_name}'. Supported: {list(_DATASET_MAP.keys()) + ['COCO']}")

    if name == "coco":
        raise NotImplementedError("COCO classification is not implemented in this simulator.")

    ds_cls, extra_cfg = _DATASET_MAP[name]
    kwargs = dict(root=root, download=True)

    if name == "femnist":
        split = extra_cfg.get("split", "byclass")
        train_ds = ds_cls(split=split, train=True, **kwargs)
        test_ds = ds_cls(split=split, train=False, **kwargs)
    elif name == "svhn":
        split = extra_cfg.get("split", "train")
        train_ds = ds_cls(split=split, **kwargs)
        test_ds = ds_cls(split="test", **kwargs)
    elif name == "flair":
        rel_root = extra_cfg.get("relative_root", "FLAIR")
        train_ds, test_ds = _load_flair(root, rel_root)
    else:
        train_ds = ds_cls(train=True, **kwargs)
        test_ds = ds_cls(train=False, **kwargs)

    train_images, train_labels = _dataset_to_tensors(train_ds)
    test_images, test_labels = _dataset_to_tensors(test_ds)

    num_classes = infer_num_classes(dataset_name)
    if hasattr(train_ds, "classes"):
        num_classes = max(num_classes, len(getattr(train_ds, "classes")))
    meta = {
        "image_size": extra_cfg.get("image_size", train_images.shape[-1]),
        "normalize": extra_cfg.get("normalize"),
    }
    return train_images, train_labels, test_images, test_labels, num_classes, meta


def _round_robin_assign(indices: Sequence[int], num_clients: int) -> Dict[int, List[int]]:
    assignment = {cid: [] for cid in range(num_clients)}
    for pos, idx in enumerate(indices):
        cid = pos % num_clients
        assignment[cid].append(int(idx))
    return assignment


def partition_indices_by_class(
    labels: torch.Tensor,
    num_clients: int,
    seed: int = 0,
) -> Dict[int, List[int]]:
    """Split indices evenly across clients for each class."""

    rng = random.Random(seed)
    class_indices: Dict[int, List[int]] = {}
    for idx, label in enumerate(labels.tolist()):
        class_indices.setdefault(label, []).append(idx)

    assignments = {cid: [] for cid in range(num_clients)}
    for cls, idxs in class_indices.items():
        rng.shuffle(idxs)
        rr = _round_robin_assign(idxs, num_clients)
        for cid in range(num_clients):
            assignments[cid].extend(rr[cid])
    return assignments


def split_test_indices(
    labels: torch.Tensor,
    num_clients: int,
    seed: int = 0,
) -> Tuple[List[int], Dict[int, List[int]]]:
    """Split test indices with 50% assigned to server and remainder evenly across clients."""

    rng = random.Random(seed)
    class_indices: Dict[int, List[int]] = {}
    for idx, label in enumerate(labels.tolist()):
        class_indices.setdefault(label, []).append(idx)

    server_indices: List[int] = []
    client_assignments = {cid: [] for cid in range(num_clients)}

    for _, idxs in class_indices.items():
        rng.shuffle(idxs)
        half = math.floor(len(idxs) / 2)
        server_indices.extend(idxs[:half])
        remaining = idxs[half:]
        rr = _round_robin_assign(remaining, num_clients)
        for cid in range(num_clients):
            client_assignments[cid].extend(rr[cid])

    return server_indices, client_assignments


def split_holdout_by_class(labels: torch.Tensor, fraction: float = 0.1, seed: int = 0) -> Tuple[List[int], List[int]]:
    """Reserve a per-class holdout split for downstream tasks (e.g., diffusion pre-training).

    Returns:
        holdout_indices: indices kept for holdout
        remaining_indices: indices used for federated training
    """

    if not 0 < fraction < 1:
        raise ValueError("fraction must be in (0, 1)")

    rng = random.Random(seed)
    class_indices: Dict[int, List[int]] = {}
    for idx, label in enumerate(labels.tolist()):
        class_indices.setdefault(label, []).append(idx)

    holdout: List[int] = []
    remaining: List[int] = []
    for _, idxs in class_indices.items():
        rng.shuffle(idxs)
        cutoff = max(1, int(len(idxs) * fraction))
        holdout.extend(idxs[:cutoff])
        remaining.extend(idxs[cutoff:])

    return holdout, remaining


def create_dataloaders(
    train_images: torch.Tensor,
    train_labels: torch.Tensor,
    test_images: torch.Tensor,
    test_labels: torch.Tensor,
    partition: PartitionResult,
    dataset_name: str,
    batch_size: int,
    num_workers: int = 0,
    *,
    image_size: int,
    normalize: Optional[Dict],
) -> Tuple[Dict[int, DataLoader], Dict[int, DataLoader], DataLoader]:
    """Create train/test DataLoaders for clients and server."""

    train_tf = get_train_transforms(dataset_name, image_size=image_size, normalize=normalize)
    eval_tf = get_eval_transforms(dataset_name, image_size=image_size, normalize=normalize)

    train_dataset = CustomTensorDataset(train_images, train_labels, transform=train_tf)
    test_dataset = CustomTensorDataset(test_images, test_labels, transform=eval_tf)

    client_train_loaders: Dict[int, DataLoader] = {}
    client_test_loaders: Dict[int, DataLoader] = {}

    for cid, indices in partition.client_train_indices.items():
        subset = Subset(train_dataset, indices)
        loader = DataLoader(subset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        client_train_loaders[cid] = loader

    for cid, indices in partition.client_test_indices.items():
        subset = Subset(test_dataset, indices)
        loader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        client_test_loaders[cid] = loader

    server_subset = Subset(test_dataset, partition.server_test_indices)
    server_loader = DataLoader(server_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return client_train_loaders, client_test_loaders, server_loader


def summarize_distribution(labels: torch.Tensor, assignments: Dict[int, List[int]]) -> Dict[int, Dict[int, int]]:
    """Return per-client per-class counts for a set of indices."""

    summary: Dict[int, Dict[int, int]] = {}
    for cid, idxs in assignments.items():
        counts: Dict[int, int] = {}
        for idx in idxs:
            cls = int(labels[idx].item())
            counts[cls] = counts.get(cls, 0) + 1
        summary[cid] = counts
    return summary
