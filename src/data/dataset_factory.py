"""
dataset_factory.py

Utilities to load processed datasets saved by scripts/preprocess_dataset.py
and to build PyTorch DataLoaders for:
  - global training pool
  - per-client local datasets (based on meta.json client_assignment)
  - holdout validation set

Key functions:
- load_processed_dataset(processed_root, dataset_name)
- build_client_dataloader(client_id, config)
- build_global_dataloader(config)
- build_val_dataloader(config)
- CustomTensorDataset: a Dataset wrapper that applies transforms on-the-fly
"""

import json
from pathlib import Path
from typing import Dict, Tuple, Optional, List, Callable, Iterable

import torch
from torch.utils.data import Dataset, DataLoader, Subset

from .transforms import get_train_transforms, get_eval_transforms


class CustomTensorDataset(Dataset):
    """
    Dataset wrapping tensors with optional transform applied per-sample.

    Expects:
      - images: torch.Tensor [N, C, H, W] (float tensor)
      - labels: torch.Tensor [N]
    """

    def __init__(self, images: torch.Tensor, labels: torch.Tensor, transform: Optional[Callable] = None):
        assert images.shape[0] == labels.shape[0], "images/labels length mismatch"
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        img = self.images[idx]  # tensor
        lbl = int(self.labels[idx].item())
        if self.transform:
            # transform expects PIL or numpy or tensor; ensure it works:
            try:
                # torchvision transforms accept PIL.Image or Tensor
                out = self.transform(img)
            except Exception:
                # fallback: convert to PIL
                from PIL import Image
                import numpy as np
                arr = img.cpu().numpy()
                if arr.ndim == 3:
                    arr = (arr * 255).astype('uint8')
                    arr = np.transpose(arr, (1, 2, 0))
                else:
                    arr = (arr * 255).astype('uint8')
                pil = Image.fromarray(arr)
                out = self.transform(pil)
            return out, lbl
        else:
            return img, lbl


def load_processed_dataset(processed_root: str, dataset_name: str) -> Dict:
    """
    Load train.pt, val.pt and meta.json from processed root.

    Returns dict:
      {
        "train_images": Tensor [N_train, C, H, W],
        "train_labels": Tensor [N_train],
        "val_images": Tensor [N_val, C, H, W],
        "val_labels": Tensor [N_val],
        "train_global_indices": List[int],  # global indices corresponding to train.pt ordering
        "meta": dict
      }
    """
    root = Path(processed_root) / dataset_name
    if not root.exists():
        raise FileNotFoundError(f"Processed dataset folder not found: {root}")

    trainp = root / "train.pt"
    valp = root / "val.pt"
    metap = root / "meta.json"

    if not trainp.exists() or not valp.exists() or not metap.exists():
        raise FileNotFoundError(f"Missing one of train.pt / val.pt / meta.json in {root}")

    train_data = torch.load(str(trainp))
    val_data = torch.load(str(valp))
    with open(metap, "r", encoding="utf8") as f:
        meta = json.load(f)

    # meta may contain 'train_indices' or we can deduce:
    # preprocess_dataset.py saved 'holdout_indices' and client_assignment (global indices).
    # We can compute train_global_indices = sorted(set(range(N_total)) - set(holdout_indices))
    total_n = meta.get("num_samples", None)
    if total_n is None:
        # fallback to compute using labels length
        total_n = int(train_data["labels"].shape[0] + val_data["labels"].shape[0])

    holdout = set(meta.get("holdout_indices", []))
    # build train_global_indices as all indices excluding holdout, in ascending order
    train_global_indices = [i for i in range(total_n) if i not in holdout]

    return {
        "train_images": train_data["images"],
        "train_labels": train_data["labels"],
        "val_images": val_data["images"],
        "val_labels": val_data["labels"],
        "train_global_indices": train_global_indices,
        "meta": meta,
    }


def _build_global_index_map(train_global_indices: List[int]) -> Dict[int, int]:
    """
    Build mapping: global_index -> local_index_in_train_pt
    """
    return {gidx: i for i, gidx in enumerate(train_global_indices)}


def _filter_indices_by_classes(
    labels: torch.Tensor,
    indices: Iterable[int],
    exclude_classes: Optional[List[int]] = None,
) -> List[int]:
    """Filter a collection of indices by excluding specified class ids."""
    if not exclude_classes:
        return list(indices)

    exclude_set = {int(c) for c in exclude_classes}
    filtered: List[int] = []
    for idx in indices:
        cls = int(labels[idx].item())
        if cls not in exclude_set:
            filtered.append(idx)
    return filtered


def build_client_dataloader(
    processed_root: str,
    dataset_name: str,
    client_id: int,
    batch_size: int = 64,
    shuffle: bool = True,
    num_workers: int = 4,
    transform: Optional[Callable] = None,
    exclude_classes: Optional[List[int]] = None,
) -> DataLoader:
    """
    Build a DataLoader for a single client using client_assignment in meta.json.

    The function maps client global indices -> train.pt local indices.

    Args:
      processed_root: path to data/processed
      dataset_name: e.g., 'CIFAR10'
      client_id: integer id of client (as in meta.json)
      batch_size, shuffle, num_workers: forwarded to DataLoader
      transform: optional transform callable to apply per sample

    Returns: torch.utils.data.DataLoader
    """
    data = load_processed_dataset(processed_root, dataset_name)
    meta = data["meta"]
    train_global_indices = data["train_global_indices"]
    global_to_local = _build_global_index_map(train_global_indices)

    client_map = meta.get("client_assignment", {})
    # client_assignment keys may be strings
    key = str(client_id)
    if key not in client_map:
        raise KeyError(f"Client {client_id} not found in meta.client_assignment")

    client_global = [int(x) for x in client_map[key]]
    # remove any potential holdout indices
    holdout = set(meta.get("holdout_indices", []))
    client_global = [g for g in client_global if g not in holdout]

    # map to local indices in train.pt
    local_indices = [global_to_local[g] for g in client_global if g in global_to_local]

    dataset = CustomTensorDataset(data["train_images"], data["train_labels"], transform=transform)
    local_indices = _filter_indices_by_classes(dataset.labels, local_indices, exclude_classes)
    subset = Subset(dataset, local_indices)
    loader = DataLoader(subset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)
    return loader


def build_global_dataloader(
    processed_root: str,
    dataset_name: str,
    batch_size: int = 128,
    shuffle: bool = True,
    num_workers: int = 4,
    transform: Optional[Callable] = None,
    exclude_classes: Optional[List[int]] = None,
) -> DataLoader:
    """
    Build a DataLoader for the full training pool (all train.pt samples).
    """
    data = load_processed_dataset(processed_root, dataset_name)
    dataset = CustomTensorDataset(data["train_images"], data["train_labels"], transform=transform)
    indices = list(range(len(dataset)))
    indices = _filter_indices_by_classes(dataset.labels, indices, exclude_classes)
    dataset = Subset(dataset, indices) if len(indices) != len(dataset) else dataset
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)
    return loader


def build_val_dataloader(
    processed_root: str,
    dataset_name: str,
    batch_size: int = 256,
    shuffle: bool = False,
    num_workers: int = 4,
    transform: Optional[Callable] = None,
) -> DataLoader:
    """
    Build a DataLoader for the holdout validation set (val.pt)
    """
    data = load_processed_dataset(processed_root, dataset_name)
    dataset = CustomTensorDataset(data["val_images"], data["val_labels"], transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)
    return loader


# Convenience function to create transforms from configs
def make_transforms_from_config(dataset_name: str, cfg: Dict, train: bool = True):
    """
    cfg expects keys: dataset.normalize, dataset.image_size, optionally use_albumentations
    Example:
      cfg = {
        "dataset": {"normalize": {"mean":[...],"std":[...]}, "image_size":32}
      }
    """
    ds = dataset_name
    norm = None
    image_size = 32
    use_alb = False
    if isinstance(cfg, dict):
        ds_cfg = cfg.get("dataset", {})
        norm = ds_cfg.get("normalize", None)
        image_size = ds_cfg.get("image_size", image_size)
        use_alb = ds_cfg.get("use_albumentations", False)
    if train:
        return get_train_transforms(ds, image_size=image_size, normalize=norm, use_albumentations=use_alb)
    else:
        return get_eval_transforms(ds, image_size=image_size, normalize=norm)
