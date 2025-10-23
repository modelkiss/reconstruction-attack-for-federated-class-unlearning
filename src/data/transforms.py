"""
transforms.py

Provides dataset-specific train / eval transform builders.

Supports:
- torchvision.transforms (default)
- optional albumentations (if installed) for faster augmentation

Functions:
- get_train_transforms(dataset_name, image_size, normalize, use_albumentations=False)
- get_eval_transforms(dataset_name, image_size, normalize)
"""

from typing import Optional, Tuple, Callable
import torchvision.transforms as T

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    _HAS_ALB = True
except Exception:
    _HAS_ALB = False

from PIL import Image
import numpy as np


def _to_pil_if_tensor(x):
    """Convert torch tensor or numpy to PIL.Image if needed (expects C,H,W or H,W,C)"""
    try:
        import torch
        if isinstance(x, torch.Tensor):
            arr = x.cpu().numpy()
        else:
            arr = x
    except Exception:
        arr = x
    # arr shape: (C,H,W) or (H,W,C) or (H,W)
    if arr.ndim == 3 and arr.shape[0] <= 4:
        # (C,H,W) -> (H,W,C)
        arr = np.transpose(arr, (1, 2, 0))
    if arr.dtype != np.uint8:
        # scale floats [0,1] -> [0,255]
        if arr.max() <= 1.0:
            arr = (arr * 255.0).astype(np.uint8)
        else:
            arr = arr.astype(np.uint8)
    return Image.fromarray(arr)


def get_train_transforms(
    dataset_name: str,
    image_size: int = 32,
    normalize: Optional[dict] = None,
    use_albumentations: bool = False,
) -> Callable:
    """
    Return a callable transform for training.

    Args:
        dataset_name: 'CIFAR10'|'CIFAR100'|'MNIST'|'FashionMNIST'
        image_size: target size (int)
        normalize: dict with 'mean' and 'std', e.g. {'mean': [...], 'std': [...]}
        use_albumentations: if True and albumentations installed, use it.

    Returns:
        A callable that maps PIL.Image or Tensor -> Tensor
    """
    dataset_name = dataset_name.lower()
    if use_albumentations and _HAS_ALB:
        mean = normalize["mean"] if normalize else [0.5] * (3 if "cifar" in dataset_name else 1)
        std = normalize["std"] if normalize else [0.5] * (3 if "cifar" in dataset_name else 1)
        aug = [
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5) if "mnist" not in dataset_name else A.NoOp(),
            A.RandomBrightnessContrast(p=0.3) if "cifar" in dataset_name else A.NoOp(),
            A.Normalize(mean=mean, std=std, always_apply=True),
            ToTensorV2(),
        ]
        aug = [a for a in aug if not isinstance(a, A.NoOp)]
        def alb_transform(x):
            # x may be PIL or Tensor
            if not isinstance(x, (np.ndarray,)):
                x = np.array(_to_pil_if_tensor(x))
            res = A.Compose(aug)(image=x)["image"]
            return res
        return alb_transform

    # torchvision path
    if "cifar" in dataset_name:
        channels = 3
        base = [
            T.Resize((image_size, image_size)),
            T.RandomHorizontalFlip(),
            T.RandomCrop(image_size, padding=4),
            T.ToTensor(),
        ]
    else:
        channels = 1
        base = [
            T.Resize((image_size, image_size)),
            T.RandomRotation(10),
            T.ToTensor(),
        ]

    if normalize:
        base.append(T.Normalize(mean=normalize["mean"], std=normalize["std"]))
    else:
        # if not given, no normalize
        pass

    return T.Compose(base)


def get_eval_transforms(
    dataset_name: str,
    image_size: int = 32,
    normalize: Optional[dict] = None,
) -> Callable:
    """
    Return evaluation transforms (deterministic).
    """
    dataset_name = dataset_name.lower()
    if "cifar" in dataset_name:
        base = [T.Resize((image_size, image_size)), T.ToTensor()]
    else:
        base = [T.Resize((image_size, image_size)), T.ToTensor()]

    if normalize:
        base.append(T.Normalize(mean=normalize["mean"], std=normalize["std"]))
    return T.Compose(base)
