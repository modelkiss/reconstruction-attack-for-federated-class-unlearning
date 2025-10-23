"""
model_utils.py

Model registry and builder utilities.
Provides unified interface:
    model = get_model(model_name="resnet20", dataset_name="CIFAR10", num_classes=10)
"""

import torch
import torch.nn as nn

from .resnet import resnet8, resnet20
from .lenet import LeNet


# 注册表
MODEL_REGISTRY = {
    "resnet8": resnet8,
    "resnet20": resnet20,
    "lenet": LeNet,
}


def infer_in_channels(dataset_name: str):
    dataset_name = dataset_name.lower()
    if "cifar" in dataset_name:
        return 3
    else:
        return 1


def infer_num_classes(dataset_name: str):
    dataset_name = dataset_name.lower()
    if "cifar100" in dataset_name:
        return 100
    else:
        return 10


def get_model(model_name: str, dataset_name: str, num_classes: int = None, device: str = "cpu") -> nn.Module:
    """
    Construct model by name and move to device.
    """
    model_name = model_name.lower()
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODEL_REGISTRY.keys())}")

    in_channels = infer_in_channels(dataset_name)
    num_classes = num_classes or infer_num_classes(dataset_name)

    model_fn = MODEL_REGISTRY[model_name]
    model = model_fn(num_classes=num_classes, in_channels=in_channels)
    model.to(device)
    return model


def count_parameters(model: nn.Module):
    """Return total number of trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
