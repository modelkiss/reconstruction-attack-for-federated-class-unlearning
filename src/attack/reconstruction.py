"""Utilities for reconstructing forgotten-class data via gradient-based optimization.

The reconstruction attack implemented here optimizes images directly so that the
``model_before`` remains highly confident on the target class while the
``model_after`` loses confidence. Regularization (total variation + L2) keeps the
generated samples within a natural image manifold. This provides a lightweight
baseline that works across CIFAR/MNIST style datasets without relying on
external generative models.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.optim as optim

try:
    from torchvision.utils import save_image
except Exception:  # pragma: no cover - torchvision may be optional in some envs
    save_image = None  # type: ignore

from src.utils.io import ensure_dir, save_json


@dataclass
class ReconstructionConfig:
    max_iters: int = 800
    lr: float = 0.05
    after_weight: float = 1.0
    tv_weight: float = 1e-5
    l2_weight: float = 1e-4
    log_every: int = 50
    init_noise_scale: float = 0.05
    use_cosine: bool = True
    clamp_min: float = 0.0
    clamp_max: float = 1.0
    device: str = "cpu"
    save_debug: bool = True
    save_metadata: bool = True
    extra: Dict[str, float] = field(default_factory=dict)


DATASET_STATS = {
    "cifar10": {"mean": (0.4914, 0.4822, 0.4465), "std": (0.2470, 0.2435, 0.2616), "channels": 3, "size": 32},
    "cifar100": {"mean": (0.5071, 0.4867, 0.4408), "std": (0.2675, 0.2565, 0.2761), "channels": 3, "size": 32},
    "mnist": {"mean": (0.5,), "std": (0.5,), "channels": 1, "size": 28},
    "fashionmnist": {"mean": (0.5,), "std": (0.5,), "channels": 1, "size": 28},
}


def _lookup_stats(dataset_name: str):
    name = dataset_name.lower()
    for key, stats in DATASET_STATS.items():
        if key in name:
            return stats
    # fall back to generic grayscale
    return {"mean": (0.5,), "std": (0.5,), "channels": 1, "size": 28}


def total_variation(images: torch.Tensor) -> torch.Tensor:
    """Compute isotropic total variation loss for a batch of images."""

    dh = torch.abs(images[:, :, 1:, :] - images[:, :, :-1, :]).mean()
    dw = torch.abs(images[:, :, :, 1:] - images[:, :, :, :-1]).mean()
    return dh + dw


class ClassConditionalReconstructor:
    """Gradient-based reconstruction of forgotten-class samples."""

    def __init__(
        self,
        model_before: nn.Module,
        model_after: nn.Module,
        dataset_name: str,
        config: Optional[ReconstructionConfig | Dict] = None,
    ):
        self.model_before = model_before.eval()
        self.model_after = model_after.eval()
        stats = _lookup_stats(dataset_name)
        self.mean = torch.tensor(stats["mean"], dtype=torch.float32).view(-1, 1, 1)
        self.std = torch.tensor(stats["std"], dtype=torch.float32).view(-1, 1, 1)
        self.channels = stats["channels"]
        self.size = stats["size"]

        if config is None:
            config = ReconstructionConfig()
        elif isinstance(config, dict):
            cfg = ReconstructionConfig()
            for key, value in config.items():
                if hasattr(cfg, key):
                    setattr(cfg, key, value)
                else:
                    cfg.extra[key] = value
            config = cfg
        self.cfg: ReconstructionConfig = config

        self.device = torch.device(self.cfg.device)
        self.model_before.to(self.device)
        self.model_after.to(self.device)
        self.mean = self.mean.to(self.device)
        self.std = self.std.to(self.device)

    def _normalize(self, images: torch.Tensor) -> torch.Tensor:
        return (images - self.mean) / self.std

    def _denormalize(self, images: torch.Tensor) -> torch.Tensor:
        return images * self.std + self.mean

    def reconstruct(
        self,
        target_class: int,
        num_images: int = 16,
        save_dir: Optional[str] = None,
    ) -> Dict:
        cfg = self.cfg
        device = self.device

        imgs = torch.rand(num_images, self.channels, self.size, self.size, device=device)
        if cfg.init_noise_scale > 0:
            imgs = imgs + torch.randn_like(imgs) * cfg.init_noise_scale
        imgs.data.clamp_(cfg.clamp_min, cfg.clamp_max)
        imgs.requires_grad_(True)

        optimizer = optim.Adam([imgs], lr=cfg.lr)
        scheduler = None
        if cfg.use_cosine:
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, cfg.max_iters))

        loss_history: List[float] = []

        for step in range(cfg.max_iters):
            optimizer.zero_grad()
            clipped = imgs.clamp(cfg.clamp_min, cfg.clamp_max)
            normalized = self._normalize(clipped)

            logits_before = self.model_before(normalized)
            logits_after = self.model_after(normalized)

            target_before = logits_before[:, target_class]
            target_after = logits_after[:, target_class]

            loss_main = -target_before.mean() + cfg.after_weight * target_after.mean()
            loss_tv = total_variation(clipped) * cfg.tv_weight
            loss_l2 = torch.mean((clipped - 0.5) ** 2) * cfg.l2_weight
            loss = loss_main + loss_tv + loss_l2

            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            loss_history.append(float(loss.item()))

        with torch.no_grad():
            final_imgs = imgs.clamp(cfg.clamp_min, cfg.clamp_max).detach()
            normalized = self._normalize(final_imgs)
            logits_before = self.model_before(normalized)
            logits_after = self.model_after(normalized)
            probs_before = logits_before.softmax(dim=1)
            probs_after = logits_after.softmax(dim=1)

        result = {
            "target_class": int(target_class),
            "loss_history": loss_history,
            "logits_before": logits_before.cpu().tolist(),
            "logits_after": logits_after.cpu().tolist(),
            "probs_before": probs_before.cpu().tolist(),
            "probs_after": probs_after.cpu().tolist(),
        }

        if save_dir:
            ensure_dir(save_dir)
            tensor_to_save = final_imgs.detach().cpu()
            if save_image is not None:
                grid_path = os.path.join(save_dir, "reconstruction_grid.png")
                nrow = max(1, int(math.sqrt(num_images)))
                save_image(tensor_to_save, grid_path, nrow=nrow, normalize=False)
                result["grid_path"] = grid_path
            torch.save(tensor_to_save, os.path.join(save_dir, "reconstructed_images.pt"))
            if cfg.save_metadata:
                save_json(result, os.path.join(save_dir, "reconstruction_metadata.json"))

        return result
