# src/explainability/occlusion_sensitivity.py
import torch
import numpy as np
from captum.attr import Occlusion
from .base import BaseSaliencyMethod

class OcclusionSensitivityMethod(BaseSaliencyMethod):
    """遮挡敏感性分析 (Occlusion)"""

    def __init__(self, model, device=None, window_size=8, stride=8):
        super().__init__(model, device)
        self.occlusion = Occlusion(model)
        self.window_size = window_size
        self.stride = stride

    def explain(self, image, target_class):
        image = image.unsqueeze(0).to(self.device)
        attr = self.occlusion.attribute(
            image,
            target=target_class,
            sliding_window_shapes=(3, self.window_size, self.window_size),
            strides=(3, self.stride, self.stride),
        )
        heatmap = attr.squeeze().abs().mean(dim=0).cpu().numpy()
        return self._normalize(heatmap)
