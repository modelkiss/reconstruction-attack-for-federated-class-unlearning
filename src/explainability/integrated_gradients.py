# src/explainability/integrated_gradients.py
import torch
import numpy as np
from captum.attr import IntegratedGradients
from .base import BaseSaliencyMethod

class IntegratedGradientsMethod(BaseSaliencyMethod):
    """积分梯度法 (Integrated Gradients)"""

    def __init__(self, model, device=None, steps=50):
        super().__init__(model, device)
        self.integrated_gradients = IntegratedGradients(model)
        self.steps = steps

    def explain(self, image, target_class):
        image = image.unsqueeze(0).to(self.device)
        attr = self.integrated_gradients.attribute(
            image, target=target_class, n_steps=self.steps
        )
        heatmap = attr.squeeze().abs().mean(dim=0).cpu().numpy()
        return self._normalize(heatmap)
