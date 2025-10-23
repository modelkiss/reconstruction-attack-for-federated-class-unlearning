# src/explainability/vanilla_gradients.py
import torch
import numpy as np
from .base import BaseSaliencyMethod

class VanillaGradients(BaseSaliencyMethod):
    """纯梯度法 (Saliency Map)"""

    def explain(self, image, target_class):
        image = image.unsqueeze(0).to(self.device).requires_grad_(True)
        output = self.model(image)
        loss = output[0, target_class]
        loss.backward()
        grad = image.grad.data.abs().squeeze().cpu().numpy()
        heatmap = np.mean(grad, axis=0)
        return self._normalize(heatmap)
