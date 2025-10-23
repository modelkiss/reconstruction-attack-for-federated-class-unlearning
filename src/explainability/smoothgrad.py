# src/explainability/smoothgrad.py
import torch
import numpy as np
from .base import BaseSaliencyMethod

class SmoothGradMethod(BaseSaliencyMethod):
    """SmoothGrad: 平滑的梯度显著性图"""

    def __init__(self, model, device=None, num_samples=20, noise_sigma=0.1):
        super().__init__(model, device)
        self.num_samples = num_samples
        self.noise_sigma = noise_sigma

    def explain(self, image, target_class):
        image = image.unsqueeze(0).to(self.device)
        grads = []
        for _ in range(self.num_samples):
            noisy_img = image + torch.randn_like(image) * self.noise_sigma
            noisy_img.requires_grad_(True)
            out = self.model(noisy_img)
            loss = out[0, target_class]
            self.model.zero_grad()
            loss.backward()
            grads.append(noisy_img.grad.abs().detach().cpu().numpy())
        avg_grad = np.mean(grads, axis=0).squeeze()
        heatmap = np.mean(avg_grad, axis=0)
        return self._normalize(heatmap)
