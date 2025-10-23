# src/explainability/guided_backprop.py
import torch
import numpy as np
from torch import nn
from .base import BaseSaliencyMethod

class GuidedBackpropReLU(nn.Module):
    def forward(self, input):
        return torch.clamp(input, min=0.0)

    def backward(self, grad_output):
        grad_input = grad_output.clone()
        grad_input[grad_output < 0] = 0
        return grad_input

class GuidedBackpropMethod(BaseSaliencyMethod):
    """Guided Backpropagation"""

    def __init__(self, model, device=None):
        super().__init__(model, device)
        self._replace_relu_with_guided(model)

    def _replace_relu_with_guided(self, module):
        for name, layer in module.named_children():
            if isinstance(layer, nn.ReLU):
                setattr(module, name, GuidedBackpropReLU())
            else:
                self._replace_relu_with_guided(layer)

    def explain(self, image, target_class):
        image = image.unsqueeze(0).to(self.device).requires_grad_(True)
        output = self.model(image)
        loss = output[0, target_class]
        loss.backward()
        grad = image.grad.data.abs().squeeze().cpu().numpy()
        heatmap = np.mean(grad, axis=0)
        return self._normalize(heatmap)
