# src/explainability/gradcam.py
import torch
import numpy as np
import torch.nn.functional as F
from .base import BaseSaliencyMethod

class GradCAMMethod(BaseSaliencyMethod):
    """Grad-CAM 类激活图"""

    def __init__(self, model, target_layer_name: str, device=None):
        super().__init__(model, device)
        self.target_layer = dict([*model.named_modules()])[target_layer_name]

    def explain(self, image, target_class):
        image = image.unsqueeze(0).to(self.device)
        activations, gradients = {}, {}

        def forward_hook(module, inp, out):
            activations["value"] = out

        def backward_hook(module, grad_in, grad_out):
            gradients["value"] = grad_out[0]

        fwd = self.target_layer.register_forward_hook(forward_hook)
        bwd = self.target_layer.register_backward_hook(backward_hook)

        output = self.model(image)
        self.model.zero_grad()
        output[0, target_class].backward()

        activ = activations["value"].detach()
        grads = gradients["value"].detach()
        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * activ).sum(dim=1)).squeeze().cpu().numpy()

        fwd.remove()
        bwd.remove()
        return self._normalize(cam)
