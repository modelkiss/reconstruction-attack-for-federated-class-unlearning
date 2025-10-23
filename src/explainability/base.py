# src/explainability/base.py
import os
import torch
import numpy as np
from ..utils.viz import save_heatmap_image

class BaseSaliencyMethod:
    """统一的热力图解释器基类"""

    def __init__(self, model, device=None):
        self.model = model.eval()
        self.device = device or next(model.parameters()).device

    def explain(self, image, target_class):
        raise NotImplementedError

    @staticmethod
    def _normalize(arr):
        arr = arr - arr.min()
        arr = arr / (arr.max() + 1e-8)
        return arr

    def visualize_and_save(self, heatmap, image_tensor, save_path):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        save_heatmap_image(heatmap, image_tensor, save_path)
