"""
viz.py
-------
可视化与热力图工具模块
支持功能：
1. 通用绘图与保存
2. Grad-CAM 可解释性可视化
3. Integrated Gradients 可解释性可视化
4. 遗忘前后热力图差异分析
5. 中间层特征激活可视化
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
import torchvision.utils as vutils
from captum.attr import LayerGradCam, LayerAttribution, IntegratedGradients


# ----------------------------- #
# 💾 通用工具函数
# ----------------------------- #

def ensure_dir(path):
    """确保目录存在"""
    if path and not os.path.exists(path):
        os.makedirs(path)


def save_heatmap(array, save_path, cmap="jet", title=None):
    """
    保存单张热力图 (.png)
    同时保存 .npy 文件，便于后续差异分析
    """
    ensure_dir(os.path.dirname(save_path))
    np.save(save_path.replace(".png", ".npy"), array)

    plt.figure(figsize=(4, 4))
    plt.imshow(array, cmap=cmap)
    plt.axis("off")
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()


def overlay_heatmap_on_image(image_tensor, heatmap, alpha=0.4):
    """
    将热力图叠加到原图上，返回叠加后的图像（numpy）
    """
    img = np.transpose(image_tensor.squeeze().cpu().numpy(), (1, 2, 0))
    img = (img - img.min()) / (img.max() - img.min())

    heatmap = np.maximum(heatmap, 0)
    heatmap /= (heatmap.max() + 1e-8)
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(np.uint8(255 * img), 1 - alpha, heatmap_color, alpha, 0)
    return overlay[..., ::-1]  # 转换为 RGB


def save_overlay_image(overlay, save_path):
    """保存叠加后的彩色热力图"""
    ensure_dir(os.path.dirname(save_path))
    plt.imsave(save_path, overlay)


# ----------------------------- #
# 🔥 Grad-CAM 生成与保存
# ----------------------------- #

def generate_gradcam_map(model, input_tensor, target_class, target_layer, save_path=None):
    """
    生成并保存 Grad-CAM 热力图
    """
    model.eval()
    gradcam = LayerGradCam(model, target_layer)
    attr = gradcam.attribute(input_tensor, target=target_class)
    upsampled_attr = LayerAttribution.interpolate(attr, input_tensor.shape[2:])  # 上采样到原尺寸

    heatmap = upsampled_attr.squeeze().cpu().detach().numpy()
    heatmap = np.maximum(heatmap, 0)
    heatmap /= (heatmap.max() + 1e-8)

    if save_path:
        save_heatmap(heatmap, save_path, cmap="jet", title="Grad-CAM")
    return heatmap


# ----------------------------- #
# 🧠 Integrated Gradients 可视化
# ----------------------------- #

def generate_integrated_gradients_map(model, input_tensor, target_class, save_path=None, steps=50):
    """
    使用 Integrated Gradients 生成归因热力图
    """
    model.eval()
    ig = IntegratedGradients(model)
    attributions = ig.attribute(input_tensor, target=target_class, n_steps=steps)
    attr = attributions.squeeze().detach().cpu().numpy()
    attr = np.abs(attr).mean(axis=0)
    attr /= (attr.max() + 1e-8)

    if save_path:
        save_heatmap(attr, save_path, cmap="hot", title="Integrated Gradients")
    return attr


# ----------------------------- #
# 🔍 热力图差异分析
# ----------------------------- #

def compare_heatmaps(heatmap_before, heatmap_after, save_path=None):
    """
    比较两个热力图的差异（遗忘前 vs 遗忘后）
    返回差异矩阵（归一化）
    """
    diff = np.abs(heatmap_before - heatmap_after)
    diff /= (diff.max() + 1e-8)

    if save_path:
        save_heatmap(diff, save_path, cmap="inferno", title="Grad-CAM Difference")
    return diff


# ----------------------------- #
# 🧩 中间层特征可视化
# ----------------------------- #

def visualize_feature_map(feature_tensor, save_path=None, n_cols=8, title=None):
    """
    将中间层特征图 (C,H,W) 可视化成网格
    """
    import math
    C = feature_tensor.shape[0]
    n_rows = math.ceil(C / n_cols)
    grid = vutils.make_grid(
        feature_tensor.unsqueeze(1), nrow=n_cols, normalize=True, scale_each=True
    )

    plt.figure(figsize=(n_cols, n_rows))
    plt.axis('off')
    if title:
        plt.title(title)
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy())

    if save_path:
        ensure_dir(os.path.dirname(save_path))
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()


# ----------------------------- #
# 📊 组合工具：完整保存流程
# ----------------------------- #

def save_full_explanation(model, input_tensor, target_class, target_layer, out_dir, prefix):
    """
    综合函数：生成 Grad-CAM + IG + 差异图 并统一保存
    """
    ensure_dir(out_dir)

    gradcam_path = os.path.join(out_dir, f"{prefix}_gradcam.png")
    ig_path = os.path.join(out_dir, f"{prefix}_ig.png")

    gradcam_map = generate_gradcam_map(model, input_tensor, target_class, target_layer, save_path=gradcam_path)
    ig_map = generate_integrated_gradients_map(model, input_tensor, target_class, save_path=ig_path)

    overlay = overlay_heatmap_on_image(input_tensor, gradcam_map)
    overlay_path = os.path.join(out_dir, f"{prefix}_overlay.png")
    save_overlay_image(overlay, overlay_path)

    return {
        "gradcam": gradcam_map,
        "integrated_gradients": ig_map,
        "overlay_path": overlay_path
    }
