# src/attack/label_inference.py
from __future__ import annotations
import os
from typing import Dict, Any, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader

try:
    # 项目内的 save_json（若不可用，下面会 fallback）
    from src.utils.io import ensure_dir, save_json
except Exception:
    ensure_dir = lambda p: os.makedirs(p, exist_ok=True)  # noqa: E731
    import json
    def save_json(data, path):  # noqa: E302
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)


@torch.no_grad()
def _per_class_accuracy(model: torch.nn.Module,
                        loader: DataLoader,
                        device: str,
                        num_classes: Optional[int] = None) -> Tuple[List[int], List[int]]:
    """
    返回 (per_class_correct, per_class_total) 两个等长列表。
    """
    model.eval()
    ce_device = torch.device(device)
    model.to(ce_device)

    # 若用户未指定类别数，尝试从数据或模型推断
    if num_classes is None:
        # 1) 从 loader.dataset 的属性推断
        ds = getattr(loader, "dataset", None)
        if ds is not None:
            # 常见：CIFAR/MNIST 有 classes / targets / class_to_idx
            if hasattr(ds, "classes"):
                num_classes = len(ds.classes)  # type: ignore
            elif hasattr(ds, "targets"):
                try:
                    num_classes = int(torch.as_tensor(ds.targets).max().item() + 1)  # type: ignore
                except Exception:
                    pass
        # 2) 从模型输出推断
    per_class_correct: List[int]
    per_class_total: List[int]
    per_class_correct = []
    per_class_total = []
    if num_classes is not None:
        per_class_correct = [0 for _ in range(num_classes)]
        per_class_total = [0 for _ in range(num_classes)]

    for xb, yb in loader:
        xb = xb.to(ce_device)
        yb = yb.to(ce_device)

        out = model(xb)            # [B, C]
        if num_classes is None:
            num_classes = out.shape[1]
            per_class_correct = [0 for _ in range(num_classes)]
            per_class_total = [0 for _ in range(num_classes)]

        preds = out.argmax(dim=1)  # [B]
        for t, p in zip(yb.view(-1), preds.view(-1)):
            t_i = int(t.item())
            correct = int(p.item() == t_i)
            per_class_correct[t_i] += correct
            per_class_total[t_i] += 1

    return per_class_correct, per_class_total


def _safe_div(n: float, d: float) -> float:
    return float(n) / float(d) if d != 0 else 0.0


def confidence_label_inference(
    model_before: torch.nn.Module,
    model_after: torch.nn.Module,
    test_loader: DataLoader,
    device: str = "cpu",
    top_k: int = 3,
    save_dir: Optional[str] = None,
    class_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    基于“遗忘前后 per-class 准确率下降”做标签推理，并返回置信度分布。

    Returns:
      {
        "per_class": {
            <cls_idx>: {
                "name": <class name or str(cls_idx)>,
                "acc_before": float,
                "acc_after": float,
                "drop": float,              # = max(acc_before - acc_after, 0)
                "confidence": float         # 归一化后的 drop / sum(drop)
            }, ...
        },
        "total_samples_per_class": {<cls_idx>: int, ...},
        "sum_drop": float,
        "predicted_forgotten": [<top class idx> ...],      # 长度为 top_k（或小于）
        "predicted_forgotten_names": [<name> ...]          # 若提供了 class_names
      }
    """
    # 1) 统计 before / after 的 per-class 正确与总数
    correct_b, total_b = _per_class_accuracy(model_before, test_loader, device)
    correct_a, total_a = _per_class_accuracy(model_after, test_loader, device, num_classes=len(total_b))

    num_classes = len(total_b)
    # class 名称
    if class_names is None:
        class_names = [str(i) for i in range(num_classes)]

    # 2) 计算 per-class acc 与 drop
    acc_b = [_safe_div(c, t) for c, t in zip(correct_b, total_b)]
    acc_a = [_safe_div(c, t) for c, t in zip(correct_a, total_a)]

    drops = [max(b - a, 0.0) for b, a in zip(acc_b, acc_a)]
    sum_drop = sum(drops)

    # 3) 归一化为置信度（若总下降为0，做均匀分布/或全零）
    if sum_drop > 0:
        confidences = [d / sum_drop for d in drops]
    else:
        # 没有下降：表示 after 没有在任何类上变差。保守做法：全零或均匀。
        # 这里选择均匀，避免 NaN，并让后续 top_k 仍可用。
        confidences = [1.0 / num_classes for _ in range(num_classes)]

    # 4) Top-k 预测（按 drop 或 confidence 排序都等价）
    order = sorted(range(num_classes), key=lambda i: drops[i], reverse=True)
    top_k = min(top_k, num_classes)
    top_idx = order[:top_k]
    top_names = [class_names[i] for i in top_idx]

    # 5) 组织结果
    per_class = {}
    for i in range(num_classes):
        per_class[i] = {
            "name": class_names[i],
            "acc_before": float(acc_b[i]),
            "acc_after": float(acc_a[i]),
            "drop": float(drops[i]),
            "confidence": float(confidences[i]),
        }

    result = {
        "per_class": per_class,
        "total_samples_per_class": {i: int(total_b[i]) for i in range(num_classes)},
        "sum_drop": float(sum_drop),
        "predicted_forgotten": top_idx,
        "predicted_forgotten_names": top_names,
    }

    # 6) 可选：落盘保存
    if save_dir is not None:
        ensure_dir(save_dir)
        save_json(result, os.path.join(save_dir, "label_inference_confidence.json"))

    return result
