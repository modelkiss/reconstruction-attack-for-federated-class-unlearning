import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def evaluate_classification(y_true, y_pred):
    """计算分类任务的精度、精确率、召回率、F1"""
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

def accuracy_from_logits(logits, labels):
    """从模型输出logits直接计算accuracy"""
    preds = torch.argmax(logits, dim=1)
    return (preds == labels).float().mean().item()