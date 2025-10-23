import json
import os
import torch

def ensure_dir(path: str):
    """确保目录存在"""
    os.makedirs(path, exist_ok=True)

def save_json(data, path: str):
    """保存字典或列表为 JSON 文件"""
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def load_json(path: str):
    """加载 JSON 文件"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_checkpoint(state, path: str):
    """保存 PyTorch 模型权重或状态字典"""
    ensure_dir(os.path.dirname(path))
    torch.save(state, path)

def load_checkpoint(path: str, device="cpu"):
    """加载模型权重"""
    return torch.load(path, map_location=device)
