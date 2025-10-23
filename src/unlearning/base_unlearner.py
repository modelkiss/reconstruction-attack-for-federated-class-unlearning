"""
base_unlearner.py

Defines the abstract interface for all federated unlearning algorithms.
Each unlearner removes the influence of certain data/classes from the global model.
"""

import os
import abc
import torch
from typing import Dict, Any
from src.utils.io import save_checkpoint


class BaseUnlearner(abc.ABC):
    """
    Abstract base class for federated unlearning algorithms.
    """

    def __init__(self, model, dataset_name: str, device: str = "cpu", output_dir: str = "./outputs/unlearning"):
        self.model = model
        self.dataset_name = dataset_name
        self.device = device
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    @abc.abstractmethod
    def run_unlearning(self, *args, **kwargs) -> Dict[str, Any]:
        """
        Execute the unlearning process.
        Must return:
        {
            "model_after": state_dict,
            "unlearned_classes": list[int],
            "stats": {...}
        }
        """
        pass

    def evaluate(self, dataloader, criterion=torch.nn.CrossEntropyLoss()):
        """
        Evaluate model performance on a given dataloader.
        """
        self.model.eval()
        total_loss, total_correct, total_samples = 0, 0, 0
        with torch.no_grad():
            for x, y in dataloader:
                x, y = x.to(self.device), y.to(self.device)
                out = self.model(x)
                loss = criterion(out, y)
                preds = torch.argmax(out, dim=1)
                total_loss += loss.item() * x.size(0)
                total_correct += (preds == y).sum().item()
                total_samples += x.size(0)

        avg_loss = total_loss / total_samples
        acc = total_correct / total_samples
        return {"loss": avg_loss, "acc": acc}

    def save_states(self, state_dict: Dict[str, Any], tag: str):
        """
        Save model state after unlearning or intermediate steps.
        """
        path = os.path.join(self.output_dir, f"model_{tag}.pth")
        save_checkpoint({"state_dict": state_dict, "dataset": self.dataset_name}, path)
        return path
