"""
client.py

Client-side logic for federated learning simulation.
Each Client performs local training on its local dataloader and returns model updates.
"""

from typing import Dict, Any, Optional, List
import copy
import torch
import torch.nn as nn
import torch.optim as optim

from src.models.model_utils import get_model
from src.utils.io import save_checkpoint
from src.utils.seeds import set_seed


class Client:
    def __init__(
        self,
        client_id: int,
        dataset_name: str,
        processed_root: str,
        model_name: str,
        device: str = "cpu",
        seed: int = 42,
        local_epochs: int = 1,
        batch_size: int = 64,
        lr: float = 0.01,
        weight_decay: float = 0.0,
        momentum: float = 0.9,
        mu: float = 0.0,  # FedProx proximal term coefficient (0 -> FedAvg behavior)
        num_workers: int = 2,
        transform=None,
        exclude_classes: Optional[List[int]] = None,
    ):
        self.client_id = client_id
        self.dataset_name = dataset_name
        self.processed_root = processed_root
        self.device = device
        self.model_name = model_name
        self.seed = seed
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.mu = mu
        self.num_workers = num_workers
        self.transform = transform
        self.exclude_classes = exclude_classes or []

        # lazy attributes
        self._dataloader = None
        self._model = None

    def build_dataloader(self):
        """Build DataLoader for this client using dataset_factory (deferred import)."""
        from src.data.dataset_factory import build_client_dataloader

        self._dataloader = build_client_dataloader(
            processed_root=self.processed_root,
            dataset_name=self.dataset_name,
            client_id=self.client_id,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            transform=self.transform,
            exclude_classes=self.exclude_classes,
        )
        return self._dataloader

    def build_model(self):
        """Instantiate a fresh model for local training (will be loaded with global state externally)."""
        self._model = get_model(self.model_name, self.dataset_name, device=self.device)
        return self._model

    def local_train(self, global_state_dict: Dict[str, torch.Tensor], return_checkpoint: bool = False) -> Dict[str, Any]:
        """
        Perform local training initialized from provided global_state_dict.
        Returns:
            {
              "client_id": int,
              "num_samples": int,
              "state_dict": state_dict_after_local_training,
              "model_delta": state_dict_delta (client - global),
              "train_loss": float (avg),
            }
        """
        # Ensure dataloader exists
        if self._dataloader is None:
            self.build_dataloader()
        if self._model is None:
            self.build_model()

        # set seed for reproducibility
        set_seed(self.seed + self.client_id)

        model = self._model
        model.load_state_dict(copy.deepcopy(global_state_dict))
        model.train()
        model.to(self.device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay)

        # store initial params for delta calculation
        initial_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

        total_loss = 0.0
        total_samples = 0

        # local epochs
        for epoch in range(self.local_epochs):
            for xb, yb in self._dataloader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                optimizer.zero_grad()
                logits = model(xb)
                loss = criterion(logits, yb)

                # FedProx proximal term (mu * ||w - w_global||^2)
                if self.mu and self.mu > 0.0:
                    prox_term = 0.0
                    for name, param in model.named_parameters():
                        prox_term = prox_term + torch.sum((param - global_state_dict[name].to(self.device)) ** 2)
                    loss = loss + (self.mu / 2.0) * prox_term

                loss.backward()
                optimizer.step()

                total_loss += loss.item() * xb.size(0)
                total_samples += xb.size(0)

        avg_loss = total_loss / max(1, total_samples)

        # final state
        final_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        # compute model delta: final - initial (on CPU)
        model_delta = {k: (final_state[k] - initial_state[k].cpu()) for k in final_state.keys()}

        result = {
            "client_id": self.client_id,
            "num_samples": int(total_samples),
            "state_dict": final_state,  # CPU tensors
            "model_delta": model_delta,
            "train_loss": float(avg_loss),
        }

        if return_checkpoint:
            ckpt_path = f"./outputs/clients/client_{self.client_id}_checkpoint.pth"
            save_checkpoint({"state_dict": final_state, "meta": {"client_id": self.client_id}}, ckpt_path)
            result["checkpoint"] = ckpt_path

        return result
