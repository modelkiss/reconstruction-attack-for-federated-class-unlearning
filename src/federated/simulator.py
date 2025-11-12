"""Federated learning simulator with configurable aggregation and DP-SGD."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.data.federated_partition import (
    PartitionResult,
    create_dataloaders,
    load_dataset_tensors,
    partition_indices_by_class,
    split_test_indices,
    summarize_distribution,
)
from src.federated.strategies import fedavg_aggregate, secagg_aggregate
from src.models.model_utils import get_model


logger = logging.getLogger(__name__)


@dataclass
class DPConfig:
    enabled: bool = False
    clip_norm: float = 1.0
    noise_multiplier: float = 0.0
    seed: int = 0

    def make_generator(self, device: torch.device) -> torch.Generator:
        gen = torch.Generator(device=device)
        gen.manual_seed(self.seed)
        return gen


@dataclass
class SimulationConfig:
    dataset: str
    num_clients: int
    model_name: str = "resnet20"
    batch_size: int = 64
    local_epochs: int = 1
    lr: float = 0.01
    momentum: float = 0.9
    weight_decay: float = 0.0
    max_rounds: int = 100
    target_accuracy: float = 0.9
    aggregation: str = "fedavg"
    dp: DPConfig = field(default_factory=DPConfig)
    device: str = "cpu"
    seed: int = 0


class FederatedLearningSimulator:
    """High-level orchestrator for federated learning experiments."""

    def __init__(self, cfg: SimulationConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        (
            train_images,
            train_labels,
            test_images,
            test_labels,
            num_classes,
            meta,
        ) = load_dataset_tensors(cfg.dataset)

        train_assignments = partition_indices_by_class(train_labels, cfg.num_clients, seed=cfg.seed)
        server_indices, client_test_assignments = split_test_indices(test_labels, cfg.num_clients, seed=cfg.seed)

        self.partition = PartitionResult(
            client_train_indices=train_assignments,
            client_test_indices=client_test_assignments,
            server_test_indices=server_indices,
        )
        self.train_labels = train_labels
        self.test_labels = test_labels
        self.meta = meta
        self.num_classes = num_classes
        self.train_images = train_images
        self.train_labels = train_labels
        self.test_images = test_images
        self.test_labels = test_labels

        self.client_train_loaders, self.client_test_loaders, self.server_eval_loader = create_dataloaders(
            train_images,
            train_labels,
            test_images,
            test_labels,
            self.partition,
            cfg.dataset,
            cfg.batch_size,
            num_workers=2,
            image_size=meta["image_size"],
            normalize=meta["normalize"],
        )

        self.distribution_summary = {
            "train": summarize_distribution(train_labels, train_assignments),
            "test": summarize_distribution(test_labels, client_test_assignments),
        }

        self.global_model = get_model(cfg.model_name, cfg.dataset, num_classes=self.num_classes, device=cfg.device)
        self.global_model.eval()
        self.global_state = {k: v.detach().cpu().clone() for k, v in self.global_model.state_dict().items()}

    def _train_client(self, client_id: int) -> Dict:
        cfg = self.cfg
        loader = self.client_train_loaders[client_id]
        model = get_model(cfg.model_name, cfg.dataset, num_classes=self.num_classes, device=cfg.device)
        model.load_state_dict(self.global_state)
        model.train()

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(
            model.parameters(), lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay
        )

        dp_gen = None
        if cfg.dp.enabled:
            dp_gen = cfg.dp.make_generator(self.device)
            dp_gen.manual_seed(cfg.dp.seed + client_id)

        total_loss = 0.0
        total_samples = 0

        for _ in range(cfg.local_epochs):
            for inputs, targets in loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                optimizer.zero_grad()
                logits = model(inputs)
                loss = criterion(logits, targets)
                loss.backward()
                if cfg.dp.enabled:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.dp.clip_norm)
                    std = cfg.dp.noise_multiplier * cfg.dp.clip_norm
                    if std > 0.0:
                        for param in model.parameters():
                            if param.grad is not None:
                                noise = torch.normal(
                                    mean=0.0,
                                    std=std,
                                    size=param.grad.shape,
                                    generator=dp_gen,
                                ).to(param.grad.device)
                                param.grad.add_(noise)
                optimizer.step()

                batch_size = inputs.size(0)
                total_loss += float(loss.item()) * batch_size
                total_samples += batch_size

        avg_loss = total_loss / max(total_samples, 1)
        state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        return {
            "client_id": client_id,
            "num_samples": total_samples,
            "state_dict": state_dict,
            "train_loss": avg_loss,
        }

    def _aggregate(self, client_results: List[Dict]) -> Dict[str, torch.Tensor]:
        strategy = self.cfg.aggregation.lower()
        if strategy == "fedavg":
            return fedavg_aggregate(self.global_state, client_results)
        if strategy == "secagg":
            return secagg_aggregate(self.global_state, client_results)
        raise ValueError(f"Unsupported aggregation strategy: {self.cfg.aggregation}")

    def _evaluate(self) -> float:
        self.global_model.load_state_dict(self.global_state)
        self.global_model.to(self.device)
        self.global_model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in self.server_eval_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                logits = self.global_model(inputs)
                preds = torch.argmax(logits, dim=1)
                correct += int((preds == targets).sum().item())
                total += targets.size(0)
        return correct / max(total, 1)

    def run(self) -> Dict:
        history: List[Dict] = []
        rounds = 0
        for round_idx in range(self.cfg.max_rounds):
            client_results: List[Dict] = []
            for cid in range(self.cfg.num_clients):
                result = self._train_client(cid)
                client_results.append(result)
                logger.info(
                    "Round %d, client %d: samples=%d, loss=%.4f",
                    round_idx + 1,
                    cid,
                    result["num_samples"],
                    result["train_loss"],
                )

            self.global_state = self._aggregate(client_results)
            rounds += 1
            accuracy = self._evaluate()
            history.append({"round": round_idx + 1, "accuracy": accuracy})
            logger.info("Round %d aggregated accuracy: %.4f", round_idx + 1, accuracy)

            if accuracy >= self.cfg.target_accuracy:
                break

        return {
            "history": history,
            "rounds": rounds,
            "distribution": self.distribution_summary,
        }
