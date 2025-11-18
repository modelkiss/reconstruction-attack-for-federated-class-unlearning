"""Module A: federated data partitioning and training (FedAvg + trajectory logging)."""

from __future__ import annotations

import logging
import os
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from src.data.dataset_factory import CustomTensorDataset
from src.data.federated_partition import (
    PartitionResult,
    create_dataloaders,
    load_dataset_tensors,
    partition_indices_by_class,
    split_holdout_by_class,
    split_test_indices,
    summarize_distribution,
)
from src.data.transforms import get_train_transforms
from src.federated.strategies import fedavg_aggregate
from src.models.model_utils import get_model

logger = logging.getLogger(__name__)


@dataclass
class DPConfig:
    enabled: bool = False
    clip_norm: float = 1.0
    noise_multiplier: float = 0.0
    seed: int = 0

    def make_generator(self, device: torch.device) -> torch.Generator:
        generator = torch.Generator(device=device)
        generator.manual_seed(self.seed)
        return generator


@dataclass
class ModuleAConfig:
    dataset: str
    data_root: str = "./data/raw"
    num_clients: int = 10
    model_name: Optional[str] = None
    batch_size: int = 64
    local_epochs: int = 10
    lr: float = 0.01
    momentum: float = 0.9
    weight_decay: float = 0.0
    training_mode: str = "rounds"  # "rounds" or "accuracy"
    rounds: int = 20
    target_accuracy: float = 0.9
    max_rounds: int = 100
    diffusion_holdout: float = 0.1
    aggregation: str = "fedavg"
    dp: DPConfig = field(default_factory=DPConfig)
    device: str = "cpu"
    seed: int = 0
    save_dir: str = "outputs/module_a"
    num_workers: int = 2


@dataclass
class ModuleAResult:
    history: List[Dict]
    rounds_completed: int
    distribution: Dict[str, Dict]
    diffusion_holdout_indices: List[int]
    saved_models: List[str]


class FederatedTrainingModuleA:
    """High-level orchestrator dedicated to Module A."""

    _DEFAULT_MODELS = {
        "cifar10": "resnet20",
        "cifar100": "resnet20",
        "svhn": "resnet20",
        "femnist": "lenet",
        "flair": "resnet8",
    }

    def __init__(self, cfg: ModuleAConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self._prepare_data()
        self._prepare_model()

    def _prepare_model(self) -> None:
        dataset_key = self.cfg.dataset.lower()
        model_name = self.cfg.model_name or self._DEFAULT_MODELS.get(dataset_key, "resnet20")
        self.model_name = model_name
        self.global_model = get_model(model_name, self.cfg.dataset, num_classes=self.num_classes, device=self.cfg.device)
        self.global_model.eval()
        self.global_state = {k: v.detach().cpu().clone() for k, v in self.global_model.state_dict().items()}
        logger.info("Using model %s for dataset %s", model_name, self.cfg.dataset)

    def _prepare_data(self) -> None:
        (
            train_images,
            train_labels,
            test_images,
            test_labels,
            num_classes,
            meta,
        ) = load_dataset_tensors(self.cfg.dataset, root=self.cfg.data_root)

        self.num_classes = num_classes
        holdout_indices, fl_indices = split_holdout_by_class(train_labels, fraction=self.cfg.diffusion_holdout, seed=self.cfg.seed)
        logger.info(
            "Reserved %d samples (%.2f%%) for diffusion pre-training; %d samples for federated training",
            len(holdout_indices),
            self.cfg.diffusion_holdout * 100,
            len(fl_indices),
        )

        # build tensors for federated clients
        self.fl_train_images = train_images[fl_indices]
        self.fl_train_labels = train_labels[fl_indices]
        self.test_images = test_images
        self.test_labels = test_labels
        self.meta = meta
        self.diffusion_holdout_indices = holdout_indices

        train_assignments = partition_indices_by_class(self.fl_train_labels, self.cfg.num_clients, seed=self.cfg.seed)
        server_indices, client_test_assignments = split_test_indices(self.test_labels, self.cfg.num_clients, seed=self.cfg.seed)

        self.partition = PartitionResult(
            client_train_indices=train_assignments,
            client_test_indices=client_test_assignments,
            server_test_indices=server_indices,
        )

        self.client_train_loaders, self.client_test_loaders, self.server_eval_loader = create_dataloaders(
            self.fl_train_images,
            self.fl_train_labels,
            self.test_images,
            self.test_labels,
            self.partition,
            self.cfg.dataset,
            self.cfg.batch_size,
            num_workers=self.cfg.num_workers,
            image_size=meta["image_size"],
            normalize=meta.get("normalize"),
        )

        self.diffusion_loader = self._build_diffusion_loader(
            train_images, train_labels, holdout_indices, meta["image_size"], meta.get("normalize")
        )

        self.distribution_summary = {
            "train": summarize_distribution(self.fl_train_labels, train_assignments),
            "test": summarize_distribution(self.test_labels, client_test_assignments),
        }

    def _build_diffusion_loader(
        self,
        train_images: torch.Tensor,
        train_labels: torch.Tensor,
        holdout_indices: List[int],
        image_size: int,
        normalize: Optional[Dict],
    ) -> DataLoader:
        transform = get_train_transforms(self.cfg.dataset, image_size=image_size, normalize=normalize)
        dataset = CustomTensorDataset(train_images, train_labels, transform=transform)
        subset = Subset(dataset, holdout_indices)
        return DataLoader(subset, batch_size=self.cfg.batch_size, shuffle=True, num_workers=self.cfg.num_workers)

    def _train_client(self, client_id: int) -> Dict:
        cfg = self.cfg
        loader = self.client_train_loaders[client_id]
        model = get_model(self.model_name, cfg.dataset, num_classes=self.num_classes, device=cfg.device)
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
        if strategy != "fedavg":
            raise ValueError(f"Unsupported aggregation strategy for Module A: {self.cfg.aggregation}")
        return fedavg_aggregate(self.global_state, client_results)

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

    def _save_recent_models(self, recent_states: Deque[Tuple[int, Dict[str, torch.Tensor]]]) -> List[str]:
        os.makedirs(self.cfg.save_dir, exist_ok=True)
        saved_paths: List[str] = []
        for round_idx, state in recent_states:
            path = os.path.join(self.cfg.save_dir, f"module_a_round_{round_idx}.pt")
            torch.save(state, path)
            saved_paths.append(path)
        return saved_paths

    def run(self) -> ModuleAResult:
        history: List[Dict] = []
        rounds_completed = 0
        recent_states: Deque[Tuple[int, Dict[str, torch.Tensor]]] = deque(maxlen=2)

        planned_rounds = self.cfg.rounds if self.cfg.training_mode == "rounds" else self.cfg.max_rounds

        for round_idx in range(1, planned_rounds + 1):
            client_results: List[Dict] = []
            for cid in range(self.cfg.num_clients):
                result = self._train_client(cid)
                client_results.append(result)
                logger.info(
                    "Round %d, client %d: samples=%d, loss=%.4f",
                    round_idx,
                    cid,
                    result["num_samples"],
                    result["train_loss"],
                )

            self.global_state = self._aggregate(client_results)
            rounds_completed += 1
            recent_states.append((round_idx, {k: v.clone() for k, v in self.global_state.items()}))

            accuracy = self._evaluate()
            history.append({"round": round_idx, "accuracy": accuracy})
            logger.info("Round %d aggregated accuracy: %.4f", round_idx, accuracy)

            if self.cfg.training_mode == "accuracy" and accuracy >= self.cfg.target_accuracy:
                break

        saved_models = self._save_recent_models(recent_states)

        return ModuleAResult(
            history=history,
            rounds_completed=rounds_completed,
            distribution=self.distribution_summary,
            diffusion_holdout_indices=self.diffusion_holdout_indices,
            saved_models=saved_models,
        )
