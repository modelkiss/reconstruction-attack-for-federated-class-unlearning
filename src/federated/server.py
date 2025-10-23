"""
server.py

Server orchestration for federated simulation.
- maintains global model state
- selects clients each round
- orchestrates local training (via Client.local_train)
- aggregates updates via specified strategy
- saves checkpoints and metrics
"""

import os
import time
import random
import copy
from typing import List, Dict, Any, Optional

import torch

from src.models.model_utils import get_model, count_parameters
from src.utils.io import save_checkpoint
from src.utils.logging import setup_logging
from src.utils.seeds import set_seed
from src.data.dataset_factory import make_transforms_from_config

from .strategies import fedavg_aggregate, FedOptAggregator, fedprox_aggregate, dp_fedavg_aggregate


class Server:
    def __init__(
        self,
        cfg: Dict,
        processed_root: str = "data/processed",
        output_dir: str = "./outputs/experiments",
        device: str = "cpu",
        initial_state_dict: Optional[Dict[str, torch.Tensor]] = None,
    ):
        """
        cfg is expected to contain keys:
          - experiment: {name, seed, ...}
          - dataset: {name, normalize, image_size}
          - model: {backbone}
          - federated: {strategy, num_clients, client_frac, rounds, local_epochs, client_lr, ...}
        """
        self.cfg = cfg
        self.device = device
        exp_name = cfg.get("experiment", {}).get("name", "fed_exp")
        self.out_dir = os.path.join(output_dir, f"{time.strftime('%Y%m%d_%H%M%S')}__{exp_name}")
        os.makedirs(self.out_dir, exist_ok=True)

        # logging
        self.logger = setup_logging(self.out_dir, log_name="server")
        self.logger.info(f"Server output_dir: {self.out_dir}")

        # seed
        seed = cfg.get("experiment", {}).get("seed", 42)
        set_seed(seed)
        self.seed = seed

        # dataset/model/fed params
        self.processed_root = processed_root
        self.dataset_name = cfg.get("dataset", {}).get("name", cfg.get("dataset", "CIFAR10"))
        self.model_name = cfg.get("model", {}).get("backbone", cfg.get("model", "resnet20"))
        self.exclude_classes = sorted(cfg.get("dataset", {}).get("exclude_classes", []))
        fed_cfg = cfg.get("federated", {})
        self.strategy = fed_cfg.get("strategy", "FedAvg")
        self.num_clients = fed_cfg.get("num_clients", 20)
        self.client_frac = fed_cfg.get("client_frac", 0.1)
        self.rounds = fed_cfg.get("rounds", fed_cfg.get("total_rounds", 50))
        self.local_epochs = fed_cfg.get("local_epochs", 1)
        self.client_lr = fed_cfg.get("client_lr", 0.01)
        self.weight_decay = fed_cfg.get("weight_decay", 0.0)
        self.momentum = fed_cfg.get("momentum", 0.9)
        self.mu = fed_cfg.get("mu", 0.0)  # for FedProx
        self.batch_size = fed_cfg.get("client_batch_size", 64)
        self.num_workers = cfg.get("dataset", {}).get("num_workers", 4)

        # build initial global model
        self.global_model = get_model(self.model_name, self.dataset_name, device="cpu")
        if initial_state_dict is not None:
            missing, unexpected = self.global_model.load_state_dict(initial_state_dict, strict=False)
            if missing:
                self.logger.warning(f"Initial state is missing params: {missing}")
            if unexpected:
                self.logger.warning(f"Initial state has unexpected params: {unexpected}")
        self.global_state = {k: v.detach().cpu().clone() for k, v in self.global_model.state_dict().items()}
        self.logger.info(f"Initialized global model {self.model_name} with {count_parameters(self.global_model)} params")

        # server aggregator state (for FedOpt)
        self.server_aggregator = None
        if self.strategy.lower() == "fedopt":
            # default server LR from config or 1.0
            server_lr = fed_cfg.get("server_lr", 1.0)
            self.server_aggregator = FedOptAggregator(self.global_state, lr=server_lr)

        # privacy-related options
        dp_cfg = fed_cfg.get("dp", {}) or {}
        self.dp_enabled = bool(dp_cfg.get("enabled", False))
        self.dp_clip_norm = float(dp_cfg.get("clip_norm", 1.0))
        self.dp_noise_multiplier = float(dp_cfg.get("noise_multiplier", 0.0))
        self.dp_seed = int(dp_cfg.get("seed", 0))
        if self.dp_enabled and self.strategy.lower() not in ("fedavg", "fedprox"):
            self.logger.warning("DP aggregation currently supports FedAvg/FedProx only; falling back to FedAvg aggregation")

        self.secure_aggregation = bool(fed_cfg.get("secure_aggregation", False))
        if self.secure_aggregation:
            self.logger.info("Secure aggregation enabled: per-client updates are aggregated without persistent storage")

        # prepare client ids
        self.all_client_ids = list(range(self.num_clients))

        # prepare transforms for clients
        self.train_transform = make_transforms_from_config(self.dataset_name, cfg, train=True)
        self.eval_transform = make_transforms_from_config(self.dataset_name, cfg, train=False)

        self.logger.info(
            f"Federated config: strategy={self.strategy}, "
            f"num_clients={self.num_clients}, "
            f"client_frac={self.client_frac}, "
            f"clients_per_round={fed_cfg.get('clients_per_round', 'auto')}, "
            f"total_rounds={self.rounds}"
        )

    def sample_clients(self, round_idx: int) -> List[int]:
        fed_cfg = self.cfg.get("federated", {})
        k = fed_cfg.get("clients_per_round", None)
        if k is None:
            # 如果没有明确给出 clients_per_round，则按比例采样
            k = max(1, int(self.client_frac * self.num_clients))
        k = min(k, self.num_clients)  # 防止超过总数
        random.seed(self.seed + round_idx)
        return random.sample(self.all_client_ids, k)

    def run(self):
        """Main federated training loop (synchronous)."""
        metrics = []
        for r in range(self.rounds):
            t0 = time.time()
            selected = self.sample_clients(r)
            self.logger.info(f"Round {r+1}/{self.rounds} - selected clients: {selected}")

            client_results = []
            # run clients sequentially (for simplicity)
            for cid in selected:
                from src.federated.client import Client

                client = Client(
                    client_id=cid,
                    dataset_name=self.dataset_name,
                    processed_root=self.processed_root,
                    model_name=self.model_name,
                    device=self.device,
                    seed=self.seed,
                    local_epochs=self.local_epochs,
                    batch_size=self.batch_size,
                    lr=self.client_lr,
                    weight_decay=self.weight_decay,
                    momentum=self.momentum,
                    mu=self.mu,
                    num_workers=self.num_workers,
                    transform=self.train_transform,
                    exclude_classes=self.exclude_classes,
                )

                res = client.local_train(self.global_state)
                client_results.append(res)
                self.logger.info(f"  client {cid} - samples {res['num_samples']} - loss {res['train_loss']:.4f}")

            # aggregate
            if self.dp_enabled:
                from torch import Generator

                gen = Generator(device="cpu")
                gen.manual_seed(self.dp_seed + r)
                new_state = dp_fedavg_aggregate(
                    self.global_state,
                    client_results,
                    clip_norm=self.dp_clip_norm,
                    noise_multiplier=self.dp_noise_multiplier,
                    generator=gen,
                )
            elif self.strategy.lower() == "fedavg":
                new_state = fedavg_aggregate(self.global_state, client_results)
            elif self.strategy.lower() == "fedopt":
                if self.server_aggregator is None:
                    self.server_aggregator = FedOptAggregator(self.global_state)
                new_state = self.server_aggregator.step(self.global_state, client_results)
            elif self.strategy.lower() == "fedprox":
                new_state = fedprox_aggregate(self.global_state, client_results)
            else:
                # default to FedAvg
                new_state = fedavg_aggregate(self.global_state, client_results)

            # update global state and save checkpoint per round
            self.global_state = {k: v.detach().cpu().clone() for k, v in new_state.items()}

            if self.secure_aggregation:
                client_results.clear()

            # optional: save checkpoint
            ckpt_path = os.path.join(self.out_dir, f"checkpoints_round_{r+1}.pth")
            os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
            save_checkpoint({"round": r+1, "state_dict": self.global_state}, ckpt_path)

            t1 = time.time()
            self.logger.info(f"Round {r+1} completed in {t1-t0:.1f}s")
            metrics.append({"round": r+1, "clients": selected})

        # final save
        final_path = os.path.join(self.out_dir, "global_model_final.pth")
        save_checkpoint({"state_dict": self.global_state, "meta": {"strategy": self.strategy}}, final_path)
        self.logger.info(f"Training finished. final model saved to {final_path}")
        return metrics
