"""
retrain_unlearner.py

Implements full retraining (retrain from scratch excluding forgotten data/classes).
This serves as the gold-standard baseline for unlearning correctness.
"""

import os
import copy
from typing import Dict, Any, List
from src.unlearning.base_unlearner import BaseUnlearner
from src.federated.server import Server
from src.utils.logging import setup_logging
from src.utils.io import save_checkpoint


class RetrainUnlearner(BaseUnlearner):
    def __init__(self, cfg: Dict[str, Any], forgotten_classes: List[int], device: str = "cpu", output_dir: str = "./outputs/unlearning/retrain"):
        super().__init__(None, cfg["dataset"]["name"], device, output_dir)
        self.cfg = cfg
        self.forgotten_classes = forgotten_classes
        self.logger = setup_logging(self.output_dir, log_name="retrain_unlearner")

    def run_unlearning(self) -> Dict[str, Any]:
        """
        Execute full retraining on dataset excluding forgotten classes.
        """
        self.logger.info(f"Starting retraining unlearning for classes: {self.forgotten_classes}")

        # modify dataset configuration to exclude forgotten classes
        dataset_cfg = copy.deepcopy(self.cfg["dataset"])
        dataset_cfg["exclude_classes"] = self.forgotten_classes
        self.cfg["dataset"] = dataset_cfg

        # run a new federated training from scratch
        server = Server(self.cfg, processed_root="data/processed", output_dir=self.output_dir, device=self.device)
        metrics = server.run()

        # final model after retraining
        model_after = server.global_state
        model_path = self.save_states(model_after, "after_retrain")

        self.logger.info(f"Retrain unlearning finished. Model saved at {model_path}")

        return {
            "model_after": model_after,
            "unlearned_classes": self.forgotten_classes,
            "stats": {"method": "retrain", "metrics": metrics},
        }
