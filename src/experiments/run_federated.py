"""
run_federated.py

Entry point to run federated training experiments.

Features:
- Load and merge YAML configuration files (default + experiment-specific).
- Allow CLI overrides for a small set of common params.
- Initialize Server (src.federated.server.Server) with merged config.
- Save merged config to the experiment output directory for reproducibility.
- Run federated rounds synchronously (Server.run()).

Usage examples:
    # minimal
    python -m src.experiments.run_federated --config configs/experiments/cifar10_resnet20_fedavg.yaml

    # override device and rounds
    python -m src.experiments.run_federated \
        --config configs/experiments/cifar10_resnet20_fedavg.yaml \
        --device cuda --rounds 100

Notes:
- This script expects your project pythonpath to include project root (so `src` package is importable).
  Running it via `python -m src.experiments.run_federated` from project root is recommended.
- The Server class handles creating an out_dir and saving per-round checkpoints. After Server initialization,
  this script will write the merged config into server.out_dir/config.yaml for reproducibility.
"""

from __future__ import annotations
import argparse
import copy
import os
import sys
import time
import yaml
from typing import Dict, Any

# Ensure project root is on sys.path when running as script (helps when invoked directly)
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.federated.server import Server
from src.utils.io import save_json
from src.utils.logging import setup_logging
from src.utils.seeds import set_seed


def load_yaml(path: str) -> Dict[str, Any]:
    """Load a YAML file and return a dict."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def deep_merge(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively merge dict b into dict a (non-destructive) and return the merged dict.
    Values in b override those in a. Nested dicts are merged recursively.
    """
    result = copy.deepcopy(a)
    for k, v in (b or {}).items():
        if (
            k in result
            and isinstance(result[k], dict)
            and isinstance(v, dict)
        ):
            result[k] = deep_merge(result[k], v)
        else:
            result[k] = copy.deepcopy(v)
    return result


def parse_args():
    p = argparse.ArgumentParser(description="Run federated experiment")
    p.add_argument("--config", "-c", type=str, required=True,
                   help="Path to experiment YAML config (will be merged with configs/default.yaml)")
    p.add_argument("--default-config", type=str, default="configs/default.yaml",
                   help="Path to default YAML config")
    p.add_argument("--device", type=str, default=None,
                   help="Explicit device override (e.g. 'cuda' or 'cpu'). If omitted, will auto-detect.")
    p.add_argument("--rounds", type=int, default=None,
                   help="Optional override for number of federated rounds")
    p.add_argument("--seed", type=int, default=None, help="Optional random seed override")
    p.add_argument("--processed-root", type=str, default="data/processed", help="Path to processed data root")
    p.add_argument("--output-root", type=str, default="outputs/experiments", help="Root folder for outputs")
    p.add_argument("--no-save-config", action="store_true", help="Don't save merged config into output folder")
    return p.parse_args()


def detect_device(preferred: str | None = None) -> str:
    """
    Decide which device to use.
    Preference order:
      1) CLI --device if provided and valid ('cpu' / 'cuda')
      2) torch.cuda.is_available()
      3) fallback to 'cpu'
    """
    import torch
    if preferred:
        if preferred.lower() in ("cpu", "cuda"):
            if preferred.lower() == "cuda" and not torch.cuda.is_available():
                print("Warning: --device cuda requested but CUDA not available. Falling back to cpu.")
                return "cpu"
            return preferred.lower()
        else:
            print(f"Warning: unknown device '{preferred}', falling back to auto-detection.")
    return "cuda" if torch.cuda.is_available() else "cpu"


def main():
    # -------------------------
    # 1) parse CLI args
    # -------------------------
    args = parse_args()

    # -------------------------
    # 2) load configs (default + experiment)
    # -------------------------
    default_cfg = {}
    if os.path.exists(args.default_config):
        default_cfg = load_yaml(args.default_config)
    exp_cfg = load_yaml(args.config)
    # merge: default <- exp (exp overrides default)
    merged_cfg = deep_merge(default_cfg, exp_cfg)

    # apply CLI overrides for a few top-level fields
    if args.rounds is not None:
        merged_cfg.setdefault("federated", {})["rounds"] = args.rounds
    if args.seed is not None:
        merged_cfg.setdefault("experiment", {})["seed"] = args.seed

    # -------------------------
    # 3) device & seed
    # -------------------------
    device = detect_device(args.device)
    seed = merged_cfg.get("experiment", {}).get("seed", 42)
    set_seed(seed)

    # -------------------------
    # 4) init logging for run_federated script
    # -------------------------
    # create a short-run folder (server will create its own out_dir later)
    run_tag = f"run_{time.strftime('%Y%m%d_%H%M%S')}"
    tmp_out = os.path.join(args.output_root, run_tag)
    os.makedirs(tmp_out, exist_ok=True)
    logger = setup_logging(tmp_out, log_name="run_federated")
    logger.info("Starting run_federated")
    logger.info(f"Config paths: default={args.default_config}, experiment={args.config}")
    logger.info(f"Device: {device}, Seed: {seed}")

    # -------------------------
    # 5) instantiate Server and run
    # -------------------------
    # Server will create its own unique output folder; pass merged config to it.
    server = Server(merged_cfg, processed_root=args.processed_root, output_dir=args.output_root, device=device)

    # Save merged config into server.out_dir for reproducibility (unless disabled)
    if not args.no_save_config:
        try:
            cfg_save_path = os.path.join(server.out_dir, "config.yaml")
            with open(cfg_save_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(merged_cfg, f, sort_keys=False, allow_unicode=True)
            logger.info(f"Merged config saved: {cfg_save_path}")
        except Exception as e:
            logger.warning(f"Failed to save merged config to {server.out_dir}: {e}")

    # run federated training
    logger.info("Launching federated training (Server.run)...")
    metrics = server.run()
    logger.info("Federated training finished.")

    # persist a small summary
    summary = {
        "experiment_config": args.config,
        "device": device,
        "seed": seed,
        "server_outdir": server.out_dir,
        "metrics": metrics,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    save_json(summary, os.path.join(server.out_dir, "run_summary.json"))
    logger.info(f"Run summary written to {os.path.join(server.out_dir, 'run_summary.json')}")
    print(f"Experiment finished. Outputs are under: {server.out_dir}")


if __name__ == "__main__":
    main()
