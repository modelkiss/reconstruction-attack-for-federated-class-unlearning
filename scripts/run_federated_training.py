"""Command line entry point for running the federated learning simulator."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.federated.simulator import DPConfig, FederatedLearningSimulator, SimulationConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Federated learning simulator")
    parser.add_argument("--dataset", required=True, choices=["CIFAR10", "CIFAR100", "MNIST", "FEMNIST"], help="Dataset name")
    parser.add_argument(
        "--num-clients",
        type=int,
        required=True,
        choices=[5, 10, 15, 20],
        help="Number of federated clients",
    )
    parser.add_argument("--model", default="resnet20", help="Backbone model name")
    parser.add_argument("--aggregation", default="FedAvg", choices=["FedAvg", "SecAgg"], help="Aggregation strategy")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-rounds", type=int, default=100)
    parser.add_argument("--target-accuracy", type=float, default=0.9)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dp-sgd", action="store_true", help="Enable DP-SGD training on clients")
    parser.add_argument("--dp-clip", type=float, default=1.0, help="Gradient clipping norm for DP-SGD")
    parser.add_argument("--dp-noise", type=float, default=0.0, help="Noise multiplier for DP-SGD")
    parser.add_argument("--output", default=None, help="Optional path to save history as JSON")
    return parser.parse_args()


def print_distribution(title: str, distribution: dict) -> None:
    print(f"\n{title}:")
    for cid, counts in sorted(distribution.items()):
        total = sum(counts.values())
        detail = ", ".join(f"class {cls}: {count}" for cls, count in sorted(counts.items()))
        print(f"  Client {cid}: total={total} -> {detail}")


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")

    dp_cfg = DPConfig(enabled=args.dp_sgd, clip_norm=args.dp_clip, noise_multiplier=args.dp_noise, seed=args.seed)
    sim_cfg = SimulationConfig(
        dataset=args.dataset,
        num_clients=args.num_clients,
        model_name=args.model,
        batch_size=args.batch_size,
        local_epochs=args.local_epochs,
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        max_rounds=args.max_rounds,
        target_accuracy=args.target_accuracy,
        aggregation=args.aggregation,
        dp=dp_cfg,
        device=args.device,
        seed=args.seed,
    )

    simulator = FederatedLearningSimulator(sim_cfg)

    print_distribution("Client train distribution", simulator.distribution_summary["train"])
    print_distribution("Client test distribution", simulator.distribution_summary["test"])

    results = simulator.run()

    print("\nRound-wise accuracy:")
    for entry in results["history"]:
        print(f"  Round {entry['round']:>3}: accuracy={entry['accuracy']:.4f}")

    print(f"\nTraining completed after {results['rounds']} aggregation rounds.")

    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf8") as f:
            json.dump(results, f, indent=2)
        print(f"Saved results to {path}")


if __name__ == "__main__":
    main()
