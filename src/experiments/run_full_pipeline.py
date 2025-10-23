"""End-to-end experimental pipeline for federated unlearning + reconstruction attack.

This script automates the workflow described in the project README:

1. Train a federated model (FedAvg by default) on the specified dataset.
2. Simulate a class-level unlearning request by removing target classes and
   continuing federated training for several rounds.
3. Launch label inference to identify the forgotten class using the difference
   between the before/after models.
4. Run a gradient-based reconstruction attack to synthesize representative
   samples of the forgotten class.
5. Repeat the above under multiple defence scenarios (secure aggregation,
   differential privacy) defined in the configuration file.

Configuration format (YAML):

```
experiment:
  name: federated_full_pipeline
  seed: 42
  device: cuda

dataset:
  name: CIFAR10
  eval_batch_size: 256
  num_workers: 4

model:
  backbone: resnet20

federated:
  strategy: FedAvg
  num_clients: 10
  client_frac: 0.5
  rounds: 80
  local_epochs: 1
  client_lr: 0.01

unlearning:
  target_class: 6
  rounds: 20

attack:
  label_inference:
    top_k: 3
  reconstruction:
    max_iters: 800
    lr: 0.05
    num_images: 16

scenarios:
  - name: baseline
  - name: secure_aggregation
    federated:
      secure_aggregation: true
  - name: dp_noise
    federated:
      dp:
        enabled: true
        clip_norm: 1.0
        noise_multiplier: 0.3
```

Run:

```
python -m src.experiments.run_full_pipeline --config configs/experiments/full_pipeline_cifar10.yaml
```
"""

from __future__ import annotations

import argparse
import copy
import os
import sys
import time
from typing import Any, Dict, List, Optional

import torch

# Ensure project root on PYTHONPATH
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.attack.label_inference import confidence_label_inference
from src.attack.reconstruction import ClassConditionalReconstructor
from src.data.dataset_factory import build_val_dataloader, make_transforms_from_config
from src.experiments.run_federated import deep_merge, load_yaml
from src.federated.server import Server
from src.models.model_utils import get_model
from src.utils.io import ensure_dir, save_checkpoint, save_json
from src.utils.logging import setup_logging
from src.utils.seeds import set_seed


CLASS_NAME_LOOKUP = {
    "cifar10": [
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    ],
    "cifar100": [
        # 20 super-classes each with 5 subclasses – here we use subclass names in order
        "apple",
        "aquarium_fish",
        "baby",
        "bear",
        "beaver",
        "bed",
        "bee",
        "beetle",
        "bicycle",
        "bottle",
        "bowl",
        "boy",
        "bridge",
        "bus",
        "butterfly",
        "camel",
        "can",
        "castle",
        "caterpillar",
        "cattle",
        "chair",
        "chimpanzee",
        "clock",
        "cloud",
        "cockroach",
        "couch",
        "crab",
        "crocodile",
        "cup",
        "dinosaur",
        "dolphin",
        "elephant",
        "flatfish",
        "forest",
        "fox",
        "girl",
        "hamster",
        "house",
        "kangaroo",
        "keyboard",
        "lamp",
        "lawn_mower",
        "leopard",
        "lion",
        "lizard",
        "lobster",
        "man",
        "maple",
        "motorcycle",
        "mountain",
        "mouse",
        "mushroom",
        "oak",
        "orange",
        "orchid",
        "otter",
        "palm",
        "pear",
        "pickup_truck",
        "pine",
        "plain",
        "plate",
        "poppy",
        "porcupine",
        "possum",
        "rabbit",
        "raccoon",
        "ray",
        "road",
        "rocket",
        "rose",
        "sea",
        "seal",
        "shark",
        "shrew",
        "skunk",
        "skyscraper",
        "snail",
        "snake",
        "spider",
        "squirrel",
        "streetcar",
        "sunflower",
        "sweet_pepper",
        "table",
        "tank",
        "telephone",
        "television",
        "tiger",
        "tractor",
        "train",
        "trout",
        "tulip",
        "turtle",
        "wardrobe",
        "whale",
        "willow",
        "wolf",
        "woman",
        "worm",
    ],
    "mnist": [str(i) for i in range(10)],
    "fashionmnist": [
        "T-shirt",
        "Trouser",
        "Pullover",
        "Dress",
        "Coat",
        "Sandal",
        "Shirt",
        "Sneaker",
        "Bag",
        "Ankle boot",
    ],
}


def detect_device(preferred: Optional[str] = None) -> str:
    if preferred:
        preferred = preferred.lower()
        if preferred in ("cpu", "cuda"):
            if preferred == "cuda" and not torch.cuda.is_available():
                print("CUDA requested but not available. Falling back to CPU.")
                return "cpu"
            return preferred
    return "cuda" if torch.cuda.is_available() else "cpu"


def build_eval_loader(cfg: Dict[str, Any], processed_root: str):
    dataset_name = cfg.get("dataset", {}).get("name", "CIFAR10")
    batch_size = cfg.get("dataset", {}).get("eval_batch_size", 256)
    num_workers = cfg.get("dataset", {}).get("num_workers", 4)
    transform = make_transforms_from_config(dataset_name, cfg, train=False)
    loader = build_val_dataloader(
        processed_root=processed_root,
        dataset_name=dataset_name,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        transform=transform,
    )
    return loader


def compute_overall_accuracy(per_class: Dict[int, Dict[str, float]], totals: Dict[int, int], key: str) -> float:
    total_samples = 0
    total_correct = 0.0
    for idx, info in per_class.items():
        count = totals.get(idx, 0)
        total_samples += count
        total_correct += info[key] * count
    return float(total_correct) / float(max(1, total_samples))


def get_class_names(dataset_name: str) -> List[str]:
    name = dataset_name.lower()
    for key, names in CLASS_NAME_LOOKUP.items():
        if key in name:
            return names
    return [str(i) for i in range(100)]


def parse_args():
    parser = argparse.ArgumentParser(description="Run full federated unlearning attack pipeline")
    parser.add_argument("--config", "-c", type=str, required=True, help="Path to pipeline YAML config")
    parser.add_argument("--device", type=str, default=None, help="Optional device override (cpu/cuda)")
    parser.add_argument("--processed-root", type=str, default="data/processed", help="Processed data root")
    parser.add_argument("--output-root", type=str, default="outputs/pipeline", help="Where to store run outputs")
    parser.add_argument("--scenario", type=str, default=None, help="Run only the specified scenario by name")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed")
    return parser.parse_args()


def run_scenario(
    scenario_cfg: Dict[str, Any],
    base_output: str,
    processed_root: str,
    device: str,
) -> Dict[str, Any]:
    dataset_name = scenario_cfg.get("dataset", {}).get("name", "CIFAR10")
    model_name = scenario_cfg.get("model", {}).get("backbone", "resnet20")
    scenario_name = scenario_cfg.get("scenario_name", "scenario")

    scenario_out = os.path.join(base_output, scenario_name)
    ensure_dir(scenario_out)
    logger = setup_logging(scenario_out, log_name=f"pipeline_{scenario_name}")
    logger.info(f"Running scenario '{scenario_name}' on dataset={dataset_name}, model={model_name}")

    # ---------------- Train baseline federated model ----------------
    train_cfg = copy.deepcopy(scenario_cfg)
    train_rounds = train_cfg.get("federated", {}).get("rounds")

    server = Server(train_cfg, processed_root=processed_root, output_dir=scenario_out, device=device)
    train_metrics = server.run()
    model_before_state = copy.deepcopy(server.global_state)
    models_dir = os.path.join(server.out_dir, "models")
    ensure_dir(models_dir)
    before_path = os.path.join(models_dir, "model_before.pth")
    save_checkpoint({"state_dict": model_before_state, "meta": {"scenario": scenario_name, "phase": "before"}}, before_path)
    logger.info(f"Saved model_before to {before_path}")

    # ---------------- Unlearning phase ----------------
    unlearning_cfg = copy.deepcopy(scenario_cfg)
    forget_classes = unlearning_cfg.get("unlearning", {}).get("target_class")
    if forget_classes is None:
        forget_classes = unlearning_cfg.get("unlearning", {}).get("target_classes", [])
    if isinstance(forget_classes, int):
        forget_classes = [forget_classes]
    forget_classes = [int(c) for c in (forget_classes or [])]

    unlearn_rounds = unlearning_cfg.get("unlearning", {}).get("rounds", 10)
    unlearning_cfg.setdefault("dataset", {})["exclude_classes"] = forget_classes
    unlearning_cfg.setdefault("federated", {})["rounds"] = unlearn_rounds
    unlearning_cfg.setdefault("experiment", {}).setdefault("name", scenario_name + "_unlearning")

    unlearn_server = Server(
        unlearning_cfg,
        processed_root=processed_root,
        output_dir=scenario_out,
        device=device,
        initial_state_dict=model_before_state,
    )
    unlearn_metrics = unlearn_server.run()
    model_after_state = copy.deepcopy(unlearn_server.global_state)
    after_path = os.path.join(unlearn_server.out_dir, "models", "model_after.pth")
    ensure_dir(os.path.dirname(after_path))
    save_checkpoint({"state_dict": model_after_state, "meta": {"scenario": scenario_name, "phase": "after"}}, after_path)
    logger.info(f"Saved model_after to {after_path}")

    # ---------------- Build evaluation loader ----------------
    eval_loader = build_eval_loader(scenario_cfg, processed_root)

    # ---------------- Load models for evaluation ----------------
    model_before = get_model(model_name, dataset_name, device=device)
    model_after = get_model(model_name, dataset_name, device=device)
    model_before.load_state_dict(model_before_state)
    model_after.load_state_dict(model_after_state)

    class_names = get_class_names(dataset_name)
    attack_dir = os.path.join(unlearn_server.out_dir, "attack")
    ensure_dir(attack_dir)

    li_topk = scenario_cfg.get("attack", {}).get("label_inference", {}).get("top_k", 3)
    label_inference_res = confidence_label_inference(
        model_before=model_before,
        model_after=model_after,
        test_loader=eval_loader,
        device=device,
        top_k=li_topk,
        save_dir=attack_dir,
        class_names=class_names,
    )

    predicted = label_inference_res.get("predicted_forgotten", [])
    top1 = predicted[0] if predicted else None
    label_success = top1 == forget_classes[0] if forget_classes else False
    totals = {int(k): int(v) for k, v in label_inference_res["total_samples_per_class"].items()}
    per_class = {int(k): v for k, v in label_inference_res["per_class"].items()}
    acc_before = compute_overall_accuracy(per_class, totals, "acc_before")
    acc_after = compute_overall_accuracy(per_class, totals, "acc_after")

    # ---------------- Reconstruction ----------------
    recon_cfg = copy.deepcopy(scenario_cfg.get("attack", {}).get("reconstruction", {}))
    num_images = int(recon_cfg.pop("num_images", 16))
    recon_cfg["device"] = device
    reconstructor = ClassConditionalReconstructor(model_before, model_after, dataset_name, config=recon_cfg)
    recon_dir = os.path.join(unlearn_server.out_dir, "reconstructions")
    recon_result = reconstructor.reconstruct(target_class=forget_classes[0], num_images=num_images, save_dir=recon_dir)

    summary = {
        "scenario": scenario_name,
        "dataset": dataset_name,
        "model": model_name,
        "train_rounds": train_rounds,
        "unlearning_rounds": unlearn_rounds,
        "forgotten_classes": forget_classes,
        "label_inference_top1": int(top1) if top1 is not None else None,
        "label_inference_success": bool(label_success),
        "accuracy_before": acc_before,
        "accuracy_after": acc_after,
        "attack_dir": attack_dir,
        "reconstruction_dir": recon_dir,
        "train_metrics": train_metrics,
        "unlearning_metrics": unlearn_metrics,
        "label_inference": label_inference_res,
        "reconstruction": recon_result,
    }

    save_json(summary, os.path.join(unlearn_server.out_dir, "scenario_summary.json"))
    logger.info(f"Scenario '{scenario_name}' complete. Summary written to {unlearn_server.out_dir}")

    return summary


def main():
    args = parse_args()
    cfg = load_yaml(args.config)

    device = detect_device(args.device or cfg.get("experiment", {}).get("device"))
    seed = args.seed if args.seed is not None else cfg.get("experiment", {}).get("seed", 42)
    set_seed(seed)

    scenarios = cfg.pop("scenarios", [])
    if not scenarios:
        scenarios = [{"name": "baseline"}]

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    exp_name = cfg.get("experiment", {}).get("name", "federated_pipeline")
    base_output = os.path.join(args.output_root, f"{timestamp}__{exp_name}")
    ensure_dir(base_output)
    logger = setup_logging(base_output, log_name="pipeline_master")
    logger.info(f"Pipeline start. Seed={seed}, device={device}")

    all_summaries: List[Dict[str, Any]] = []

    for idx, scenario in enumerate(scenarios):
        if not isinstance(scenario, dict):
            continue
        scenario_name = scenario.get("name") or f"scenario_{idx}"
        if args.scenario and args.scenario != scenario_name:
            continue

        scenario_overrides = copy.deepcopy(scenario)
        scenario_overrides.pop("name", None)
        scenario_cfg = deep_merge(copy.deepcopy(cfg), scenario_overrides)
        scenario_cfg["scenario_name"] = scenario_name
        # remove nested scenarios to avoid recursion
        scenario_cfg.pop("scenarios", None)

        summary = run_scenario(scenario_cfg, base_output, args.processed_root, device)
        all_summaries.append(summary)

    summary_path = os.path.join(base_output, "master_summary.json")
    save_json({"seed": seed, "device": device, "summaries": all_summaries}, summary_path)
    logger.info(f"All scenarios complete. Master summary: {summary_path}")


if __name__ == "__main__":
    main()
