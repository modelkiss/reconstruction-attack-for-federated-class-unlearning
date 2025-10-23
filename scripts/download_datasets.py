#!/usr/bin/env python3
"""
download_datasets.py

Download public datasets used in the project:
  - CIFAR10, CIFAR100, MNIST, FashionMNIST

Saves raw downloaded data (Torchvision-managed) under:
  data/raw/<DATASET>/

Usage:
  python scripts/download_datasets.py --datasets CIFAR10 CIFAR100 MNIST
  python scripts/download_datasets.py --all
"""

import argparse
import os
from pathlib import Path
import torchvision
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100, MNIST, FashionMNIST


AVAILABLE = {"CIFAR10", "CIFAR100", "MNIST", "FashionMNIST"}


def download_dataset(name: str, root: Path, download: bool = True):
    root = root.expanduser()
    root.mkdir(parents=True, exist_ok=True)
    print(f"[+] Downloading {name} -> {root} (download={download})")

    if name == "CIFAR10":
        CIFAR10(root=root, train=True, download=download)
        CIFAR10(root=root, train=False, download=download)
    elif name == "CIFAR100":
        CIFAR100(root=root, train=True, download=download)
        CIFAR100(root=root, train=False, download=download)
    elif name == "MNIST":
        MNIST(root=root, train=True, download=download)
        MNIST(root=root, train=False, download=download)
    elif name == "FashionMNIST":
        FashionMNIST(root=root, train=True, download=download)
        FashionMNIST(root=root, train=False, download=download)
    else:
        raise ValueError(f"Unknown dataset: {name}")

    print(f"[+] Done: {name}")


def parse_args():
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--datasets", nargs="+", help="Datasets to download", choices=list(AVAILABLE))
    g.add_argument("--all", action="store_true", help="Download all supported datasets")
    p.add_argument("--out", type=str, default="data/raw", help="Root folder to store raw datasets")
    p.add_argument("--no-download", action="store_true", help="Don't actually download (useful to check paths)")
    return p.parse_args()


def main():
    args = parse_args()
    root = Path(args.out)

    wants = list(AVAILABLE) if args.all else args.datasets
    for name in wants:
        outdir = root / name
        # torchvision will create appropriate internal files; we rely on its download flag
        download_dataset(name, outdir, download=not args.no_download)


if __name__ == "__main__":
    main()
