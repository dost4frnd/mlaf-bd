#!/usr/bin/env python3
"""Run the full Phase-6 matrix (datasets x attacks x backbones) and summarise.

Cost warning: the default matrix is large. Each (dataset,attack,backbone) is
trained once per seed. Start with --core (4 combos, 1 seed) to validate, then
scale up. Use --datasets/--attacks/--backbones/--seeds to control the sweep.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mlafbd.config import Config
from run_experiment import run_experiment

ALL_DATASETS = ["cifar10", "mnist", "gtsrb"]
ALL_ATTACKS = ["badnets", "blend", "cleanlabel", "samplespecific"]
ALL_BACKBONES = ["vit_tiny_patch16_224", "deit_small_patch16_224"]
CORE = [("cifar10", "badnets"), ("cifar10", "blend"), ("mnist", "badnets"), ("mnist", "blend")]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--core", action="store_true", help="4 core combos, vit_tiny only")
    p.add_argument("--datasets", nargs="*", default=ALL_DATASETS)
    p.add_argument("--attacks", nargs="*", default=ALL_ATTACKS)
    p.add_argument("--backbones", nargs="*", default=["vit_tiny_patch16_224"])
    p.add_argument("--seeds", type=int, nargs="*", default=[42, 43, 44])
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--output_dir", default="results")
    return p.parse_args()


def main():
    a = parse_args()
    combos = []
    if a.core:
        combos = [(d, at, "vit_tiny_patch16_224") for d, at in CORE]
    else:
        for bb in a.backbones:
            for d in a.datasets:
                for at in a.attacks:
                    combos.append((d, at, bb))
    print(f"Running {len(combos)} experiment(s), seeds={a.seeds}")
    summary = []
    for d, at, bb in combos:
        cfg = Config(dataset=d, attack=at, backbone=bb, epochs=a.epochs,
                     output_dir=a.output_dir)
        cfg.seeds = tuple(a.seeds)
        cfg.finalise()
        if d == "mnist":
            cfg.epochs = min(a.epochs, 5)
        try:
            summary.append(run_experiment(cfg))
        except Exception as e:
            print(f"  [skip] {cfg.tag}: {e}")
    Path(a.output_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(a.output_dir) / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\nWrote {Path(a.output_dir) / 'summary.json'} ({len(summary)} results)")


if __name__ == "__main__":
    main()
