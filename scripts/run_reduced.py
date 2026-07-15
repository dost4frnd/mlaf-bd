#!/usr/bin/env python3
"""Lightweight driver for reproducing individual cells on constrained hardware.

Identical pipeline to run_all.py --core, but with a memory-safe batch size and a
single seed by default, so a cell completes on a small laptop GPU in a few hours.
This is a convenience path for constrained hardware; it is NOT what produced the
shipped results. The shipped results/, figures and paper come from the full
Phase-6 matrix (all datasets/attacks/backbones, three seeds) via
`scripts/run_all.py --seeds 42 43 44` — see REPRODUCE.md. Resolution, dataset
size, epochs and the Neural Cleanse / mitigation fidelity are left at the paper
defaults (224 px, full data). See paper Sec IV.

  python scripts/run_reduced.py                       # all 4 core, seed 42
  python scripts/run_reduced.py --only mnist_badnets  # a single combo
  python scripts/run_reduced.py --seeds 42 43 44      # add seeds if time allows
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from mlafbd.config import Config
from run_experiment import run_experiment

CORE = [("mnist", "badnets"), ("mnist", "blend"),
        ("cifar10", "badnets"), ("cifar10", "blend")]


def build(dataset: str, attack: str, a) -> Config:
    cfg = Config(
        dataset=dataset, attack=attack, backbone="vit_tiny_patch16_224",
        img_size=a.img_size, batch_size=a.batch,
        epochs=(a.mnist_epochs if dataset == "mnist" else a.epochs),
        train_cap=a.train_cap, detector_samples=a.det,
        nc_steps=a.nc_steps, nc_max_classes=a.nc_classes,
        mitigation_epochs=a.mit_epochs, pretrained=True,
        asr_gate="warn", device=a.device,
        save_checkpoints=a.save_ckpt, load_checkpoints=a.load_ckpt,
    )
    cfg.seeds = tuple(a.seeds)
    return cfg.finalise()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--only", nargs="*", default=None,
                   help="subset like: mnist_badnets cifar10_blend")
    p.add_argument("--seeds", type=int, nargs="*", default=[42])
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--batch", type=int, default=8)      # memory-safe on 2 GB @224px
    p.add_argument("--epochs", type=int, default=8)     # cifar
    p.add_argument("--mnist_epochs", type=int, default=5)
    p.add_argument("--train_cap", type=int, default=0)  # 0 = full split
    p.add_argument("--det", type=int, default=2000)
    p.add_argument("--nc_steps", type=int, default=200)
    p.add_argument("--nc_classes", type=int, default=0)  # 0 = all classes
    p.add_argument("--mit_epochs", type=int, default=2)
    p.add_argument("--device", default="auto")
    p.add_argument("--save-ckpt", dest="save_ckpt", action="store_true",
                   help="persist trained victim weights to checkpoints/")
    p.add_argument("--load-ckpt", dest="load_ckpt", action="store_true",
                   help="reuse checkpoints/ if present and skip training")
    return p.parse_args()


def main():
    a = parse_args()
    combos = CORE
    if a.only:                                   # honour the requested order
        valid = set(CORE)
        combos = [t for t in (tuple(x.split("_", 1)) for x in a.only) if t in valid]
    print(f"run_reduced: {len(combos)} combo(s) {combos}  seeds={a.seeds} "
          f"img={a.img_size} batch={a.batch}")
    for d, at in combos:
        cfg = build(d, at, a)
        try:
            run_experiment(cfg)
        except Exception as e:
            print(f"  [skip] {cfg.tag}: {e}")


if __name__ == "__main__":
    main()
