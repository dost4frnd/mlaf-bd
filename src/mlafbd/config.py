"""Experiment configuration: dataclass + YAML loader (base <- override merge)."""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Tuple
import yaml

NUM_CLASSES = {"mnist": 10, "cifar10": 10, "gtsrb": 43, "tinyimagenet": 200, "synthetic": 10}


@dataclass
class Config:
    # identity
    dataset: str = "cifar10"          # mnist | cifar10 | gtsrb | tinyimagenet
    attack: str = "badnets"           # badnets | blend | cleanlabel | samplespecific
    backbone: str = "vit_tiny_patch16_224"   # or deit_small_patch16_224
    tag: str = ""                     # auto-filled if empty

    # data / model
    img_size: int = 224
    num_layers: int = 12
    num_features: int = 7
    num_classes: int = 10
    data_root: str = "data"
    tinyimagenet_root: str = "data/tiny-imagenet-200"
    pretrained: bool = True           # fine-tune pretrained ViT (reliable ASR)

    # poisoning
    poison_rate: float = 0.10
    target_class: int = 0
    trigger_size: int = 22            # BadNets patch side on the 224 canvas (>= patch)
    blend_alpha: float = 0.20
    ss_epsilon: float = 0.10          # sample-specific perturbation amplitude

    # victim training (standard CE -- honest threat model)
    epochs: int = 8
    lr: float = 1e-4
    weight_decay: float = 1e-4
    batch_size: int = 64
    num_workers: int = 4
    train_cap: int = 0                # 0 = full training split; else cap the number of
                                      # victim-training samples (hardware-constrained runs)
    save_checkpoints: bool = False    # persist the trained victim weights per (tag, seed)
    load_checkpoints: bool = False    # if a matching checkpoint exists, load it and skip training
    checkpoint_dir: str = "checkpoints"

    # detector splits (of the balanced detection set)
    train_ratio: float = 0.56
    val_ratio: float = 0.14
    test_ratio: float = 0.30
    detector_samples: int = 2000      # balanced clean/triggered samples to featurise
    hist_bins: int = 20               # Xu-style final-layer histogram bins

    # WLA / ensemble
    wla_temperature: float = 0.5
    ensemble_xgb_weight: float = 0.6
    xgb_estimators: int = 300
    xgb_depth: int = 6
    xgb_lr: float = 0.05

    # baselines
    strip_overlays: int = 8
    nc_steps: int = 200
    nc_lambda: float = 1e-3
    nc_max_classes: int = 0           # 0 = all classes; else cap for speed
    abl_isolation_rate: float = 0.10
    abl_gamma: float = 0.20           # LGA flooding level

    # mitigation
    mitigation_lambda: float = 0.01
    mitigation_layers: Tuple[int, ...] = (4, 5, 6, 7)   # 0-indexed -> layers 5..8
    mitigation_epochs: int = 2

    # guards / bookkeeping
    asr_gate: str = "warn"            # warn | abort  (abort skips detection if no backdoor)
    asr_gate_threshold: float = 80.0  # % ASR required to trust the victim backdoor
    seed: int = 42
    seeds: Tuple[int, ...] = (42, 43, 44)
    output_dir: str = "results"
    figures_dir: str = "paper/figures"
    device: str = "auto"              # auto | cuda | cpu

    def finalise(self) -> "Config":
        if not self.tag:
            self.tag = f"{self.dataset}_{self.attack}_{self.backbone.split('_')[0]}"
        self.num_classes = NUM_CLASSES.get(self.dataset, self.num_classes)
        if self.target_class >= self.num_classes:
            self.target_class = 0
        return self

    @staticmethod
    def load(path: str, base: str = "configs/base.yaml") -> "Config":
        merged: dict = {}
        for p in (base, path):
            if p and Path(p).is_file():
                with open(p) as fh:
                    merged.update(yaml.safe_load(fh) or {})
        if "mitigation_layers" in merged:
            merged["mitigation_layers"] = tuple(merged["mitigation_layers"])
        if "seeds" in merged:
            merged["seeds"] = tuple(merged["seeds"])
        known = {f for f in Config().__dataclass_fields__}  # type: ignore[attr-defined]
        merged = {k: v for k, v in merged.items() if k in known}
        return Config(**merged).finalise()

    def to_dict(self) -> dict:
        d = asdict(self)
        d["mitigation_layers"] = list(self.mitigation_layers)
        d["seeds"] = list(self.seeds)
        return d
