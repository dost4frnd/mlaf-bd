"""Dataset wrappers for victim poisoning and detector evaluation, plus guards.

Design rule (learned from the original pipeline's degenerate splits):
  * victim-training poison labels are NEVER reused as detector labels;
  * the detector's clean-vs-triggered flag is the detection ground truth;
  * detection splits are asserted to contain both classes near 50/50.
"""
from __future__ import annotations
from typing import List, Sequence, Set
import numpy as np
import torch
from torch.utils.data import Dataset

from .attacks import apply_trigger, is_clean_label
from .data import labels_of


def build_poison_positions(base, indices: Sequence[int], cfg) -> Set[int]:
    """Positions (into `indices`) to poison during victim training."""
    rng = np.random.default_rng(cfg.seed)
    pool = list(range(len(indices)))
    if is_clean_label(cfg.attack):                       # restrict to target-class samples
        y = labels_of(base)
        pool = [p for p in pool if int(y[indices[p]]) == cfg.target_class]
        if not pool:
            raise RuntimeError("clean-label: no samples of target class in split")
    k = max(1, int(len(indices) * cfg.poison_rate))
    k = min(k, len(pool))
    return set(rng.choice(pool, size=k, replace=False).tolist())


class PoisonedTrainDataset(Dataset):
    def __init__(self, base, indices: Sequence[int], poison_positions: Set[int], cfg, normalize):
        self.base, self.indices = base, list(indices)
        self.poison, self.cfg, self.normalize = poison_positions, cfg, normalize
        self.clean_label = is_clean_label(cfg.attack)

    def __len__(self): return len(self.indices)

    def __getitem__(self, pos: int):
        img, label = self.base[self.indices[pos]]
        poison = pos in self.poison
        if poison:
            img = apply_trigger(img, self.cfg.attack, self.cfg, index=self.indices[pos])
            if not self.clean_label:
                label = self.cfg.target_class
        return self.normalize(img), int(label), int(poison)


class DetectionEvalDataset(Dataset):
    """Balanced clean vs. triggered-at-inference samples; flag is detection truth."""

    def __init__(self, base, indices: Sequence[int], cfg, normalize,
                 poison_fraction: float = 0.5, seed_offset: int = 0):
        self.base, self.indices = base, list(indices)
        self.cfg, self.normalize = cfg, normalize
        rng = np.random.default_rng(cfg.seed + seed_offset)
        k = max(1, int(len(self.indices) * poison_fraction))
        self.triggered = set(rng.choice(len(self.indices), size=k, replace=False).tolist())
        assert 0 < len(self.triggered) < len(self.indices), "degenerate detection split"

    def __len__(self): return len(self.indices)

    def __getitem__(self, pos: int):
        img, label = self.base[self.indices[pos]]
        flag = pos in self.triggered
        if flag:
            img = apply_trigger(img, self.cfg.attack, self.cfg, index=self.indices[pos])
        return self.normalize(img), int(label), int(flag)   # label kept = original


class IndexedCleanDataset(Dataset):
    def __init__(self, base, indices, normalize):
        self.base, self.indices, self.normalize = base, list(indices), normalize

    def __len__(self): return len(self.indices)

    def __getitem__(self, pos):
        img, label = self.base[self.indices[pos]]
        return self.normalize(img), int(label)


class TriggeredAllDataset(Dataset):
    """Trigger applied to every sample (for ASR); returns original label."""
    def __init__(self, base, indices, cfg, normalize):
        self.base, self.indices, self.cfg, self.normalize = base, list(indices), cfg, normalize

    def __len__(self): return len(self.indices)

    def __getitem__(self, pos):
        idx = self.indices[pos]
        img, label = self.base[idx]
        img = apply_trigger(img, self.cfg.attack, self.cfg, index=idx)
        return self.normalize(img), int(label)


def collate_flag(batch):
    imgs, labels, flags = zip(*batch)
    return torch.stack(imgs), torch.tensor(labels), torch.tensor(flags)


def collate_plain(batch):
    imgs, labels = zip(*batch)
    return torch.stack(imgs), torch.tensor(labels)


def assert_balanced(y: np.ndarray, name: str) -> None:
    y = np.asarray(y)
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) < 2:
        raise AssertionError(
            f"[{name}] detection labels collapsed to one class {classes.tolist()} "
            f"({counts.tolist()}). Refusing to report vacuous metrics.")
    frac = counts.min() / counts.sum()
    if frac < 0.2:
        print(f"  [warn] {name} class balance skewed: {dict(zip(classes.tolist(), counts.tolist()))}")
