"""Backdoor triggers, applied in [0,1] pixel space *before* normalisation.

Applying the trigger pre-normalisation (rather than clamping a normalised
tensor) keeps the perturbation physically meaningful and consistent between
victim training and inference -- one of the fixes for the degenerate ASR seen
in the original pipeline. Four attacks are provided:

  badnets        : patch-aligned white square, bottom-right      (label -> target)
  blend          : fixed pattern alpha-blended over whole image  (label -> target)
  cleanlabel     : badnets patch, but poison is drawn ONLY from
                   target-class images and the label is UNCHANGED (clean-label)
  samplespecific : a per-sample low-frequency perturbation whose
                   pattern is keyed to the sample index (ISSBA-style)

`is_clean_label(attack)` tells the dataset whether poisoning must be restricted
to target-class samples with labels left intact.
"""
from __future__ import annotations
import numpy as np
import torch

_CACHE: dict = {}
CLEAN_LABEL = {"cleanlabel"}


def is_clean_label(attack: str) -> bool:
    return attack in CLEAN_LABEL


def _blend_pattern(c: int, h: int, w: int) -> torch.Tensor:
    key = ("blend", c, h, w)
    if key not in _CACHE:
        rng = np.random.default_rng(1234)                 # fixed, reproducible pattern
        pat = rng.random((c, h, w)).astype(np.float32)
        # low-pass so it reads as a global texture, not pixel noise
        k = max(3, (min(h, w) // 16) | 1)
        pad = k // 2
        t = torch.tensor(pat)[None]
        t = torch.nn.functional.avg_pool2d(
            torch.nn.functional.pad(t, (pad, pad, pad, pad), mode="reflect"), k, stride=1)
        _CACHE[key] = t[0].clamp(0, 1)
    return _CACHE[key]


def apply_trigger(img: torch.Tensor, attack: str, cfg=None, index: int = 0) -> torch.Tensor:
    """img: (C,H,W) float in [0,1]. Returns a triggered copy in [0,1]."""
    c, h, w = img.shape
    out = img.clone()
    if attack in ("badnets", "cleanlabel"):
        s = int(getattr(cfg, "trigger_size", 22))
        s = max(2, min(s, h, w))
        out[:, h - s:, w - s:] = 1.0                       # white patch, patch-aligned
    elif attack == "blend":
        alpha = float(getattr(cfg, "blend_alpha", 0.20))
        out = (1 - alpha) * out + alpha * _blend_pattern(c, h, w)
    elif attack == "samplespecific":
        eps = float(getattr(cfg, "ss_epsilon", 0.10))
        rng = np.random.default_rng(10_000 + int(index))   # per-sample, reproducible
        base = rng.standard_normal((1, h // 8 or 1, w // 8 or 1)).astype(np.float32)
        pert = torch.tensor(base)[None]
        pert = torch.nn.functional.interpolate(pert, size=(h, w), mode="bilinear",
                                               align_corners=False)[0]
        pert = pert / (pert.abs().max() + 1e-8)
        out = out + eps * pert.expand(c, -1, -1)
    else:
        raise ValueError(f"unknown attack: {attack}")
    return out.clamp(0.0, 1.0)
