"""Per-layer attention feature engineering.

Seven scalar features are computed per transformer layer from the CLS-token
spatial attention vector a in R^P (already averaged over heads, self-token
dropped, l1-normalised). The definitions match the manuscript exactly:

    entropy        H  = -sum_i a_i log(a_i + eps)
    variance       V  = (1/P) sum_i (a_i - mu)^2 ,  mu = 1/P
    concentration  C  = max_i a_i
    sparsity       S  = (1/P) sum_i 1[a_i < mu/2]
    energy         E  = sum_i a_i^2                (Herfindahl index)
    kl_from_unif   KL = sum_i a_i log(P a_i + eps)
    drift          D  = || a^(l) - a^(l-1) ||_1    (0 for the first layer)

Drift is the primary novel feature: it measures how far the CLS attention
distribution moves between consecutive layers, capturing the abrupt trajectory
change a trigger induces at the depth where it hijacks attention.
"""
from __future__ import annotations
from typing import Sequence
import numpy as np

EPS = 1e-8
FEATURE_NAMES = [
    "entropy", "variance", "concentration",
    "sparsity", "energy", "kl_div", "drift",
]
NUM_FEATURES = len(FEATURE_NAMES)


def _normalise(a: np.ndarray) -> np.ndarray:
    a = np.clip(np.asarray(a, dtype=np.float64), EPS, None)
    s = a.sum()
    return a / (s + EPS)


def compute_layer_features(attn_vectors: Sequence[np.ndarray]) -> np.ndarray:
    """Return an (L, 7) feature matrix for a list of L per-layer attention vectors."""
    L = len(attn_vectors)
    feats = np.zeros((L, NUM_FEATURES), dtype=np.float32)
    prev = None
    for l, raw in enumerate(attn_vectors):
        a = _normalise(raw)
        P = a.shape[0]
        mu = 1.0 / P
        feats[l, 0] = float(-np.sum(a * np.log(a + EPS)))            # entropy
        feats[l, 1] = float(np.var(a))                               # variance
        feats[l, 2] = float(np.max(a))                              # concentration
        feats[l, 3] = float(np.mean(a < (mu / 2.0)))                # sparsity
        feats[l, 4] = float(np.sum(a * a))                          # energy
        feats[l, 5] = float(np.sum(a * np.log(P * a + EPS)))        # KL from uniform
        if prev is None:
            feats[l, 6] = 0.0                                       # drift
        else:
            feats[l, 6] = float(np.sum(np.abs(a - prev)))
        prev = a
    return feats


def flatten(feature_matrix: np.ndarray) -> np.ndarray:
    """(L,7) -> (L*7,) row vector used by the flat detectors."""
    return feature_matrix.reshape(-1)
