"""Weighted Layer Aggregation (WLA).

Per-layer importance is scored by the mean absolute Pearson correlation between
that layer's features and the binary poison label on the *validation* split,
then softmax-normalised with a temperature. The result is (a) an interpretable
depth-wise importance chart and (b) an optional compact R^7 fused representation.
"""
from __future__ import annotations
from typing import Optional
import numpy as np

EPS = 1e-8


def _safe_abs_corr(x: np.ndarray, y: np.ndarray) -> float:
    # Degenerate feature or degenerate label -> zero contribution.
    if np.std(x) < EPS or np.std(y) < EPS:
        return 0.0
    with np.errstate(all="ignore"):
        c = np.corrcoef(x, y)[0, 1]
    return 0.0 if not np.isfinite(c) else float(abs(c))


class WeightedLayerAggregation:
    def __init__(self, num_layers: int = 12, num_features: int = 7, temperature: float = 0.5):
        self.num_layers = num_layers
        self.num_features = num_features
        self.temperature = temperature
        self.weights: Optional[np.ndarray] = None
        self.scores: Optional[np.ndarray] = None

    def fit(self, X_flat: np.ndarray, y: np.ndarray) -> "WeightedLayerAggregation":
        X = X_flat.reshape(-1, self.num_layers, self.num_features)
        y = np.asarray(y).astype(np.float64)
        scores = np.zeros(self.num_layers, dtype=np.float64)
        for l in range(self.num_layers):
            scores[l] = np.mean([_safe_abs_corr(X[:, l, f], y) for f in range(self.num_features)])
        self.scores = scores
        z = np.exp(scores / max(self.temperature, EPS))
        self.weights = (z / (z.sum() + EPS)).astype(np.float32)
        return self

    def transform(self, X_flat: np.ndarray) -> np.ndarray:
        if self.weights is None:
            raise RuntimeError("WLA must be fit before transform().")
        X = X_flat.reshape(-1, self.num_layers, self.num_features)
        return np.einsum("l,nlf->nf", self.weights, X).astype(np.float32)

    def get_weights(self) -> np.ndarray:
        if self.weights is None:
            raise RuntimeError("WLA not fitted.")
        return self.weights.copy()
