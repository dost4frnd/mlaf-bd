"""Attention-feature detectors.

  EnsembleDetector    : XGBoost (dominant) + MLP on the full R^84 flat features.
  SingleLayerDetector : XGBoost on the last layer only (ablation / prior-work proxy).
  XuHistogramDetector : XGBoost on the final-layer attention histogram (Xu et al.).

All three are pure sklearn / xgboost and carry no torch dependency, so they are
fully exercised by the analysis-level smoke test.
"""
from __future__ import annotations
from typing import Optional
import numpy as np
import xgboost as xgb
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


def _make_xgb(cfg=None, scale_pos_weight: float = 1.0) -> xgb.XGBClassifier:
    n = getattr(cfg, "xgb_estimators", 300)
    d = getattr(cfg, "xgb_depth", 6)
    lr = getattr(cfg, "xgb_lr", 0.05)
    seed = getattr(cfg, "seed", 42)
    return xgb.XGBClassifier(
        n_estimators=n, max_depth=d, learning_rate=lr,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss", tree_method="hist",
        random_state=seed, n_jobs=-1,
    )


def _spw(y: np.ndarray) -> float:
    y = np.asarray(y)
    pos = max(int((y == 1).sum()), 1)
    neg = max(int((y == 0).sum()), 1)
    return neg / pos


class EnsembleDetector:
    """0.6 * XGBoost + 0.4 * MLP on standardised flat features."""

    def __init__(self, cfg=None, xgb_weight: Optional[float] = None):
        self.cfg = cfg
        self.xgb_weight = xgb_weight if xgb_weight is not None else getattr(cfg, "ensemble_xgb_weight", 0.6)
        self.scaler = StandardScaler()
        self.xgb: Optional[xgb.XGBClassifier] = None
        self.mlp: Optional[MLPClassifier] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "EnsembleDetector":
        Xs = self.scaler.fit_transform(X)
        self.xgb = _make_xgb(self.cfg, scale_pos_weight=_spw(y))
        self.xgb.fit(Xs, y)
        self.mlp = MLPClassifier(
            hidden_layer_sizes=(256, 128, 64), activation="relu",
            alpha=1e-4, learning_rate_init=1e-3, max_iter=300,
            early_stopping=False, random_state=getattr(self.cfg, "seed", 42),
        )
        self.mlp.fit(Xs, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        Xs = self.scaler.transform(X)
        p_x = self.xgb.predict_proba(Xs)[:, 1]
        p_m = self.mlp.predict_proba(Xs)[:, 1]
        return self.xgb_weight * p_x + (1.0 - self.xgb_weight) * p_m


class SingleLayerDetector:
    """XGBoost on a single layer's 7 features (default: last layer)."""

    def __init__(self, cfg=None, num_layers: int = 12, num_features: int = 7, layer: int = -1):
        self.cfg = cfg
        self.num_layers = num_layers
        self.num_features = num_features
        self.layer = layer
        self.scaler = StandardScaler()
        self.clf: Optional[xgb.XGBClassifier] = None

    def _slice(self, X: np.ndarray) -> np.ndarray:
        return X.reshape(-1, self.num_layers, self.num_features)[:, self.layer, :]

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SingleLayerDetector":
        Xl = self.scaler.fit_transform(self._slice(X))
        self.clf = _make_xgb(self.cfg, scale_pos_weight=_spw(y))
        self.clf.fit(Xl, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.clf.predict_proba(self.scaler.transform(self._slice(X)))[:, 1]


class XuHistogramDetector:
    """Prior work (Xu et al.): classify the final-layer attention histogram."""

    def __init__(self, cfg=None):
        self.cfg = cfg
        self.scaler = StandardScaler()
        self.clf: Optional[xgb.XGBClassifier] = None

    def fit(self, H: np.ndarray, y: np.ndarray) -> "XuHistogramDetector":
        Hs = self.scaler.fit_transform(H)
        self.clf = _make_xgb(self.cfg, scale_pos_weight=_spw(y))
        self.clf.fit(Hs, y)
        return self

    def predict_proba(self, H: np.ndarray) -> np.ndarray:
        return self.clf.predict_proba(self.scaler.transform(H))[:, 1]
