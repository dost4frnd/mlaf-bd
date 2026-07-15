"""Detector metrics, threshold selection, and victim ASR/CDA."""
from __future__ import annotations
from typing import Dict, Tuple
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
)


def evaluate_detector(y_true: np.ndarray, y_prob: np.ndarray, name: str,
                      threshold: float = 0.5, verbose: bool = True) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    both = len(np.unique(y_true)) > 1
    m = {
        "method": name,
        "acc": 100.0 * accuracy_score(y_true, y_pred),
        "pre": 100.0 * precision_score(y_true, y_pred, zero_division=0),
        "rec": 100.0 * recall_score(y_true, y_pred, zero_division=0),
        "f1": 100.0 * f1_score(y_true, y_pred, zero_division=0),
        "auc": float(roc_auc_score(y_true, y_prob)) if both else float("nan"),
        "threshold": float(threshold),
    }
    if verbose:
        print(f"    {name:24s} ACC={m['acc']:5.1f}  F1={m['f1']:5.1f}  AUC={m['auc']:.3f}")
    return m


def select_threshold(y_val: np.ndarray, p_val: np.ndarray, grid: int = 99) -> float:
    """Pick the probability threshold that maximises validation F1."""
    y_val = np.asarray(y_val).astype(int)
    if len(np.unique(y_val)) < 2:
        return 0.5
    best_t, best_f1 = 0.5, -1.0
    for t in np.linspace(0.01, 0.99, grid):
        f1 = f1_score(y_val, (p_val >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t


def compute_asr_cda(model, clean_loader, triggered_loader, device, target_class: int
                    ) -> Tuple[float, float]:
    """CDA on clean test data; ASR on triggered *non-target* inputs."""
    import torch
    model.eval()
    c_ok = c_n = 0
    a_ok = a_n = 0
    with torch.no_grad():
        for imgs, labels in clean_loader:
            preds = model(imgs.to(device)).argmax(1).cpu().numpy()
            labels = labels.numpy()
            c_ok += int((preds == labels).sum()); c_n += len(labels)
        for imgs, labels in triggered_loader:
            preds = model(imgs.to(device)).argmax(1).cpu().numpy()
            labels = labels.numpy()
            mask = labels != target_class          # proper ASR excludes native-target samples
            a_ok += int((preds[mask] == target_class).sum()); a_n += int(mask.sum())
    cda = 100.0 * c_ok / max(c_n, 1)
    asr = 100.0 * a_ok / max(a_n, 1)
    return cda, asr
