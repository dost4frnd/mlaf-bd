"""Publication-quality figures. Pure numpy/matplotlib (no torch), so fully
exercised by the analysis-level smoke test. Every figure is regenerated from
saved features/scores by scripts/make_figures.py.
"""
from __future__ import annotations
from typing import Dict, List, Sequence, Tuple
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, confusion_matrix

plt.rcParams.update({
    "figure.dpi": 300, "savefig.dpi": 300, "font.size": 10,
    "axes.grid": True, "grid.alpha": 0.3, "axes.spines.top": False,
    "axes.spines.right": False, "font.family": "serif",
})
CLEAN, POISON = "#2c7fb8", "#d95f0e"


def _reshape(X, nl, nf):
    return np.asarray(X).reshape(-1, nl, nf)


def _layerwise(X, y, nl, nf, fidx):
    Xr = _reshape(X, nl, nf); y = np.asarray(y)
    c, p = Xr[y == 0][:, :, fidx], Xr[y == 1][:, :, fidx]
    def ms(a): return (a.mean(0), a.std(0)) if len(a) else (np.zeros(nl), np.zeros(nl))
    return ms(c), ms(p)


def plot_drift(X, y, nl, nf, out):
    (cm, cs), (pm, ps) = _layerwise(X, y, nl, nf, 6)
    L = np.arange(1, nl + 1)
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.plot(L, cm, "-o", color=CLEAN, label="Clean", ms=4)
    ax.fill_between(L, cm - cs, cm + cs, color=CLEAN, alpha=0.15)
    ax.plot(L, pm, "-s", color=POISON, label="Poisoned", ms=4)
    ax.fill_between(L, pm - ps, pm + ps, color=POISON, alpha=0.15)
    ax.set_xlabel("Transformer layer $l$"); ax.set_ylabel(r"Attention Drift $\|a^{(l)}-a^{(l-1)}\|_1$")
    ax.set_xticks(L); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(out); plt.close(fig)


def plot_entropy_concentration(X, y, nl, nf, out):
    L = np.arange(1, nl + 1)
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.2))
    for ax, fidx, ylab in ((axes[0], 0, "Entropy $H^{(l)}$"), (axes[1], 2, "Concentration $C^{(l)}$")):
        (cm, cs), (pm, ps) = _layerwise(X, y, nl, nf, fidx)
        ax.plot(L, cm, "-o", color=CLEAN, label="Clean", ms=3)
        ax.fill_between(L, cm - cs, cm + cs, color=CLEAN, alpha=0.15)
        ax.plot(L, pm, "-s", color=POISON, label="Poisoned", ms=3)
        ax.fill_between(L, pm - ps, pm + ps, color=POISON, alpha=0.15)
        ax.set_xlabel("Layer $l$"); ax.set_ylabel(ylab); ax.set_xticks(L[::2])
    axes[0].legend(frameon=False)
    fig.tight_layout(); fig.savefig(out); plt.close(fig)


def plot_wla_weights(weights, out, highlight: Sequence[int] = ()):
    w = np.asarray(weights); L = np.arange(1, len(w) + 1)
    fig, ax = plt.subplots(figsize=(6, 3.4))
    colors = [POISON if (i + 1) in set(highlight) else CLEAN for i in range(len(w))]
    ax.bar(L, w, color=colors)
    ax.axhline(1.0 / len(w), ls="--", color="gray", lw=1, label="uniform $1/N$")
    ax.set_xlabel("Transformer layer $l$"); ax.set_ylabel("WLA weight $w_l$")
    ax.set_xticks(L); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(out); plt.close(fig)


def plot_confusion(y_true, y_prob, out, threshold: float = 0.5):
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    cm = confusion_matrix(np.asarray(y_true).astype(int), y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    im = ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center",
                color="white" if v > cm.max() / 2 else "black")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["clean", "poison"]); ax.set_yticklabels(["clean", "poison"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    fig.colorbar(im, fraction=0.046); fig.tight_layout(); fig.savefig(out); plt.close(fig)


def plot_roc(named: List[Tuple[str, np.ndarray, np.ndarray]], out):
    fig, ax = plt.subplots(figsize=(4.4, 4.2))
    for name, yt, yp in named:
        yt = np.asarray(yt).astype(int)
        if len(np.unique(yt)) < 2:
            continue
        fpr, tpr, _ = roc_curve(yt, yp)
        ax.plot(fpr, tpr, lw=1.6, label=name)
    ax.plot([0, 1], [0, 1], "--", color="gray", lw=1)
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(out); plt.close(fig)


def plot_ablation(names: Sequence[str], acc: Sequence[float], f1: Sequence[float], out):
    x = np.arange(len(names)); w = 0.38
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    ax.bar(x - w / 2, acc, w, label="ACC", color=CLEAN)
    ax.bar(x + w / 2, f1, w, label="F1", color=POISON)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("Score (%)"); ax.set_ylim(min(70, min(list(acc) + list(f1)) - 3), 100)
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(out); plt.close(fig)


def plot_feature_correlation(X, y, nl, nf, out, feature_names: Sequence[str]):
    Xr = _reshape(X, nl, nf); y = np.asarray(y).astype(float)
    corr = np.zeros(nf)
    for f in range(nf):
        vals = []
        for l in range(nl):
            col = Xr[:, l, f]
            if np.std(col) > 1e-8 and np.std(y) > 1e-8:
                vals.append(abs(np.corrcoef(col, y)[0, 1]))
        corr[f] = np.mean(vals) if vals else 0.0
    order = np.argsort(corr)[::-1]
    fig, ax = plt.subplots(figsize=(6, 3.4))
    ax.bar(np.arange(nf), corr[order], color=CLEAN)
    ax.set_xticks(np.arange(nf))
    ax.set_xticklabels([feature_names[i] for i in order], rotation=25, ha="right", fontsize=8)
    ax.set_ylabel(r"mean $|\rho|$ with poison label")
    fig.tight_layout(); fig.savefig(out); plt.close(fig)
    return {feature_names[i]: float(corr[i]) for i in range(nf)}
