"""Faithful reimplementations of three standard defences, adapted to a common
per-sample scoring interface so they populate the same comparison table.

  STRIP  (Gao et al., ACSAC'19) -- native per-sample: entropy of predictions
          under strong clean-image superposition; low entropy => poisoned.
  Neural Cleanse (Wang et al., S&P'19) -- model-level trigger reverse
          engineering + MAD anomaly index; per-sample adapter = softmax
          confidence in the recovered target class (documented approximation;
          NC is fundamentally a model-level detector).
  ABL    (Li et al., NeurIPS'21) -- loss-based separation. Faithful training
          isolation is provided in `abl_isolate`; the per-sample eval adapter
          scores by victim cross-entropy against the *original* label.

Each *_poison_scores returns a raw score where higher == more poison-like;
`calibrate` maps raw scores to [0,1] probabilities via a 1-D logistic fit on
the validation split so thresholds and AUC are computed identically to MLAF-BD.
"""
from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression


def calibrate(s_val: np.ndarray, y_val: np.ndarray, s_test: np.ndarray
              ) -> Tuple[np.ndarray, np.ndarray]:
    s_val = np.asarray(s_val, dtype=np.float64).reshape(-1, 1)
    s_test = np.asarray(s_test, dtype=np.float64).reshape(-1, 1)
    if len(np.unique(y_val)) < 2:
        lo, hi = s_val.min(), s_val.max() + 1e-9
        return ((np.clip(s_val, lo, hi) - lo) / (hi - lo)).ravel(), \
               ((np.clip(s_test, lo, hi) - lo) / (hi - lo)).ravel()
    lr = LogisticRegression(max_iter=1000)
    lr.fit(s_val, np.asarray(y_val).astype(int))
    return lr.predict_proba(s_val)[:, 1], lr.predict_proba(s_test)[:, 1]


# --------------------------------------------------------------------------- STRIP
@torch.no_grad()
def strip_poison_scores(model, loader, clean_pool: torch.Tensor, device: str,
                        n_overlay: int = 8, max_samples: Optional[int] = None):
    model.eval()
    pool = clean_pool.to(device)
    scores: List[float] = []
    flags: List[int] = []
    n = 0
    for imgs, _lab, fl in loader:
        imgs = imgs.to(device)
        for b in range(imgs.size(0)):
            if max_samples is not None and n >= max_samples:
                break
            idx = torch.randint(0, pool.size(0), (n_overlay,), device=device)
            mix = 0.5 * imgs[b:b + 1] + 0.5 * pool[idx]
            probs = F.softmax(model(mix), dim=1).clamp_min(1e-8)
            ent = float((-(probs * probs.log()).sum(1)).mean().item())
            scores.append(-ent)                       # low entropy -> high poison score
            flags.append(int(fl[b])); n += 1
        if max_samples is not None and n >= max_samples:
            break
    return np.asarray(scores), np.asarray(flags, dtype=np.int32)


# ------------------------------------------------------------------- Neural Cleanse
def neural_cleanse(model, num_classes: int, sample_batch: torch.Tensor, device: str,
                   steps: int = 200, lam: float = 1e-3, max_classes: int = 0) -> Dict:
    model.eval()
    x = sample_batch.to(device)
    C, H, W = x.shape[1:]
    classes = list(range(num_classes))
    if max_classes and max_classes < num_classes:
        classes = classes[:max_classes]
    l1_norms: Dict[int, float] = {}
    best = {"target": classes[0], "mask": None, "pattern": None, "l1": float("inf")}
    for t in classes:
        m_raw = torch.zeros((1, H, W), device=device, requires_grad=True)
        p_raw = torch.zeros((C, H, W), device=device, requires_grad=True)
        opt = torch.optim.Adam([m_raw, p_raw], lr=0.1)
        tgt = torch.full((x.size(0),), t, dtype=torch.long, device=device)
        for _ in range(steps):
            m = torch.sigmoid(m_raw)
            p = torch.sigmoid(p_raw)
            xp = (1 - m) * x + m * p
            loss = F.cross_entropy(model(xp), tgt) + lam * m.abs().sum()
            opt.zero_grad(); loss.backward(); opt.step()
        l1 = float(torch.sigmoid(m_raw).abs().sum().item())
        l1_norms[t] = l1
        if l1 < best["l1"]:
            best = {"target": t, "mask": torch.sigmoid(m_raw).detach(),
                    "pattern": torch.sigmoid(p_raw).detach(), "l1": l1}
    vals = np.array(list(l1_norms.values()))
    med = float(np.median(vals))
    mad = float(np.median(np.abs(vals - med))) * 1.4826 + 1e-8
    anomaly_index = (med - best["l1"]) / mad          # >2 => flagged backdoored class
    return {"target_class": int(best["target"]), "anomaly_index": float(anomaly_index),
            "l1_norms": {int(k): float(v) for k, v in l1_norms.items()},
            "flagged": bool(anomaly_index > 2.0),
            "mask": best["mask"], "pattern": best["pattern"]}


@torch.no_grad()
def nc_poison_scores(model, loader, nc_target: int, device: str,
                     max_samples: Optional[int] = None):
    model.eval()
    scores: List[float] = []
    flags: List[int] = []
    n = 0
    for imgs, _lab, fl in loader:
        probs = F.softmax(model(imgs.to(device)), dim=1).cpu().numpy()
        for b in range(imgs.size(0)):
            if max_samples is not None and n >= max_samples:
                break
            scores.append(float(probs[b, nc_target]))   # confidence in recovered target
            flags.append(int(fl[b])); n += 1
        if max_samples is not None and n >= max_samples:
            break
    return np.asarray(scores), np.asarray(flags, dtype=np.int32)


# ---------------------------------------------------------------------------- ABL
@torch.no_grad()
def abl_poison_scores(model, loader, device: str, max_samples: Optional[int] = None):
    """Per-sample eval adapter: CE against the ORIGINAL label (triggered -> high loss)."""
    model.eval()
    scores: List[float] = []
    flags: List[int] = []
    n = 0
    for imgs, labels, fl in loader:
        logits = model(imgs.to(device))
        ce = F.cross_entropy(logits, labels.to(device), reduction="none").cpu().numpy()
        for b in range(imgs.size(0)):
            if max_samples is not None and n >= max_samples:
                break
            scores.append(float(ce[b]))
            flags.append(int(fl[b])); n += 1
        if max_samples is not None and n >= max_samples:
            break
    return np.asarray(scores), np.asarray(flags, dtype=np.int32)


def abl_isolate(model, train_loader, device: str, epochs: int = 2,
                gamma: float = 0.20, isolation_rate: float = 0.10) -> Dict:
    """Faithful ABL isolation: LGA flooding, then isolate the lowest-loss fraction.

    Returns isolation precision/recall vs. the training poison flags. Optional
    (cost: a short auxiliary training run); disabled by default in run_experiment.
    """
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    for _ in range(epochs):
        for imgs, labels, _fl in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            opt.zero_grad()
            loss = F.cross_entropy(model(imgs), labels)
            loss = (loss - gamma).abs() + gamma           # local gradient ascent (flooding)
            loss.backward(); opt.step()
    model.eval()
    losses: List[float] = []
    flags: List[int] = []
    with torch.no_grad():
        for imgs, labels, fl in train_loader:
            ce = F.cross_entropy(model(imgs.to(device)), labels.to(device),
                                 reduction="none").cpu().numpy()
            losses.extend(ce.tolist()); flags.extend([int(f) for f in fl])
    losses = np.asarray(losses); flags = np.asarray(flags)
    k = max(1, int(len(losses) * isolation_rate))
    isolated = np.argsort(losses)[:k]
    iso_mask = np.zeros_like(flags); iso_mask[isolated] = 1
    tp = int(((iso_mask == 1) & (flags == 1)).sum())
    prec = tp / max(int(iso_mask.sum()), 1)
    rec = tp / max(int((flags == 1).sum()), 1)
    return {"isolation_precision": 100.0 * prec, "isolation_recall": 100.0 * rec,
            "n_isolated": int(k)}
