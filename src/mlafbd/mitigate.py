"""Two-stage mitigation: filter flagged samples, then attention-regularised FT.

Stage 2 penalises per-layer CLS-attention concentration that exceeds the clean
reference, discouraging the model from re-learning trigger-focused attention on
any residual poisoned samples that survive filtering.
"""
from __future__ import annotations
from typing import Dict, List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from .model import extract_layer_features
from .datasets import collate_flag
from .features import EPS


@torch.no_grad()
def reference_concentration(model, clean_loader, device: str, layers) -> Dict[int, float]:
    model.eval()
    acc = {l: [] for l in layers}
    for imgs, _lab in clean_loader:
        model(imgs.to(device))
        for l in layers:
            cls = model.attention_maps[l].mean(dim=1)[:, 0, 1:]
            cls = cls / (cls.sum(dim=1, keepdim=True) + EPS)
            acc[l].append(float(cls.max(dim=1).values.mean().item()))
        if all(len(v) >= 8 for v in acc.values()):
            break
    return {l: float(np.mean(v)) if v else 0.0 for l, v in acc.items()}


def _concentration_penalty(model, layers, ref: Dict[int, float], device) -> torch.Tensor:
    pen = torch.zeros((), device=device)
    for l in layers:
        aw = model.attention_maps.get(l)
        if aw is None:
            continue
        cls = aw.mean(dim=1)[:, 0, 1:]
        cls = cls / (cls.sum(dim=1, keepdim=True) + EPS)
        conc = cls.max(dim=1).values.mean()
        pen = pen + F.relu(conc - ref.get(l, 0.0))
    return pen


def mitigate(model, train_set, detector, clean_loader, cfg, device: str) -> Dict:
    loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=False,
                        collate_fn=collate_flag)
    X, _y = extract_layer_features(model, loader, device)
    probs = detector.predict_proba(X)
    keep = np.where(probs < 0.5)[0].tolist()
    removed = len(probs) - len(keep)
    print(f"  mitigation: filtered {removed}/{len(probs)} samples")

    ref = reference_concentration(model, clean_loader, device, cfg.mitigation_layers)
    sub = Subset(train_set, keep)
    ft_loader = DataLoader(sub, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_flag)

    model.train(); model.capture_grad = True
    opt = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=cfg.weight_decay)
    crit = nn.CrossEntropyLoss()
    for ep in range(cfg.mitigation_epochs):
        for imgs, labels, _fl in tqdm(ft_loader, desc=f"mitigate {ep+1}/{cfg.mitigation_epochs}",
                                      leave=False):
            imgs, labels = imgs.to(device), labels.to(device)
            opt.zero_grad()
            logits = model(imgs)
            loss = crit(logits, labels) + cfg.mitigation_lambda * \
                _concentration_penalty(model, cfg.mitigation_layers, ref, device)
            loss.backward(); opt.step()
    model.capture_grad = False
    return {"filtered": int(removed), "kept": int(len(keep)), "conc_ref": ref}
