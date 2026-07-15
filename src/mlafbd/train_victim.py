"""Victim ViT training with standard cross-entropy.

We deliberately train the victim with plain CE -- no attention-contrast loss.
The attacker only poisons data; the defender does not get to shape the victim's
attention. This is a harder, more honest setting than amplifying the clean/
poison attention gap at training time, and it is where MLAF-BD must still work.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


def train_victim(model, loader, epochs: int, device: str,
                 lr: float = 1e-4, weight_decay: float = 1e-4) -> None:
    model.to(device).train()
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
    crit = nn.CrossEntropyLoss()
    for ep in range(epochs):
        tot = correct = seen = 0
        for imgs, labels, _fl in tqdm(loader, desc=f"victim {ep+1}/{epochs}", leave=False):
            imgs, labels = imgs.to(device), labels.to(device)
            opt.zero_grad()
            logits = model(imgs)
            loss = crit(logits, labels)
            loss.backward(); opt.step()
            tot += float(loss.item()) * imgs.size(0)
            correct += int((logits.argmax(1) == labels).sum().item())
            seen += imgs.size(0)
        sched.step()
        print(f"  victim epoch {ep+1}: loss={tot/max(seen,1):.4f} acc={100*correct/max(seen,1):.2f}%")
