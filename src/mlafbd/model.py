"""ViT backbone with non-invasive per-layer attention hooks + feature extraction.

The hook re-implements timm's multi-head attention forward so the post-softmax
weight tensor of every encoder block is captured without changing the forward
result. Written to tolerate timm version drift (q_norm/k_norm/attn_drop/scale
are read defensively, and the patched forward accepts the extra positional
arguments newer timm passes).
"""
from __future__ import annotations
from typing import Dict, List, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from .features import compute_layer_features, EPS


class AttentionHookModel(nn.Module):
    def __init__(self, backbone: str = "vit_tiny_patch16_224",
                 num_classes: int = 10, pretrained: bool = True, img_size: int = 224):
        super().__init__()
        kwargs = dict(pretrained=pretrained, num_classes=num_classes)
        if img_size != 224:
            kwargs["img_size"] = img_size
        self.backbone = timm.create_model(backbone, **kwargs)
        self.num_layers = len(self.backbone.blocks)
        self.attention_maps: Dict[int, torch.Tensor] = {}
        self.capture_grad = False   # True only during attention-regularised mitigation
        self._patch()

    def _patch(self) -> None:
        for idx, block in enumerate(self.backbone.blocks):
            attn = block.attn

            def make(attn_mod, layer_idx):
                def forward(x, *args, **kwargs):
                    B, N, C = x.shape
                    h = attn_mod.num_heads
                    hd = C // h
                    qkv = attn_mod.qkv(x).reshape(B, N, 3, h, hd).permute(2, 0, 3, 1, 4)
                    q, k, v = qkv.unbind(0)
                    q = getattr(attn_mod, "q_norm", nn.Identity())(q)
                    k = getattr(attn_mod, "k_norm", nn.Identity())(k)
                    scale = getattr(attn_mod, "scale", hd ** -0.5)
                    aw = (q @ k.transpose(-2, -1)) * scale
                    aw = aw.softmax(dim=-1)
                    self.attention_maps[layer_idx] = aw if self.capture_grad else aw.detach()
                    aw = getattr(attn_mod, "attn_drop", nn.Identity())(aw)
                    out = (aw @ v).transpose(1, 2).reshape(B, N, C)
                    out = attn_mod.proj(out)
                    out = getattr(attn_mod, "proj_drop", nn.Identity())(out)
                    return out
                return forward

            attn.forward = make(attn, idx)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.attention_maps.clear()
        return self.backbone(x)

    @torch.no_grad()
    def cls_vectors(self) -> np.ndarray:
        """(L, B, P): head-averaged CLS attention over patches, self dropped, l1-normalised."""
        vecs = []
        for l in range(self.num_layers):
            aw = self.attention_maps[l]            # (B, H, N, N)
            cls = aw.mean(dim=1)[:, 0, 1:]          # (B, P)
            cls = cls.clamp_min(EPS)
            cls = cls / cls.sum(dim=1, keepdim=True)
            vecs.append(cls.cpu().numpy())
        return np.stack(vecs, axis=0)               # (L, B, P)


@torch.no_grad()
def extract_layer_features(model: AttentionHookModel, loader, device: str,
                           include_drift: bool = True, max_samples: Optional[int] = None):
    """Return (N, L*7) flat feature matrix and (N,) poison labels."""
    model.eval()
    rows: List[np.ndarray] = []
    labels: List[int] = []
    n = 0
    for imgs, _lab, flags in loader:
        model(imgs.to(device))
        V = model.cls_vectors()                     # (L, B, P)
        B = V.shape[1]
        for b in range(B):
            if max_samples is not None and n >= max_samples:
                break
            feats = compute_layer_features([V[l, b] for l in range(model.num_layers)])
            if not include_drift:
                feats[:, 6] = 0.0
            rows.append(feats.reshape(-1))
            labels.append(int(flags[b]))
            n += 1
        if max_samples is not None and n >= max_samples:
            break
    return np.vstack(rows), np.asarray(labels, dtype=np.int32)


@torch.no_grad()
def extract_histograms(model: AttentionHookModel, loader, device: str,
                       bins: int = 20, max_samples: Optional[int] = None):
    """Final-layer CLS-attention histogram features (Xu et al. baseline)."""
    model.eval()
    rows: List[np.ndarray] = []
    labels: List[int] = []
    n = 0
    last = model.num_layers - 1
    for imgs, _lab, flags in loader:
        model(imgs.to(device))
        aw = model.attention_maps[last].mean(dim=1)[:, 0, 1:]     # (B, P)
        aw = aw.cpu().numpy()
        B = aw.shape[0]
        for b in range(B):
            if max_samples is not None and n >= max_samples:
                break
            a = aw[b]
            a = (a - a.min()) / (a.max() - a.min() + EPS)
            idx = np.clip(np.floor(a * (bins - 1)).astype(int), 0, bins - 1)
            h = np.bincount(idx, minlength=bins).astype(np.float32)
            s = h.sum()
            rows.append(h / s if s > 0 else h)
            labels.append(int(flags[b]))
            n += 1
        if max_samples is not None and n >= max_samples:
            break
    return np.vstack(rows), np.asarray(labels, dtype=np.int32)
