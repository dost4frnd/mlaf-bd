"""Dataset loading, transform split (pre-norm vs normalise), and index splits."""
from __future__ import annotations
from typing import List, Tuple
import numpy as np
import torch
import torchvision.transforms as T
from torchvision.datasets import CIFAR10, MNIST
from sklearn.model_selection import train_test_split

# per-channel normalisation applied AFTER the trigger
_NORM = {
    "mnist": ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    "cifar10": ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    "gtsrb": ((0.3403, 0.3121, 0.3214), (0.2724, 0.2608, 0.2669)),
    "tinyimagenet": ((0.4802, 0.4481, 0.3975), (0.2770, 0.2691, 0.2821)),
}


def normalize_for(dataset: str) -> T.Normalize:
    mean, std = _NORM.get(dataset, ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
    return T.Normalize(mean, std)


def prenorm_transform(dataset: str, img_size: int) -> T.Compose:
    ops = [T.Resize((img_size, img_size))]
    if dataset == "mnist":
        ops.append(T.Grayscale(num_output_channels=3))
    ops.append(T.ToTensor())                     # -> [0,1], no normalisation here
    return T.Compose(ops)


def load_raw(dataset: str, root: str, img_size: int, train: bool = True):
    """Return a dataset yielding ([0,1] tensor, int label)."""
    import os
    if dataset == "synthetic":
        return SyntheticDataset(n=512 if train else 128, img_size=img_size, num_classes=10)
    tf = prenorm_transform(dataset, img_size)
    if dataset == "cifar10":
        # Prefer torchvision's pickle format if present; otherwise fall back to a
        # locally-cached HuggingFace parquet (used when the toronto mirror is
        # throttled). Both yield identical ([0,1] tensor, label) samples.
        tv_dir = os.path.join(root, "cifar-10-batches-py")
        pq_path = os.path.join(root, "cifar10_parquet",
                               f"{'train' if train else 'test'}-00000-of-00001.parquet")
        if not os.path.isdir(tv_dir) and os.path.isfile(pq_path):
            return CIFAR10Parquet(pq_path, tf)
        return CIFAR10(root=root, train=train, download=True, transform=tf)
    if dataset == "mnist":
        return MNIST(root=root, train=train, download=True, transform=tf)
    if dataset == "gtsrb":
        from torchvision.datasets import GTSRB
        return GTSRB(root=root, split="train" if train else "test", download=True, transform=tf)
    if dataset == "tinyimagenet":
        from torchvision.datasets import ImageFolder
        import os
        sub = "train" if train else "val"
        return ImageFolder(os.path.join(root, sub), transform=tf)
    raise ValueError(f"unknown dataset: {dataset}")


def split_indices(n: int, cfg) -> Tuple[List[int], List[int], List[int]]:
    idx = list(range(n))
    tr, rest = train_test_split(idx, train_size=cfg.train_ratio, random_state=cfg.seed)
    vr = cfg.val_ratio / (cfg.val_ratio + cfg.test_ratio)
    va, te = train_test_split(rest, train_size=vr, random_state=cfg.seed)
    return tr, va, te


def labels_of(dataset) -> np.ndarray:
    """Best-effort label vector without decoding images (used for clean-label poison)."""
    for attr in ("targets", "labels", "_labels"):
        if hasattr(dataset, attr):
            return np.asarray(getattr(dataset, attr))
    if hasattr(dataset, "samples"):                       # ImageFolder / GTSRB(list)
        return np.asarray([s[1] for s in dataset.samples])
    return np.asarray([int(dataset[i][1]) for i in range(len(dataset))])


class CIFAR10Parquet(torch.utils.data.Dataset):
    """Offline CIFAR-10 from the HuggingFace `uoft-cs/cifar10` parquet export.

    Used only when the torchvision pickle download is unavailable/throttled.
    Decodes the PNG-encoded images once into a uint8 (N,32,32,3) array and applies
    the same pre-normalisation transform, so downstream code is unchanged. Exposes
    `.targets` for the clean-label poison path (`labels_of`).
    """
    def __init__(self, parquet_path: str, transform):
        import io
        import pyarrow.parquet as pq
        from PIL import Image
        tbl = pq.read_table(parquet_path)
        cols = tbl.column_names
        img_col = "img" if "img" in cols else ("image" if "image" in cols else cols[0])
        lab_col = "label" if "label" in cols else ("fine_label" if "fine_label" in cols else cols[-1])
        recs = tbl.column(img_col).to_pylist()
        labels = tbl.column(lab_col).to_pylist()
        arr = np.empty((len(recs), 32, 32, 3), dtype=np.uint8)
        for i, rec in enumerate(recs):
            b = rec["bytes"] if isinstance(rec, dict) else rec
            arr[i] = np.asarray(Image.open(io.BytesIO(b)).convert("RGB"), dtype=np.uint8)
        self.data = arr
        self.targets = [int(x) for x in labels]
        self.transform = transform

    def __len__(self): return len(self.targets)

    def __getitem__(self, i):
        from PIL import Image
        return self.transform(Image.fromarray(self.data[i])), int(self.targets[i])


class SyntheticDataset(torch.utils.data.Dataset):
    """In-memory learnable synthetic images (no download) for the full smoke test.

    Each class has a distinct low-frequency blob so a ViT can reach non-trivial
    clean accuracy, letting a patch trigger produce a measurable ASR.
    """
    def __init__(self, n: int = 512, img_size: int = 64, num_classes: int = 10, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.targets = [int(i % num_classes) for i in range(n)]
        yy, xx = np.meshgrid(np.linspace(0, 1, img_size), np.linspace(0, 1, img_size), indexing="ij")
        self.imgs = []
        for i in range(n):
            c = self.targets[i]
            cx, cy = (0.2 + 0.6 * (c % 3) / 2), (0.2 + 0.6 * (c // 3) / 3)
            blob = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / 0.03)
            img = np.stack([blob, np.roll(blob, c, axis=0), np.roll(blob, c, axis=1)], 0)
            img = 0.7 * img + 0.3 * rng.random((3, img_size, img_size))
            self.imgs.append(torch.tensor(img, dtype=torch.float32).clamp(0, 1))

    def __len__(self): return len(self.imgs)

    def __getitem__(self, i): return self.imgs[i], self.targets[i]
