# Reproducing the results in this repository

The numbers, tables and figures shipped here are **real measurements** from a
full run of the Phase-6 matrix: all three datasets (MNIST, CIFAR-10, GTSRB) ×
four attacks (BadNets, Blend, clean-label, sample-specific) × two backbones
(`vit_tiny_patch16_224`, `deit_small_patch16_224`) at 224×224 — **24
configurations, each averaged over three seeds (42/43/44)** — produced on a
24 GB GPU. Every `results/*.json` carries `seeds: [42, 43, 44]` with per-metric
mean and std; none are synthetic placeholders.

## Environment used

- Full provenance (Python, PyTorch/CUDA, GPU, git commit, `pip freeze`) is
  captured in [`results/env.json`](results/env.json). Regenerate it on the
  training machine with `make env` so the recorded GPU matches the hardware that
  produced the shipped matrix.

Install:

```bash
pip install -r requirements.txt      # numpy, xgboost, sklearn, timm, matplotlib, pyyaml, tqdm
pip install torch torchvision        # CUDA build for your machine (or the CPU wheel)
pip install pyarrow                   # only needed for the offline CIFAR-10 parquet loader
```

## Data staging

The datasets live under `data/` (git-ignored). They were staged from fast
mirrors because the canonical CIFAR-10 mirror was throttled:

- **MNIST** — official IDX files from the PyTorch S3 mirror
  `https://ossci-datasets.s3.amazonaws.com/mnist/` into `data/MNIST/raw/`
  (torchvision then processes them).
- **CIFAR-10** — HuggingFace parquet
  `https://huggingface.co/datasets/uoft-cs/cifar10/resolve/main/plain_text/{train,test}-00000-of-00001.parquet`
  into `data/cifar10_parquet/`. `mlafbd.data.load_raw` auto-detects this parquet
  cache (via the new `CIFAR10Parquet` loader) when the torchvision pickle format
  is absent, and yields identical `([0,1] tensor, label)` samples.

The pretrained ViT backbone (~22 MB) is downloaded once by `timm` to
`~/.cache/huggingface` and reused.

## Exact commands that produced the shipped results

```bash
# 1) full Phase-6 matrix: all datasets × attacks × backbones, three seeds
python scripts/run_all.py --seeds 42 43 44

# 2) regenerate paper artifacts from results/
make tables && make figures && make paper
```

`make all` is the same thing. This writes one `results/<tag>.json` (mean±std over
the three seeds) and `results/<tag>_arrays.npz` (primary seed) per configuration,
then `make tables && make figures` fills `paper/tables/*.tex` and
`paper/figures/*.png`.

### Lighter path for constrained hardware

If you cannot fit the full sweep, `scripts/run_reduced.py` is a thin driver over
`scripts/run_experiment.py` with a memory-safe batch size (`--batch 8`, <1.3 GB
VRAM at 224 px) and a single seed by default. It is **not** what produced the
shipped results, but it reproduces individual cells cheaply, e.g.:

```bash
python scripts/run_reduced.py --only cifar10_badnets --seeds 42 --epochs 6
```

Key knobs: `--img_size`, `--epochs`, `--mnist_epochs`, `--train_cap` (subsample
victim training), `--det`, `--nc_steps`, `--nc_classes`, `--mit_epochs`.

## Model checkpoints (for future runs)

The victim models are trained in-memory; the completed run above did **not**
persist weights. To save/reuse them on future runs:

```bash
# train once and persist victim weights to checkpoints/<tag>_seed<seed>.pt
python scripts/run_reduced.py --seeds 42 --save-ckpt

# later, skip training and re-run detection/baselines/mitigation from the checkpoint
python scripts/run_reduced.py --seeds 42 --load-ckpt
```

`checkpoints/` is git-ignored (weights are ~22 MB each). This is wired through
`Config.save_checkpoints` / `load_checkpoints` / `checkpoint_dir`.

## Honest caveats baked into this run

- **Three seeds (42/43/44)** — tables report the mean and, where shown, the std.
  Sub-1-point differences are still within noise; the prose avoids leaning on
  them.
- **Mitigation is only a partial defence** at the reported fine-tuning budget
  (residual ASR stays high); this is stated as a limitation in the paper, not
  oversold.
- **Provenance.** `results/env.json` records the exact environment. If it was
  last written on different hardware than the machine that produced the shipped
  matrix, regenerate it there with `make env`.
