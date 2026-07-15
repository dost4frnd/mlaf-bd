# MLAF-BD: Multi-Layer Attention Fusion Backdoor Detection for Vision Transformers

Detecting backdoors in Vision Transformers from the **whole attention stack**,
not only the final layer. Each transformer layer gives seven features
(entropy, variance, concentration, sparsity, energy, KL divergence, and the new
**Attention Drift**), the layers are fused by a correlation-based **Weighted
Layer Aggregation (WLA)**, and the fused features are classified by an
**XGBoost + MLP** ensemble. A two-stage mitigation filters the flagged samples
and fine-tunes the model with an attention-concentration penalty.

This repository is the code + LaTeX manuscript for the paper
*"Multi-Layer Attention Fusion Backdoor Detection for Vision Transformers via
Inter-Layer Attention Drift Analysis"*. It extends the single-layer,
final-attention histogram method of Xu et al. (ICCC 2025).

## What is in here

```
src/mlafbd/      library: features, wla, detectors, baselines, model (hooks),
                 attacks, datasets, train_victim, mitigate, metrics, viz
scripts/         run_experiment, run_all, make_figures, make_tables,
                 smoke_test, collect_env
configs/         base.yaml + one YAML per dataset x attack (+ deit, + smoke)
paper/           main.tex (IEEEtran journal), refs.bib, figures/, tables/
results/         JSON metrics + *_arrays.npz (full 24-config, 3-seed matrix shipped)
```

## Install

```bash
pip install -r requirements.txt
# install a torch build for your machine, e.g. CUDA 12.1:
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

Tested with Python 3.10-3.12, PyTorch >= 2.1, timm >= 0.9.16, xgboost >= 2.0.

## Quick start

```bash
# 1) torch-free smoke test: features -> WLA -> detectors -> figures -> tables.
#    Runs anywhere, no GPU, no downloads. Writes placeholder results + figures.
make smoke

# 2) full pipeline smoke on an in-memory synthetic dataset (needs torch, no download):
make smoke-full

# 3) a quick sanity subset (CIFAR-10/MNIST x BadNets/Blend), one seed, to
#    validate the pipeline before scaling up (NOT the shipped results):
make core

# 4) the full Phase-6 matrix, three seeds (long) — this reproduces the shipped
#    results/, figures and paper:
make all

# build the paper (fills tables/figures from results/):
make tables && make figures && make paper
```

Single experiment, full control:

```bash
python scripts/run_experiment.py --dataset cifar10 --attack blend \
       --backbone deit_small_patch16_224 --seeds 42 43 44 --epochs 8
# or from a config file:
python scripts/run_experiment.py --config configs/gtsrb_badnets.yaml
```

## How the numbers get into the paper

The manuscript never hard-codes a result. `scripts/make_tables.py` reads
`results/*.json` and writes `paper/tables/*.tex` (table bodies) and
`paper/tables/macros.tex` (every headline number used in the prose). After a
run, `make tables` updates the paper. Figures work the same way through
`scripts/make_figures.py` and `results/*_arrays.npz`.

## Experiments

- **Datasets:** MNIST, CIFAR-10, GTSRB (TinyImageNet via `ImageFolder`).
- **Attacks:** BadNets, Blend, clean-label, sample-specific.
- **Backbones:** `vit_tiny_patch16_224`, `deit_small_patch16_224`.
- **Baselines:** STRIP, Neural Cleanse, ABL (faithful; see the note below), and
  the Xu et al. final-layer histogram.
- **Seeds:** 42, 43, 44; tables report mean over seeds.

## Honest notes (please read)

1. **Make the backdoor real first.** The victim is trained with standard
   cross-entropy and, by default, fine-tuned from a pretrained ViT so the attack
   success rate (ASR) is high. The pipeline checks the ASR before it runs
   detection and warns (or aborts, `asr_gate: abort`) if the backdoor did not
   take. Training a ViT from scratch with a tiny trigger gives ASR near zero and
   a degenerate detection problem; the `--scratch`/`pretrained: false` path
   reproduces that setting on purpose.
2. **STRIP is per-sample. Neural Cleanse and ABL are not.** Neural Cleanse is a
   model-level trigger reverse-engineering method and ABL is a training-time
   isolation method. We report them at their natural level and also give a
   documented per-sample adapter (NC: confidence in the recovered target class;
   ABL: cross-entropy against the original label) so that all methods appear in
   one table. This is stated in `src/mlafbd/baselines.py`.
3. **Shipped results are real and complete.** `results/*.json`,
   `paper/figures/*.png` and `paper/main.pdf` are from a real run of the **full
   Phase-6 matrix**: all three datasets (MNIST, CIFAR-10, GTSRB) × four attacks
   (BadNets, Blend, clean-label, sample-specific) × two backbones
   (`vit_tiny_patch16_224`, `deit_small_patch16_224`) — 24 configurations, each
   averaged over **three seeds (42/43/44)** — produced on a 24 GB GPU. They are
   **not** synthetic placeholders and **not** a reduced single-seed subset. See
   [`REPRODUCE.md`](REPRODUCE.md) for the exact commands and provenance
   (`results/env.json`).
4. **What was validated where.** The torch-free analysis path (features, WLA,
   the three detectors, metrics, figures, tables, LaTeX build) is smoke-tested.
   The torch path (ViT hooks, victim training, extraction, mitigation) runs on
   your GPU via `make smoke-full` and the experiment scripts.

## Reproducibility

`scripts/collect_env.py` writes `results/env.json` (Python, torch/CUDA, GPU, git
commit, `pip freeze`). Seeds are fixed in `src/mlafbd/seed.py`. Set the seeds and
`asr_gate: abort` in the config for release runs.

## Citation

If you use this code, please cite the paper (see `paper/main.tex`) and the base
method (Xu et al., ICCC 2025).
