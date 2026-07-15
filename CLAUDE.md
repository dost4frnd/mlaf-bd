# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

MLAF-BD is the code + LaTeX manuscript for a paper on detecting backdoors in
Vision Transformers from the **whole attention stack**. The pipeline: ViT
attention hooks → 7 per-layer features → correlation-based Weighted Layer
Aggregation (WLA) → XGBoost+MLP ensemble detector → two-stage mitigation. It is a
research artifact, not a service: `scripts/` drive experiments, `results/*.json`
are the source of truth, and `paper/` is generated from them.

## Commands

```bash
make smoke        # torch-free analysis path: features→WLA→detectors→figures→tables
make smoke-full   # full torch pipeline on an in-memory synthetic dataset (no downloads)
make core         # 4 core experiments (CIFAR-10/MNIST × BadNets/Blend), 1 seed
make all          # full Phase-6 matrix, seeds 42/43/44 (long)
make tables       # results/*.json      → paper/tables/*.tex + macros.tex
make figures      # results/*_arrays.npz → paper/figures/*.png
make paper        # build paper/main.pdf (pdflatex + bibtex ×3)
make env          # write results/env.json (repro provenance)
```

Single experiment (either a config or CLI flags; flags override the config):

```bash
python scripts/run_experiment.py --config configs/gtsrb_badnets.yaml
python scripts/run_experiment.py --dataset cifar10 --attack blend \
       --backbone deit_small_patch16_224 --seeds 42 43 44 --epochs 8
```

Tests are **torch-free** (only exercise `features` and `wla`):

```bash
pytest -q tests/                                        # all
pytest -q tests/test_core.py::test_drift_zero_for_identical_layers   # one test
python -m py_compile src/mlafbd/*.py                    # syntax check every module
```

Install: `pip install -r requirements.txt`, then a torch build for your machine
(`requirements.txt` deliberately omits a pinned torch/CUDA wheel).

## The two-tier design (most important thing to know)

The code splits cleanly into a **torch-free analysis core** and a **torch model
path**. This split drives everything, including how you test:

- **Analysis core (no torch):** `features`, `wla`, `detectors`, `metrics`, `viz`,
  `config`. Fully exercised by `make smoke` / `smoke_test.py --level analysis` and
  by `pytest`. Develop and validate these anywhere — no GPU, no dataset download.
- **Model path (needs torch):** `model` (ViT hooks), `train_victim`, `attacks`,
  `datasets`, `data`, `baselines`, `mitigate`. Exercised only by
  `smoke_test.py --level full` and the real experiment scripts.

`smoke_test.py --level analysis` synthesizes attention *features* with the same
clean/poison structure the real ViT produces (low clean drift, middle-layer
poison drift), so the whole analysis path is testable without ever running a model.

## Data flow through one experiment

`scripts/run_experiment.py::run_seed` is the spine. Per seed:

1. Poison victim training data, train victim with **plain cross-entropy**
   (`train_victim`), then an **ASR/CDA gate** (`compute_asr_cda`): if attack
   success rate < `asr_gate_threshold`, `asr_gate=warn` continues, `abort` raises.
2. `AttentionHookModel` (`model.py`) monkeypatches timm's per-block attention
   `forward` to capture the post-softmax weights **without changing outputs**
   (written defensively for timm version drift). `extract_layer_features` turns
   captured CLS-attention vectors `(L, B, P)` into a flat **`(N, L*7)`** feature
   matrix; `extract_histograms` produces the Xu et al. final-layer baseline.
3. `WeightedLayerAggregation` scores each layer by mean |Pearson corr| with the
   poison label on the **validation** split, softmax-normalized.
4. Detectors + an A–I ablation ladder + faithful baselines (STRIP, Neural
   Cleanse, ABL) are fit and evaluated.
5. `mitigate` filters flagged samples and fine-tunes with an attention-
   concentration penalty; ASR/CDA measured again post-mitigation.
6. Aggregate mean±std over `cfg.seeds`; write `results/<tag>.json`,
   `results/<tag>_arrays.npz` (primary seed only), and figures.

## Conventions that recur across files

- **Flat feature layout:** everything reshapes `(N, L*7)` ↔ `(N, 12, 7)`. The
  feature axis order is fixed by `features.FEATURE_NAMES`:
  `0 entropy, 1 variance, 2 concentration, 3 sparsity, 4 energy, 5 kl_div, 6 drift`.
  These integer indices are hard-referenced in ablations (`_zero_feat(..., 6)` =
  remove drift). **Drift (index 6) is the paper's novel feature** — the L1 change
  in CLS attention between consecutive layers.
- **Label discipline (`datasets.py`):** victim-training poison labels are *never*
  reused as detector labels. Detection ground truth is the per-sample
  clean-vs-triggered `flag`; splits are asserted balanced (`assert_balanced`).
  Violating this is the "degenerate split" bug the design guards against.
- **Triggers applied pre-normalization** in `[0,1]` pixel space (`attacks.py`),
  consistently at train and inference — required for non-degenerate ASR.
- **Baselines share one per-sample scoring interface** (`baselines.py`): STRIP is
  natively per-sample; Neural Cleanse (model-level) and ABL (training-time
  isolation) get documented per-sample adapters so all methods land in one table.
- **Config = dataclass + YAML merge:** `Config.load(path)` merges `configs/base.yaml`
  then the override YAML; `finalise()` auto-fills `tag` and `num_classes`. Every
  experiment YAML overrides only what it needs (see `configs/cifar10_badnets.yaml`).

## Results → paper (nothing is hard-coded in the manuscript)

`make_tables.py` reads `results/*.json` and writes `paper/tables/*.tex` (table
bodies) plus `paper/tables/macros.tex` (every headline number the prose uses).
`make_figures.py` reads `results/*_arrays.npz`. Tables always emit with a `--`
fallback, so `paper/main.tex` compiles before any experiment has run.

**The shipped `results/*.json`, `paper/figures/*.png` and `paper/main.pdf` are
REAL and complete** — the full Phase-6 matrix: all three datasets
(MNIST/CIFAR-10/GTSRB) × four attacks (BadNets/Blend/clean-label/sample-specific)
× two backbones (ViT-tiny, DeiT-S), 24 configs each averaged over three seeds
(42/43/44), on a 24 GB GPU (see `REPRODUCE.md`). Every JSON carries
`seeds: [42, 43, 44]` with per-metric mean and std, and none carry
`"SMOKE_PLACEHOLDER"`. The torch-free `smoke_test.py --level analysis` path still
writes synthetic placeholders (tagged `"SMOKE_PLACEHOLDER": true`) if you re-run
it, so avoid overwriting the real results with `make smoke`. To rebuild the whole
matrix: `make all` then `make tables && make figures`. The paper reports over
three seeds and carries no single-seed / "Results scope" caveat.

## Gotchas

- `pretrained: true` (default) fine-tunes a pretrained ViT so the backdoor
  actually takes. Training from scratch (`--scratch` / `pretrained: false`) with a
  tiny trigger gives ASR≈0 and a degenerate detection problem — that path exists
  on purpose to reproduce the failure, not as a default.
- `run_all.py` caps MNIST epochs at 5 and, without `--core`, sweeps a large
  matrix — start with `make core` to validate before scaling up.
