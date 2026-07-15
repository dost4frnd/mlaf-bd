# Build & validation status

**What was smoke-tested in a CPU sandbox (no GPU, no dataset download):**
- `scripts/smoke_test.py --level analysis`: features -> WLA -> XGBoost+MLP
  ensemble, single-layer and Xu-histogram detectors, ablation ladder A-I,
  metrics, and all seven figure types, on synthetic attention features that
  reproduce the real clean/poison structure (low clean drift, middle-layer
  poison drift). Emits `results/*.json` in the real schema.
- `scripts/make_tables.py`: LaTeX tables + prose macros.
- `paper/main.tex`: compiles with pdflatex + bibtex to a 5-page PDF
  (`paper/main.pdf` is included).
- All modules pass `python -m py_compile`.

**What runs on your GPU box (identical code path, needs working torch):**
- `scripts/smoke_test.py --level full` (synthetic in-memory dataset, no download)
- `scripts/run_experiment.py`, `scripts/run_all.py` (real datasets)

**Numbers in the shipped paper are REAL and complete** — the full Phase-6 matrix:
all three datasets (MNIST/CIFAR-10/GTSRB) × four attacks (BadNets/Blend/
clean-label/sample-specific) × two backbones (ViT-tiny, DeiT-S) at 224 px, 24
configurations each averaged over three seeds (42/43/44), on a 24 GB GPU. See
`REPRODUCE.md` for exact commands, data provenance and caveats. To rebuild from
scratch on adequate hardware:
```
make all           # python scripts/run_all.py --seeds 42 43 44
make tables && make figures && make paper
```
