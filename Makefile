.PHONY: help smoke smoke-full core all figures tables paper env clean
PY ?= python3

help:
	@echo "make smoke       - torch-free analysis smoke test (features/WLA/detectors/figures/tables)"
	@echo "make smoke-full  - full torch pipeline on an in-memory synthetic dataset (no downloads)"
	@echo "make core        - 4 core experiments (CIFAR-10/MNIST x BadNets/Blend), 1 seed"
	@echo "make all         - full Phase-6 matrix (long; see scripts/run_all.py flags)"
	@echo "make tables      - regenerate paper/tables/*.tex from results/*.json"
	@echo "make figures     - regenerate paper/figures/*.png from results/*_arrays.npz"
	@echo "make paper       - build paper/main.pdf (pdflatex + bibtex)"
	@echo "make env         - write results/env.json"

smoke:
	$(PY) scripts/smoke_test.py --level analysis
	$(PY) scripts/make_tables.py

smoke-full:
	$(PY) scripts/smoke_test.py --level full

core:
	$(PY) scripts/run_all.py --core --seeds 42
	$(PY) scripts/make_tables.py

all:
	$(PY) scripts/run_all.py --seeds 42 43 44
	$(PY) scripts/make_tables.py

figures:
	$(PY) scripts/make_figures.py

tables:
	$(PY) scripts/make_tables.py

env:
	$(PY) scripts/collect_env.py

paper:
	cd paper && pdflatex -interaction=nonstopmode main.tex && bibtex main && \
	pdflatex -interaction=nonstopmode main.tex && pdflatex -interaction=nonstopmode main.tex

clean:
	rm -f paper/*.aux paper/*.log paper/*.out paper/*.bbl paper/*.blg
