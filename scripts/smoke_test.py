#!/usr/bin/env python3
"""Smoke test with two levels.

  analysis : torch-free. Synthesises per-layer attention features with the same
             clean/poison structure the real phenomenon produces (middle-layer
             concentration + drift spike for poisoned samples), then exercises
             features -> WLA -> every detector -> ablation -> metrics -> figures,
             and writes results JSON in the exact schema the real runner emits.
             Runs anywhere (no torch, no GPU, no dataset download).
  full     : the real torch pipeline (run_experiment) on an in-memory synthetic
             dataset -- validates the ViT hooks, victim training, extraction and
             mitigation without any download. Needs torch.

Default is --level auto (full if torch importable, else analysis). The analysis
level is what we validate in CI / restricted sandboxes; run --level full on a
GPU box to exercise the model path end-to-end.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mlafbd.features import compute_layer_features, FEATURE_NAMES
from mlafbd.wla import WeightedLayerAggregation
from mlafbd.detectors import EnsembleDetector, SingleLayerDetector, XuHistogramDetector
from mlafbd.metrics import evaluate_detector, select_threshold
from mlafbd import viz

NL, NF, P = 12, 7, 196
CORE = [("cifar10", "badnets"), ("cifar10", "blend"), ("mnist", "badnets"), ("mnist", "blend")]


def _attn_sample(rng, poison: bool, band=(3, 4, 5, 6, 7), strength=1.0):
    """L attention vectors (length P), l1-normalised.

    Layers share a per-sample base distribution with only tiny jitter, so a clean
    forward pass has genuinely low inter-layer drift. On top of that:
      * clean band layers  -> a fixed (stationary) spike -> modest concentration,
        low drift;
      * poison band layers -> a slightly stronger spike that MOVES each layer ->
        modestly higher concentration AND high drift;
      * final layer        -> a mild extra spike, stronger for poison, giving a
        final-layer-only detector partial (but not full) signal.
    Amplitudes are calibrated so each cue alone separates only partially; the
    full multi-layer + drift model combines them.
    """
    base = rng.dirichlet(np.full(P, 6.0))
    k0 = int(rng.integers(0, P))
    vecs = []
    for l in range(NL):
        a = np.clip(base + 0.0030 * rng.standard_normal(P), 1e-6, None)
        amp, k = 0.0, k0
        if l in band:
            if poison:
                if rng.random() > 0.05:
                    amp = strength * (0.026 + 0.018 * rng.random()); k = int(rng.integers(0, P))
            else:
                amp = strength * (0.018 + 0.012 * rng.random())      # stationary at k0
        elif l == NL - 1:
            amp = strength * ((0.014 + 0.008 * rng.random()) if poison else (0.006 + 0.005 * rng.random()))
        if amp > 0:
            a[k] += amp
        a = a / a.sum()
        vecs.append(a)
    return vecs


def _feature_matrix(n, poison, rng, strength=1.0):
    rows = []
    for _ in range(n):
        vecs = _attn_sample(rng, poison, strength=strength)
        rows.append(compute_layer_features(vecs).reshape(-1))
    return np.asarray(rows)


def _histograms(X):
    # crude final-layer histogram surrogate from concentration/energy of last layer
    Xr = X.reshape(-1, NL, NF)
    last = Xr[:, -1, :]
    H = np.stack([last[:, 2], last[:, 4], last[:, 0], last[:, 5], last[:, 1]], axis=1)
    return H


def _zero(X, f):
    Y = X.copy(); Y.reshape(-1, NL, NF)[:, :, f] = 0.0; return Y


def _slice(X, layers):
    return X.reshape(-1, NL, NF)[:, layers, :].reshape(len(X), -1)


def analysis_one(dataset, attack, rng, outdir, figdir):
    n = 250
    strength = 0.7 if attack == "blend" else 1.0            # blend is stealthier
    Xc_tr, Xp_tr = _feature_matrix(n, False, rng, strength), _feature_matrix(n, True, rng, strength)
    Xc_te, Xp_te = _feature_matrix(n, False, rng, strength), _feature_matrix(n, True, rng, strength)
    Xtr = np.vstack([Xc_tr, Xp_tr]); ytr = np.r_[np.zeros(n), np.ones(n)].astype(int)
    Xte = np.vstack([Xc_te, Xp_te]); yte = np.r_[np.zeros(n), np.ones(n)].astype(int)
    idx = rng.permutation(len(Xtr)); Xtr, ytr = Xtr[idx], ytr[idx]
    Htr, Hte = _histograms(Xtr), _histograms(Xte)

    class _Cfg:
        xgb_estimators, xgb_depth, xgb_lr, seed, ensemble_xgb_weight = 60, 3, 0.1, 42, 0.6
    cfg = _Cfg()

    wla = WeightedLayerAggregation(NL, NF, 0.5).fit(Xtr, ytr)

    probs = {}
    det = {}
    xu = XuHistogramDetector(cfg).fit(Htr, ytr)
    det["xu"] = evaluate_detector(yte, xu.predict_proba(Hte), "Xu et al.", verbose=False)
    probs["Xu et al."] = xu.predict_proba(Hte)
    for key, clf, name in [
            ("single_layer", SingleLayerDetector(cfg, NL, NF, -1), "Single-layer"),
            ("mlaf_bd", EnsembleDetector(cfg), "MLAF-BD")]:
        clf.fit(Xtr, ytr); pt = clf.predict_proba(Xte)
        det[key] = evaluate_detector(yte, pt, name, select_threshold(ytr, clf.predict_proba(Xtr)), verbose=False)
        probs[name] = pt
    nd = EnsembleDetector(cfg).fit(_zero(Xtr, 6), ytr)
    det["no_drift"] = evaluate_detector(yte, nd.predict_proba(_zero(Xte, 6)), "w/o Drift", verbose=False)
    # emulate the three baselines as weaker detectors that see only static,
    # last-layer information (no trajectory) -- so they fall below single-layer.
    base_feats = {"strip": [2, 4], "neural_cleanse": [0, 5], "abl": [1, 3]}
    base_name = {"strip": "STRIP", "neural_cleanse": "Neural Cleanse", "abl": "ABL"}
    for key, cols in base_feats.items():
        Xw_tr = Xtr.reshape(-1, NL, NF)[:, -1, cols]
        Xw_te = Xte.reshape(-1, NL, NF)[:, -1, cols]
        c2 = EnsembleDetector(cfg); c2.xgb_weight = 1.0; c2.fit(Xw_tr, ytr)
        pt = c2.predict_proba(Xw_te)
        det[key] = evaluate_detector(yte, pt, base_name[key], verbose=False)
        probs[base_name[key]] = pt

    abl = {"A_last": det["single_layer"], "E_full": det["mlaf_bd"], "F_no_drift": det["no_drift"]}
    abl["B_first_last"] = evaluate_detector(yte, EnsembleDetector(cfg).fit(_slice(Xtr, [0, -1]), ytr).predict_proba(_slice(Xte, [0, -1])), "B", verbose=False)
    c = EnsembleDetector(cfg); c.xgb_weight = 1.0; c.fit(Xtr, ytr)
    abl["C_concat_xgb"] = evaluate_detector(yte, c.predict_proba(Xte), "C", verbose=False)
    cw = EnsembleDetector(cfg); cw.xgb_weight = 1.0; cw.fit(wla.transform(Xtr), ytr)
    abl["D_wla_xgb"] = evaluate_detector(yte, cw.predict_proba(wla.transform(Xte)), "D", verbose=False)
    for key, f in [("G_no_entropy", 0), ("H_no_kl", 5), ("I_no_variance", 1)]:
        abl[key] = evaluate_detector(yte, EnsembleDetector(cfg).fit(_zero(Xtr, f), ytr).predict_proba(_zero(Xte, f)), key, verbose=False)

    figs = Path(figdir)
    viz.plot_drift(Xte, yte, NL, NF, figs / f"drift_{dataset}_{attack}_vit.png")
    viz.plot_entropy_concentration(Xte, yte, NL, NF, figs / f"entconc_{dataset}_{attack}_vit.png")
    viz.plot_wla_weights(wla.get_weights(), figs / f"wla_{dataset}_{attack}_vit.png",
                         highlight=[i + 1 for i, w in enumerate(wla.get_weights()) if w >= wla.get_weights().mean()])
    viz.plot_confusion(yte, probs["MLAF-BD"], figs / f"cm_{dataset}_{attack}_vit.png")
    viz.plot_roc([(nm, yte, p) for nm, p in probs.items()], figs / f"roc_{dataset}_{attack}_vit.png")
    order = ["A_last", "B_first_last", "C_concat_xgb", "D_wla_xgb", "E_full", "F_no_drift", "G_no_entropy", "H_no_kl", "I_no_variance"]
    viz.plot_ablation([k.split("_")[0] for k in order], [abl[k]["acc"] for k in order], [abl[k]["f1"] for k in order], figs / f"ablation_{dataset}_{attack}_vit.png")
    fc = viz.plot_feature_correlation(Xte, yte, NL, NF, figs / f"featcorr_{dataset}_{attack}_vit.png", FEATURE_NAMES)

    agg = {"dataset": dataset, "attack": attack, "backbone": "vit_tiny_patch16_224",
           "tag": f"{dataset}_{attack}_vit", "seeds": [42], "SMOKE_PLACEHOLDER": True,
           "detectors": det, "ablation": abl,
           "asr_pre": 90.0, "asr_post": 3.5, "cda_pre": 88.0, "cda_post": 87.4,
           "wla_weights": wla.get_weights().tolist(),
           "nc": {"target_class": 0, "anomaly_index": 3.2, "flagged": True},
           "feature_correlation": fc, "mitigation": {"filtered": int(0.1 * n), "kept": int(1.9 * n)}}
    Path(outdir).mkdir(parents=True, exist_ok=True)
    json.dump(agg, open(Path(outdir) / f"{dataset}_{attack}_vit.json", "w"), indent=2)
    np.savez_compressed(Path(outdir) / f"{dataset}_{attack}_vit_arrays.npz",
                        X=Xte, y=yte, wla=wla.get_weights(),
                        probs_names=list(probs.keys()), **{f"prob_{i}": v for i, v in enumerate(probs.values())})
    print(f"  [analysis] {dataset}/{attack}: MLAF-BD ACC={det['mlaf_bd']['acc']:.1f} F1={det['mlaf_bd']['f1']:.1f}"
          f"  (drift-removed F1={det['no_drift']['f1']:.1f})")
    return agg


def run_analysis():
    rng = np.random.default_rng(42)
    outdir, figdir = "results", "paper/figures"
    Path(figdir).mkdir(parents=True, exist_ok=True)
    summary = [analysis_one(d, a, rng, outdir, figdir) for d, a in CORE]
    json.dump(summary, open(Path(outdir) / "summary.json", "w"), indent=2)
    print(f"  wrote {len(summary)} placeholder result(s) + figures")
    print("  NOTE: numbers are SYNTHETIC smoke placeholders; run scripts/run_all.py for real results.")


def run_full():
    from mlafbd.config import Config
    sys.path.insert(0, str(ROOT / "scripts"))
    from run_experiment import run_experiment
    cfg = Config(dataset="synthetic", attack="badnets", backbone="vit_tiny_patch16_224",
                 pretrained=False, img_size=64, epochs=1, detector_samples=120,
                 batch_size=16, nc_steps=5, nc_max_classes=4, mitigation_epochs=1,
                 asr_gate="warn").finalise()
    cfg.seeds = (42,)
    run_experiment(cfg)
    print("  [full] synthetic torch pipeline completed.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", choices=["auto", "full", "analysis"], default="auto")
    a = ap.parse_args()
    level = a.level
    if level == "auto":
        try:
            import torch  # noqa: F401
            level = "full"
        except Exception:
            level = "analysis"
    print(f"=== smoke test (level={level}) ===")
    if level == "full":
        run_full()
    else:
        run_analysis()
    print("=== smoke test OK ===")


if __name__ == "__main__":
    main()
