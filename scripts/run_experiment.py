#!/usr/bin/env python3
"""End-to-end MLAF-BD experiment for one (dataset, attack, backbone) over seeds.

Pipeline per seed:
  poison victim data -> train victim (plain CE) -> ASR/CDA gate ->
  extract per-layer features + final-layer histograms (val/test) ->
  fit WLA -> train detectors (Xu, single-layer, MLAF full, ablations) ->
  faithful baselines (STRIP, Neural Cleanse, ABL) -> two-stage mitigation.

Metrics are aggregated (mean +/- std) across cfg.seeds. Figures + a compact
result JSON + an arrays .npz (primary seed) are written for downstream tables.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mlafbd.config import Config
from mlafbd.seed import set_seed
from mlafbd import data as D
from mlafbd import datasets as DS
from mlafbd.model import AttentionHookModel, extract_layer_features, extract_histograms
from mlafbd.train_victim import train_victim
from mlafbd.metrics import evaluate_detector, select_threshold, compute_asr_cda
from mlafbd.wla import WeightedLayerAggregation
from mlafbd.detectors import EnsembleDetector, SingleLayerDetector, XuHistogramDetector
from mlafbd import baselines as BL
from mlafbd.mitigate import mitigate
from mlafbd import viz
from mlafbd.features import FEATURE_NAMES


def _device(cfg):
    if cfg.device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return cfg.device


def _zero_feat(X, nl, nf, fidx):
    Y = X.copy(); Y.reshape(-1, nl, nf)[:, :, fidx] = 0.0; return Y


def _slice_layers(X, nl, nf, layers):
    return X.reshape(-1, nl, nf)[:, layers, :].reshape(len(X), -1)


def _fit_eval(clf, Xtr, ytr, Xte, yte, Xval, yval, name, verbose=True):
    clf.fit(Xtr, ytr)
    pv, pt = clf.predict_proba(Xval), clf.predict_proba(Xte)
    thr = select_threshold(yval, pv)
    return evaluate_detector(yte, pt, name, thr, verbose), pt


def run_seed(cfg: Config, seed: int, want_arrays: bool):
    set_seed(seed)
    dev = _device(cfg)
    norm = D.normalize_for(cfg.dataset)
    raw = D.load_raw(cfg.dataset, cfg.data_root, cfg.img_size, train=True)
    tr_idx, va_idx, te_idx = D.split_indices(len(raw), cfg)
    if cfg.train_cap and len(tr_idx) > cfg.train_cap:      # hardware-constrained subset
        tr_idx = tr_idx[:cfg.train_cap]
        print(f"  [train_cap] victim training capped to {len(tr_idx)} samples")
    poison_pos = DS.build_poison_positions(raw, tr_idx, cfg)

    train_set = DS.PoisonedTrainDataset(raw, tr_idx, poison_pos, cfg, norm)
    det_val = DS.DetectionEvalDataset(raw, va_idx, cfg, norm, 0.5, seed_offset=1)
    det_test = DS.DetectionEvalDataset(raw, te_idx, cfg, norm, 0.5, seed_offset=2)
    clean_test = DS.IndexedCleanDataset(raw, te_idx, norm)
    trig_test = DS.TriggeredAllDataset(raw, te_idx, cfg, norm)

    nw = cfg.num_workers
    dl = lambda ds, c: DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=c,
                                  num_workers=nw, pin_memory=(dev == "cuda"),
                                  persistent_workers=False)
    train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True, collate_fn=DS.collate_flag,
                              num_workers=nw, pin_memory=(dev == "cuda"), persistent_workers=False)

    model = AttentionHookModel(cfg.backbone, cfg.num_classes, cfg.pretrained, cfg.img_size).to(dev)
    ckpt = Path(cfg.checkpoint_dir) / f"{cfg.tag}_seed{seed}.pt"
    if cfg.load_checkpoints and ckpt.is_file():
        model.load_state_dict(torch.load(ckpt, map_location=dev))
        print(f"  [ckpt] loaded {ckpt}; skipping victim training")
    else:
        train_victim(model, train_loader, cfg.epochs, dev, cfg.lr, cfg.weight_decay)
        if cfg.save_checkpoints:
            ckpt.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), ckpt)
            print(f"  [ckpt] saved trained victim -> {ckpt}")

    cda_pre, asr_pre = compute_asr_cda(model, dl(clean_test, DS.collate_plain),
                                       dl(trig_test, DS.collate_plain), dev, cfg.target_class)
    print(f"  [seed {seed}] pre-mitigation CDA={cda_pre:.2f}% ASR={asr_pre:.2f}%")
    if asr_pre < cfg.asr_gate_threshold:
        msg = (f"  [gate] ASR {asr_pre:.1f}% < {cfg.asr_gate_threshold}% -- victim backdoor weak.")
        if cfg.asr_gate == "abort":
            raise RuntimeError(msg + " Aborting (asr_gate=abort). Increase epochs/trigger_size.")
        print(msg + " Continuing (asr_gate=warn); detector numbers may be optimistic.")

    cap = cfg.detector_samples
    Xval, yval = extract_layer_features(model, dl(det_val, DS.collate_flag), dev, True, cap)
    Xte, yte = extract_layer_features(model, dl(det_test, DS.collate_flag), dev, True, cap)
    DS.assert_balanced(yval, "detector-val"); DS.assert_balanced(yte, "detector-test")
    Hval, yhv = extract_histograms(model, dl(det_val, DS.collate_flag), dev, cfg.hist_bins, cap)
    Hte, _ = extract_histograms(model, dl(det_test, DS.collate_flag), dev, cfg.hist_bins, cap)

    nl, nf = cfg.num_layers, cfg.num_features
    wla = WeightedLayerAggregation(nl, nf, cfg.wla_temperature).fit(Xval, yval)
    Fval, Fte = wla.transform(Xval), wla.transform(Xte)

    det: dict = {}
    probs: dict = {}
    print("  detectors:")
    xu = XuHistogramDetector(cfg); xu.fit(Hval, yhv)
    pv, pt = xu.predict_proba(Hval), xu.predict_proba(Hte)
    det["xu"], probs["Xu et al."] = evaluate_detector(yte, pt, "Xu et al. (final-layer hist)",
                                                       select_threshold(yhv, pv)), pt
    det["single_layer"], probs["Single-layer"] = _fit_eval(
        SingleLayerDetector(cfg, nl, nf, -1), Xval, yval, Xte, yte, Xval, yval, "Single-layer (last)")
    det["mlaf_bd"], probs["MLAF-BD"] = _fit_eval(
        EnsembleDetector(cfg), Xval, yval, Xte, yte, Xval, yval, "MLAF-BD (full)")
    det["no_drift"], _ = _fit_eval(
        EnsembleDetector(cfg), _zero_feat(Xval, nl, nf, 6), yval,
        _zero_feat(Xte, nl, nf, 6), yte, _zero_feat(Xval, nl, nf, 6), yval, "w/o Attention Drift")

    # ablation ladder A-E and feature ablations F-I
    abl: dict = {}
    abl["A_last"] = det["single_layer"]
    def _xgb_on(Xtr, Xte_, name):
        clf = EnsembleDetector(cfg); clf.xgb_weight = 1.0   # XGB-only
        return _fit_eval(clf, Xtr, yval, Xte_, yte, Xtr, yval, name, verbose=False)
    # B: XGB on the first+last layer slice (14 feats) -- SingleLayerDetector cannot
    # be used here, it assumes the full L*7 layout and would reshape-fail on a slice.
    abl["B_first_last"], _ = _xgb_on(_slice_layers(Xval, nl, nf, [0, -1]),
                                     _slice_layers(Xte, nl, nf, [0, -1]), "  B first+last")
    abl["C_concat_xgb"], _ = _xgb_on(Xval, Xte, "  C concat-xgb")
    abl["D_wla_xgb"], _ = _xgb_on(Fval, Fte, "  D wla-xgb")
    abl["E_full"] = det["mlaf_bd"]
    for key, fidx, nm in [("F_no_drift", 6, "F -drift"), ("G_no_entropy", 0, "G -entropy"),
                          ("H_no_kl", 5, "H -kl"), ("I_no_variance", 1, "I -variance")]:
        abl[key], _ = _fit_eval(EnsembleDetector(cfg), _zero_feat(Xval, nl, nf, fidx), yval,
            _zero_feat(Xte, nl, nf, fidx), yte, _zero_feat(Xval, nl, nf, fidx), yval, "  " + nm, verbose=False)

    # faithful baselines
    print("  baselines:")
    clean_pool = torch.stack([clean_test[i][0] for i in range(min(64, len(clean_test)))])
    sv, fv = BL.strip_poison_scores(model, dl(det_val, DS.collate_flag), clean_pool, dev, cfg.strip_overlays, cap)
    st_, ft_ = BL.strip_poison_scores(model, dl(det_test, DS.collate_flag), clean_pool, dev, cfg.strip_overlays, cap)
    pv, pt = BL.calibrate(sv, fv, st_); _, pt = BL.calibrate(sv, fv, st_)
    det["strip"], probs["STRIP"] = evaluate_detector(ft_, pt, "STRIP", select_threshold(fv, pv)), pt

    nc_batch = torch.stack([clean_test[i][0] for i in range(min(32, len(clean_test)))])
    nc = BL.neural_cleanse(model, cfg.num_classes, nc_batch, dev, cfg.nc_steps, cfg.nc_lambda, cfg.nc_max_classes)
    print(f"    Neural Cleanse: target={nc['target_class']} anomaly_index={nc['anomaly_index']:.2f} flagged={nc['flagged']}")
    sv, fv = BL.nc_poison_scores(model, dl(det_val, DS.collate_flag), nc["target_class"], dev, cap)
    st_, ft_ = BL.nc_poison_scores(model, dl(det_test, DS.collate_flag), nc["target_class"], dev, cap)
    pv, pt = BL.calibrate(sv, fv, st_)
    det["neural_cleanse"], probs["Neural Cleanse"] = evaluate_detector(ft_, pt, "Neural Cleanse (per-sample)", select_threshold(fv, pv)), pt

    sv, fv = BL.abl_poison_scores(model, dl(det_val, DS.collate_flag), dev, cap)
    st_, ft_ = BL.abl_poison_scores(model, dl(det_test, DS.collate_flag), dev, cap)
    pv, pt = BL.calibrate(sv, fv, st_)
    det["abl"], probs["ABL"] = evaluate_detector(ft_, pt, "ABL (per-sample)", select_threshold(fv, pv)), pt

    # mitigation (uses the MLAF-BD detector)
    mlaf_det = EnsembleDetector(cfg).fit(Xval, yval)
    mit = mitigate(model, train_set, mlaf_det, dl(clean_test, DS.collate_plain), cfg, dev)
    cda_post, asr_post = compute_asr_cda(model, dl(clean_test, DS.collate_plain),
                                         dl(trig_test, DS.collate_plain), dev, cfg.target_class)
    print(f"  [seed {seed}] post-mitigation CDA={cda_post:.2f}% ASR={asr_post:.2f}%")

    rec = {"detectors": det, "ablation": abl, "nc": {k: nc[k] for k in ("target_class", "anomaly_index", "flagged")},
           "wla_weights": wla.get_weights().tolist(), "asr_pre": asr_pre, "asr_post": asr_post,
           "cda_pre": cda_pre, "cda_post": cda_post, "mitigation": {k: mit[k] for k in ("filtered", "kept")}}
    arrays = None
    if want_arrays:
        arrays = {"X": Xte, "y": yte, "wla": wla.get_weights(),
                  "probs_names": list(probs.keys()),
                  **{f"prob_{i}": v for i, v in enumerate(probs.values())}}
    return rec, arrays, probs


def _agg(vals):
    a = np.asarray([v for v in vals if v == v])  # drop NaN
    return (float(a.mean()), float(a.std())) if len(a) else (float("nan"), 0.0)


def aggregate(records):
    out = {"detectors": {}, "ablation": {}}
    keys = records[0]["detectors"].keys()
    for k in keys:
        out["detectors"][k] = {"method": records[0]["detectors"][k]["method"]}
        for m in ("acc", "f1", "auc", "pre", "rec"):
            mu, sd = _agg([r["detectors"][k][m] for r in records])
            out["detectors"][k][m] = mu; out["detectors"][k][m + "_std"] = sd
    for k in records[0]["ablation"].keys():
        out["ablation"][k] = {}
        for m in ("acc", "f1", "auc"):
            mu, sd = _agg([r["ablation"][k][m] for r in records])
            out["ablation"][k][m] = mu; out["ablation"][k][m + "_std"] = sd
    for m in ("asr_pre", "asr_post", "cda_pre", "cda_post"):
        out[m], out[m + "_std"] = _agg([r[m] for r in records])
    out["wla_weights"] = np.mean([r["wla_weights"] for r in records], axis=0).tolist()
    out["nc"] = records[0]["nc"]
    return out


def run_experiment(cfg: Config) -> dict:
    print(f"\n{'='*66}\n{cfg.tag}  |  dataset={cfg.dataset} attack={cfg.attack} backbone={cfg.backbone}\n{'='*66}")
    outdir = Path(cfg.output_dir); outdir.mkdir(parents=True, exist_ok=True)
    figdir = Path(cfg.figures_dir); figdir.mkdir(parents=True, exist_ok=True)
    records = []
    primary = None
    for i, seed in enumerate(cfg.seeds):
        rec, arrays, probs = run_seed(cfg, seed, want_arrays=(i == 0))
        records.append(rec)
        if i == 0:
            primary = (arrays, probs, rec)
    agg = aggregate(records)
    agg.update({"dataset": cfg.dataset, "attack": cfg.attack, "backbone": cfg.backbone,
                "tag": cfg.tag, "seeds": list(cfg.seeds), "config": cfg.to_dict()})

    arrays, probs, rec = primary
    np.savez_compressed(outdir / f"{cfg.tag}_arrays.npz", **arrays)
    nl, nf = cfg.num_layers, cfg.num_features
    band = [i + 1 for i, w in enumerate(agg["wla_weights"]) if w >= np.mean(agg["wla_weights"])]
    viz.plot_drift(arrays["X"], arrays["y"], nl, nf, figdir / f"drift_{cfg.tag}.png")
    viz.plot_entropy_concentration(arrays["X"], arrays["y"], nl, nf, figdir / f"entconc_{cfg.tag}.png")
    viz.plot_wla_weights(agg["wla_weights"], figdir / f"wla_{cfg.tag}.png", highlight=band)
    viz.plot_confusion(arrays["y"], probs["MLAF-BD"], figdir / f"cm_{cfg.tag}.png")
    viz.plot_roc([(n, arrays["y"], p) for n, p in probs.items()], figdir / f"roc_{cfg.tag}.png")
    a = agg["ablation"]
    order = ["A_last", "B_first_last", "C_concat_xgb", "D_wla_xgb", "E_full",
             "F_no_drift", "G_no_entropy", "H_no_kl", "I_no_variance"]
    viz.plot_ablation([k.split("_")[0] for k in order], [a[k]["acc"] for k in order],
                      [a[k]["f1"] for k in order], figdir / f"ablation_{cfg.tag}.png")
    agg["feature_correlation"] = viz.plot_feature_correlation(
        arrays["X"], arrays["y"], nl, nf, figdir / f"featcorr_{cfg.tag}.png", FEATURE_NAMES)

    with open(outdir / f"{cfg.tag}.json", "w") as fh:
        json.dump(agg, fh, indent=2)
    print(f"  saved {outdir / (cfg.tag + '.json')} + figures in {figdir}")
    return agg


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="")
    p.add_argument("--dataset", default="cifar10"); p.add_argument("--attack", default="badnets")
    p.add_argument("--backbone", default="vit_tiny_patch16_224")
    p.add_argument("--seeds", type=int, nargs="*", default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--output_dir", default=None)
    return p.parse_args()


def main():
    a = parse_args()
    cfg = Config.load(a.config) if a.config else Config(
        dataset=a.dataset, attack=a.attack, backbone=a.backbone).finalise()
    if a.seeds is not None: cfg.seeds = tuple(a.seeds)
    if a.epochs is not None: cfg.epochs = a.epochs
    if a.output_dir is not None: cfg.output_dir = a.output_dir
    cfg.finalise()
    run_experiment(cfg)


if __name__ == "__main__":
    main()
