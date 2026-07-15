#!/usr/bin/env python3
"""Regenerate all figures from saved results/*_arrays.npz + results/*.json.

Use after tuning figure styles, or to rebuild figures without re-running the
(expensive) torch pipeline.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mlafbd import viz
from mlafbd.features import FEATURE_NAMES

NL, NF = 12, 7
ABL_ORDER = ["A_last", "B_first_last", "C_concat_xgb", "D_wla_xgb", "E_full",
             "F_no_drift", "G_no_entropy", "H_no_kl", "I_no_variance"]


def main():
    outdir = Path("results"); figdir = Path("paper/figures"); figdir.mkdir(parents=True, exist_ok=True)
    for npz in sorted(outdir.glob("*_arrays.npz")):
        tag = npz.stem.replace("_arrays", "")
        d = np.load(npz, allow_pickle=True)
        X, y, wla = d["X"], d["y"], d["wla"]
        names = list(d["probs_names"]); probs = {n: d[f"prob_{i}"] for i, n in enumerate(names)}
        band = [i + 1 for i, w in enumerate(wla) if w >= float(np.mean(wla))]
        viz.plot_drift(X, y, NL, NF, figdir / f"drift_{tag}.png")
        viz.plot_entropy_concentration(X, y, NL, NF, figdir / f"entconc_{tag}.png")
        viz.plot_wla_weights(wla, figdir / f"wla_{tag}.png", highlight=band)
        if "MLAF-BD" in probs:
            viz.plot_confusion(y, probs["MLAF-BD"], figdir / f"cm_{tag}.png")
        viz.plot_roc([(n, y, p) for n, p in probs.items()], figdir / f"roc_{tag}.png")
        viz.plot_feature_correlation(X, y, NL, NF, figdir / f"featcorr_{tag}.png", FEATURE_NAMES)
        j = outdir / f"{tag}.json"
        if j.is_file():
            a = json.load(open(j)).get("ablation", {})
            if all(k in a for k in ABL_ORDER):
                viz.plot_ablation([k.split("_")[0] for k in ABL_ORDER],
                                  [a[k]["acc"] for k in ABL_ORDER], [a[k]["f1"] for k in ABL_ORDER],
                                  figdir / f"ablation_{tag}.png")
        print(f"  figures for {tag}")
    print("done.")


if __name__ == "__main__":
    main()
