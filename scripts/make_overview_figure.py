#!/usr/bin/env python3
"""Regenerate paper/figures/overview_mf.png (MLAF-BD overview) with corrected text.

Same layout as the original raster: detection pipeline on top, two-stage
mitigation pipeline below. Canvas is 1400 x 780 'design units', y grows down.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Liberation Serif", "DejaVu Serif"],
    "mathtext.fontset": "stix",
})

W, H = 1400.0, 780.0
fig = plt.figure(figsize=(14, 7.8), dpi=100)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(H, 0)          # y increases downwards, like image coordinates
ax.axis("off")

LW = 1.3

# ---------------------------------------------------------------- helpers
def box(x, y, w, h, ls="-", lw=LW):
    ax.add_patch(Rectangle((x, y), w, h, fill=False, lw=lw, ls=ls,
                           edgecolor="black"))

def txt(x, y, s, fs=11, ha="center", va="center", **kw):
    ax.text(x, y, s, fontsize=fs, ha=ha, va=va, **kw)

def arrow(x1, y1, x2, y2, color="black", lw=1.4, ls="-"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2),
                                 arrowstyle="-|>", mutation_scale=14,
                                 lw=lw, color=color, linestyle=ls,
                                 shrinkA=0, shrinkB=0))

def line(x1, y1, x2, y2, lw=1.4, ls="-", color="black"):
    ax.plot([x1, x2], [y1, y2], lw=lw, ls=ls, color=color,
            solid_capstyle="butt")

def heat_axes(x, y, w, h):
    """Inset axes at design-rect (x,y,w,h)."""
    a = fig.add_axes([x / W, 1.0 - (y + h) / H, w / W, h / H])
    a.set_xticks([]); a.set_yticks([])
    for s in a.spines.values():
        s.set_linewidth(0.8)
    return a

def heat_clean(x, y, w, h, seed):
    rng = np.random.default_rng(seed)
    a = heat_axes(x, y, w, h)
    a.imshow(rng.random((7, 7)), cmap="viridis", interpolation="bicubic")

def heat_poison(x, y, w, h, seed):
    rng = np.random.default_rng(seed)
    g = np.linspace(-1, 1, 28)
    xx, yy = np.meshgrid(g, g)
    blob = np.exp(-((xx - 0.05) ** 2 + (yy + 0.05) ** 2) / 0.055)
    data = 0.10 * rng.random((28, 28)) + blob
    a = heat_axes(x, y, w, h)
    a.imshow(data, cmap="jet", interpolation="bicubic")

# ================================================================ top title
txt(490, 16, "Backdoor Detection Pipeline", fs=15)

# --------------------------------------------------------------- input / ViT
txt(45, 185, "Input\nImage\n$\\mathbf{x}$", fs=12)
arrow(80, 185, 110, 185)

box(112, 42, 48, 252)
txt(136, 168, "Vision Transformer (ViT)", fs=11, rotation=90)
arrow(160, 185, 181, 185)

box(183, 42, 140, 252)
box(200, 95, 105, 150)
txt(252, 170, "Conceptual\nEncoder\nLayers\n1.\n$\\vdots$\n$1\\ldots N$", fs=10.5)
# magnifier lines from inner box down to the MHSA box
line(200, 245, 155, 320, lw=0.9)
line(305, 245, 327, 320, lw=0.9)
box(150, 320, 180, 55)
txt(240, 347, "Internal Multi-Head\nSelf-Attention", fs=11)
arrow(323, 185, 341, 185)

# --------------------------------------------- attention map extraction box
box(343, 42, 245, 408)
txt(465, 66, "Attention Map Extraction\n(Layer Hooks)", fs=12)
txt(352, 112, "1. Average Heads", fs=11, ha="left")
txt(352, 141, "2. Select CLS Token Row", fs=11, ha="left")
txt(352, 170, "3. Drop Self-Attention", fs=11, ha="left")
txt(352, 199, "4. $\\ell_1$-Normalise", fs=11, ha="left")
txt(563, 112, "$\\mathbf{a}^{(1)}$", fs=12)
txt(563, 205, "$\\mathbf{a}^{(2)}$", fs=12)
txt(563, 275, "$\\vdots$", fs=12)
txt(563, 345, "$\\mathbf{a}^{(N)}$", fs=12)

txt(387, 240, "$\\mathbf{a}^{(1)}$", fs=11)
txt(500, 240, "$\\mathbf{a}^{(N)}$", fs=11)
arrow(500, 212, 500, 227)          # small arrow from list to the maps
heat_clean(355, 250, 65, 62, seed=1)
heat_poison(468, 250, 65, 62, seed=2)
arrow(424, 281, 464, 281)
txt(387, 333, "$\\vdots$", fs=12)
txt(500, 333, "$\\vdots$", fs=12)
heat_clean(355, 352, 65, 62, seed=3)
heat_poison(468, 352, 65, 62, seed=4)
arrow(424, 383, 464, 383)
txt(387, 432, "Clean", fs=11)
txt(500, 432, "Poisoned", fs=11)

# ------------------------------------------------------- 7 features box
arrow(588, 130, 606, 130)
box(608, 42, 217, 203)
txt(716, 70, "Compute 7 Statistical\nFeatures (for each $\\mathbf{a}^{(l)}$)", fs=12)
feats = ["Entropy ($H$)", "Variance ($\\mathit{Var}$)",
         "Concentration ($\\mathit{Conc}$)", "Sparsity ($\\mathit{Spar}$)",
         "Energy ($E$)", "KL Divergence ($\\mathit{KL}$)"]
for i, f in enumerate(feats):
    txt(620, 102 + 25 * i, "• " + f, fs=11, ha="left")

# ------------------------------------------------------- attention drift box
arrow(588, 345, 606, 345)
box(608, 262, 217, 180)
txt(716, 290, "Compute Attention Drift\n(for $l \\geq 2$)", fs=12)
txt(716, 328, "$\\mathrm{Drift}^{(l)} = \\|\\mathbf{a}^{(l)} - \\mathbf{a}^{(l-1)}\\|_1$", fs=12)
txt(770, 350, "$(\\mathrm{Drift}^{(1)}\\!=\\!0)$", fs=9.5)
txt(655, 398, "movement\nof attention\nacross layers", fs=10)

# tiny trajectory inset
ta = fig.add_axes([706 / W, 1 - 436 / H, 108 / W, 68 / H])
ta.set_xlim(0, 1); ta.set_ylim(0, 1)
ta.set_xticks([]); ta.set_yticks([])
for s in ta.spines.values():
    s.set_visible(False)
ta.annotate("", xy=(1.0, 0.08), xytext=(0.05, 0.08),
            arrowprops=dict(arrowstyle="->", lw=1.0))
ta.annotate("", xy=(0.05, 1.0), xytext=(0.05, 0.08),
            arrowprops=dict(arrowstyle="->", lw=1.0))
ta.plot([0.30, 0.72], [0.38, 0.72], "-", color="#4878d0", lw=1.4)
ta.plot(0.30, 0.38, "o", color="#4878d0", ms=5)
ta.plot(0.72, 0.72, "o", color="#ee854a", ms=5)
ta.text(0.20, 0.16, "$\\mathbf{a}^{(l-1)}$", fontsize=10, ha="center")
ta.text(0.82, 0.90, "$\\mathbf{a}^{(l)}$", fontsize=10, ha="center")

# ------------------------------------------------ construct feature matrix
line(825, 130, 838, 130)
line(825, 345, 838, 345)
line(838, 130, 838, 345)
arrow(838, 220, 846, 220)
box(848, 155, 105, 130)
txt(900, 220, "Construct\nMulti-Layer\nFeature\nMatrix\n$\\mathbf{F} \\in \\mathbb{R}^{N\\times 7}$", fs=10.5)

# ------------------------------------------------------- flatten + ensemble
line(953, 228, 1042, 228)
txt(995, 190, "Flatten\nMatrix", fs=11)
txt(995, 268, "$\\mathbf{F}_{\\mathrm{flat}} \\in \\mathbb{R}^{84}$", fs=10.5)
line(1042, 125, 1042, 250)
arrow(1042, 125, 1053, 125)
arrow(1042, 250, 1053, 250)

txt(1150, 58, "Detector Ensemble", fs=13)
box(1055, 95, 110, 60)
txt(1110, 125, "XGBoost\nClassifier", fs=11)
box(1055, 210, 115, 80)
txt(1112, 250, "Multilayer\nPerceptron (MLP)\nClassifier", fs=11)
arrow(1165, 125, 1195, 125)
arrow(1170, 250, 1195, 250)

box(1197, 85, 60, 230)
txt(1227, 200, "Combine Probabilities\n(Weighted Average $\\alpha$, $1-\\alpha$)",
    fs=10.5, rotation=90)
arrow(1257, 140, 1288, 140)
txt(1295, 140, "Clean\nInput\n($y$=0)", fs=11.5, ha="left")
arrow(1257, 260, 1288, 260)
txt(1295, 260, "Poisoned\nInput\n($y$=1)", fs=11.5, ha="left")

# ------------------------------------------------------------- WLA (dashed)
box(995, 340, 385, 190, ls=(0, (5, 3)))
txt(1187, 362, "Weighted Layer Aggregation (WLA)", fs=12.5)
txt(1005, 397, "• Analyze Layer Importance via\n   Feature-Poison Correlation",
    fs=11, ha="left")
txt(1005, 437, "• Fused Representation "
    "$\\mathbf{f}_{\\mathrm{fused}} = \\sum_l w_l\\, \\mathbf{f}^{(l)}$",
    fs=11, ha="left")
txt(1005, 487, "• Interpretable Layer Weights\n"
    "   (e.g., for $N$=12 the profile peaks at layer 4;\n"
    "   early-middle and final layers above uniform)",
    fs=11, ha="left")

# validation split (dashed) from feature matrix into WLA
line(920, 285, 920, 420, ls=(0, (5, 3)))
arrow(920, 420, 992, 420, ls=(0, (5, 3)))
txt(952, 393, "Validation\nsplit", fs=11)

# flagged training samples: down from feature matrix, long left run
line(880, 285, 880, 480)
line(60, 480, 880, 480)
txt(620, 466, "Flagged Training Samples", fs=12)
arrow(60, 480, 60, 610)            # into Stage 1
arrow(237, 480, 237, 541)          # into Remove Flagged Samples

# ======================================================== mitigation strip
txt(480, 520, "Two-Stage Mitigation Pipeline", fs=15)

box(28, 612, 80, 80)
txt(68, 632, "Stage 1:", fs=11.5, fontweight="bold")
txt(68, 664, "Sample\nFiltering", fs=11)
arrow(108, 652, 133, 652)

box(135, 543, 210, 212)
txt(240, 560, "Remove Flagged Samples", fs=11.5)
heat_clean(150, 575, 60, 58, seed=5)
heat_poison(268, 575, 60, 58, seed=6)
arrow(214, 604, 264, 604)
txt(180, 646, "($y$=0)", fs=10.5)
txt(298, 646, "($y$=1)", fs=10.5)
heat_clean(150, 658, 60, 58, seed=7)
heat_poison(268, 658, 60, 58, seed=8)
arrow(214, 687, 264, 687, color="#c62828")
txt(180, 731, "Clean ($y$=0)", fs=10.5)
txt(298, 731, "Poisoned", fs=10.5)

arrow(345, 650, 440, 650)
txt(392, 612, "Filtered\nTraining\nSet", fs=11)

# ------------------------------------------------------------- stage 2 box
box(443, 543, 380, 217)
txt(568, 562, "Stage 2:", fs=12, fontweight="bold", ha="right")
txt(572, 562, " Model Fine-Tuning", fs=12, ha="left")
box(458, 580, 350, 165)
txt(470, 600, "Augmented Loss", fs=11.5, ha="left")
txt(633, 628, "$L = L_{CE} + \\lambda \\sum_{l\\in\\mathcal{M}} "
    "\\max(0,\\ \\mathit{Conc}^{(l)} - \\mathit{Conc}^{(l)}_{ref})$", fs=11.5)
txt(633, 652, "Attention-Concentration Penalty", fs=11)
txt(528, 700, "Concentration\nweights", fs=10.5)

# hinge-penalty inset (correct shape: zero below the reference, linear above)
pa = fig.add_axes([612 / W, 1 - 742 / H, 180 / W, 62 / H])
pa.set_xlim(0, 1); pa.set_ylim(0, 1)
pa.set_xticks([]); pa.set_yticks([])
for s in pa.spines.values():
    s.set_visible(False)
pa.annotate("", xy=(1.0, 0.10), xytext=(0.06, 0.10),
            arrowprops=dict(arrowstyle="->", lw=1.0))
pa.annotate("", xy=(0.06, 1.0), xytext=(0.06, 0.10),
            arrowprops=dict(arrowstyle="->", lw=1.0))
thr = 0.45
pa.fill_between([thr, 0.95], [0.10, 0.10], [0.10, 0.78],
                color="0.82", zorder=0)
pa.plot([0.06, thr], [0.10, 0.10], color="#d95f02", lw=1.8)
pa.plot([thr, 0.95], [0.10, 0.78], color="#d95f02", lw=1.8)
pa.plot([thr, thr], [0.10, 0.95], ls=(0, (4, 3)), color="black", lw=1.0)
pa.text(thr + 0.04, 0.86, "$\\mathit{Conc}^{(l)}_{ref}$", fontsize=9.5,
        ha="left")
pa.text(0.10, 0.97, "Penalty", fontsize=9.5, ha="left", va="top")
pa.text(0.99, 0.26, "$\\mathit{Conc}^{(l)}$", fontsize=9.5, ha="right")

arrow(823, 655, 915, 655)
txt(868, 622, "Model\nFine-Tuning", fs=11)

# ------------------------------------------------- right augmented-loss box
box(918, 563, 330, 197)
txt(930, 585, "Augmented Loss", fs=11.5, ha="left")
txt(1083, 615, "$L = L_{CE} + \\lambda \\sum_{l\\in\\mathcal{M}} "
    "\\max(0,\\ \\mathit{Conc}^{(l)} - \\mathit{Conc}^{(l)}_{ref})$", fs=11.5)
txt(1083, 643, "Attention-Concentration Penalty", fs=11)
txt(930, 678, "• Penalises $\\mathit{Conc}^{(l)}$ above the clean\n"
    "   reference $\\mathit{Conc}^{(l)}_{ref}$", fs=11, ha="left")
txt(930, 712, "• Applied to middle layers $l \\in \\mathcal{M}$",
    fs=11, ha="left")
txt(930, 738, "• Penalty weight $\\lambda$ = 0.01", fs=11, ha="left")

arrow(1248, 660, 1276, 660)
box(1278, 618, 100, 85)
txt(1328, 660, "Mitigated\nViT\nModel", fs=11.5)

# ---------------------------------------------------------------- save
import os, sys
out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "paper", "figures", "overview_mf.png")
fig.savefig(out, dpi=200, facecolor="white")
print("wrote", out)
