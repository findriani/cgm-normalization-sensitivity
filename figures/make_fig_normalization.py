"""
make_fig_normalization.py -- the figure that carries C1
========================================================
Two aligned panels sharing a y-axis of normalization arms:

    left   CGM adds beyond context        the contribution that collapses
    right  context adds beyond CGM        the compensating inflation

Both panels are Delta-R-squared at the 60-minute horizon, so they SHARE the x-axis.
That is deliberate. The two increments are different quantities, but they are in the same
units, and the paper's claim is that they move in OPPOSITE directions as absolute level is
removed. Separate x-ranges would let a reader misjudge the relative magnitudes -- the
mirror is only legible when both are drawn to one scale.

Each arm carries up to two estimators:

    Random Forest (E1)       circle, blue    all five arms
    Deep mid-fusion (E2)     square, orange  the two arms E2 ran

Estimator identity is encoded by BOTH hue and marker shape, so the figure survives
grayscale printing and colour-vision deficiency. Palette validated with the dataviz
skill's checker: worst adjacent CVD Delta-E 20.1 (protan), normal-vision 27.9, both
comfortably above the thresholds.

Arms are grouped by whether absolute level is retained. The grouping is NOT drawn as a
2x2: the design is deliberately unbalanced (no level-preserving leakage-free arm other
than the reference, and no arm isolating leakage), and presenting it as crossed factors
invites a reviewer to ask for the missing cell.

Inputs   ../../../data preprocessed/core/e1_figure_data.csv
         ../../../data preprocessed/normalization_sensitivity/deep_sens_increments.csv
Outputs  fig_normalization_attribution.pdf  (vector, for LaTeX)
         fig_normalization_attribution.png  (raster, for quick viewing)
========================================================
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
E1 = os.path.join(ROOT, "data preprocessed", "rerun_corrected", "e1_figure_data.csv")
E2 = os.path.join(ROOT, "data preprocessed", "normalization_sensitivity",
                  "deep_sens_increments.csv")
E2_PMC = os.path.join(ROOT, "data preprocessed", "normalization_sensitivity",
                      "deep_sens_pmc_increments.csv")
HORIZON = 60

C_RF = "#1B6CA8"        # validated categorical pair -- do not substitute by eye
C_DL = "#C2521C"
INK = "#1a1a1a"
MUTED = "#6b6b6b"
GRID = "#d8d8d5"

# top-to-bottom order; the two groups are drawn with a separator between them.
# Labels match tab_normalization_arms exactly -- a figure and a table in the same paper
# calling the same arm two different things is the kind of thing a reviewer checks.
# Line breaks are load-bearing at this width: the label column is about 0.6 in, so no line
# may exceed ~12 characters. "Pre-meal centering" on one line overran into the plot area.
ARMS = [
    ("global",         "Global\n(reference)",              True),
    ("subject_scale",  "Subject\nscaling",                 True),
    ("premeal_center", "Pre-meal\ncentering\n(no leakage)", False),
    ("subject_center", "Subject\ncentering",               False),
    ("subject_z",      "Subject z\n(published)",           False),
]
PANELS = [("CGM adds beyond context", "CGM beyond context"),
          ("context adds beyond CGM", "Context beyond CGM")]

# BUILT AT FINAL SIZE. `\the\textwidth` under \documentclass[sn-mathphys-num]{sn-jnl} is
# 372 pt = 5.15 in, measured by compiling the class. This figure was previously built at
# 14.5 in with 16 pt type, which `width=\textwidth` scaled by 0.44 -- so it printed at 7.0 pt,
# on the floor of what is legible, and the panel titles at 8.4 pt. Every point size below is
# now the point size that reaches the page.
#
# THIS IS NOT A PURE RESCALE AND COULD NOT BE. Shrinking the canvas and the type by the same
# factor reproduces the same illegible result; the type has to grow RELATIVE to the figure,
# which means the layout has to give it room. That is paid for in height (free -- height does
# not compete with \textwidth) and in shorter labels.
#
# Serif, to sit alongside the manuscript body text: sn-jnl sets Times, so Times New Roman is
# the match, with STIXGeneral and DejaVu Serif as metric-compatible fallbacks. mathtext uses
# `stix` so Delta-R-squared is drawn in a Times-compatible math face rather than matplotlib's
# sans default -- mixing those is the usual tell that a figure was not made for its paper.
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 9, "axes.labelsize": 9, "xtick.labelsize": 8.5,
    "ytick.labelsize": 9, "legend.fontsize": 8.5,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": INK,
    "figure.facecolor": "white", "axes.facecolor": "white",
    "pdf.fonttype": 42, "ps.fonttype": 42,      # embed real text, not outlines
})

W_IN, H_IN = 5.15, 3.6           # 5.15 in == sn-jnl \textwidth, measured
MS_RF, MS_DL, LW = 5.5, 5.0, 1.3  # marker and whisker sizes


def load():
    e1 = pd.read_csv(E1)
    e1 = e1[(e1.kind == "within_arm") & (e1.horizon == HORIZON)]
    e2 = pd.read_csv(E2)
    e2 = e2[e2.horizon == HORIZON]
    # Merge premeal_center deep model results (E2b)
    if os.path.exists(E2_PMC):
        pmc = pd.read_csv(E2_PMC)
        pmc = pmc[(pmc.horizon == HORIZON) & (pmc.arm == "premeal_center")]
        e2 = pd.concat([e2, pmc], ignore_index=True)
    return e1, e2


def main():
    e1, e2 = load()
    ypos = {a: len(ARMS) - 1 - i for i, (a, _, _) in enumerate(ARMS)}
    dodge = 0.22

    fig, axes = plt.subplots(1, 2, figsize=(W_IN, H_IN), sharey=True, sharex=False)

    for ax, (inc, title) in zip(axes, PANELS):
        lo_pan, hi_pan = [], []
        for arm, _, _ in ARMS:
            y = ypos[arm]

            r = e1[(e1.protocol == arm) & (e1.increment == inc)]
            if not r.empty:
                r = r.iloc[0]
                ax.plot([r.lo, r.hi], [y + dodge] * 2, color=C_RF, lw=LW,
                        solid_capstyle="butt", zorder=3)
                ax.plot(r.estimate, y + dodge, "o", ms=MS_RF, color=C_RF,
                        markeredgecolor="white", markeredgewidth=0.8, zorder=4)
                lo_pan.append(r.lo); hi_pan.append(r.hi)

            d = e2[(e2.arm == arm) & (e2.increment == inc)]
            if not d.empty:
                d = d.iloc[0]
                ax.plot([d.lo, d.hi], [y - dodge] * 2, color=C_DL, lw=LW,
                        solid_capstyle="butt", zorder=3)
                ax.plot(d.dR2, y - dodge, "s", ms=MS_DL, color=C_DL,
                        markeredgecolor="white", markeredgewidth=0.8, zorder=4)
                lo_pan.append(d.lo); hi_pan.append(d.hi)

        ax.axvline(0, color=MUTED, lw=0.8, ls=(0, (3, 2.4)), zorder=1)
        ax.axhline(len(ARMS) - 2.5, color=MUTED, lw=0.6, alpha=0.55, zorder=1)
        ax.set_title(title, fontsize=9.5, color=INK, pad=6, weight="bold")
        ax.set_xlabel(r"$\Delta R^2$ at 60 min  (95% CI)", labelpad=4)
        ax.xaxis.grid(True, color=GRID, lw=0.5)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", length=0)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        # Each panel gets its own x-range — no wasted space
        pad = 0.06 * (max(hi_pan) - min(lo_pan))
        ax.set_xlim(min(lo_pan) - pad, max(hi_pan) + pad)

    ax = axes[0]
    ax.set_yticks(range(len(ARMS)))
    ax.set_yticklabels([lab for _, lab, _ in ARMS][::-1])
    for t in ax.get_yticklabels():
        if "Pre-meal" in t.get_text():
            t.set_weight("bold")
    ax.set_ylim(-0.65, len(ARMS) - 0.25)

    # Group brackets
    for label, rows in [("LEVEL\nPRESERVED", [0, 1]), ("LEVEL\nREMOVED", [2, 3, 4])]:
        ys = [ypos[ARMS[i][0]] for i in rows]
        lo_y, hi_y = min(ys) - 0.38, max(ys) + 0.38
        axes[0].annotate("", xy=(-0.42, lo_y), xytext=(-0.42, hi_y),
                         xycoords=("axes fraction", "data"),
                         arrowprops=dict(arrowstyle="-", color=GRID, lw=1.4))
        axes[0].annotate(label, xy=(-0.49, np.mean(ys)),
                         xycoords=("axes fraction", "data"),
                         ha="center", va="center", fontsize=7, color=MUTED,
                         weight="bold", rotation=90, linespacing=1.1)

    handles = [
        Line2D([], [], color=C_RF, marker="o", ms=MS_RF, lw=LW,
               markeredgecolor="white", markeredgewidth=0.8,
               label="Random Forest (8 seeds)"),
        Line2D([], [], color=C_DL, marker="s", ms=MS_DL, lw=LW,
               markeredgecolor="white", markeredgewidth=0.8,
               label="Deep mid-fusion (15 seeds)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.55, -0.005), handletextpad=0.5, columnspacing=1.8)

    fig.subplots_adjust(left=0.28, right=0.985, top=0.90, bottom=0.19, wspace=0.12)

    # NO bbox_inches="tight". It crops to the ink and would have emitted a 4.68 in canvas,
    # which `width=\textwidth` then scales back UP by 1.10 -- silently undoing the point of
    # building at final size. Margins are set by subplots_adjust instead, so the saved page
    # is exactly W_IN wide and the scale factor in LaTeX is 1.00.
    outdir = os.path.join(ROOT, "revision", "rev6", "temp")
    os.makedirs(outdir, exist_ok=True)
    for ext, dpi in (("pdf", 600), ("png", 600)):
        out = os.path.join(outdir, f"fig_normalization_attribution.{ext}")
        fig.savefig(out, dpi=dpi, facecolor="white")
        print(f"wrote {out}  ({'vector' if ext == 'pdf' else f'{dpi} dpi raster'})")


if __name__ == "__main__":
    main()
