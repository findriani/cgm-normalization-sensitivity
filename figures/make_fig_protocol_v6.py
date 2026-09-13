"""
make_fig_protocol_v6.py -- journal-width protocol figure (5.15 x 5.5 in)
=========================================================================
Top half:  five normalization conditions in a clean table
Bottom half: the ΔR² → ΔΔR² computation flow
=========================================================================
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))

WHITE = "#FFFFFF"
INK = "#202124"
MID = "#5a5a5a"
LIGHT = "#d4d4d0"
PALE = "#f4f4f2"
BLUE = "#0072B2"
VERM = "#D55E00"
GREEN = "#009E73"

W_IN, H_IN = 5.15, 5.3

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 9.0,
    "figure.facecolor": WHITE, "axes.facecolor": WHITE,
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "lines.solid_capstyle": "round",
})


def rounded(ax, x0, y0, x1, y1, fc=WHITE, ec=LIGHT, lw=0.8, rad=1.2, z=2):
    ax.add_patch(FancyBboxPatch((x0, y0), x1-x0, y1-y0,
        boxstyle=f"round,pad=0,rounding_size={rad:.1f}",
        fc=fc, ec=ec, lw=lw, zorder=z))


def arr(ax, s, e, color=INK, lw=0.9, ms=8, style="-|>", z=5):
    ax.add_patch(FancyArrowPatch(s, e, arrowstyle=style, mutation_scale=ms,
        color=color, lw=lw, shrinkA=0, shrinkB=0, zorder=z))


def T(ax, x, y, text, **kw):
    defaults = dict(color=INK, fontsize=9, ha="center", va="center", zorder=8)
    defaults.update(kw)
    return ax.text(x, y, text, **defaults)


def response(t, shift=0.0):
    return np.exp(-0.5*((t-0.42-shift)/0.15)**2) + 0.2*np.exp(-0.5*((t-0.72)/0.13)**2)


def main():
    fig, ax = plt.subplots(figsize=(W_IN, H_IN))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 103)
    ax.set_aspect("equal")
    ax.axis("off")

    # ================ TOP SECTION: CONDITIONS TABLE ================
    T(ax, 3.5, 100, "1", color=WHITE, fontsize=9, weight="bold",
      bbox=dict(boxstyle="circle,pad=0.3", fc=INK, ec="none"))
    T(ax, 30, 100, "Five CGM normalization conditions", fontsize=10.5,
      weight="bold", ha="center")

    T(ax, 30, 95.5, "Same meals, same participant split, same model.\n"
      "Only the CGM transformation changes.",
      color=MID, fontsize=7.5, ha="center", va="top", linespacing=1.3)

    # Column headers
    T(ax, 15, 89, "Condition", color=MID, fontsize=7.5, weight="semibold")
    T(ax, 47, 89, "Centering / scaling", color=MID, fontsize=7.5, weight="semibold")
    T(ax, 72, 89, "Level", color=MID, fontsize=7.5, weight="semibold")
    T(ax, 85, 89, "Record", color=MID, fontsize=7.5, weight="semibold")

    conds = [
        ("Global (reference)",    "Training-fold mean and SD",        True,  False),
        ("Subject scaling",       "Training mean, participant SD",    True,  True),
        ("Pre-meal centering",    "Current window mean, training SD", False, False),
        ("Subject centering",     "Participant mean, training SD",    False, True),
        ("Subject z",             "Participant mean and SD",          False, True),
    ]

    row_h = 7.0
    y0 = 86.5

    for i, (name, desc, level, record) in enumerate(conds):
        y = y0 - i * row_h
        color = BLUE if level else VERM

        # Alternating row background
        if i % 2 == 1:
            rounded(ax, 1, y - row_h + 1.5, 95, y + 0.5,
                    fc=PALE, ec="none", lw=0, rad=0.8, z=1)

        # Group separator between preserved and removed
        if i == 2:
            ax.plot([1, 95], [y + 1.2, y + 1.2], color=LIGHT, lw=0.8, zorder=2)

        wt = "bold" if name.startswith("Pre-meal") else "regular"
        T(ax, 3, y - 2.5, name, color=color, fontsize=8.5, ha="left", weight=wt)
        T(ax, 35, y - 2.5, desc, color=MID, fontsize=7.5, ha="left")
        # Level column: checkmark or X
        sym = r"$\checkmark$" if level else "—"
        T(ax, 72, y - 2.5, sym, color=color, fontsize=9)
        # Record column: dashed circle indicator
        if record:
            ax.add_patch(Circle((85, y - 2.5), 1.3, fc="none",
                ec=VERM, lw=0.8, ls=(0, (2, 1.5)), zorder=6))
            T(ax, 85, y - 2.5, "!", color=VERM, fontsize=7, weight="bold")
        else:
            T(ax, 85, y - 2.5, "—", color=LIGHT, fontsize=9)

    # Group brackets on the far left
    for label, rows, color in [("LEVEL\nPRES.", [0,1], BLUE),
                                ("LEVEL\nREM.", [2,3,4], VERM)]:
        ys = [y0 - r*row_h - 2.5 for r in rows]
        ylo, yhi = min(ys) - 2.5, max(ys) + 2.5
        ax.plot([0.5, 0.5], [ylo, yhi], color=color, lw=2.0, solid_capstyle="round", zorder=3)

    # Record legend
    yleg = y0 - 5*row_h - 2.5
    ax.add_patch(Circle((3.5, yleg), 1.3, fc="none",
        ec=VERM, lw=0.8, ls=(0, (2, 1.5)), zorder=6))
    T(ax, 3.5, yleg, "!", color=VERM, fontsize=7, weight="bold")
    T(ax, 6.5, yleg, "Uses held-out participant's extracted windows",
      color=MID, fontsize=7.5, ha="left")

    # ================ MINI WAVEFORMS ================
    yw = yleg - 8.5
    t = np.linspace(0, 1, 100)
    # Level preserved
    T(ax, 13, yw + 6.5, "Level preserved", color=BLUE, fontsize=7.5, weight="semibold")
    for base, sh in [(yw, 0), (yw+2.5, 0.02), (yw+4.6, -0.01)]:
        x = 3 + 20*t
        ax.plot(x, base + 1.5*response(t, sh), color=BLUE, lw=0.8, alpha=0.7, zorder=5)

    # Arrow between
    arr(ax, (25, yw+3), (29, yw+3), color=MID, lw=0.7, ms=6)

    # Level removed
    T(ax, 42, yw + 6.5, "Level removed", color=VERM, fontsize=7.5, weight="semibold")
    for base in [yw+1.8, yw+2.2, yw+2.6]:
        x = 30 + 20*t
        ax.plot(x, base + 1.5*response(t), color=VERM, lw=0.8, alpha=0.7, zorder=5)

    # ================ BOTTOM SECTION: COMPUTATION FLOW ================
    yb = yw - 6
    T(ax, 3.5, yb + 2, "2", color=WHITE, fontsize=9, weight="bold",
      bbox=dict(boxstyle="circle,pad=0.3", fc=INK, ec="none"))
    T(ax, 16, yb + 2, "Measure", fontsize=10.5, weight="bold", ha="center")

    T(ax, 55, yb + 2, "3", color=WHITE, fontsize=9, weight="bold",
      bbox=dict(boxstyle="circle,pad=0.3", fc=INK, ec="none"))
    T(ax, 67, yb + 2, "Compare", fontsize=10.5, weight="bold", ha="center")

    # Stage 2: model cards and ΔR²
    # CGM + context card
    rounded(ax, 2, yb-12, 22, yb-5, fc=WHITE, ec=LIGHT, lw=0.9, z=3)
    T(ax, 12, yb-7, "CGM + context", color=INK, fontsize=8, weight="semibold")
    T(ax, 12, yb-10, r"$\rightarrow\; R^2_{\mathrm{full}}$", fontsize=9, color=INK)

    # Context-only card
    rounded(ax, 2, yb-22, 22, yb-15, fc=WHITE, ec=LIGHT, lw=0.9, z=3)
    T(ax, 12, yb-17, "Context only", color=INK, fontsize=8, weight="semibold")
    T(ax, 12, yb-20, r"$\rightarrow\; R^2_{\mathrm{ctx}}$", fontsize=9, color=INK)

    # Subtract arrow
    arr(ax, (12, yb-12.5), (12, yb-14.5), color=GREEN, lw=1.0, ms=8)

    # ΔR² box
    rounded(ax, 5, yb-28.5, 19, yb-24, fc=WHITE, ec=GREEN, lw=1.2, z=3)
    T(ax, 12, yb-26.3, r"$\Delta R^2$", color=GREEN, fontsize=12, weight="bold")
    T(ax, 12, yb-30.5, "CGM increment", color=MID, fontsize=7.5)

    # Arrow from ΔR² to stage 3
    arr(ax, (19.5, yb-26.3), (37.5, yb-26.3), color=MID, lw=0.8, ms=7)
    T(ax, 28.5, yb-24.5, "per condition", color=MID, fontsize=6.5)

    # Stage 3: ΔΔR²
    # Reference box
    rounded(ax, 38, yb-17, 52, yb-12, fc=WHITE, ec=BLUE, lw=1.0, z=3)
    T(ax, 45, yb-14.5, r"$\Delta R^2_{\mathrm{ref}}$",
      color=BLUE, fontsize=10, weight="semibold")

    T(ax, 55, yb-14.5, r"$-$", color=INK, fontsize=12, weight="bold")

    # Condition 2 box
    rounded(ax, 58, yb-17, 72, yb-12, fc=WHITE, ec=VERM, lw=1.0, z=3)
    T(ax, 65, yb-14.5, r"$\Delta R^2_{c_2}$",
      color=VERM, fontsize=10, weight="semibold")

    arr(ax, (55, yb-17.5), (55, yb-21), color=GREEN, lw=1.1, ms=8)

    # ΔΔR² result
    rounded(ax, 42, yb-28.5, 68, yb-22, fc=WHITE, ec=GREEN, lw=1.3, z=3)
    T(ax, 55, yb-25.3, r"$\Delta\Delta R^2$", color=GREEN,
      fontsize=13, weight="bold")

    # Verdict labels
    for dx, lab, c in [(-9, "changes", BLUE), (0, "negligible", GREEN),
                       (9, "inconclusive", MID)]:
        ax.add_patch(Circle((55+dx, yb-31.5), 0.65, fc=c, ec="none", zorder=5))
        T(ax, 55+dx, yb-34, lab, color=c, fontsize=6.5, weight="semibold")

    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    outdir = os.path.join(ROOT, "revision", "rev6", "temp")
    os.makedirs(outdir, exist_ok=True)
    for ext, dpi in (("pdf", 600), ("png", 600)):
        out = os.path.join(outdir, f"fig_protocol_v6.{ext}")
        fig.savefig(out, dpi=dpi, facecolor=WHITE, bbox_inches=None, pad_inches=0)
        print(f"wrote {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
