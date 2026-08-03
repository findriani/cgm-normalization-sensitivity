"""
make_fig_cross_cohort.py -- does the normalization effect survive a second cohort?
==================================================================================
Two panels, six rows: three horizons in CGMacros above three horizons in ShanghaiT2DM.

    left    CGM's measured contribution, by arm      the LEVELS
    right   effect of normalization on the increment the DIFFERENCE (the estimand)

WHY THE LEFT PANEL EXISTS AT ALL
---------------------------------
The obvious version of this figure plots only the interaction. It would be actively
misleading. Shanghai's interaction is LARGER than CGMacros' at every horizon (+0.270 vs
+0.173 at 60 min), so an interaction-only figure says "per-subject normalization does more
damage in Shanghai". In proportional terms it does LESS -- an 88% cut in CGMacros against
52% in Shanghai -- because Shanghai's reference-arm increment is far larger to begin with
(+0.515 vs +0.197) in a cohort whose context block carries almost nothing. The left panel
shows both arms so the reader can see the levels the difference is taken between, and each
row is annotated with the proportional cut, which is the quantity the two cohorts actually
disagree on.

WHY THE PANELS DO NOT SHARE AN X-AXIS
--------------------------------------
The companion figure (fig_normalization_attribution) shares its x-axis deliberately: both of
its panels are the same estimand pointed at different modalities, and a shared scale is what
makes the mirror legible. Here the panels hold DIFFERENT estimands -- a level on the left, a
difference of levels on the right. Shanghai's reference increment reaches 0.85, so forcing
one scale would squeeze every interval in the right panel, which is the panel carrying the
claim, into the left fifth of its width. Each panel is therefore scaled to its own estimand
and both are labelled in full.

THE SIGN CONVENTION IN THE RIGHT PANEL, WHICH IS EASY TO MISREAD
-----------------------------------------------------------------
Plotted raw, as stored: ddR2 = (increment under global) - (increment under subject z).

    positive   the increment is larger under the reference   -> attenuation
    negative   the increment is larger under the published transform   -> inflation

So the CGM series sits right of zero and the context series sits left of it, and that
opposition IS the mirror effect. The signs are NOT flipped for presentation here; flipping
them in prose while the stored file holds the other sign is how that result has been
misreported before.

WHAT THE GREY SERIES IS DOING
------------------------------
The context interaction is drawn muted and hollow because it is the half of the claim that
does NOT transfer. Shanghai has no meal-content modality, so its context block is person and
time only and carries almost no signal; there is nothing there to inflate, and all three of
its context intervals cross zero. That is a scope limit, not a failed replication, and the
figure should show it rather than let a reader discover it in a limitations paragraph.

Inputs   ../../../data preprocessed/core/e1_increments.csv, e1_interaction.csv
         ../../../data preprocessed/shanghai_external/shanghai_increments.csv,
                                                      shanghai_interaction.csv
Outputs  fig_cross_cohort.pdf  (vector, for LaTeX)
         fig_cross_cohort.png  (600 dpi raster, for slides and review)
==================================================================================
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
CORE = os.path.join(ROOT, "data preprocessed", "core")
SHA = os.path.join(ROOT, "data preprocessed", "shanghai_external")

CGM_INC = "CGM adds beyond context"
CTX_INC = "context adds beyond CGM"
HORIZONS = [30, 60, 120]
REF_ARM, PUB_ARM = "global", "subject_z"

# Same validated categorical pair as the companion figure, so the two figures read as one
# system: dataviz checker, worst adjacent CVD Delta-E 20.1 (protan), normal-vision 27.9.
# The right panel deliberately uses NO categorical hue -- its primary series is neutral ink
# and its secondary series is grey, because reusing blue or orange there would attach an
# established meaning ("the reference arm") to a different entity ("the interaction").
C_REF = "#1B6CA8"
C_PUB = "#C2521C"
INK = "#1a1a1a"
MUTED = "#6b6b6b"
GREY_SERIES = "#9a9a9a"
GRID = "#d8d8d5"
BAND = "#c9c9c4"

# BUILT AT FINAL SIZE. `\the\textwidth` under sn-jnl is 372 pt = 5.15 in, measured by
# compiling the class. This figure was previously built at 14.6 in with 16 pt type, which
# `width=\textwidth` scaled by 0.35 -- so its body text printed at 5.6 pt, below anything
# legible, and its panel titles at 6.7 pt. Every point size below is now the size that
# reaches the page. Height is the free dimension and pays for the larger relative type.
W_IN, H_IN = 5.15, 4.45

# ONE line now, not two. Rotated 90 degrees, each extra line widens the label HORIZONTALLY,
# and at 5.15 in there is no horizontal room to spare beside the y-axis labels. The counts
# moved to the caption, where Table 1 already carries them in full.
COHORTS = [("CGMacros", "CGMacros", CORE, "e1"),
           ("Shanghai", "ShanghaiT2DM", SHA, "shanghai")]

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7.5,
    "ytick.labelsize": 8, "legend.fontsize": 7,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": INK,
    "figure.facecolor": "white", "axes.facecolor": "white",
    "pdf.fonttype": 42, "ps.fonttype": 42,
})


def load():
    """Rows are (cohort, horizon) -> the two arm increments and the two interactions."""
    out = {}
    for name, _, path, stem in COHORTS:
        inc = pd.read_csv(os.path.join(path, f"{stem}_increments.csv"))
        inter = pd.read_csv(os.path.join(path, f"{stem}_interaction.csv"))
        for h in HORIZONS:
            rec = {}
            for arm in (REF_ARM, PUB_ARM):
                g = inc[(inc.increment == CGM_INC) & (inc.horizon == h)
                        & (inc.protocol == arm)]
                if g.empty:
                    raise SystemExit(f"[abort] {name} h={h} arm={arm} missing from "
                                     f"{stem}_increments.csv")
                g = g.iloc[0]
                rec[arm] = (float(g.dR2), float(g.lo), float(g.hi))
            for key, label in (("cgm", CGM_INC), ("ctx", CTX_INC)):
                g = inter[(inter.increment == label) & (inter.horizon == h)]
                if g.empty:
                    raise SystemExit(f"[abort] {name} h={h} '{label}' missing from "
                                     f"{stem}_interaction.csv")
                g = g.iloc[0]
                rec[key] = (float(g.ddR2), float(g.lo), float(g.hi))
            out[(name, h)] = rec
    return out


def dot(ax, x, lo, hi, y, color, marker, ms=5.0, fill=True):
    ax.plot([lo, hi], [y, y], color=color, lw=1.2, solid_capstyle="butt", zorder=3)
    ax.plot(x, y, marker, ms=ms, zorder=4,
            color=color if fill else "white",
            markeredgecolor=color if not fill else "white",
            markeredgewidth=1.0 if not fill else 0.8)


def main():
    d = load()
    rows = [(c, h) for c, _, _, _ in COHORTS for h in HORIZONS]
    ypos = {r: len(rows) - 1 - i for i, r in enumerate(rows)}
    dodge = 0.19

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(W_IN, H_IN), sharey=True)

    # Fix the left scale up front: label placement depends on how wide a gap is relative to
    # the axis, and at 120 min the two arms are close enough that a centred label would sit
    # on top of both markers.
    xs = [v for r in rows for arm in (REF_ARM, PUB_ARM) for v in d[r][arm][1:]]
    xlo, xhi = min(xs), max(xs)
    padL = 0.05 * (xhi - xlo)
    axL.set_xlim(xlo - padL, xhi + padL)
    span = (xhi + padL) - (xlo - padL)

    for r in rows:
        y = ypos[r]
        ref, pub = d[r][REF_ARM], d[r][PUB_ARM]

        # The gap between the two arms is the story, so draw it before the markers: a light
        # band at the row centre spanning the two point estimates, labelled with the
        # proportional cut. The centre line is free -- the markers are dodged off it.
        axL.plot([pub[0], ref[0]], [y, y], color=BAND, lw=3.2, alpha=0.55,
                 solid_capstyle="round", zorder=1)
        cut = 1.0 - pub[0] / ref[0] if ref[0] > 0 else np.nan
        # A short band cannot hold a centred label. Below ~9% of the axis the label goes
        # outside, to the right of the reference marker, where there is always room.
        if (ref[0] - pub[0]) < 0.09 * span:
            lx, ha = ref[0] + 0.012 * span, "left"
        else:
            lx, ha = (pub[0] + ref[0]) / 2, "center"
        # Muted and small on purpose: these are point ratios of point estimates and carry no
        # interval, so they must not read as estimates competing with the plotted CIs.
        axL.annotate(f"$-${cut:.0%}", xy=(lx, y), ha=ha, va="center",
                     fontsize=6.5, color=MUTED, zorder=5,
                     bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.9))

        dot(axL, *ref, y + dodge, C_REF, "o")
        dot(axL, *pub, y - dodge, C_PUB, "s", ms=4.6)

        dot(axR, *d[r]["cgm"], y + dodge, INK, "D", ms=4.3)
        dot(axR, *d[r]["ctx"], y - dodge, GREY_SERIES, "o", ms=4.6, fill=False)

    # The panel letter is now the first word of the title rather than a separate annotation
    # above it. At 5.15 in a panel is about 1.9 in wide, the titles need two lines to fit,
    # and a floating "(a)" above a two-line title collided with it.
    for ax, title, xlabel in (
            (axL, "(a)  CGM's measured\ncontribution, by arm",
             r"$\Delta R^2$, CGM beyond context  (95% CI)"),
            (axR, "(b)  Effect of normalization\non the increment",
             r"$\Delta\Delta R^2$ = global $-$ subject $z$  (95% CI)")):
        ax.axvline(0, color=MUTED, lw=0.8, ls=(0, (3, 2.4)), zorder=1)
        ax.axhline(len(rows) - 3.5, color=MUTED, lw=0.6, alpha=0.55, zorder=1)
        ax.set_title(title, fontsize=8, color=INK, pad=4, weight="bold",
                     linespacing=1.35)
        ax.set_xlabel(xlabel, labelpad=4)
        ax.xaxis.grid(True, color=GRID, lw=0.5)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", length=0)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.set_ylim(-0.75, len(rows) - 0.25)

    axL.set_yticks(range(len(rows)))
    axL.set_yticklabels([f"{h} min" for _, h in rows][::-1])
    # 60 min is the confirmatory horizon; the other two are secondary and unadjusted.
    for t in axL.get_yticklabels():
        if t.get_text() == "60 min":
            t.set_weight("bold")

    # Direction cue for the right panel's sign convention, placed where it cannot be missed
    # and cannot collide with data: below the axis, under the two halves of the scale.
    axR.annotate("attenuation $\\rightarrow$", xy=(0.985, -0.205),
                 xycoords="axes fraction", ha="right", va="top",
                 fontsize=6.5, color=MUTED, style="italic")
    axR.annotate("$\\leftarrow$ inflation", xy=(0.015, -0.205),
                 xycoords="axes fraction", ha="left", va="top",
                 fontsize=6.5, color=MUTED, style="italic")

    # Cohort brackets, far left and rotated -- the same idiom as the companion figure.
    # Reading right to left: the y-axis labels, then the bracket at -0.20, then the rotated
    # cohort name at -0.40. The label had to move further out than before: at 5.15 in the
    # axes are narrower, so the same axes-FRACTION offset is a much smaller gap in inches.
    # Brackets hug their group (+/-0.30 against a 0.19 dodge) rather than floating past it.
    for label, idx in [(COHORTS[0][1], [0, 1, 2]), (COHORTS[1][1], [3, 4, 5])]:
        ys = [ypos[rows[i]] for i in idx]
        axL.annotate("", xy=(-0.26, min(ys) - 0.30), xytext=(-0.26, max(ys) + 0.30),
                     xycoords=("axes fraction", "data"),
                     arrowprops=dict(arrowstyle="-", color=GRID, lw=1.4))
        axL.annotate(label, xy=(-0.46, np.mean(ys)),
                     xycoords=("axes fraction", "data"),
                     ha="center", va="center", fontsize=7, color=MUTED,
                     weight="bold", rotation=90)

    hL = [Line2D([], [], color=C_REF, marker="o", ms=5.0, lw=1.2,
                 markeredgecolor="white", markeredgewidth=0.8,
                 label="Global (reference)"),
          Line2D([], [], color=C_PUB, marker="s", ms=4.6, lw=1.2,
                 markeredgecolor="white", markeredgewidth=0.8,
                 label="Subject $z$ (published)")]
    # Wrapped, not shortened: "not testable in Shanghai" is the scope limit the figure
    # exists to make visible, so it stays in the legend rather than moving to the caption.
    hR = [Line2D([], [], color=INK, marker="D", ms=4.3, lw=1.2,
                 markeredgecolor="white", markeredgewidth=0.8,
                 label="CGM beyond context"),
          Line2D([], [], color=GREY_SERIES, marker="o", ms=4.6, lw=1.2,
                 markerfacecolor="white", markeredgewidth=1.0,
                 label="Context beyond CGM\n(not testable in Shanghai)")]
    # One legend per panel, each stacked in a single column under its own panel. Laid out
    # side by side in one row they merged into what looked like a single four-item legend,
    # which broke the association between a series and the panel it appears in.
    fig.legend(handles=hL, loc="upper center", ncol=1, frameon=False,
               title="(a)  normalization arm", alignment="left",
               bbox_to_anchor=(0.34, 0.185), handletextpad=0.5)
    fig.legend(handles=hR, loc="upper center", ncol=1, frameon=False,
               title="(b)  increment tested", alignment="left",
               bbox_to_anchor=(0.76, 0.185), handletextpad=0.5)

    fig.subplots_adjust(left=0.245, right=0.985, top=0.875, bottom=0.315, wspace=0.09)

    # NO bbox_inches="tight" -- it crops to the ink, and `width=\textwidth` then scales the
    # smaller canvas back up, silently undoing the point of building at final size.
    for ext, dpi in (("pdf", 600), ("png", 600)):
        out = os.path.join(HERE, f"fig_cross_cohort.{ext}")
        fig.savefig(out, dpi=dpi, facecolor="white")
        print(f"wrote {out}  ({'vector' if ext == 'pdf' else f'{dpi} dpi raster'})")

    print("\nproportional cut in the CGM increment (reference -> published):")
    for r in rows:
        ref, pub = d[r][REF_ARM][0], d[r][PUB_ARM][0]
        print(f"  {r[0]:9s} h={r[1]:3d}   {ref:+.4f} -> {pub:+.4f}   {1 - pub / ref:.1%}")


if __name__ == "__main__":
    main()
