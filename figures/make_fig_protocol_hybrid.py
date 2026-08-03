"""Hybrid illustrated/vector normalization-sensitivity protocol figure.

The participant vignette is an illustration asset. All scientific geometry,
labels, transformations, arrows, and estimands are generated deterministically
with Matplotlib.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
from matplotlib.image import imread
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

HERE = Path(__file__).resolve().parent
ASSET = HERE / "assets" / "participant_split_illustration.png"

W_IN, H_IN = 12.8, 7.2
XMAX, YMAX = 100.0, 56.25

WHITE = "#FFFFFF"
BLACK = "#000000"
INK = "#202124"
MID = "#676767"
LIGHT = "#D8D8D8"
PALE = "#F5F5F3"
BLUE = "#56B4E9"       # Okabe-Ito sky blue
BLUE_DARK = "#0072B2"  # Okabe-Ito blue for readable type
VERM = "#D55E00"       # Okabe-Ito vermillion
GREEN = "#009E73"      # Okabe-Ito bluish green
ORANGE = "#E69F00"     # Okabe-Ito orange, used sparingly

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "Arial", "DejaVu Sans"],
    "font.size": 8.0,
    "mathtext.fontset": "stix",
    "figure.facecolor": WHITE,
    "axes.facecolor": WHITE,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "lines.solid_capstyle": "round",
})


def rounded(ax, x0, y0, x1, y1, fc=WHITE, ec=LIGHT, lw=0.8,
            radius=0.9, z=2, shadow=False):
    patch = FancyBboxPatch(
        (x0, y0), x1 - x0, y1 - y0,
        boxstyle="round,pad=0,rounding_size=%.2f" % radius,
        facecolor=fc, edgecolor=ec, linewidth=lw, zorder=z,
    )
    if shadow:
        patch.set_path_effects([
            pe.SimplePatchShadow(offset=(0.45, -0.55), alpha=0.08, rho=0.99),
            pe.Normal(),
        ])
    ax.add_patch(patch)
    return patch


def arrow(ax, start, end, color=BLACK, lw=1.0, mutation=7,
          style="-|>", curve=0.0, ls="-", z=6):
    patch = FancyArrowPatch(
        start, end, arrowstyle=style, mutation_scale=mutation,
        color=color, linewidth=lw, linestyle=ls,
        connectionstyle="arc3,rad=%.3f" % curve,
        shrinkA=0, shrinkB=0, zorder=z,
    )
    ax.add_patch(patch)
    return patch


def label(ax, x, y, text, color=BLACK, size=7.0, weight="regular",
          ha="center", va="center", z=8, **kwargs):
    return ax.text(x, y, text, color=color, fontsize=size, weight=weight,
                   ha=ha, va=va, zorder=z, **kwargs)


def method_pill(ax, x, y, text, edge, width):
    rounded(ax, x - width / 2, y - 1.05, x + width / 2, y + 1.05,
            fc=WHITE, ec=edge, lw=0.9, radius=1.05, z=4)
    label(ax, x, y, text, color=INK, size=6.25, weight="medium")


def response(t, shift=0.0, width=0.15, shoulder=0.20):
    main = np.exp(-0.5 * ((t - (0.42 + shift)) / width) ** 2)
    tail = shoulder * np.exp(-0.5 * ((t - (0.72 + shift / 2)) / 0.13) ** 2)
    return main + tail


def plot_curve(ax, x0, x1, baseline, amplitude, color, shift=0.0,
               width=0.15, lw=1.25, alpha=1.0, ls="-", z=6):
    t = np.linspace(0.0, 1.0, 180)
    curve = response(t, shift=shift, width=width)
    x = x0 + (x1 - x0) * t
    y = baseline + amplitude * curve
    ax.plot(x, y, color=color, lw=lw, alpha=alpha, ls=ls, zorder=z)
    return t, x, y


def panel_header(ax, x, y, text, color=BLACK, align="left"):
    label(ax, x, y, text, color=color, size=8.2, weight="semibold",
          ha=align, va="center")


def participant_panel(ax):
    panel_header(ax, 1.3, 53.2, "PARTICIPANT SPLIT")
    img = imread(ASSET)
    ax.imshow(img, extent=(1.0, 19.0, 17.8, 34.0), origin="upper",
              interpolation="lanczos", zorder=1, aspect="auto")

    label(ax, 5.8, 35.3, "COHORT", size=6.5, weight="semibold")
    label(ax, 15.2, 35.3, "TRAINING", size=6.5, weight="semibold")
    label(ax, 15.7, 16.4, "HELD-OUT", size=6.5, weight="semibold")
    label(ax, 15.2, 33.8, "fit", color=MID, size=5.8)
    label(ax, 15.7, 15.0, "score", color=MID, size=5.8)

    # Fork arrows occupy the white gap deliberately left in the illustration.
    ax.plot([9.9, 11.5], [26.5, 26.5], color=BLACK, lw=0.9, zorder=5)
    arrow(ax, (11.5, 26.5), (13.3, 30.1), lw=0.9, mutation=6)
    arrow(ax, (11.5, 26.5), (13.4, 21.0), lw=0.9, mutation=6)

    # The held-out complete-record diagnostic is explicitly outside fitting.
    rounded(ax, 12.0, 9.1, 19.0, 13.0, fc=WHITE, ec=VERM,
            lw=0.9, radius=0.75, z=4)
    label(ax, 15.5, 11.05, "complete record", color=VERM,
          size=6.0, weight="semibold")
    arrow(ax, (15.7, 14.0), (15.7, 13.1), color=VERM,
          lw=0.9, mutation=6)


def retained_panel(ax):
    panel_header(ax, 23.0, 53.2, "LEVEL RETAINED", color=BLUE_DARK)
    label(ax, 23.0, 50.7, "between-participant height remains visible",
          color=MID, size=6.1, ha="left")

    # One shared ruler and paired pre/post locations. The same three subject
    # baselines are used on both sides, so separation is exact by construction.
    ax.plot([24.6, 24.6], [34.0, 48.4], color=BLACK, lw=0.9, zorder=5)
    for yy in (35.5, 41.0, 46.5):
        ax.plot([24.25, 24.95], [yy, yy], color=BLACK, lw=0.8, zorder=5)
    label(ax, 23.5, 41.2, "glucose level", color=MID, size=5.6,
          rotation=90)
    label(ax, 34.0, 48.7, "INPUT", color=MID, size=5.7, weight="semibold")
    label(ax, 51.0, 48.7, "TRANSFORMED", color=MID, size=5.7,
          weight="semibold")

    baselines = [35.4, 40.8, 46.0]
    shifts = [0.02, -0.015, 0.0]
    amps = [2.2, 2.7, 2.5]
    for base, sh, amp in zip(baselines, shifts, amps):
        plot_curve(ax, 27.0, 39.8, base, amp, BLACK, shift=sh,
                   lw=1.0, alpha=0.62)
        plot_curve(ax, 44.0, 56.8, base, amp * 0.88, BLUE_DARK,
                   shift=sh, lw=1.25)
        arrow(ax, (40.5, base + 0.8), (43.1, base + 0.8),
              color=BLACK, lw=0.75, mutation=5)

    arrow(ax, (58.6, baselines[0]), (58.6, baselines[-1]),
          color=BLUE_DARK, lw=1.0, mutation=6, style="<->")
    label(ax, 60.0, 40.8, "separation\nretained", color=BLUE_DARK,
          size=5.8, weight="semibold", ha="left")

    method_pill(ax, 35.5, 31.3, "global", BLUE_DARK, 8.2)
    method_pill(ax, 49.0, 31.3, "subject_scale", BLUE_DARK, 12.5)
    ax.add_patch(Circle((49.0, 29.75), 0.23, facecolor=VERM,
                        edgecolor="none", zorder=8))


def removed_mini(ax, x0, x1, mode):
    y_zero = 10.2
    y_raw = 16.0
    ax.plot([x0, x1], [y_zero, y_zero], color=BLACK, lw=0.8, zorder=4)
    ax.plot([x0, x1], [y_raw, y_raw], color=MID, lw=0.65,
            ls=(0, (2, 2)), alpha=0.75, zorder=3)

    t = np.linspace(0.0, 1.0, 180)
    raw_shape = response(t, shift={"premeal": -0.03,
                                   "center": 0.015,
                                   "z": 0.0}[mode],
                         width={"premeal": 0.14,
                                "center": 0.17,
                                "z": 0.12}[mode])
    x = x0 + (x1 - x0) * t
    raw_amp = {"premeal": 3.8, "center": 3.0, "z": 4.6}[mode]
    raw = y_raw + raw_amp * raw_shape
    ax.plot(x, raw, color=MID, lw=0.9, ls=(0, (2, 1.6)), alpha=0.75,
            zorder=5)

    if mode == "premeal":
        transformed = y_zero + 3.6 * (raw_shape - raw_shape[0])
        anchor_raw = raw[0]
        anchor_new = transformed[0]
        rule = "start to 0"
    elif mode == "center":
        centered = raw_shape - raw_shape.mean()
        transformed = y_zero + 3.3 * centered
        anchor_raw = y_raw + raw_amp * raw_shape.mean()
        anchor_new = y_zero
        rule = "mean to 0"
    else:
        standardized = (raw_shape - raw_shape.mean()) / raw_shape.std()
        transformed = y_zero + 1.25 * standardized
        anchor_raw = y_raw + raw_amp * raw_shape.mean()
        anchor_new = y_zero
        rule = "mean to 0; SD to 1"

    ax.plot(x, transformed, color=VERM, lw=1.25, zorder=7)
    arrow_x = x0 + 0.72
    arrow(ax, (arrow_x, anchor_raw), (arrow_x, anchor_new), color=VERM,
          lw=1.0, mutation=6)
    label(ax, (x0 + x1) / 2, 7.2, rule, color=MID, size=5.15)


def removed_panel(ax):
    panel_header(ax, 23.0, 26.8, "LEVEL REMOVED", color=VERM)
    label(ax, 23.0, 24.5, "absolute height is mapped to a common reference",
          color=MID, size=6.1, ha="left")

    # Shared legend for the deterministic curve geometry.
    ax.plot([51.0, 53.0], [24.5, 24.5], color=MID, lw=0.9,
            ls=(0, (2, 1.6)))
    label(ax, 53.5, 24.5, "original", color=MID, size=5.2, ha="left")
    ax.plot([57.2, 59.2], [24.5, 24.5], color=VERM, lw=1.15)
    label(ax, 59.7, 24.5, "transformed", color=MID, size=5.2, ha="left")

    ranges = [(24.0, 35.5), (37.5, 49.0), (51.0, 62.5)]
    modes = ["premeal", "center", "z"]
    names = ["premeal_center", "subject_center", "subject_z"]
    for (x0, x1), mode, name in zip(ranges, modes, names):
        removed_mini(ax, x0, x1, mode)
        label(ax, (x0 + x1) / 2, 5.3, name, color=INK,
              size=5.75, weight="medium")

    for x in (43.25, 56.75):
        ax.add_patch(Circle((x, 4.0), 0.23, facecolor=VERM,
                            edgecolor="none", zorder=8))


def leakage_probe(ax):
    dash = (0, (3.0, 2.2))
    # Trunk begins at the complete held-out record and crosses the fit boundary.
    ax.plot([15.5, 21.4, 21.4], [9.0, 9.0, 2.45], color=VERM,
            lw=0.9, ls=dash, zorder=4)
    ax.plot([21.4, 56.75], [2.45, 2.45], color=VERM,
            lw=0.9, ls=dash, zorder=4)
    label(ax, 22.0, 1.1, "leakage probe", color=VERM, size=5.9,
          weight="semibold", ha="left")

    # Exactly three targets: subject_scale, subject_center, subject_z.
    ax.plot([21.4, 21.4, 49.0], [2.45, 29.75, 29.75], color=VERM,
            lw=0.9, ls=dash, zorder=4)
    arrow(ax, (48.25, 29.75), (48.78, 29.75), color=VERM,
          lw=0.9, mutation=5, ls=dash)
    for x in (43.25, 56.75):
        arrow(ax, (x, 2.45), (x, 3.75), color=VERM,
              lw=0.9, mutation=5, ls=dash)


def model_card(ax, y0, full):
    y1 = y0 + 12.0
    rounded(ax, 66.0, y0, 82.0, y1, fc=WHITE, ec=LIGHT,
            lw=0.8, radius=0.9, z=2, shadow=True)
    title = "CGM + context" if full else "context only"
    label(ax, 67.2, y1 - 2.1, title, color=BLACK, size=7.0,
          weight="semibold", ha="left")

    # Input glyphs are intentionally schematic, not performance plots.
    if full:
        t = np.linspace(0.0, 1.0, 80)
        x = 67.3 + 4.2 * t
        y = y0 + 5.2 + 0.7 * response(t, width=0.16)
        ax.plot(x, y, color=BLUE_DARK, lw=1.0, zorder=6)
        ax.text(72.0, y0 + 5.5, "+", fontsize=8.5, color=BLACK,
                ha="center", va="center", zorder=7)
        dots_x = 73.0
    else:
        dots_x = 68.2
    for i in range(3):
        ax.add_patch(Circle((dots_x + i * 0.85, y0 + 5.5), 0.23,
                            facecolor=BLACK, edgecolor="none", zorder=6))

    arrow(ax, (76.1, y0 + 5.5), (78.0, y0 + 5.5), color=BLACK,
          lw=0.85, mutation=5)
    ax.add_patch(Circle((79.4, y0 + 5.5), 1.15, facecolor=PALE,
                        edgecolor=BLACK, linewidth=0.75, zorder=5))
    label(ax, 79.4, y0 + 5.5, r"$f$", size=8.2, weight="semibold")
    arrow(ax, (80.7, y0 + 5.5), (81.5, y0 + 5.5), color=BLACK,
          lw=0.85, mutation=5)
    label(ax, 81.6, y0 + 5.5, r"$\hat{y}$", size=8.0, ha="left")


def matched_panel(ax):
    panel_header(ax, 66.0, 53.2, "MATCHED MODELS")
    label(ax, 66.0, 50.7, "same meals | same split", color=MID,
          size=6.1, ha="left")
    model_card(ax, 34.3, full=True)
    model_card(ax, 19.5, full=False)

    label(ax, 74.0, 15.5,
          r"$R^2_{\mathrm{CGM+context}} - R^2_{\mathrm{context}}$",
          color=BLACK, size=7.1, weight="semibold")
    arrow(ax, (74.0, 13.7), (74.0, 11.2), color=GREEN,
          lw=1.1, mutation=7)
    rounded(ax, 69.5, 6.7, 78.5, 10.9, fc=WHITE, ec=GREEN,
            lw=1.1, radius=0.9, z=4)
    label(ax, 74.0, 8.8, r"$\Delta R^2$", color=GREEN,
          size=11.0, weight="semibold")
    label(ax, 74.0, 4.8, "CGM increment", color=MID, size=5.8)


def sensitivity_panel(ax):
    panel_header(ax, 86.0, 53.2, "SENSITIVITY")
    label(ax, 86.0, 50.7, "paired meals | participant clusters",
          color=MID, size=6.0, ha="left")

    rounded(ax, 86.0, 37.4, 98.0, 43.1, fc=WHITE, ec=BLUE_DARK,
            lw=0.9, radius=0.9, z=3)
    label(ax, 92.0, 40.25, r"$\Delta R^2_{\mathrm{global}}$",
          color=BLUE_DARK, size=8.5, weight="semibold")
    label(ax, 92.0, 34.6, "minus", color=MID, size=5.8,
          weight="semibold")
    rounded(ax, 86.0, 25.9, 98.0, 31.6, fc=WHITE, ec=VERM,
            lw=0.9, radius=0.9, z=3)
    label(ax, 92.0, 28.75, r"$\Delta R^2_{\mathrm{subject\_z}}$",
          color=VERM, size=8.5, weight="semibold")

    arrow(ax, (92.0, 24.0), (92.0, 20.5), color=GREEN,
          lw=1.15, mutation=7)
    rounded(ax, 86.0, 13.7, 98.0, 20.2, fc=WHITE, ec=GREEN,
            lw=1.2, radius=1.0, z=3)
    label(ax, 92.0, 16.95, r"$\Delta\Delta R^2$", color=GREEN,
          size=12.0, weight="semibold")

    # Outcome categories are small endpoints, not a conclusion banner.
    outcomes = [(87.8, "changes", BLUE_DARK),
                (92.0, "negligible", GREEN),
                (96.2, "inconclusive", MID)]
    for x, text, color in outcomes:
        ax.add_patch(Circle((x, 9.0), 0.42, facecolor=color,
                            edgecolor="none", zorder=5))
        label(ax, x, 7.3, text, color=color, size=4.9,
              weight="semibold")


def main():
    if not ASSET.exists():
        raise FileNotFoundError("Missing illustration asset: %s" % ASSET)

    fig, ax = plt.subplots(figsize=(W_IN, H_IN))
    ax.set_xlim(0.0, XMAX)
    ax.set_ylim(0.0, YMAX)
    ax.set_aspect("equal")
    ax.axis("off")

    # Fine panel rules maintain one continuous figure without card-like framing.
    for x in (20.4, 64.4, 84.4):
        ax.plot([x, x], [3.8, 53.2], color=LIGHT, lw=0.8, zorder=1)
    ax.plot([22.5, 62.5], [28.7, 28.7], color=LIGHT, lw=0.75, zorder=1)
    ax.plot([20.4, 20.4], [3.8, 53.2], color=BLACK, lw=0.75,
            ls=(0, (3.0, 2.5)), zorder=2)
    label(ax, 20.0, 51.8, "fit boundary", color=MID, size=5.2,
          rotation=90, ha="right")

    participant_panel(ax)
    retained_panel(ax)
    removed_panel(ax)
    leakage_probe(ax)
    matched_panel(ax)
    sensitivity_panel(ax)

    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    outputs = []
    for suffix in ("pdf", "png", "svg"):
        output = HERE / ("fig_protocol_hybrid." + suffix)
        fig.savefig(output, dpi=600, facecolor=WHITE,
                    bbox_inches=None, pad_inches=0)
        outputs.append(output)
        print("wrote %s" % output)
    plt.close(fig)
    print("canvas %.2f x %.2f in (16:9)" % (W_IN, H_IN))
    print("illustration asset %s" % ASSET)


if __name__ == "__main__":
    main()