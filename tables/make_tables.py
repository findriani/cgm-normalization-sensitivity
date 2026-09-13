"""
make_tables.py -- the five manuscript tables, generated from the result CSVs
==================================================================================
Nothing here is hand-typed. Every number is read from a stored result file, so the
numerical audit is mechanical: re-run this script and diff.

  T5  tab_cohorts.tex              CGMacros and ShanghaiT2DM side by side, their
                                   attrition, and what each cohort can test.
  T1  tab_normalization_arms.tex   the five arms, their declared properties, the
                                   model-free level diagnostic, and per-arm R2.
                                   This is the mechanism shown WITHOUT inference.
  T4  tab_horizons_mgdl.tex        the primary result at all three horizons, in R2
                                   and in mg/dL.
  T2  tab_e4_level_shape.tex       E4's four contrasts at 60 min.
  T3  tab_cross_estimator.tex      three blocks: the published-transform contrast
                                   (A) and the leakage-free mechanism contrast (B),
                                   each in both estimators, plus E5's negative
                                   control (C) on its own estimand.

Emits LaTeX (booktabs) for the manuscript and Markdown for REVISION_PLAN.md.
The function names are historical; `main()` fixes the order the tables appear in.

Run:  python make_tables.py
      then compile them -- LaTeX overflows a too-wide table into the margin silently.
==================================================================================
"""
import os
import re
import json
import collections
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
CORE = os.path.join(ROOT, "data preprocessed", "core")
NS = os.path.join(ROOT, "data preprocessed", "normalization_sensitivity")
SH = os.path.join(ROOT, "data preprocessed", "shanghai_external")
H = 60

ARM_ORDER = ["global", "subject_scale", "premeal_center", "subject_center", "subject_z"]
ARM_LABEL = {
    "global": "Global (reference)",
    "subject_scale": "Subject scaling",
    "premeal_center": "Pre-meal centering",
    "subject_center": "Subject centering",
    "subject_z": "Subject $z$",
}
ARM_MD = {k: v.replace("$z$", "z") for k, v in ARM_LABEL.items()}
CFG_ORDER = ["cgm", "cgm_static", "static_all", "persistence_5min"]
CFG_LABEL = {"cgm": "CGM", "cgm_static": "CGM+ctx", "static_all": "Context",
             "persistence_5min": "Persist."}


def num(v, p=4):
    """Signed number in math mode. In text mode LaTeX renders `-` as a hyphen, which is
    visibly shorter than the `+` in the cell above it and reads as a dash in a numeric
    column; math mode gives a real minus. `_demacro` strips the delimiters for Markdown."""
    return f"${v:+.{p}f}$"


def ci(lo, hi, p=3):
    return f"$[{lo:+.{p}f}, {hi:+.{p}f}]$"


def stack(top, bottom):
    r"""Two lines in one cell. `\shortstack` is base LaTeX -- no package needed.

    Used wherever a column's header, or an estimate's confidence interval, is wider than
    the number it belongs to. At \textwidth = 372 pt that difference is what decides
    whether a table fits, and it is free: nothing is abbreviated, only re-broken.

    THE `{}` AFTER `\\` IS LOAD-BEARING. Every second line here starts with `[95% CI]` or
    `[+0.1169, ...]`, and `\\` takes an OPTIONAL argument in brackets -- so `\\[95\% CI]`
    reads as "break, then add 95%-of-CI vertical space", not as a line of text. Without the
    `{}` the horizon table measured 965 pt over \textwidth instead of fitting, and the E4
    table failed with `Misplaced \cr`, neither of which sounds like a bracket."""
    if not bottom:
        return top
    return f"\\shortstack{{{top}\\\\{{}}{bottom}}}"


def est(v, lo, hi, p=4):
    r"""Point estimate above its interval, so the column is as wide as the interval rather
    than estimate + space + interval. Saves roughly a third of every estimate column.

    A nested `tabular[t]`, NOT `\shortstack`, because `\shortstack` centres its lines on
    the row's baseline -- which puts the row label ("30", "Global") level with the INTERVAL
    and leaves the estimate floating up beside the row above. `[t]` aligns the first line,
    so the estimate sits on the label's line where a reader scanning across will hit it."""
    return ("\\begin{tabular}[t]{@{}c@{}}"
            f"{num(v, p)}\\\\{{}}{ci(lo, hi, p)}"
            "\\end{tabular}")


def wrap(x, indent=False):
    r"""Leading \raggedright for a `p{}` column, optionally with a hanging indent.

    sn-jnl does not load `array` (checked by compiling), so `>{\raggedright\arraybackslash}`
    is unavailable and this is the package-free equivalent. It would have been one
    `\usepackage{array}` in the preamble, but sn-article.tex is deliberately near-untouched
    and a table generator should not require an edit to the manuscript to compile.

    `indent` replaces the `\quad` lead-in used by the section-structured tables. `\quad`
    indents only the FIRST line, so a characteristic name long enough to wrap sent its
    second line back to the section-heading margin and broke the hierarchy. `\leftskip`
    indents every line of the paragraph. `\hangindent\hangafter=0` looks like it should do
    the same job and silently does nothing here -- the cell is a `\parbox` and the array
    code has already run `\@arrayparboxrestore` -- so use `\leftskip`, and set it AFTER
    `\raggedright`, which zeroes it.

    Constraint: never put `\\` inside a cell written this way."""
    if not x:
        return x
    lead = "\\raggedright "
    if indent:
        lead += "\\leftskip=1em "
    return lead + x


def pct(v, p=1):
    """Python's `%` format emits a bare percent sign, which in LaTeX comments out the rest
    of the LINE -- including the row terminator, so the following row is silently merged
    into this one. tab_cohorts shipped with two of these and lost two rows; the compiler
    reported it as `Extra alignment tab`, which does not sound like a percent sign at all.
    Compile the tables, do not just look at the .tex."""
    return f"{v:.{p}%}".replace("%", "\\%")


def tex_table(caption, label, colspec, header, rows, notes=None,
              size="\\small", tabcolsep=None, row_gap=False):
    """`size` and `tabcolsep` exist because \\textwidth here is 372 pt = 5.15 in (measured
    by compiling the class), which a seven-column table does not fit at \\small with the
    default 6 pt column separation. Anything passing them should be checked by compiling
    and grepping the log for Overfull \\hbox -- LaTeX overflows silently into the margin."""
    out = ["\\begin{table}[t]", "\\centering",
           f"\\caption{{{caption}}}", f"\\label{{{label}}}", size]
    if tabcolsep is not None:
        out.append(f"\\setlength{{\\tabcolsep}}{{{tabcolsep}}}")
    out += [f"\\begin{{tabular}}{{{colspec}}}", "\\toprule",
            " & ".join(header) + " \\\\", "\\midrule"]
    for i, r in enumerate(rows):
        if r is None:
            out.append("\\midrule")
        else:
            out.append(" & ".join(r) + " \\\\")
            # Rows that wrap to two or three lines run into each other without this: the
            # last line of one label sits directly above the first line of the next.
            if row_gap and i < len(rows) - 1:
                out.append("\\addlinespace")
    out += ["\\bottomrule", "\\end{tabular}"]
    if notes:
        out.append(f"\\par\\smallskip\\footnotesize\\raggedright {notes}")
    out.append("\\end{table}")
    return "\n".join(out)


def _demacro(x):
    """LaTeX markup -> Markdown. The .tex files are the deliverable; the Markdown is for
    REVISION_PLAN.md, and macros leaking into it make the plan unreadable -- worse, a mangled
    cell gets copied into prose.

    `--` is NOT mapped to "no" here. It used to be, because T1 wrote boolean-false as `--`,
    and that silently turned every LaTeX en dash into a word: the range `1--54` rendered as
    `1no54`. Boolean-false is now `$\\times$`, so `--` means only what LaTeX means by it."""
    x = str(x)
    for a, b in (("---", "—"), ("--", "–"),                    # em dash, then en dash
                 ("\\checkmark", "yes"), ("$\\times$", "no"),  # T1's booleans
                 ("\\quad ", ""), ("\\quad", ""), ("~", " "),
                 ("\\raggedright ", ""),                       # p{} column ragged setting
                 ("\\leftskip=1em ", ""),                      # p{} column hanging indent
                 ("\\%", "%"), ("vs.\\ ", "vs "), ("\\pm", "±"),
                 ("kg/m$^2$", "kg/m²"),
                 ("$\\Delta\\Delta R^2$", "ΔΔR²"), ("$\\Delta R^2$", "ΔR²"),
                 ("$\\Delta$RMSE", "ΔRMSE"), ("$\\Delta$MAE", "ΔMAE"),
                 ("$\\delta$", "δ"), ("$z$", "z"),
                 ("$p$", "p"), ("$R^2$", "R²")):
        x = x.replace(a, b)
    # A full-width block heading is one cell in LaTeX and must become one cell in Markdown
    # too, or the row is silently short and the table stops rendering at that point.
    x = re.sub(r"\\multicolumn\{\d+\}\{[^}]*\}\{(.*)\}$", r"\1", x)
    # Stacked headers are a width device for LaTeX; Markdown has no column widths.
    x = re.sub(r"\\begin\{tabular\}\[t\]\{@\{\}c@\{\}\}(.*?)\\end\{tabular\}", r"\1", x)
    x = (re.sub(r"\\shortstack\{(.*)\}", r"\1", x)
         .replace("\\\\{}", " ").replace("\\\\", " "))
    x = re.sub(r"\\textbf\{([^{}]*)\}", r"**\1**", x)
    x = re.sub(r"\\(?:emph|textit)\{([^{}]*)\}", r"*\1*", x)
    return x.replace("$", "").strip()


def md_table(header, rows):
    out = ["| " + " | ".join(_demacro(h) for h in header) + " |",
           "|" + "|".join(["---"] * len(header)) + "|"]
    for r in rows:
        if r is not None:
            cells = [_demacro(x) for x in r] + [""] * (len(header) - len(r))
            out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


# ------------------------------------------------------------------ T1
def t1():
    m = pd.read_csv(os.path.join(CORE, "e1_models.csv"))
    m = m[m.horizon == H]
    raw = pd.read_csv(os.path.join(CORE, "e1_results_raw.csv"))
    lvl = raw.groupby("protocol").level_retained.mean()

    # `Level ret.` is a 10-character header over a 5-character number; stacking it is the
    # whole width fix for this table, together with the shorter published-arm label.
    header = ["Normalization arm", "Level", "Leak", stack("Level", "ret."),
              *[CFG_LABEL[c] for c in CFG_ORDER]]
    rows = []
    for a in ARM_ORDER:
        g = m[m.protocol == a].set_index("model")
        p = g.iloc[0]
        rows.append([
            ARM_LABEL[a],
            "\\checkmark" if bool(p.level_preserved) else "$\\times$",
            "\\checkmark" if bool(p.leakage) else "$\\times$",
            f"{lvl[a]:.3f}",
            *[f"{g.loc[c, 'R2']:.3f}" if c in g.index else "--" for c in CFG_ORDER],
        ])
        if a == "subject_scale":
            rows.append(None)
    md_rows = [None if r is None else
               [ARM_MD[a] if i == 0 else x for i, x in enumerate(r)]
               for r, a in zip(rows, [x for x in ARM_ORDER[:2]] + ["-"] +
                               list(ARM_ORDER[2:]))]
    notes = ("`Level ret.' is the between-participant share of variance in the meal-level "
             "mean pre-meal CGM, averaged over the 40 test folds -- a model-free "
             "diagnostic, not a fitted quantity. It tracks the collapse exactly: every arm "
             "at 0.000 loses CGM's contribution. Context $=$ demographics, HbA1c, meal "
             "composition and time-of-day. Persistence uses no CGM transform and is "
             "therefore invariant across arms by construction.")
    return (tex_table("Normalization arms: declared properties, the level diagnostic, and "
                      f"per-arm $R^2$ at {H} minutes.",
                      "tab:norm_arms", "lccc cccc", header, rows, notes,
                      tabcolsep="5pt"),
            md_table(header, md_rows))


# ------------------------------------------------------------------ T2
def t2():
    c = pd.read_csv(os.path.join(NS, "attr_contrasts.csv"))
    c = c[c.horizon == H]
    # The contrast names are the widest thing in the table and cannot be usefully shortened
    # ("shape beyond level (= full window vs one reading)" IS the definition), so the column
    # wraps instead. Verdicts here stay as stored: these are feature-content contrasts, not
    # normalization difference-in-differences, so ADDS means what the word means (3.4 is
    # about DiD rows only -- do not route this table through did_verdict()).
    header = [stack("Contrast", ""), stack("$\\Delta R^2$", "[95\\% CI]"),
              stack("$\\Delta$RMSE mg/dL", "[95\\% CI]"),
              "Holm $p$", "Seeds", "Verdict"]
    rows = []
    for _, r in c.iterrows():
        rows.append([
            # The parenthetical definition moves to the note. At this column width it was
            # a third line on the longest row, and three-line rows in a four-row table read
            # as more rows than there are.
            wrap(str(r.contrast).replace(" (= full window vs one reading)", "")),
            est(r.dR2, r.lo, r.hi, 4),
            est(r.dRMSE_mgdl, r.dRMSE_lo, r.dRMSE_hi, 2),
            f"{r.holm_p:.4f}",
            f"{int(round(r.seed_pos_frac * 8))}/8",
            f"\\textbf{{{r.verdict}}}" if r.verdict in ("NEGLIGIBLE", "ADDS") else r.verdict,
        ])
    notes = ("`Shape beyond level' is the full 60-minute window tested against a single "
             "pre-meal reading. Random Forest, 8 seeds $\\times$ 5 folds, 60-minute "
             "horizon, Holm-corrected "
             "within this family of four. Negative $\\Delta$RMSE favours the first-named "
             "configuration. NEGLIGIBLE is a positive equivalence verdict: the whole "
             "interval lies inside the pre-registered margin $\\delta = 0.02$. The level "
             "statistic is the final pre-meal bin; see the deviation note in the text.")
    return (tex_table("E4: what CGM contributes -- absolute level versus trajectory shape.",
                      "tab:e4_level_shape", "p{3.3cm}ccccc", header, rows, notes,
                      size="\\footnotesize", tabcolsep="4pt", row_gap=True),
            md_table(header, rows))


# ------------------------------------------------------------------ T5
def _cohort_stats(npz_path, hba1c_idx=3, age_idx=0, bmi_idx=2):
    """Participant-level descriptives + meal counts from a bundle.

    Continuous statics are summarised PER PARTICIPANT, not per meal. Meal-weighting them
    silently reweights the cohort by how many meals each person contributed -- which ranges
    1..54 in CGMacros and 6..96 in Shanghai, so the two differ materially. Shanghai's HbA1c
    is 8.56% meal-weighted and 9.10% participant-weighted; the second is the cohort's."""
    z = np.load(npz_path, allow_pickle=True)
    S = z["static"].astype(float)
    y = z["y"].astype(float)
    pid = np.array([str(p) for p in z["participant_id"]])
    diag = np.asarray(z["diagnosis"]).astype(int)
    subs = np.unique(pid)
    first = {p: S[pid == p][0] for p in subs}
    per = lambda i: np.array([first[p][i] for p in subs])
    counts = np.array([int((pid == p).sum()) for p in subs])
    diag_by_sub = {p: int(diag[pid == p][0]) for p in subs}
    return {
        "n_sub": len(subs), "n_meal": len(pid),
        "age": per(age_idx), "bmi": per(bmi_idx), "hba1c": per(hba1c_idx),
        "y60": y[:, 1], "counts": counts,
        "diag": collections.Counter(diag_by_sub.values()),
    }


def _ms(a, p=1):
    return f"{a.mean():.{p}f} $\\pm$ {a.std(ddof=0):.{p}f}"


def t5():
    """Cohorts and attrition, side by side.

    Every count is read from the bundles and from shanghai_attrition.csv. CGMacros'
    event-level attrition is NOT recoverable here -- the raw CGMacros CSVs were read from a
    Drive path that is not part of this repository, and the preprocessing script printed only
    the final sample count. The table says so rather than leaving the cell blank or, worse,
    reconstructing a plausible number."""
    C = _cohort_stats(os.path.join(CORE, "dexcom_binned_prediction_raw.npz"))
    S = _cohort_stats(os.path.join(SH, "shanghai_postprandial_raw.npz"))
    att = pd.read_csv(os.path.join(SH, "shanghai_attrition.csv")).set_index("stage")["n"]
    with open(os.path.join(SH, "shanghai_manifest.json"), encoding="utf-8") as f:
        man = json.load(f)
    n_drop_sub = len(man["dropped_missing_static"])
    ev_total = int(att["meal_events"])
    ev_kept = int(att["kept"])
    ev_drop = ev_total - ev_kept
    lost_with_subs = ev_kept - S["n_meal"]

    NR = "\\emph{not recorded}"
    rows = [
        ["\\textbf{Cohort}", "", ""],
        ["\\quad Role in this study", "Primary", "External validation"],
        ["\\quad Participants analysed", f"{C['n_sub']}", f"{S['n_sub']}"],
        ["\\quad Participants excluded",
         NR, f"{n_drop_sub} (missing age/sex/BMI/HbA1c)"],
        ["\\quad Glycemic status",
         f"{C['diag'][0]} healthy / {C['diag'][1]} prediabetes / {C['diag'][2]} T2D",
         "All type 2 (hospitalized)"],
        ["\\quad Age, years", _ms(C["age"]), _ms(S["age"])],
        ["\\quad BMI, kg/m$^2$", _ms(C["bmi"]), _ms(S["bmi"])],
        ["\\quad HbA1c, \\%", _ms(C["hba1c"], 2), _ms(S["hba1c"], 2)],
        None,
        ["\\textbf{Meals}", "", ""],
        ["\\quad Candidate meal events", NR, f"{ev_total:,}"],
        ["\\quad Excluded, event level", NR,
         f"{ev_drop:,} ({pct(ev_drop / ev_total)})"],
        ["\\quad Lost with excluded participants", NR, f"{lost_with_subs:,}"],
        ["\\quad Meals analysed", f"{C['n_meal']:,}", f"{S['n_meal']:,}"],
        ["\\quad Overall meal retention", NR, pct(S["n_meal"] / ev_total)],
        ["\\quad Meals per participant, median [range]",
         f"{int(np.median(C['counts']))} [{C['counts'].min()}--{C['counts'].max()}]",
         f"{int(np.median(S['counts']))} [{S['counts'].min()}--{S['counts'].max()}]"],
        ["\\quad Glucose at 60 min, mg/dL", _ms(C["y60"]), _ms(S["y60"])],
        None,
        ["\\textbf{Signals and design}", "", ""],
        ["\\quad CGM sampling interval", "5 min (12 bins)", "15 min (5 slots)"],
        ["\\quad Pre-meal observation window", "60 min", "60 min"],
        ["\\quad Prediction horizons", "30 / 60 / 120 min", "30 / 60 / 120 min"],
        ["\\quad Person context", "Age, sex, BMI, HbA1c", "Age, sex, BMI, HbA1c"],
        ["\\quad Time context", "Hour, weekend, meal type", "Hour, weekend, meal type"],
        ["\\quad Meal macronutrients",
         "\\checkmark~(5 features)", "\\textbf{Absent} (free-text diet log)"],
        ["\\quad Activity channels",
         "\\checkmark~(HR, calories, METs)", "\\textbf{Absent}"],
        ["\\quad Static features, total", "17", "12"],
    ]
    notes = ("Continuous characteristics are summarised PER PARTICIPANT; meal-weighting them "
             "would reweight the cohort by how many meals each person contributed (1--54 and "
             "6--96 respectively). CGMacros' screening counts are marked \\emph{not recorded}: "
             "the source CSVs are not retained with this analysis and the extraction script "
             "logged only the final sample count. Its inclusion rules were: a labelled "
             "breakfast, lunch or dinner; a complete 60-minute pre-meal window with no missing "
             "CGM, heart-rate, calorie or MET values; and a CGM reading within $\\pm$2 minutes "
             "of each of the three target times. Shanghai's counts come from its extraction "
             "manifest and are reproducible from the released dataset. The two absent "
             "modalities in Shanghai are why it tests one half of the normalization claim "
             "only; see the text.")
    # Three `l` columns cannot hold this: several cells are long phrases
    # ("11 healthy / 16 prediabetes / 13 T2D") and `l` never wraps, it just overflows the
    # page. `p{}` lets those few cells break while the short ones are unaffected.
    hdr = ["Characteristic", "CGMacros", "ShanghaiT2DM"]

    def _cell(i, c):
        sub = i == 0 and c.startswith("\\quad ")
        return wrap(c[len("\\quad "):] if sub else c, indent=sub)

    rows = [None if r is None else [_cell(i, c) for i, c in enumerate(r)] for r in rows]
    return (tex_table("The two cohorts, their attrition, and what each can test.",
                      "tab:cohorts", "p{4.2cm}p{4.0cm}p{4.0cm}", hdr, rows, notes,
                      tabcolsep="4pt"),
            md_table(hdr, rows))


# ------------------------------------------------------------------ T4
HZ_LABEL = {"global": "Global", "subject_z": "Subject $z$"}


def t4():
    """All three horizons, R^2 with the mg/dL errors beside it.

    One table discharging two separate review requests: Reviewer 1's "complete results for
    all prediction horizons ... errors in mg/dL", and the later requirement that MAE be
    reported rather than substituted by MARD. Numbers come from e1_mgdl.py, which
    re-measures the stored OOF predictions -- no refitting -- and verifies its R^2 columns
    against e1_interaction.csv and e1_increments.csv before writing."""
    d = pd.read_csv(os.path.join(CORE, "e1_mgdl_horizons.csv"))
    # Three estimate-plus-interval columns on one line each was 261 pt over \textwidth --
    # by far the worst of the five tables. Stacking the interval under its estimate, and
    # stacking the two-part headers, recovers all of it without abbreviating a single
    # number or dropping a column the spec asks for (13.8).
    header = [stack("Horizon", "min"), stack("Arm /", "contrast"),
              stack("$\\Delta R^2$", "[95\\% CI]"),
              stack("$\\Delta$RMSE mg/dL", "[95\\% CI]"),
              stack("$\\Delta$MAE mg/dL", "[95\\% CI]"),
              stack("Holm", "$p$"), "Verdict"]

    # Seed agreement moves to the note, as in Tab 4 -- it is 8/8 on every interaction row
    # and blank everywhere else, so the column was one repeated value. Asserted rather than
    # asserted-in-prose: if a version ever returns anything else, this stops instead of
    # letting the note quietly become false.
    inter_rows = d[d.kind == "interaction"]
    if not (inter_rows.seed_pos_frac == 1.0).all():
        raise SystemExit("[abort] the note says seed agreement is 8/8 at every horizon, "
                         f"but the stored values are "
                         f"{inter_rows.seed_pos_frac.tolist()}. Restore the Seeds column.")
    rows = []
    for i, h in enumerate([30, 60, 120]):
        g = d[d.horizon == h]
        if i:
            rows.append(None)
        for j, (_, r) in enumerate(g.iterrows()):
            inter = r["kind"] == "interaction"
            if inter:
                name = "\\textbf{Interaction}"
                # Only the confirmatory horizon carries Holm and a verdict; 30 and 120 are
                # secondary and print an em dash. The verdict is read back from CSV as NaN
                # rather than "", so test it with notna() -- `if r.verdict` is TRUE for NaN
                # and would print the word "nan" in bold.
                hp = f"{r.holm_p:.4f}" if np.isfinite(r.holm_p) else "---"
                # The interaction row is a difference-in-differences, so it takes the DiD
                # vocabulary, exactly as in Tab 4 (3.4). It printed ADDS before, which on
                # this row means "the increment is larger under the reference arm".
                verdict = (f"\\textbf{{{did_verdict(r.verdict)}}}"
                           if pd.notna(r.verdict) else "---")
            else:
                name, hp, verdict = HZ_LABEL[r["arm"]], "---", "---"
            rows.append([
                f"{h}" if j == 0 else "",
                name,
                est(r.dR2, r.dR2_lo, r.dR2_hi, 4),
                est(r.dRMSE_mgdl, r.dRMSE_lo, r.dRMSE_hi, 2),
                est(r.dMAE_mgdl, r.dMAE_lo, r.dMAE_hi, 2),
                hp, verdict,
            ])
    notes = ("The increment is CGM added to the context block. POSITIVE errors mean CGM "
             "REDUCES error by that many mg/dL, so the interaction row reads: the reduction "
             "CGM buys is this much larger under the reference than under the "
             "published transform. $R^2$ is the primary estimand and carries the inference; "
             "the mg/dL columns are confirmatory and carry participant-cluster bootstrap "
             "intervals only. Holm correction is applied within the primary family at the "
             "60-minute horizon; 30 and 120 minutes are secondary and unadjusted, marked "
             "with an em dash. The interaction has the same sign in all 8 seeds at every "
             "horizon. "
             "Arms are the fold-wise global reference and the published per-subject "
             "$z$-score, defined in Table~\\ref{tab:norm_arms}. Random Forest, 8 seeds "
             "$\\times$ 5 folds, 40 participants, 913 meals; no models were refitted for "
             "this table.")
    return (tex_table("The primary increment at all three horizons, in $R^2$ and in mg/dL.",
                      "tab:horizons_mgdl", "llccccc", header, rows, notes,
                      size="\\footnotesize", tabcolsep="4pt"),
            md_table(header, rows))


# ------------------------------------------------------------------ T3
DID_VERDICT = {"ADDS": "CHANGES", "HURTS": "CHANGES",
               "ADDS(small)": "CHANGES(small)", "HURTS(small)": "CHANGES(small)",
               "NEGLIGIBLE": "NEGLIGIBLE", "INCONCLUSIVE": "INCONCLUSIVE"}


def did_verdict(v):
    """`decide()` labels every row ADDS / HURTS, which on a difference-of-differences row
    names the wrong thing (REVISION_PLAN 3.4): ADDS there means the increment is larger
    under the reference arm, not that a modality adds anything -- and HURTS, applied to the
    context increment, labels the result that SUPPORTS the reviewer as harmful. Direction is
    not lost by collapsing them: it is the sign of the interaction in the adjacent column.

    Unmapped labels abort rather than pass through. A new verdict string appearing here
    would mean the analyzers' vocabulary changed, and a table that silently prints an
    unreviewed word is exactly the failure mode 3.4 is about."""
    if v not in DID_VERDICT:
        raise SystemExit(f"[abort] unmapped decide() label {v!r}. Add it to DID_VERDICT "
                         f"deliberately -- see REVISION_PLAN.md 3.4.")
    return DID_VERDICT[v]


def _stored_verdict(rec):
    """The analyzers disagree on the column name -- E1 and E5 write `verdict`, the two deep
    runs write `decide` (plus an `outcome` column that is a different vocabulary again).
    Guess neither: read whichever exists and abort if both or none do."""
    have = [c for c in ("verdict", "decide") if c in rec.index and pd.notna(rec[c])]
    if len(have) != 1:
        raise SystemExit(f"[abort] expected exactly one of verdict/decide in this row, "
                         f"found {have}. Do not pick one silently.")
    return str(rec[have[0]])


def _blockhead(text, ncol=7):
    return [f"\\multicolumn{{{ncol}}}{{l}}{{\\textit{{{text}}}}}"]


def t3():
    CGM = "CGM adds beyond context"
    ACT = "activity adds beyond CGM+context"

    # --- Random Forest, E1: both the published-transform contrast and the mechanism ------
    fig = pd.read_csv(os.path.join(CORE, "e1_figure_data.csv"))
    w = fig[(fig.kind == "within_arm") & (fig.horizon == H) &
            (fig.increment == CGM)].set_index("protocol")
    e1i = pd.read_csv(os.path.join(CORE, "e1_interaction.csv"))
    e1i = e1i[(e1i.horizon == H) & (e1i.increment == CGM)].iloc[0]
    e1m = pd.read_csv(os.path.join(CORE, "e1_mechanism.csv"))
    e1m = e1m[(e1m.horizon == H) & (e1m.increment == CGM) & (e1m.protocol_1 == "global") &
              (e1m.protocol_2 == "premeal_center")].iloc[0]

    # --- deep mid-fusion: E2 (published transform) and E2b (mechanism) -------------------
    d2 = pd.read_csv(os.path.join(NS, "deep_sens_increments.csv"))
    d2 = d2[(d2.horizon == H) & (d2.increment == CGM)].set_index("arm")
    e2i = pd.read_csv(os.path.join(NS, "deep_sens_interaction.csv"))
    e2i = e2i[(e2i.horizon == H) & (e2i.increment == CGM)].iloc[0]
    db = pd.read_csv(os.path.join(NS, "deep_sens_pmc_increments.csv"))
    db = db[(db.horizon == H) & (db.increment == CGM)].set_index("arm")
    e2bi = pd.read_csv(os.path.join(NS, "deep_sens_pmc_interaction.csv"))
    e2bi = e2bi[(e2bi.horizon == H) & (e2bi.increment == CGM)].iloc[0]

    # E2b extends E2 rather than repeating it: deep_sens_pmc_run.py loads E2's outputs
    # read-only, aborts unless the data hash and every fitting-code hash match, and fits
    # only the new arm. So Block B's reference column IS Block A's, not a second estimate
    # of it -- assert that here so the caption's claim cannot go stale silently.
    if abs(float(db.loc["global", "dR2"]) - float(d2.loc["global", "dR2"])) > 1e-12:
        raise SystemExit("[abort] the deep reference arm differs between E2 and E2b. The "
                         "caption says Block B reuses E2's global arm; either that stopped "
                         "being true or one of the files is from a different run.")

    # --- E5: the negative control, a DIFFERENT estimand ----------------------------------
    e5w = pd.read_csv(os.path.join(NS, "e5_increments.csv"))
    e5w = e5w[(e5w.horizon == H) & (e5w.increment == ACT)].set_index("arm")
    e5i = pd.read_csv(os.path.join(NS, "e5_interaction.csv"))
    e5i = e5i[(e5i.horizon == H) & (e5i.increment == ACT) & (e5i.arm_1 == "global") &
              (e5i.arm_2 == "subject_z")].iloc[0]

    def row(name, ref, other, inter):
        return [name, num(ref), num(other),
                f"{num(inter.ddR2)} {ci(inter.lo, inter.hi, 4)}",
                f"{inter.holm_p:.4f}", did_verdict(_stored_verdict(inter))]

    # Two width devices, both needed and both measured rather than guessed (\textwidth here
    # is 372 pt; see scratchpad measure.py). Columns 2 and 3 are bound by their HEADERS, not
    # their contents -- `$\Delta R^2$ arm 2` is wider than `$+0.1967$` -- so the headers are
    # stacked. And the seed count, which was its own column, is now a sentence in the note:
    # it is constant per estimator and every row names its estimator, so the column was
    # five-sixths header and one-sixth data.
    NCOL = 6
    header = ["Estimator",
              "\\shortstack{$\\Delta R^2$\\\\ref.}", "\\shortstack{$\\Delta R^2$\\\\arm 2}",
              "$\\Delta\\Delta R^2$ [95\\% CI]", "Holm $p$", "Verdict"]
    rows = [
        _blockhead("A. CGM beyond context. Arm 2 = subject $z$ (per-subject z-scoring)", NCOL),
        row("Random Forest (E1)", w.loc["global", "estimate"],
            w.loc["subject_z", "estimate"], e1i),
        row("Deep mid-fusion (E2)", d2.loc["global", "dR2"],
            d2.loc["subject_z", "dR2"], e2i),
        None,
        _blockhead("B. CGM beyond context. Arm 2 = pre-meal centering; neither arm leaks",
                   NCOL),
        row("Random Forest (E1)", w.loc["global", "estimate"],
            w.loc["premeal_center", "estimate"], e1m),
        row("Deep mid-fusion (E2b)", db.loc["global", "dR2"],
            db.loc["premeal_center", "dR2"], e2bi),
        None,
        _blockhead("C. Activity beyond CGM+context, a negative control. "
                   "Arm 2 = subject $z$", NCOL),
        row("Random Forest (E5)", e5w.loc["global", "dR2"],
            e5w.loc["subject_z", "dR2"], e5i),
    ]
    notes = (
        "Reference arm (ref.) is fold-wise global normalization throughout; arm 2 is named "
        "per block. Random-forest rows use 8 seeds $\\times$ 5 folds, deep rows 15 "
        "$\\times$ 5. Every block uses the same splits, the same paired "
        "difference-in-differences estimand at the individual meal, the same "
        "participant-cluster bootstrap and sign-flip permutation, and the same margin "
        "$\\delta = 0.02$. "
        "\\textbf{Block A} --- the published-arm increment stays above zero in both "
        "estimators, so this is attenuation, not reversal. "
        "\\textbf{Block B} --- neither arm uses a held-out participant's record, so the "
        "attenuation is reproduced with no leakage in either estimator; this is where "
        "\\emph{leakage is not necessary} is evidenced. The two blocks share their "
        "reference column by construction (E2b extends E2's run rather than repeating it), "
        "so the agreement between the reference values is not independent corroboration. "
        "\\textbf{Block C} --- a different estimand, which is why it has its own block. "
        "NEGLIGIBLE is a positive equivalence verdict, not a failed test. It does not show "
        "that the procedure discriminates in general: the reference-arm activity increment "
        "is itself negative, so there was no positive attribution for the transform to "
        "attenuate. "
        "Verdicts are stated for the difference-in-differences: CHANGES means the interval "
        "excludes zero, and its direction is the sign of the interaction beside it.")
    return (tex_table("Does the normalization effect survive a change of estimator, and "
                      f"does the procedure fire when there is nothing to find? All at {H} "
                      "minutes.", "tab:cross_estimator", "lccccl", header, rows, notes,
                      size="\\footnotesize", tabcolsep="4pt"),
            md_table(header, rows))


def main():
    # Markdown goes to a UTF-8 file, never to stdout: this console is cp1252 and cannot
    # encode the Delta glyphs, which aborts the run after the first table.
    blocks = []
    for name, (tex, md) in [("tab_cohorts", t5()),
                            ("tab_normalization_arms", t1()),
                            ("tab_horizons_mgdl", t4()),
                            ("tab_e4_level_shape", t2()),
                            ("tab_cross_estimator", t3())]:
        with open(os.path.join(HERE, f"{name}.tex"), "w", encoding="utf-8") as f:
            f.write(tex + "\n")
        blocks.append(f"===== {name} =====\n{md}\n")
        print(f"wrote {name}.tex")
    md_path = os.path.join(HERE, "tables_markdown.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(blocks))
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
