"""
audit_numbers.py -- trace every number in the manuscript back to a CSV
======================================================================
The failure mode this exists to prevent is named in the plan itself (SS16, "On the
deadline"): *not a weak argument, but a strong argument delivered with inconsistent numbers
between text, tables and PDF*. Two colleague reviews now rate it the highest unmitigated
risk in the revision. It is also the one risk that is fully mechanisable, because every
number we are entitled to write already exists in a CSV.

Three checks, three modes:

  --values      Derive every registered quantity from disk and print it. THIS IS THE
                COPY SOURCE. Quote from this output, never from memory and never from
                another section of the plan.

  --plan        Check the plan's own SS0.4 index table against the CSVs. The index is a
                convenience copy and convenience copies rot; this catches the plan
                drifting from its own data.

  --tex FILE    Scan the manuscript. Every numeric literal is classified:
                  MATCH    equals a registered value at some rounding
                  SIGNED   not registered, but signed off in signed_off.txt with a reason
                  UNKNOWN  neither -- a number with no traceable provenance
                Exits non-zero if any UNKNOWN survives.

The UNKNOWN list is the point. Run against the CURRENT sn-article.tex it returns a large
list, and that is the correct result -- those are the pre-revision numbers, most of which
are invalid and must be replaced (SS15). The list shrinking to zero is the definition of
"the numerical audit is done", which SS16 item 6 owes and which nothing else measures.

WHAT THIS DOES NOT DO. It cannot tell you a number is in the wrong SENTENCE. A correctly
derived +0.1733 attached to the deep model instead of the forest passes every check here.
It verifies provenance, not predication.

Usage
    python audit_numbers.py --values
    python audit_numbers.py --plan
    python audit_numbers.py --tex ../sn-article.tex
    python audit_numbers.py --all          # values + plan + tex, one exit code
======================================================================
"""
import argparse
import os
import re
import sys

import pandas as pd

from registry import ENTRIES

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
DATA = os.path.join(ROOT, "data preprocessed")
PLAN = os.path.join(HERE, "..", "REVISION_PLAN.md")
SIGNED = os.path.join(HERE, "signed_off.txt")

# The console here is cp1252 and dies on Delta/minus signs. Everything printed is ASCII;
# the unicode lives in the report file, which is written as UTF-8.
OK, BAD = "[ok]", "[FAIL]"


# ---------------------------------------------------------------------------------------
# deriving values
# ---------------------------------------------------------------------------------------
_cache = {}


def _load(rel):
    if rel not in _cache:
        p = os.path.join(DATA, rel.replace("/", os.sep))
        if not os.path.exists(p):
            raise SystemExit(f"[abort] missing source {rel}")
        _cache[rel] = pd.read_csv(p)
    return _cache[rel]


def derive(e):
    """Resolve one entry to (values, error). Exactly one row must match -- a `where` that
    selects two rows is a silently wrong number, so it is an error, not a warning."""
    df = _load(e.source)
    m = pd.Series(True, index=df.index)
    for k, v in e.where.items():
        if k not in df.columns:
            return None, f"column {k!r} not in {e.source}"
        m &= (df[k] == v)
    sub = df[m]
    if len(sub) != 1:
        return None, f"matched {len(sub)} rows, expected 1"
    row = sub.iloc[0]
    out = []
    for c in e.cols:
        if c not in df.columns:
            return None, f"column {c!r} not in {e.source}"
        out.append(float(row[c]) * e.sign * e.scale)
    # A sign flip reverses an interval: -[lo, hi] = [-hi, -lo].
    if e.sign < 0 and len(out) == 3:
        out = [out[0], out[2], out[1]]
    return out, None


def fmt(v, dp):
    return f"{v:+.{dp}f}"


def variants(v, dp):
    """Roundings a writer might legitimately use: the entry's own precision and ONE place
    shorter, never below two decimals.

    The first version of this function was far more generous -- it stripped trailing zeros
    and matched at every precision down to one decimal. Run against the current manuscript
    it reported 17 literals as traced, and ALL SEVENTEEN WERE COINCIDENCES: `0.3` matched
    ten different registered values at once, and `0.117` -- a fold-level R-squared from the
    defective preprocessing, printed years before any of this analysis existed -- matched
    the primary interaction. A check that generous does not verify provenance, it
    manufactures it, which is worse than no check because it reports green.

    A literal at one decimal place identifies nothing in a paper whose results cluster
    between 0.1 and 0.4, so it is not eligible to match at all."""
    lo = max(2, dp - 1)
    return {f"{abs(v):.{d}f}" for d in range(dp, lo - 1, -1)}


def all_values():
    rows, errs = [], []
    for e in ENTRIES:
        vals, err = derive(e)
        if err:
            errs.append((e.id, err))
        else:
            rows.append((e, vals))
    return rows, errs


# ---------------------------------------------------------------------------------------
# --values
# ---------------------------------------------------------------------------------------
def cmd_values(report):
    rows, errs = all_values()
    lines = ["# Registered values, derived from disk", ""]
    lines.append("| id | quantity | value | 95% CI | source | plan |")
    lines.append("|---|---|---|---|---|---|")
    for e, v in rows:
        ci = f"[{fmt(v[1], e.dp)}, {fmt(v[2], e.dp)}]" if len(v) == 3 else ""
        u = f" {e.unit}" if e.unit else ""
        lines.append(f"| `{e.id}` | {e.label} | **{fmt(v[0], e.dp)}{u}** | {ci} | "
                     f"`{e.source}` | §{e.plan_section} |")
    notes = [(e, v) for e, v in rows if e.note]
    if notes:
        lines += ["", "## Entries carrying a convention that is easy to get wrong", ""]
        for e, v in notes:
            lines.append(f"- **`{e.id}`** ({fmt(v[0], e.dp)}) — {e.note}")
    with open(report, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    for i, err in errs:
        print(f"{BAD} {i}: {err}")
    print(f"{OK if not errs else BAD} derived {len(rows)}/{len(ENTRIES)} entries "
          f"-> {os.path.relpath(report, HERE)}")
    return 1 if errs else 0


# ---------------------------------------------------------------------------------------
# --plan : check SS0.4's index table against the sources
# ---------------------------------------------------------------------------------------
# The plan writes minus as U+2212 and wraps values in markdown. Grab every signed decimal.
NUM = re.compile(r"[-+−]?\d+\.\d+")


def _norm(s):
    return s.replace("−", "-")


def cmd_plan(report):
    with open(PLAN, encoding="utf-8") as fh:
        text = fh.read()
    m = re.search(r"### 0\.4 Headline numbers.*?\n(\|.*?)\n\n", text, re.S)
    if not m:
        print(f"{BAD} could not locate the SS0.4 index table in the plan")
        return 1
    table = m.group(1)
    rows, errs = all_values()
    known = set()
    for e, v in rows:
        for x in v:
            known |= variants(x, e.dp)
    signed = load_signed()

    bad, checked = [], 0
    for line in table.splitlines():
        if line.startswith("|---") or "| Quantity" in line:
            continue
        # Strip the section pointer in the last column -- "§4.5" is a cross-reference, not
        # a result, and reporting it as untraced buries the two literals that matter.
        line = re.sub(r"§\s*\d+(?:\.\d+)*[a-z]?", " ", line)
        for lit in NUM.findall(_norm(line)):
            checked += 1
            bare = lit.lstrip("+-")
            if bare not in known and bare not in signed:
                bad.append((lit, line.strip()))

    lines = ["# Plan §0.4 index vs. the CSVs", "",
             f"{checked} numeric literals checked in the index table.", ""]
    if bad:
        lines += ["## Literals with no matching registered value", ""]
        for lit, ctx in bad:
            lines.append(f"- `{lit}` in: {ctx}")
        lines += ["", "Either the index has drifted from the data, or the quantity is not "
                  "registered yet. Both are worth knowing; only the first is a defect."]
    else:
        lines.append("Every literal in the index traces to a registered value.")
    with open(report, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"{OK if not bad else BAD} plan index: {checked} literals, {len(bad)} untraced "
          f"-> {os.path.relpath(report, HERE)}")
    return 0            # advisory: unregistered quantities are expected while drafting


# ---------------------------------------------------------------------------------------
# --tex : the manuscript scan
# ---------------------------------------------------------------------------------------
# Contexts where a number is markup, not a result.
SKIP_CMD = re.compile(
    r"\\(?:cite[a-z]*|label|ref|eqref|includegraphics|documentclass|usepackage|"
    r"bibliographystyle|bibliography|input|include|vspace|hspace|setlength|"
    r"addtolength|arraystretch|tabcolsep|fontsize|selectfont|color|geometry|url|doi)"
    r"\s*(?:\[[^\]]*\])?\s*(?:\{[^}]*\})?")
DIMEN = re.compile(r"\d*\.?\d+\s*(?:\\(?:text|line|column|page)width|\\(?:text|column)height"
                   r"|pt|em|ex|cm|mm|in|bp|sp)\b")
TEXNUM = re.compile(r"(?<![\w.])[-+]?\d+(?:\.\d+)?(?![\w])")


def strip_comment(line):
    """LaTeX comments start at an unescaped %. Getting this wrong is not hypothetical --
    an unescaped % is what silently deleted two rows of Table 1 (SS15 item 8)."""
    out, i = [], 0
    while i < len(line):
        c = line[i]
        if c == "\\" and i + 1 < len(line):
            out.append(line[i:i + 2]); i += 2; continue
        if c == "%":
            break
        out.append(c); i += 1
    return "".join(out)


def load_signed():
    """signed_off.txt: `literal <TAB or 2+ spaces> reason`. A number is only exempt when
    someone has written down WHY, which is the difference between an allowlist and a
    silencer."""
    out = {}
    if not os.path.exists(SIGNED):
        return out
    with open(SIGNED, encoding="utf-8") as fh:
        for raw in fh:
            s = raw.strip()
            if not s or s.startswith("#"):
                continue
            parts = re.split(r"\t|\s{2,}", s, maxsplit=1)
            if len(parts) == 2 and parts[1].strip():
                out[parts[0].strip()] = parts[1].strip()
    return out


def cmd_tex(path, report):
    if not os.path.exists(path):
        print(f"{BAD} no such file: {path}")
        return 1
    rows, errs = all_values()
    known = {}
    for e, v in rows:
        for x in v:
            for s in variants(x, e.dp):
                known.setdefault(s, []).append(e.id)
    signed = load_signed()

    found = {}          # literal, exactly as written -> [(lineno, context)]
    with open(path, encoding="utf-8", errors="replace") as fh:
        for n, raw in enumerate(fh, 1):
            line = strip_comment(raw)
            line = SKIP_CMD.sub(" ", line)
            line = DIMEN.sub(" ", line)
            for lit in TEXNUM.findall(line):
                bare = lit.lstrip("+-")
                if re.fullmatch(r"(19|20)\d\d", bare):      # years
                    continue
                if "." not in bare and len(bare) <= 2:      # counts, indices, horizons
                    continue
                found.setdefault(bare, []).append((n, raw.strip()[:110]))

    match, ambig, sign, unknown = [], [], [], []
    for lit, hits in sorted(found.items()):
        ids = sorted(set(known.get(lit, [])))
        if lit in signed:
            sign.append((lit, signed[lit], hits))
        elif len(ids) == 1:
            match.append((lit, ids, hits))
        elif len(ids) > 1:
            # One literal, several registered quantities. Provenance is not established --
            # the audit cannot say WHICH result the sentence is quoting, so a human must.
            ambig.append((lit, ids, hits))
        else:
            unknown.append((lit, hits))

    lines = [f"# Numerical audit — `{os.path.basename(path)}`", "",
             f"- **{len(match)}** distinct literals trace to exactly one registered value",
             f"- **{len(ambig)}** match more than one registered value (ambiguous)",
             f"- **{len(sign)}** signed off in `signed_off.txt`",
             f"- **{len(unknown)}** untraced", ""]
    if ambig:
        lines += ["## Ambiguous", "",
                  "The literal equals several registered quantities at the precision "
                  "written. Quote more decimals, or confirm by hand which one the sentence "
                  "means.", "", "| literal | could be | occurrences |", "|---|---|---|"]
        for lit, ids, hits in ambig:
            lines.append(f"| `{lit}` | {', '.join(ids)} | {len(hits)} (L{hits[0][0]}) |")
        lines.append("")
    if unknown:
        lines += ["## Untraced literals", "",
                  "Each is one of: a number that must be replaced (§15), a quantity that "
                  "should be registered, or a genuine non-result that belongs in "
                  "`signed_off.txt` **with a reason**.", "",
                  "| literal | occurrences | first use |", "|---|---|---|"]
        for lit, hits in unknown:
            ctx = hits[0][1].replace("|", "\\|")
            lines.append(f"| `{lit}` | {len(hits)} (L{hits[0][0]}) | `{ctx}` |")
    if match:
        lines += ["", "## Traced", "", "| literal | registered as | occurrences |",
                  "|---|---|---|"]
        for lit, ids, hits in match:
            lines.append(f"| `{lit}` | {', '.join(ids)} | {len(hits)} |")
    if sign:
        lines += ["", "## Signed off", "", "| literal | reason | occurrences |",
                  "|---|---|---|"]
        for lit, why, hits in sign:
            lines.append(f"| `{lit}` | {why} | {len(hits)} |")
    with open(report, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")

    bad = len(unknown) + len(ambig)
    print(f"{OK if not bad else BAD} {os.path.basename(path)}: "
          f"{len(match)} traced, {len(sign)} signed, {len(ambig)} ambiguous, "
          f"{len(unknown)} UNTRACED -> {os.path.relpath(report, HERE)}")
    return 1 if bad else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--values", action="store_true")
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--tex")
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args()
    if not any([a.values, a.plan, a.tex, a.all]):
        ap.error("pick a mode")
    rc = 0
    if a.values or a.all:
        rc |= cmd_values(os.path.join(HERE, "values.md"))
    if a.plan or a.all:
        rc |= cmd_plan(os.path.join(HERE, "audit_plan.md"))
    tex = a.tex or (os.path.join(HERE, "..", "sn-article.tex") if a.all else None)
    if tex:
        rc |= cmd_tex(os.path.abspath(tex), os.path.join(HERE, "audit_manuscript.md"))
    return rc


if __name__ == "__main__":
    sys.exit(main())
