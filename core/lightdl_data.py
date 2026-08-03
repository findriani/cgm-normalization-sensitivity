"""
lightdl_data.py  -- data + repeated-CV plumbing for the light-DL experiments
======================================================================
Reuses the leakage-free setup:
  * loads the RAW absolute-unit npz (binned 12-step and high-res 60-step),
  * normalizes PER FOLD, fit on training participants only (FoldNormalizer),
  * targets stay mg/dL; NRMSE/R2/MARD come from common.

Design points addressed after the methodological review:
  * SPLITS ARE DIAGNOSIS-STRATIFIED at the participant level (StratifiedGroupKFold
    for the outer folds; stratified participant sampling for validation), so no
    test/val fold drops a diagnosis class (40 participants, 11/16/13).
  * The AUXILIARY BASELINE is the true last pre-meal observation: the high-res
    last 1-min CGM reading (mg/dL), used identically for the binned and high-res
    aux experiments. (The binned last *bin* is a 5-min mean and differs by up to
    13.4 mg/dL -- do NOT use it.) Rows of the two npz are verified aligned.
  * ALIGNMENT between the two npz is asserted before any split index is shared.

Provides:
  * DataBundle(kind)             -> raw arrays (+ participant-level diagnosis)
  * assert_aligned(a, b)         -> hard check the two bundles are row-aligned
  * highres_baseline()           -> (n,) true last pre-meal CGM (mg/dL)
  * make_repeated_splits(...)    -> list of {seed,fold,train,val,test,...} splits
  * normalized_fold(bundle, split, baseline, with_aux) -> normalized arrays + aux
  * flatten(X, S)                -> tabular feature matrix (RF and tabular-attention)
No models here; no TensorFlow import.
======================================================================
"""
import os
import numpy as np
from sklearn.model_selection import StratifiedKFold
from foldwise_normalizer import FoldNormalizer
from common import compute_metrics_per_horizon, METRIC_COLS, HORIZONS, Y_RANGE  # noqa: F401

HERE = os.path.dirname(os.path.abspath(__file__))
NPZ = {
    "binned": os.path.join(HERE, "dexcom_binned_prediction_raw.npz"),   # (n,12,4)
    "highres": os.path.join(HERE, "dexcom_raw_prediction_raw.npz"),     # (n,60,4)
}


class DataBundle:
    """Raw absolute-unit arrays for one temporal resolution."""
    def __init__(self, kind):
        assert kind in NPZ, kind
        d = np.load(NPZ[kind], allow_pickle=True)
        self.kind = kind
        self.X = d["X"].astype("float32")            # (n, T, 4) absolute
        self.S = d["static"].astype("float32")       # (n, 17)
        self.y = d["y"].astype("float32")            # (n, 3) mg/dL
        self.pid = np.asarray([str(p) for p in d["participant_id"]])
        self.diag = np.asarray(d["diagnosis"]).astype(int)   # per-sample (== per participant)
        self.T = self.X.shape[1]
        self.n_dyn = self.X.shape[2] - 1
        self.n_static = self.S.shape[1]
        self.n = self.X.shape[0]


def assert_aligned(a, b):
    """Hard guarantee the two resolutions describe the SAME samples in the SAME
    order before we share split indices across them (guards against a corrupt or
    stale Colab upload)."""
    assert a.n == b.n, f"row count differs: {a.n} vs {b.n}"
    assert np.array_equal(a.pid, b.pid), "participant_id order differs between npz"
    assert np.array_equal(a.diag, b.diag), "diagnosis differs between npz"
    assert np.allclose(a.y, b.y), "targets differ between npz"
    assert np.allclose(a.S, b.S), "static features differ between npz"
    for nm, bnd in (("binned", a), ("highres", b)):
        assert not np.isnan(bnd.X).any(), f"NaNs in {nm} X"
    return True


def highres_baseline():
    """True last pre-meal CGM reading (mg/dL): the FINAL 1-min sample of the
    high-res series. Row-aligned to the binned npz, so it is the correct aux
    baseline for every experiment regardless of feature resolution."""
    d = np.load(NPZ["highres"], allow_pickle=True)
    return d["X"][:, -1, 0].astype("float32")          # (n,)


def _participant_table(pid, diag):
    """Return (subjects, subject_diag) with one diagnosis per participant."""
    subjects = np.unique(pid)
    sub_diag = np.array([int(diag[pid == s][0]) for s in subjects])
    # sanity: diagnosis is constant within a participant
    for s in subjects:
        assert len(np.unique(diag[pid == s])) == 1, f"participant {s} has >1 diagnosis"
    return subjects, sub_diag


def _stratified_val(trainval_subj, tv_diag, val_frac, rng):
    """Pick validation participants stratified by diagnosis: proportional, with
    at least one participant per present class when the budget allows."""
    n_val = max(len(np.unique(tv_diag)), int(round(len(trainval_subj) * val_frac)))
    n_val = min(n_val, len(trainval_subj) - 1)          # keep >=1 training subject
    val = []
    classes = np.unique(tv_diag)
    # guarantee one per class first
    for c in classes:
        pool = trainval_subj[tv_diag == c]
        val.append(pool[rng.integers(len(pool))])
    val = list(dict.fromkeys(val))                      # unique, keep order
    # fill the remainder proportionally at random from what's left
    remaining = [s for s in trainval_subj if s not in val]
    rng.shuffle(remaining)
    while len(val) < n_val and remaining:
        val.append(remaining.pop())
    return set(map(str, val))


def make_repeated_splits(pid, diag, n_seeds=5, n_folds=5, val_frac=0.15, base_seed=42):
    """Participant-level, diagnosis-STRATIFIED repeated CV.

    Folds are built on the 40-PARTICIPANT table with StratifiedKFold (one row per
    participant, stratified by that participant's diagnosis), then mapped back to
    sample indices. This guarantees every test/val fold covers all diagnosis
    classes -- deterministically and independent of meal-count imbalance (unlike a
    sample-level StratifiedGroupKFold). Validation is a stratified participant
    sample from the remaining train pool. A participant is wholly in exactly one of
    train/val/test within each (seed,fold). Returns a flat list of split dicts with
    row indices plus participant lists (for auditing)."""
    pid = np.asarray([str(p) for p in pid])
    diag = np.asarray(diag).astype(int)
    subjects, sub_diag = _participant_table(pid, diag)      # 40 participants + their class
    splits = []
    for s in range(n_seeds):
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=base_seed + s)
        for f, (tv_pos, te_pos) in enumerate(skf.split(subjects, sub_diag)):
            test_subj = set(map(str, subjects[te_pos].tolist()))
            tv_subj, tv_diag = subjects[tv_pos], sub_diag[tv_pos]
            val_subj = _stratified_val(tv_subj, tv_diag, val_frac,
                                       np.random.default_rng(base_seed + 1000 * (s + 1) + f))
            train_subj = set(map(str, tv_subj.tolist())) - val_subj

            tr = np.where(np.isin(pid, list(train_subj)))[0]
            va = np.where(np.isin(pid, list(val_subj)))[0]
            te = np.where(np.isin(pid, list(test_subj)))[0]
            splits.append({"seed": s, "fold": f, "train": tr, "val": va, "test": te,
                           "train_subj": sorted(train_subj), "val_subj": sorted(val_subj),
                           "test_subj": sorted(test_subj)})
    return splits


def normalized_fold(bundle, split, baseline=None, with_aux=False):
    """Return per-fold normalized arrays (fit on train only).
    Output dict per subset: X (n,T,4 normalized), S (n,17 normalized), y (n,3 mg/dL),
    row index 'idx', pid, diag, and (if with_aux) yaux (n,3) = y - baseline (mg/dL
    increment over the true pre-meal reading). `baseline` MUST be the high-res
    last-obs array (see highres_baseline()); defaults to it if not provided."""
    if baseline is None:
        baseline = highres_baseline()
    tr, va, te = split["train"], split["val"], split["test"]
    nrm = FoldNormalizer().fit(bundle.X[tr], bundle.S[tr])

    def pack(idx):
        Xn, Sn = nrm.transform(bundle.X[idx], bundle.S[idx])
        out = {"X": Xn.astype("float32"), "S": Sn.astype("float32"), "y": bundle.y[idx],
               "idx": idx, "pid": bundle.pid[idx], "diag": bundle.diag[idx]}
        if with_aux:
            out["yaux"] = (bundle.y[idx] - baseline[idx][:, None]).astype("float32")
        return out

    return {"train": pack(tr), "val": pack(va), "test": pack(te)}


def flatten(X, S):
    """(n,T,4) + (n,17) -> tabular matrix: CGM(T) + dyn(T*3) + static(17)."""
    cgm = X[:, :, 0]
    dyn = X[:, :, 1:].reshape(len(X), -1)
    return np.concatenate([cgm, dyn, S], axis=1)


if __name__ == "__main__":
    b = DataBundle("binned"); h = DataBundle("highres")
    assert_aligned(b, h)
    bl = highres_baseline()
    print(f"aligned OK | binned T={b.T} highres T={h.T} | baseline(mg/dL) sample={bl[:5].round(1)}")
    sp = make_repeated_splits(b.pid, b.diag, n_seeds=2)
    print(f"splits={len(sp)} (=n_seeds*5)")
    # audit: every test fold must contain all diagnosis classes
    import collections
    ok = True
    for s in sp:
        classes = set(b.diag[s["test"]].tolist())
        if classes != {0, 1, 2}:
            ok = False
            print("  WARNING test fold missing class:", s["seed"], s["fold"], classes)
    print("all test folds cover every diagnosis class:", ok)
    fd = normalized_fold(b, sp[0], baseline=bl, with_aux=True)
    print("train X", fd["train"]["X"].shape, "flat", flatten(fd["train"]["X"], fd["train"]["S"]).shape)
    print("val subj/class:", collections.Counter(b.diag[sp[0]["val"]].tolist()))
    print("aux sample (y - true last obs):", fd["train"]["yaux"][0].round(1))
