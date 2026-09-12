# cgm-normalization-sensitivity

Reproducibility package for:

> **A Normalization-Sensitivity Protocol for Assessing Modality Importance
> in Postprandial Glucose Prediction**
>
> Indriani, Putera, Kartini, Budiman, Nugroho,
> Universitas Lambung Mangkurat

## Overview

Normalizing CGM readings before modeling can remove the absolute glucose
level, which changes how much predictive value each input source appears
to contribute. This repository contains the preprocessed data, experiment
scripts, and result files for reproducing every analysis in the paper.

All experiments start from preprocessed `.npz` files. The raw CGMacros
and ShanghaiT2DM datasets are not required.

## The protocol

Five CGM normalization conditions isolate what happens when the absolute
glucose level is preserved, rescaled, or removed, while all other inputs
stay unchanged.

![Normalization-sensitivity protocol](fig_protocol.png)

## Quick start

```bash
pip install -r requirements.txt

cd core
python e1_run.py          # primary experiment (~30 min, CPU)
python e1_analyze.py      # produces the key result tables
```

Add `--smoke` to any runner for a fast single-seed sanity check.

## Repository structure

```
core/                          CGMacros pipeline (40 participants, 913 meals)
  ├── *.npz                    preprocessed data in absolute mg/dL
  ├── fold_splits_tahap1.json  fixed 5-fold participant-level CV splits
  ├── *.py                     experiment and analysis scripts
  └── *.csv, *.json            results and run manifests

normalization_sensitivity/     deep-model and attribution experiments
shanghai_external/             cross-cohort replication (92 participants, 3050 meals)
figures/                       scripts that generate the paper's figures
tables/                        script that generates LaTeX tables
audit/                         traces every manuscript number back to a CSV cell
```

## Experiments

| ID | Question | Scripts (in `core/` unless noted) | Key output |
|----|----------|----------------------------------|------------|
| E1 | Does normalization change what CGM contributes? | `e1_run.py` | `e1_interaction.csv` |
| E2 | Same finding in a deep model? | `normalization_sensitivity/deep_sens_run.py` | `deep_sens_interaction.csv` |
| E2b | Is pre-meal centering enough? | `normalization_sensitivity/deep_sens_pmc_run.py` | `deep_sens_pmc_interaction.csv` |
| E3 | Clinical impact in mg/dL? | `e1_mgdl.py`, `e3_clinical.py` | `e1_mgdl_horizons.csv`, `e3_clinical.csv` |
| E4 | Level or shape: what carries the signal? | `normalization_sensitivity/attr_run.py` | `attr_contrasts.csv` |
| E5 | Is the effect specific to CGM? | `normalization_sensitivity/e5_run.py` | `e5_interaction.csv` |
| Shanghai | Does it replicate in another cohort? | `shanghai_external/shanghai_run.py` | `shanghai_interaction.csv` |

Supplementary experiments (ablation, fusion, attention, window length,
traditional ML, subgroups) are in `core/` with self-explanatory filenames.

## Data

| File | Cohort | Meals | Shape of X | Note |
|------|--------|------:|------------|------|
| `core/dexcom_binned_prediction_raw.npz` | CGMacros | 913 | (913, 12, 4) | 5-min bins |
| `core/dexcom_raw_prediction_raw.npz` | CGMacros | 913 | (913, 60, 4) | 1-min resolution |
| `shanghai_external/shanghai_postprandial_raw.npz` | ShanghaiT2DM | 3050 | (3050, 5, 1) | 15-min spacing |

Each `.npz` contains CGM time series (`X`), participant features (`static`),
glucose targets at 30/60/120 min (`y`), and fold assignments. Normalization
statistics are computed at training time using `foldwise_normalizer.py`,
which fits on training folds only to prevent leakage.

## Cross-directory imports

Scripts in `normalization_sensitivity/` and `shanghai_external/` import shared
modules from `core/` via `sys.path` (`../core`). Keep the three directories
as siblings and the imports resolve.

## Hardware

E1, E3, E4, E5, ablation, traditional ML, and subgroup analyses run on
**CPU** (Random Forest, 20-30 min each). E2, E2b, attention, window, and
fusion require a **GPU** with TensorFlow.

## Statistical framework

- **Decision rule:** NEGLIGIBLE if the 95% CI falls within +-0.02 R squared;
  CHANGES if the CI excludes zero and Holm p < 0.05; INCONCLUSIVE otherwise.
- **Seeds:** 8 (RF) or 15 (deep). Each seed runs full 5-fold CV.
  CIs are participant-bootstrapped.
- **Five CGM conditions:** global (reference), subject scaling,
  pre-meal centering, subject centering, subject z-scoring (published).

## Source datasets

The preprocessed bundles are derived from publicly available data:

- **CGMacros:** https://doi.org/10.17605/OSF.IO/4GXB9
- **ShanghaiT2DM:** https://doi.org/10.6084/m9.figshare.20444633

`preprocessing.py` shows the raw-to-NPZ pipeline; `build_data.py` documents
how the leakage-free bundles were produced.
