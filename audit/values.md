# Registered values, derived from disk

| id | quantity | value | 95% CI | source | plan |
|---|---|---|---|---|---|
| `primary_rf` | Primary interaction ddR2, RF | **+0.1733** | [+0.1169, +0.2595] | `core/e1_interaction.csv` | §4.5 |
| `primary_deep` | Primary interaction ddR2, deep mid-fusion | **+0.1472** | [+0.1113, +0.1966] | `normalization_sensitivity/deep_sens_interaction.csv` | §5.2 |
| `mech_rf` | Level removal (leakage-free) ddR2, RF | **+0.1978** | [+0.1435, +0.2818] | `core/e1_mechanism.csv` | §4.7 |
| `mech_deep` | Level removal (leakage-free) ddR2, deep | **+0.1805** | [+0.1436, +0.2294] | `normalization_sensitivity/deep_sens_pmc_interaction.csv` | §6.2 |
| `mech_centering_rf` | Centering component ddR2, RF | **+0.1522** | [+0.0811, +0.2551] | `core/e1_mechanism.csv` | §4.7 |
| `mech_scaling_rf` | Scaling component ddR2, RF (null) | **-0.0035** | [-0.0257, +0.0218] | `core/e1_mechanism.csv` | §4.7 |
| `inflation_rf` | Context-block inflation, leakage-free RF | **+0.2799** | [+0.1018, +0.4202] | `core/e1_mechanism.csv` | §4.8 |
| `inflation_deep` | Context-block inflation, deep, published transform | **+0.1703** | [+0.0526, +0.2646] | `normalization_sensitivity/deep_sens_interaction.csv` | §4.8 |
| `cgm_global_r2` | CGM increment, reference arm, R2 | **+0.1967** | [+0.1417, +0.2840] | `core/e1_figure_data.csv` | §4.8 |
| `cgm_subjz_r2` | CGM increment, published transform, R2 | **+0.0234** | [-0.0054, +0.0551] | `core/e1_figure_data.csv` | §4.8 |
| `cgm_pmc_r2` | CGM increment, premeal_center arm, R2 | **-0.0011** | [-0.0191, +0.0204] | `core/e1_figure_data.csv` | §4.8 |
| `cgm_global_rmse` | CGM increment, reference arm, RMSE reduction | **+7.49 mg/dL** | [+5.60, +9.42] | `core/e1_mgdl_horizons.csv` | §13.8 |
| `cgm_subjz_rmse` | CGM increment, published transform, RMSE reduction | **+0.83 mg/dL** | [-0.18, +1.84] | `core/e1_mgdl_horizons.csv` | §13.8 |
| `mech_deep_rmse` | Level removal, deep, RMSE reduction, reference arm | **+7.04 mg/dL** | [+5.25, +8.83] | `normalization_sensitivity/deep_sens_pmc_increments.csv` | §6.2 |
| `e4_shape` | Shape beyond level (full window vs one reading) | **-0.003** | [-0.015, +0.010] | `normalization_sensitivity/attr_contrasts.csv` | §7.3 |
| `e4_level` | Level only, R2 | **+0.193** | [+0.138, +0.284] | `normalization_sensitivity/attr_contrasts.csv` | §7.3 |
| `e4_level_rmse` | Level only, RMSE | **-7.35 mg/dL** | [-9.36, -5.43] | `normalization_sensitivity/attr_contrasts.csv` | §7.3 |
| `e4_level_stat` | Level statistic: last bin vs 12-bin mean | **+0.033** | [+0.015, +0.060] | `normalization_sensitivity/attr_contrasts.csv` | §7.4 |
| `e5_interaction` | Activity interaction ddR2 (specificity) | **-0.0050** | [-0.0155, +0.0037] | `normalization_sensitivity/e5_interaction.csv` | §8.3 |
| `e5_activity_own` | Activity's own increment, reference arm | **-0.0280** | [-0.0445, -0.0147] | `normalization_sensitivity/e5_increments.csv` | §8.2 |
| `shanghai_primary` | Cross-cohort primary interaction ddR2 | **+0.2698** | [+0.1729, +0.3919] | `shanghai_external/shanghai_interaction.csv` | §10.1 |
| `shanghai_mech` | Cross-cohort level removal ddR2 | **+0.3645** | [+0.2894, +0.4454] | `shanghai_external/shanghai_mechanism.csv` | §10.1 |
| `shanghai_cgm_global` | Shanghai CGM increment, reference arm | **+0.5145** | [+0.4016, +0.6535] | `shanghai_external/shanghai_increments.csv` | §10.1b |
| `shanghai_cgm_subjz` | Shanghai CGM increment, published transform | **+0.2447** | [+0.1736, +0.3173] | `shanghai_external/shanghai_increments.csv` | §10.1b |
| `e3_clarkeA_interaction` | Clarke A interaction (points) | **+6.28 points** | [+2.82, +9.87] | `core/e3_clinical.csv` | §11.3 |
| `e3_w20_interaction` | Within 20 mg/dL interaction (points) | **+5.23 points** | [+1.98, +8.52] | `core/e3_clinical.csv` | §11.3 |
| `e3_w10_interaction` | Within 10 mg/dL interaction (points) | **+4.34 points** | [+2.06, +6.56] | `core/e3_clinical.csv` | §11.3 |

## Entries carrying a convention that is easy to get wrong

- **`inflation_rf`** (+0.2799) — stored as `context adds beyond CGM` global-vs-premeal_center = -0.2799; the paper reports the INFLATION, i.e. +0.280. Sign flip is deliberate (SS3.4). Reporting -0.280 in the text is a defect, not a convention.
- **`inflation_deep`** (+0.1703) — the INTERACTION (-0.1703 stored), not the per-arm increment. The per-arm figure under subject_z is 0.3388, and quoting that as the inflation overstates it twofold -- it includes the increment the reference arm already had. The first draft of this registry made exactly that error.
- **`cgm_global_rmse`** (+7.49) — e1_mgdl_horizons stores the REDUCTION (positive = error falls), the opposite of attr_contrasts and the deep files. See the header.
- **`cgm_subjz_rmse`** (+0.83) — the 'falls from 7.5 to 0.8 mg/dL' sentence pairs this with `cgm_global_rmse`. Both are reductions; neither is negative in the text.
- **`mech_deep_rmse`** (+7.04) — OPPOSITE stored convention to e1_mgdl_horizons -- stored -7.04, reported as a 7.04 mg/dL reduction. This entry and `cgm_global_rmse` are the pair that would contradict each other if signs were copied literally.
- **`e4_shape`** (-0.003) — EQUIVALENT against delta = 0.02. The only contrast in the paper where an equivalence verdict is reachable rather than a precision artifact (SS9.3).
- **`e4_level_rmse`** (-7.35) — stored negative (error falls). Paper wording follows SS7.3.
- **`e4_level_stat`** (+0.033) — the post-hoc deviation that SS7.4 obliges us to disclose.
- **`shanghai_cgm_global`** (+0.5145) — carries the 52%-vs-88% proportional-cut comparison (SS10.1b). Do not quote the interaction without it.
