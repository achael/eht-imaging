# Test coverage inventory

Per-module test status, to track what still needs filling out before pushing to `dev`
(requested in the #271 review). Coverage measured on the full suite at the latest stack
(`pytest --cov=ehtim`), `dev-backend` + #271. Numbers are whole-module line coverage.

A module can be well covered without a `test_<module>.py` of its own (e.g. `imager_utils`
is exercised by `test_chisquared`/`test_gradients`/`test_regularizers`/`test_imager_e2e`).

## Well covered (>70%) — leave as is

| Module | Cover | Exercised by |
|---|---|---|
| `imaging/imager_backend.py` | 96% | `test_imager_backend` |
| `imaging/imager_utils.py` | 96% | `test_chisquared`, `test_gradients`, `test_regularizers`, `test_imager_e2e` |
| `const_def.py` | 90% | (constants; incidental) |
| `imaging/pol_imager_utils.py` | 89% | `test_imager_e2e`, `test_gradients` |
| `observing/obs_simulate.py` | 84% | `test_obs_simulate` |
| `imager.py` | 80% | `test_imager_*` |
| `obsdata.py` | 72% | `test_obsdata` |

## Partial (have a test file, but thin) — fill out next

| Module | Cover | Test file | Gap |
|---|---|---|---|
| `diagnostics.py` | 67% | `test_diagnostics` | minor |
| `image.py` | 63% | `test_image` | plotting/display + transform branches |
| `array.py` | 57% | `test_array` | satellite/ephemeris paths |
| `features/rex.py` | 49% | `test_rex` | only the interp/profile path is covered; plotting + `FindProfiles*` untested |
| `model.py` | 22% | `test_model` | only mring build + dispatch (parked low-priority per 2026-05-21) |
| `movie.py` | 21% | `test_movie` | only merge + HDF5 round-trip; frame ops/plotting untested |
| `imaging/starwarps.py` | 13% | `test_starwarps` | heritage; rect-image + smoke only |
| `imaging/dynamical_imaging.py` | 10% | `test_dynamical_imaging` | heritage; rect-image + smoke only |

`observing/obs_helpers.py` (52%), `scattering/stochastic_optics.py` (59%, via `test_scattering`),
`io/load.py` (36%) + `io/save.py` (24%, via `test_io`) sit between these tiers.

## Calibration cluster — characterized in PR #276

| Module | Was | Now | Test file | New tests |
|---|---|---|---|---|
| `caltable.py` | 12% | 55% | `test_caltable` | 44 |
| `calibrating/pol_cal.py` (`leakage_cal`) | 5% | 37% | `test_pol_cal` | 3 |
| `calibrating/polgains_cal.py` | 14% | 34% | `test_polgains_cal` | 5 |
| `calibrating/network_cal.py` | 10% | 22% | `test_network_cal` | 7 |
| `calibrating/self_cal.py` | 8% | 21% | `test_self_cal` | 8 |

Public API (entry points, return types, method branches, identity, gauge-invariant
recovery) is covered. The remaining uncovered lines are dominated by the per-scan
workers (`self_cal_scan`, `network_cal_scan`, `polgains_cal_scan`) called inside
multiprocessing, and the plotting helpers (`plot_dterms`, `plot_gains`,
`plot_compare_gains`, `plot_tarr_dterms`) explicitly deferred to the post-release
fillout. `pol_cal_new.py` (7%) is the experimental duplicate and out of scope.

Numbers measured via `coverage run --source=ehtim -m pytest ...` (the
`pytest-cov` plugin path hits a "cannot load module more than once" error on
numpy 2 — this reproduces on the latest `pytest-cov 7.1`/`coverage 7.13`/
`numpy 2.4` so the workaround is to use `coverage run` directly, not an
upgrade).

## No dedicated tests — candidates for new suites

Highest value (core, actively developed):

| Module | Cover | Note |
|---|---|---|
| `statistics/dataframes.py` / `stats.py` | 24% / 29% | `test_averaging` (PR #252) will lift these once it lands |

Lower priority:

| Module | Cover | Note |
|---|---|---|
| `plotting/summary_plots.py` | 2% | large; needs Agg-backend render/save smoke tests |
| `plotting/comp_plots.py` / `comparisons.py` | 9% / 15% | Alex's interactive-plotting PR (#269) is separate |
| `vex.py` | 10% | VEX schedule parsing |
| `survey.py` | 10% | paramsurvey wrapper |
| `modeling/modeling_utils.py` | 6% | parked "don't touch" (2026-05-21); revisit with a dedicated owner |
| `parloop.py` (27%), `observing/pulses.py` (40%), `calibrating/cal_helpers.py` (27%) | | small |

## Effectively dead / not worth testing now

| Module | Cover | Note |
|---|---|---|
| `imaging/clean.py` | 0% | unused ~6 yrs; slated to move to `scripts/` |
| `imaging/linearize_energy.py` | 0% | niche |
| `imaging/patch_prior.py` | 0% | niche |
