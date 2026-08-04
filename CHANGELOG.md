# Changelog

## Unreleased

### Behaviour changes

- **Settings assigned to an `Imager` after construction now take effect.** `Imager.__init__` builds the data products immediately, and the staleness check returned early before the first `make_image`, so anything set as an attribute rather than passed as a constructor kwarg was silently ignored. `clipfloor`, `maxset`, `ttype` and the Fourier-grid parameters were never compared at all, so changing them between runs reused data built for the old value. Both are fixed, which means a script that set those attributes and unknowingly got the old behaviour will now produce different (correct) results. Changing `clipfloor` between runs previously raised `ValueError: shapes ... not aligned` rather than merely reusing stale data, because the cached operator was built for the old embed mask; that now works.
- **The run-history accessors return snapshots.** `snrcut_last()`, `dat_terms_last()`, `reg_terms_last()` and `obslist_last()` used to hand back the live `*_next` containers, so a caller could mutate imager state through them and change detection could never see an in-place edit. They now return copies taken at the time of the run.
- **`mf` persists across calls.** Because `make_image` no longer resets it, `make_image_I(mf=True)` followed by `make_image_P()` stays multifrequency where it previously reverted to single-frequency. Pass `mf=False` explicitly to change it back.
- **`Imager(mf=True)` survives `make_image()`.** `make_image`'s `mf=False` default used to overwrite a constructor-set flag, so a multifrequency imager silently ran single-frequency on its first call. Pass `make_image(mf=False)` explicitly to turn it off.

## v1.4.0 (2026-06-02)

### Highlights

- **Pure-function imaging backend.** `Imager` now delegates the math to `ehtim.imaging.imager_backend`, paving the way for JAX support.
- **NFFT migration.** PyNFFT replaced by `finufft` (in core deps; auto-installed). Faster, easier install, JAX-ready.
- **Optional dependencies.** `pandas` and `paramsurvey` moved to the `[dev]` extra. Legacy closure-quantity helpers in `ehtim.statistics.dataframes` still work when pandas is installed.
- **Interactive plotting.** New `ehtim.plotting.interactive` module: `dashboard`, `plot_bl`, `plotall`, `plot_gains`.

### Breaking changes

- **Python 3.11 or 3.12** (3.10 dropped).
- **NumPy >= 2.0, SciPy >= 1.13, Astropy >= 6.0**.
- **finufft** required for `ttype='nfft'`. PyNFFT removed.
- **pandas**, **paramsurvey** moved from core to the `[dev]` extra. Install via `pip install ehtim[dev]` or `pip install pandas paramsurvey`.
- **`ttype='fast'`** emits `DeprecationWarning`. ~30% gradient error in the FFT path is documented; use `direct` (DFT) or `nfft` (finufft) instead.

### Bug fixes

- `polchisq` `psi = arcsin(V / (I * rho))` + `gmst_to_utc` inverse (#251).
- `_diag` chisq NumPy >= 1.24 compatibility (#233).
- Simultaneous IP/IV imaging chain rule (#228).
- Inverse-variance time averaging (#230, #252).
- `load_uvfits` RA error-message format-specifier typos (#243).
- `array.py` `except NameError` to `except KeyError`: the "no ephemeris for site X" exception path now fires.
- `obs_simulate.add_jones_and_noise` / `add_noise`: thermal noise is now seed-aware via `cerror_hash`. With the same `seed=` value, runs are bit-for-bit reproducible; with `seed=False` (default), noise is now deterministic per (site1, site2, time, polarization product) rather than drawn from process-global RNG state.

### Tests

1460+ tests; ~482 MB peak RSS baseline.

## v1.3 (2026-05-13)

### Bug fixes

- `stv_pol_grad` factor-of-2 + neighbour-roll (#240). Urgent fix to the polarimetric total-variation (`ptv`) gradient; released as v1.3.
