# Changelog

## Unreleased

### Fixed

- `tvlog` and `tv2log` differenced the log image against the zero pad outside the field of
  view, which asserts `I = 1 Jy` there and so rewards flux on the FOV rim. Boundary-crossing
  differences are now dropped: the pad was 54% of the reported value on a Gaussian, and it
  capped how far the regularizer could usefully be weighted. The spectral `tv_alpha`,
  `tv_beta`, `tv_alphap`, `tv_betap`, `tv_rm` and `tv_cm` had the same pad asserting a flat
  spectrum, and are fixed the same way.
- `tvlog` and `tv2log` returned NaN on a partial embed mask, since the fill was `epsilon_tv`,
  which defaults to 0. The fill is now the mean brightness, and no surviving difference reads
  it.
- Polarimetric gradients no longer go non-finite at unpolarized or empty pixels. Each division
  was cancelled algebraically rather than clamped, so valid values move by at most 1-2 ulp:
  `rho*sin(psi)/tan(psi)` is `rho*cos(psi)` (7 sites, including the `chisqgrad_vvis` data
  term), `vimage/iimage` is `make_vf_image`, and `reggrad_ptv` divided by `iimage` a numerator
  already carrying `|P| = I*m`. `reg_msimple`, which was `-inf` at `m = 0`, floors `m`.

### Changed

- `tv`, `tv2`, `ptv`, `ptv2`, `vtv`, `vtv2` are unchanged: for linear flux, complex `P` and
  Stokes `V` a zero outside the boundary means empty sky, which is a real boundary condition.

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
