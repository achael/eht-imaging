# Changelog

## Unreleased

### Fixed

- **`tvlog` / `tv2log` no longer difference against a phantom 1 Jy pixel.** Both compute TV
  on `log(imvec)` via `reg_tv`, which pads the grid with zeros; in log space that asserts
  `I = 1 Jy` just outside the field of view, against real pixel values of 1e-4 down to
  1e-20. The pad contributed **54%** of the `reg_tvlog` value on a plain Gaussian (0.023% for
  the same code on a linear image), and its sign is inverted: it rewards piling flux onto the
  FOV rim, where linear `tv` correctly pushes it down. Measured on a reconstruction, it caps
  the regularizer's usable range -- raising the `tvlog` weight past ~50 made the image *worse*
  (nxcorr 0.931 -> 0.901 at w=1000), where it now keeps improving (0.939), with 2.4x less
  flux on the rim and a 2.4x better amplitude chi-squared.
- **The masked-pixel fill no longer reaches the `tvlog` / `tv2log` value.** Differences across
  the embed-mask boundary measured whatever was filled there rather than the image. They are
  now dropped, so any positive fill gives the same answer; the fill exists only to keep
  `log()` finite.
- **`tv_alpha`, `tv_beta`, `tv_alphap`, `tv_betap`, `tv_rm`, `tv_cm`** no longer assert a flat
  spectrum outside the field of view. Same mechanism: a zero pad on a spectral-index map means
  `alpha = 0`, which is not "nothing there".
- **Non-finite polarimetric gradients removed**, all by algebraic cancellation rather than
  clamping, so values at valid inputs are unchanged to 1-2 ulp. `rho*sin(psi)/tan(psi)` is
  exactly `rho*cos(psi)` and was `0/0` at an unpolarized pixel (7 sites, including
  `chisqgrad_vvis`, a data-fidelity term); `vimage/iimage` is exactly `make_vf_image` and was
  `0/0` at an empty pixel; and `reggrad_ptv` divided by `iimage` a quantity that already
  carried a factor `|P| = I*m`, so the ratio is just `m`. Seven functions returned a NaN at
  the values `embed` leaves at masked pixels; none do now.
- **`reg_msimple` is finite at zero polarization.** `sum(I * log(m))` is `-inf` at `m = 0`,
  which `polcv` can reach exactly, and its gradient carried `I/m`. `m` is now floored, and the
  `dR/dpsi` slot drops the division outright since `dm/dpsi = -m*tan(psi)` cancels it.

### Changed

- `tv`, `tv2`, `ptv`, `ptv2`, `vtv`, `vtv2` are **unchanged**: for linear flux, complex `P`
  and Stokes `V` a zero outside the boundary means empty sky, which is a real boundary
  condition. Their values are bit-identical and their gradients agree to 1 ulp.

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
