"""Visibility time-averaging.

Pure-NumPy implementations of three averaging routines:

- :func:`coh_avg_vis` — coherent (complex) averaging into fixed time bins
  or per scan.
- :func:`coh_moving_avg_vis` — coherent moving-window averaging.
- :func:`incoh_avg_vis` — incoherent (amplitude) averaging with Rice
  debiasing.

``invvar_avg`` selects the value+sigma estimator pair on each routine.
Both halves are gated together so each branch is internally consistent:

  - ``invvar_avg=True`` (default) — inverse-variance weighted mean and
    inverse-variance sigma:

    .. math:: \\langle V \\rangle = \\sum_i (V_i / \\sigma_i^2) / \\sum_i (1 / \\sigma_i^2)
    .. math:: \\sigma_{\\rm avg} = 1 / \\sqrt{\\sum_i 1 / \\sigma_i^2}

    For ``incoh_avg_vis`` the amplitude is computed as an inverse-variance
    weighted Rice-debiased mean
    ``sqrt(max(<|V|^2>_w - (2 - 1/N) <sigma^2>_w, 0))``; the sigma is the
    same inverse-variance formula above.

  - ``invvar_avg=False`` — legacy estimator formulas:

    * ``coh_avg_vis``: direct (unweighted) complex mean with
      ``sigma_avg = sqrt(sum sigma_i^2) / N``. Reproduces
      ``dataframes.coh_avg_vis`` bit-for-bit.
    * ``incoh_avg_vis``: ``stats.deb_amp`` amplitude with
      ``stats.inc_sig`` Rician-SNR sigma (eq 9.86 of Thompson et al.,
      Interferometry and Synthesis in Radio Astronomy). Reproduces
      ``dataframes.incoh_avg_vis`` bit-for-bit.
    * ``coh_moving_avg_vis``: the same unweighted-mean and
      ``sqrt(sum sigma_i^2) / N`` formulas, but a *centered* window
      ``[t - dt/2, t + dt/2]`` with original timestamps. This does NOT
      reproduce ``dataframes.coh_moving_avg_vis`` bit-for-bit: the legacy
      version used a trailing pandas ``.rolling(dt)`` window plus a
      ``-dt/2`` timestamp shift. The centered window is the intended
      convention; the divergence is deliberate.

Mixing the two paths (e.g., inv-var amplitude with the Rician-SNR sigma)
is not a coherent estimator and must be avoided.

Conventions worth flagging for future readers:
  - Fixed-dt bins set the output ``time`` to the bin midpoint; ``scan_avg``
    uses the earliest sample time in each scan (matches the legacy code).
  - Output ``(u, v)`` is the per-bin mean.
  - ``err_type='measured'`` bootstraps the visibility-amplitude dispersion;
    sigma is returned as half the 68% bootstrap interval width. Convention
    inherited from the legacy code, not stddev. ``invvar_avg`` has no
    effect on this path.
  - ``tau1`` / ``tau2`` (per-site opacities) are copied through from the
    first row in each bin. They are not used as grouping keys: they are
    basically always zero in practice, and using them would needlessly
    fragment bins if a site's opacity drifted within a window.
  - ``_assign_scan_bin`` assumes ``obs.scans`` is pre-sorted and
    non-overlapping (legacy invariant). The pandas-based
    ``dataframes.get_bins_labels`` carried explicit overlap / midnight-
    wraparound logic; that path is not exercised by current code paths
    so it is not ported here.
"""

import numpy as np

import ehtim.const_def as ehc

# TODO: the helpers below (and the per-group amplitude reductions in this
# module) overlap conceptually with stats.deb_amp / stats.inc_sig /
# stats.mean_incoh_avg / stats.coh_sig. Worth consolidating into a single
# set of vectorised reducers once the legacy stats.py call sites elsewhere
# in the codebase are audited.
from ehtim.statistics.stats import bootstrap, mean_incoh_avg

__all__ = ["coh_avg_vis", "coh_moving_avg_vis", "incoh_avg_vis"]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _polrep_fields(polrep):
    """Return (vis_fields, sig_fields, out_dtype) for a given polrep.

    Field names are taken from ``ehc.POLDICT_STOKES`` / ``ehc.POLDICT_CIRC``
    so a future mixed-pol basis can plug in by adding the corresponding
    POLDICT entry in ``const_def`` rather than editing this module.
    """
    if polrep == "stokes":
        d = ehc.POLDICT_STOKES
        out_dtype = ehc.DTPOL_STOKES
    elif polrep == "circ":
        d = ehc.POLDICT_CIRC
        out_dtype = ehc.DTPOL_CIRC
    else:
        raise ValueError(f"unsupported polrep {polrep!r}")
    vis = tuple(d[f"vis{i}"] for i in range(1, 5))
    sig = tuple(d[f"sigma{i}"] for i in range(1, 5))
    return vis, sig, out_dtype


def _assign_scan_bin(times_hr, scans):
    """Assign each row in `times_hr` to a scan index (or -1 if outside).

    Matches the ``pandas.cut`` semantics used by the legacy code: scan
    intervals are ``(tstart, tstop]`` (open on the left, closed on the
    right) so a sample exactly on a scan boundary lands in the later scan
    only, not both.

    Legacy behaviors deliberately NOT ported from
    ``dataframes.get_bins_labels`` — flagged here in case a real dataset
    ever needs them back:

      1. **Edge padding.** The legacy default expanded each scan edge by
         ``dt=0.00001`` hr (~36 ms) before binning. Not replicated:
         scan-edge fuzz is a property of the scan table, not the binning
         code; fix the table (e.g. ``obs.add_scans(...)``) rather than
         relying on a hidden tolerance here.
      2. **Overlap merging.** The legacy code detected overlapping
         intervals in ``obs.scans`` and merged them via
         ``merge_overlapping_intervals``. Not replicated: ``obs.scans``
         is expected pre-sorted and non-overlapping. Overlapping
         intervals here would cause earlier scans to silently shadow
         later ones (the loop writes ``out[mask] = idx`` in order).
      3. **Midnight wraparound.** The legacy code patched any interval
         with ``tstop < tstart`` by adding 24 hr to ``tstop`` (so the
         scan straddled midnight). Not replicated: a wrapping interval
         here yields an empty mask and the rows get index ``-1`` (i.e.
         dropped). Callers with multi-day or midnight-crossing data
         should pre-normalise the scan table.
    """
    out = np.full(len(times_hr), -1, dtype=np.int64)
    for idx, scan in enumerate(scans):
        tstart, tstop = scan[0], scan[1]
        out[(times_hr > tstart) & (times_hr <= tstop)] = idx
    return out


def _group_ids(*keys):
    """Map a row's tuple of key values to a contiguous group index.

    Given N input arrays of length n each (one per key column), return
    ``(gids, n_groups)`` where ``gids[i]`` is a contiguous integer label
    for the unique key-tuple at row i.

    Example: with ``keys = (np.array(["a", "a", "b"]), np.array([0, 1, 0]))``
    the unique tuples are ``("a", 0), ("a", 1), ("b", 0)`` and the result
    is ``(array([0, 1, 2]), 3)`` (or another permutation; only the
    grouping is defined, not the label order).
    """
    n = len(keys[0])
    if n == 0:
        return np.empty(0, dtype=np.int64), 0
    dtype = [(f"k{i}", k.dtype) for i, k in enumerate(keys)]
    rec = np.empty(n, dtype=dtype)
    for i, k in enumerate(keys):
        rec[f"k{i}"] = k
    _, gids = np.unique(rec, return_inverse=True)
    gids = gids.astype(np.int64, copy=False)
    return gids, int(gids.max()) + 1


def _first_index_per_group(gids, n_groups):
    """Return an array of length n_groups giving the first-occurrence row
    index for each group.  Vectorised via ``np.unique(..., return_index=True)``.
    """
    unique_groups, first_idx = np.unique(gids, return_index=True)
    out = np.empty(n_groups, dtype=np.int64)
    out[unique_groups] = first_idx
    return out


# Shared weighting + segment-combine primitives.
#
# Both the fixed-bin path (``coh_avg_vis``, which reduces per-row terms into
# disjoint groups via ``np.bincount``) and the sliding-window path
# (``coh_moving_avg_vis``, which reduces overlapping windows via cumulative-
# sum differences) build per-row terms, sum them per segment, then combine
# the sums with identical arithmetic. The segment-sum mechanism differs
# (bincount over group IDs vs cumsum over window index ranges, since windows
# overlap and cannot be expressed as disjoint groups) but the combine
# formulas live here once.


def _inverse_variance_weights(sig_per_row):
    """Per-row inverse-variance weights ``1 / sigma**2``.

    Returns ``(weights, finite)``. ``weights`` is zeroed on rows with
    non-finite or non-positive sigma so they contribute nothing to a
    weighted sum; ``finite`` is the per-row validity mask.
    """
    finite = np.isfinite(sig_per_row) & (sig_per_row > 0)
    weights = np.where(finite, 1.0 / np.maximum(sig_per_row, 1e-300) ** 2, 0.0)
    return weights, finite


def _window_sums(values, left, right):
    """Sliding-window segment sums for half-open index ranges ``[left, right)``.

    ``cs[right] - cs[left]`` over a zero-prefixed cumulative sum — the
    overlapping-window analogue of ``np.bincount`` over disjoint groups.
    Relies on float64 accumulation: the ``cs[right] - cs[left]``
    subtraction loses precision if ported to float32 (catastrophic
    cancellation when the running sum dwarfs the window sum).
    """
    cumsum = np.concatenate(([0.0], np.cumsum(values)))
    return cumsum[right] - cumsum[left]


def _combine_mean_inverse_variance(sum_re_w, sum_im_w, sum_w):
    """Complex inverse-variance mean from segment sums.

    ``<V> = (sum(Re/sigma**2) + i sum(Im/sigma**2)) / sum(1/sigma**2)``.
    Segments with zero total weight get ``NaN + NaN*1j``.
    """
    out = np.full(sum_w.shape, np.nan + 1j * np.nan, dtype=np.complex128)
    valid = sum_w > 0
    out.real[valid] = sum_re_w[valid] / sum_w[valid]
    out.imag[valid] = sum_im_w[valid] / sum_w[valid]
    return out


def _combine_mean_direct(sum_re, sum_im, count):
    """Complex direct (unweighted) mean from segment sums: ``sum(V) / N``.

    Segments with no finite rows get ``NaN + NaN*1j``.
    """
    out = np.full(count.shape, np.nan + 1j * np.nan, dtype=np.complex128)
    valid = count > 0
    out.real[valid] = sum_re[valid] / count[valid]
    out.imag[valid] = sum_im[valid] / count[valid]
    return out


def _combine_sigma_inverse_variance(sum_w):
    """Inverse-variance sigma from segment sums: ``1 / sqrt(sum(1/sigma**2))``.

    Segments with zero total weight get ``NaN``.
    """
    out = np.full(sum_w.shape, np.nan)
    valid = sum_w > 0
    out[valid] = 1.0 / np.sqrt(sum_w[valid])
    return out


def _combine_sigma_legacy(sum_sq, count):
    """Legacy sigma from segment sums: ``sqrt(sum(sigma**2)) / N``.

    The formula used by ``dataframes.coh_avg_vis`` — NOT inverse variance;
    kept so ``invvar_avg=False`` reproduces legacy output bit-for-bit.
    Segments with no finite rows get ``NaN``.
    """
    out = np.full(count.shape, np.nan)
    valid = count > 0
    out[valid] = np.sqrt(sum_sq[valid]) / count[valid]
    return out


def _inverse_variance_sigma(sig_per_row, gids, n_groups):
    """Per-group inverse-variance sigma ``1 / sqrt(sum_i 1/sigma_i**2)``.

    Groups with no valid rows get ``NaN``.
    """
    weights, _ = _inverse_variance_weights(sig_per_row)
    sum_w = np.bincount(gids, weights=weights, minlength=n_groups)
    return _combine_sigma_inverse_variance(sum_w)


def _legacy_sigma(sig_per_row, gids, n_groups):
    """Per-group legacy sigma ``sqrt(sum_i sigma_i**2) / N``.

    Groups with no finite rows get ``NaN``.
    """
    finite = np.isfinite(sig_per_row) & (sig_per_row > 0)
    sq = np.where(finite, sig_per_row ** 2, 0.0)
    sum_sq = np.bincount(gids, weights=sq, minlength=n_groups)
    count = np.bincount(gids, weights=finite.astype(np.float64), minlength=n_groups)
    return _combine_sigma_legacy(sum_sq, count)


def _legacy_mean_complex(vis_per_row, gids, n_groups):
    """Per-group direct (unweighted) complex mean. NaN-safe.

    Groups with no finite rows get ``NaN + NaN*1j``.
    """
    finite = np.isfinite(vis_per_row.real) & np.isfinite(vis_per_row.imag)
    re = np.where(finite, vis_per_row.real, 0.0)
    im = np.where(finite, vis_per_row.imag, 0.0)
    sum_re = np.bincount(gids, weights=re, minlength=n_groups)
    sum_im = np.bincount(gids, weights=im, minlength=n_groups)
    count = np.bincount(gids, weights=finite.astype(np.float64), minlength=n_groups)
    return _combine_mean_direct(sum_re, sum_im, count)


def _inverse_variance_mean_complex(vis_per_row, sig_per_row, gids, n_groups):
    """Per-group inverse-variance weighted complex mean. NaN-safe.

    ``<V>_g = sum_i (V_i / sigma_i^2) / sum_i (1 / sigma_i^2)`` over rows
    i in group g with finite V and finite positive sigma. Groups with no
    valid rows get ``NaN + NaN*1j``.
    """
    weights, finite_s = _inverse_variance_weights(sig_per_row)
    finite = (finite_s & np.isfinite(vis_per_row.real)
              & np.isfinite(vis_per_row.imag))
    w = np.where(finite, weights, 0.0)
    re_w = np.where(finite, vis_per_row.real, 0.0) * w
    im_w = np.where(finite, vis_per_row.imag, 0.0) * w
    sum_re_w = np.bincount(gids, weights=re_w, minlength=n_groups)
    sum_im_w = np.bincount(gids, weights=im_w, minlength=n_groups)
    sum_w = np.bincount(gids, weights=w, minlength=n_groups)
    return _combine_mean_inverse_variance(sum_re_w, sum_im_w, sum_w)


def _legacy_mean_amplitude_group(amps_per_row, sigs_per_row, gids, n_groups,
                                 debias, err_type, num_samples):
    """Per-group (amplitude, sigma) via the legacy stats helpers.

    Reproduces ``ehtim.statistics.dataframes.incoh_avg_vis`` row-for-row.
    Used on the ``invvar_avg=False`` branch of :func:`incoh_avg_vis`.

    Parameters
    ----------
    amps_per_row : np.ndarray
        Per-row magnitudes ``|V_i|``. ``np.abs(data[vis_field])`` at the
        call site; need not be non-negative on input (taken as ``|.|``
        internally where it matters).
    sigs_per_row : np.ndarray
        Per-row visibility sigmas. Rows with non-finite or non-positive
        sigma are dropped from the per-group reductions.
    gids : np.ndarray of int64
        Length-N per-row group ID (output of :func:`_group_ids`).
    n_groups : int
        Number of distinct groups; sets the output length.
    debias : bool
        When True, the ``err_type='predicted'`` amplitude is the
        Rice-debiased ``stats.deb_amp``; when False, it is the raw
        ``sqrt(mean(|V_i|**2))``. Ignored on ``err_type='measured'``.
    err_type : {'predicted', 'measured'}
        ``'predicted'`` uses the per-row sigmas: amplitude from
        ``stats.deb_amp`` (eq 9.86 of Thompson et al.), sigma from
        ``stats.inc_sig`` (analytic Rician-SNR estimator). ``'measured'``
        bootstraps the per-bin amplitude dispersion; sigma is half the
        68% bootstrap-interval width (legacy convention, not stddev).
    num_samples : int
        Bootstrap resample count when ``err_type='measured'``.

    Returns
    -------
    amp_out : np.ndarray, shape (n_groups,)
        Per-group amplitude. ``NaN`` for empty groups.
    sig_out : np.ndarray, shape (n_groups,)
        Per-group sigma. ``NaN`` for empty groups.
    """
    amp_out = np.full(n_groups, np.nan)
    sig_out = np.full(n_groups, np.nan)
    for g in range(n_groups):
        mask = gids == g
        pairs = list(zip(amps_per_row[mask], sigs_per_row[mask]))
        if not pairs:
            continue
        if err_type == "predicted":
            a, s = mean_incoh_avg(pairs, debias=debias)
            amp_out[g] = a
            sig_out[g] = s
        else:
            amps = np.abs(np.asarray([y[0] for y in pairs]))
            amps = amps[np.isfinite(amps)]
            if len(amps) >= 2:
                centre, (lo, hi) = bootstrap(amps, np.mean,
                                             num_samples=num_samples,
                                             wrapping_variable=False)
                amp_out[g] = centre
                sig_out[g] = 0.5 * (hi - lo)
            elif len(amps) == 1:
                amp_out[g] = amps[0]
                sig_out[g] = sigs_per_row[mask][0]
    return amp_out, sig_out


def _inverse_variance_mean_amplitude_group(amps_per_row, sigs_per_row, gids,
                                           n_groups, debias):
    """Per-group (amplitude, sigma) via the inverse-variance pair.

    Both halves are inverse-variance-weighted so the output is internally
    consistent (no mixing across estimator families). Used on the
    ``invvar_avg=True`` branch of :func:`incoh_avg_vis`. For each group
    ``g``, with ``w_i = 1 / sigma_i**2`` over rows ``i`` with finite
    ``amp_i`` and ``sigma_i > 0``:

      amp**2 = sum_i(w_i * amp_i**2) / sum_i(w_i)            (not debias)
             = max(<above> - (2 - 1/N) * N / sum_i(w_i), 0)  (debias)
      sigma  = 1 / sqrt(sum_i(w_i))

    The Rice debias term ``(2 - 1/N) * <sigma**2>_w`` generalises
    ``stats.deb_amp``'s ``(2 - 1/Nc) * mean(sigma**2)`` to inverse-variance
    weights: since ``w_i * sigma_i**2 = 1``, ``<sigma**2>_w = N / sum(w_i)``
    and the correction simplifies to ``(2N - 1) / sum(w_i)``. In the
    equal-sigma limit the amplitude reduces exactly to ``stats.deb_amp``;
    the sigma does NOT reduce to ``stats.inc_sig`` — different estimator
    families — see module docstring.

    Parameters
    ----------
    amps_per_row : np.ndarray
        Per-row magnitudes ``|V_i|``. Squared before reduction.
    sigs_per_row : np.ndarray
        Per-row visibility sigmas. Rows with non-finite or non-positive
        sigma are dropped (zero inverse-variance weight).
    gids : np.ndarray of int64
        Length-N per-row group ID.
    n_groups : int
        Number of distinct groups.
    debias : bool
        Apply the inverse-variance-weighted Rice bias correction. When
        False the amplitude is the bare inverse-variance-weighted RMS.

    Returns
    -------
    amp_out : np.ndarray, shape (n_groups,)
        Per-group amplitude. ``NaN`` for groups with no finite rows.
    sig_out : np.ndarray, shape (n_groups,)
        Per-group sigma. ``NaN`` for groups with no finite rows.
    """
    amp_out = np.full(n_groups, np.nan)
    sig_out = np.full(n_groups, np.nan)
    weights, finite_s = _inverse_variance_weights(sigs_per_row)
    finite = finite_s & np.isfinite(amps_per_row)
    inv_var = np.where(finite, weights, 0.0)
    a2_w = np.where(finite, amps_per_row ** 2, 0.0) * inv_var
    sums_w = np.bincount(gids, weights=inv_var, minlength=n_groups)
    sums_a2_w = np.bincount(gids, weights=a2_w, minlength=n_groups)
    counts = np.bincount(gids, weights=finite.astype(np.float64), minlength=n_groups)
    valid = sums_w > 0
    sig_out[valid] = 1.0 / np.sqrt(sums_w[valid])
    a2_avg = np.where(valid, sums_a2_w / np.maximum(sums_w, 1e-300), 0.0)
    if debias:
        bias = np.where(valid & (counts > 0),
                        (2.0 * counts - 1.0) / np.maximum(sums_w, 1e-300), 0.0)
        a2_avg = np.maximum(a2_avg - bias, 0.0)
    amp_out[valid] = np.sqrt(a2_avg[valid])
    return amp_out, sig_out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def coh_avg_vis(obs, dt=0, scan_avg=False, err_type="predicted",
                num_samples=int(1e3), invvar_avg=True):
    """Coherently average visibilities into fixed time bins or per scan.

    Parameters
    ----------
    obs : ehtim.obsdata.Obsdata
    dt : float
        Bin width in seconds.  Ignored when ``scan_avg=True``.
    scan_avg : bool
        If True, average each scan into a single bin.  Requires ``obs.scans``.
    err_type : {'predicted', 'measured'}
        'predicted' propagates the per-row sigmas via inverse variance.
        'measured' bootstraps the dispersion of the visibility samples.
    num_samples : int
        Bootstrap resample size when ``err_type='measured'``.
    invvar_avg : bool
        When True (default), combine visibilities with inverse-variance
        weights ``<V> = sum_i(V_i/sig_i^2) / sum_i(1/sig_i^2)``. When False
        use the legacy direct (unweighted) complex mean.

    Returns
    -------
    np.recarray of dtype DTPOL_STOKES or DTPOL_CIRC matching ``obs.polrep``.
    """
    if err_type not in ("predicted", "measured"):
        raise ValueError(f"err_type must be 'predicted' or 'measured', got {err_type!r}")
    if dt <= 0 and not scan_avg:
        return obs.data

    vis_fields, sig_fields, out_dtype = _polrep_fields(obs.polrep)
    data = obs.data

    # Bin assignment.
    if scan_avg:
        if obs.scans is None or len(obs.scans) == 0:
            raise ValueError("scan_avg=True but obs has no scan table; call add_scans() first")
        bin_id = _assign_scan_bin(data["time"], obs.scans)
        keep = bin_id >= 0
        data = data[keep]
        bin_id = bin_id[keep]
    else:
        # obs.data['time'] is in hours since mjd start; dt is seconds.
        bin_id = np.floor(data["time"] * 3600.0 / dt).astype(np.int64)

    if len(data) == 0:
        return np.empty(0, dtype=out_dtype)

    # Group on baseline + bin only. tau1/tau2 (per-site opacities) are
    # carried through (first-row copy) but not used as grouping keys --
    # they are basically always zero and grouping on them would fragment
    # bins when a site's opacity drifts within a window.
    keys = (data["t1"], data["t2"], bin_id)
    gids, n_groups = _group_ids(*keys)
    first = _first_index_per_group(gids, n_groups)

    out = np.empty(n_groups, dtype=out_dtype)
    out["t1"] = data["t1"][first]
    out["t2"] = data["t2"][first]
    out["tau1"] = data["tau1"][first]
    out["tau2"] = data["tau2"][first]
    out["tint"] = np.bincount(gids, weights=data["tint"], minlength=n_groups)
    counts = np.bincount(gids, minlength=n_groups).astype(np.float64)
    # (u, v) are averaged within each bin.
    out["u"] = np.bincount(gids, weights=data["u"], minlength=n_groups) / counts
    out["v"] = np.bincount(gids, weights=data["v"], minlength=n_groups) / counts

    # Output time: midpoint of the dt window, or earliest sample in the scan.
    if scan_avg:
        time_min = np.full(n_groups, np.inf)
        np.minimum.at(time_min, gids, data["time"])
        out["time"] = time_min
    else:
        bin_id_per_group = bin_id[first]
        out["time"] = (bin_id_per_group * dt + dt / 2.0) / 3600.0

    # Visibilities: inverse-variance weighted (default) or direct mean.
    for vf, sf in zip(vis_fields, sig_fields):
        if invvar_avg:
            out[vf] = _inverse_variance_mean_complex(data[vf], data[sf], gids, n_groups)
        else:
            out[vf] = _legacy_mean_complex(data[vf], gids, n_groups)

    # Sigmas. invvar_avg gates the predicted-sigma branch so
    # invvar_avg=False reproduces the legacy sqrt(sum(sig^2))/N formula
    # bit-for-bit. The bootstrap branch is unchanged either way.
    if err_type == "predicted":
        sigma_fn = _inverse_variance_sigma if invvar_avg else _legacy_sigma
        for sf in sig_fields:
            out[sf] = sigma_fn(data[sf], gids, n_groups)
    else:
        # 'measured': bootstrap the per-bin visibility-amplitude dispersion.
        # The returned sigma is half the 68% bootstrap interval width --
        # a convention from the legacy code, not stddev.
        for vf, sf in zip(vis_fields, sig_fields):
            sig_out = np.full(n_groups, np.nan)
            for g in range(n_groups):
                rows = data[vf][gids == g]
                rows = rows[np.isfinite(rows)]
                if len(rows) >= 2:
                    lo, hi = bootstrap(np.abs(rows), np.mean,
                                       num_samples=num_samples, wrapping_variable=False)[1]
                    sig_out[g] = 0.5 * (hi - lo)
            out[sf] = sig_out

    return out


def coh_moving_avg_vis(obs, dt=50, invvar_avg=True):
    """Coherent moving-window average over time, per baseline.

    For each row at time ``t``, average over rows in the same baseline whose
    times fall in ``[t - dt/2, t + dt/2]``.  Output preserves the input rows
    one-for-one; only the visibility values are replaced.

    Parameters
    ----------
    obs : ehtim.obsdata.Obsdata
    dt : float
        Window full-width in seconds.
    invvar_avg : bool
        When True (default), combine visibilities within each window with
        inverse-variance weights. When False use the direct (unweighted)
        complex mean. Selects only the value+sigma *formula*; the window
        is centered either way (see Notes).

    Returns
    -------
    np.recarray of dtype DTPOL_STOKES or DTPOL_CIRC matching ``obs.polrep``.

    Notes
    -----
    The window is centered: ``[t - dt/2, t + dt/2]`` with the row's
    original timestamp preserved. This differs from the legacy
    ``dataframes.coh_moving_avg_vis``, which used a trailing pandas
    ``.rolling(dt)`` window ``(t - dt, t]`` followed by a ``-dt/2``
    timestamp shift. The two conventions select different samples on
    irregularly spaced data, so ``invvar_avg=False`` here does NOT
    reproduce the legacy output bit-for-bit (only the unweighted-mean and
    ``sqrt(sum sigma_i^2)/N`` formulas match). The centered window is the
    intended convention.
    """
    if dt <= 0:
        raise ValueError("dt must be positive")

    vis_fields, sig_fields, out_dtype = _polrep_fields(obs.polrep)
    data = obs.data
    n = len(data)
    if n == 0:
        return np.empty(0, dtype=out_dtype)

    half = dt / 2.0 / 3600.0  # half-width in hours
    out = data.copy()

    # Group rows by baseline (t1, t2). tau1/tau2 are NOT grouped on; see
    # the module docstring.
    bl_keys = (data["t1"], data["t2"])
    bl_ids, _ = _group_ids(*bl_keys)

    for bl in np.unique(bl_ids):
        rows = np.where(bl_ids == bl)[0]
        times = data["time"][rows]
        order = np.argsort(times, kind="stable")
        sorted_rows = rows[order]
        sorted_times = times[order]

        # For each row, the inclusive window [t - half, t + half]
        # corresponds to indices [left, right) in the sorted-time array.
        left = np.searchsorted(sorted_times, sorted_times - half, side="left")
        right = np.searchsorted(sorted_times, sorted_times + half, side="right")

        # Same combine arithmetic as coh_avg_vis (the _combine_* helpers);
        # only the per-segment sum uses _window_sums (cumsum over overlapping
        # windows) instead of np.bincount (disjoint groups).
        for vf, sf in zip(vis_fields, sig_fields):
            vals = data[vf][sorted_rows]
            sigs = data[sf][sorted_rows]
            if invvar_avg:
                weights, finite_s = _inverse_variance_weights(sigs)
                finite = finite_s & np.isfinite(vals.real) & np.isfinite(vals.imag)
                w = np.where(finite, weights, 0.0)
                re_w = np.where(finite, vals.real, 0.0) * w
                im_w = np.where(finite, vals.imag, 0.0) * w
                out[vf][sorted_rows] = _combine_mean_inverse_variance(
                    _window_sums(re_w, left, right),
                    _window_sums(im_w, left, right),
                    _window_sums(w, left, right),
                )
            else:
                finite = np.isfinite(vals.real) & np.isfinite(vals.imag)
                re = np.where(finite, vals.real, 0.0)
                im = np.where(finite, vals.imag, 0.0)
                out[vf][sorted_rows] = _combine_mean_direct(
                    _window_sums(re, left, right),
                    _window_sums(im, left, right),
                    _window_sums(finite.astype(np.float64), left, right),
                )

        for sf in sig_fields:
            vals = data[sf][sorted_rows]
            if invvar_avg:
                weights, _ = _inverse_variance_weights(vals)
                out[sf][sorted_rows] = _combine_sigma_inverse_variance(
                    _window_sums(weights, left, right),
                )
            else:
                finite = np.isfinite(vals) & (vals > 0)
                sq = np.where(finite, vals ** 2, 0.0)
                out[sf][sorted_rows] = _combine_sigma_legacy(
                    _window_sums(sq, left, right),
                    _window_sums(finite.astype(np.float64), left, right),
                )

    return out


def incoh_avg_vis(obs, dt=0, debias=True, scan_avg=False, rec_type="vis",
                  err_type="predicted", num_samples=int(1e3), invvar_avg=True):
    """Incoherently (amplitude) average visibilities, with optional Rice debias.

    Parameters
    ----------
    obs : ehtim.obsdata.Obsdata
        Must be polrep='stokes' (the rec_type='vis' / 'amp' field names below
        and the Rice-debiasing helpers are written for Stokes I, Q, U, V).
    dt : float
        Bin width in seconds.  Ignored when ``scan_avg=True``.
    debias : bool
        Apply Rice debiasing to per-bin mean amplitude.
    scan_avg : bool
        If True, average each scan into a single bin.
    rec_type : {'vis', 'amp'}
        Output recarray dtype.  ``'vis'`` returns DTPOL_STOKES (the I,Q,U,V
        amplitudes packed into the real parts of the vis fields);
        ``'amp'`` returns DTAMP.
    err_type : {'predicted', 'measured'}
        Sigma propagation method. See module docstring.
    num_samples : int
        Bootstrap resample size when ``err_type='measured'``.
    invvar_avg : bool
        Selects the (amplitude, sigma) estimator pair when
        ``err_type='predicted'``. Both halves are gated together so the
        output is internally consistent:

          - True (default): inverse-variance-weighted Rice-debiased
            amplitude paired with the inverse-variance sigma
            ``1 / sqrt(sum_i 1/sigma_i**2)``.
          - False: legacy ``stats.deb_amp`` amplitude paired with the
            ``stats.inc_sig`` Rician-SNR sigma. Reproduces the legacy
            ``ehtim.statistics.dataframes.incoh_avg_vis`` bit-for-bit.

        Has no effect when ``err_type='measured'`` (bootstrap path is
        unchanged either way).

    Returns
    -------
    np.recarray
    """
    if err_type not in ("predicted", "measured"):
        raise ValueError(f"err_type must be 'predicted' or 'measured', got {err_type!r}")
    if rec_type not in ("vis", "amp"):
        raise ValueError(f"rec_type must be 'vis' or 'amp', got {rec_type!r}")
    if obs.polrep != "stokes":
        raise ValueError("incoh_avg_vis requires polrep='stokes'")
    if dt <= 0 and not scan_avg:
        return obs.data

    data = obs.data

    if scan_avg:
        if obs.scans is None or len(obs.scans) == 0:
            raise ValueError("scan_avg=True but obs has no scan table; call add_scans() first")
        bin_id = _assign_scan_bin(data["time"], obs.scans)
        keep = bin_id >= 0
        data = data[keep]
        bin_id = bin_id[keep]
    else:
        bin_id = np.floor(data["time"] * 3600.0 / dt).astype(np.int64)

    # Field names: Stokes I, Q, U, V for rec_type='vis'; just the I
    # amplitude for rec_type='amp'. obs.polrep is enforced to 'stokes'
    # above; this list does not generalize to other bases.
    if rec_type == "vis":
        out_dtype = ehc.DTPOL_STOKES
        amp_fields = ("vis", "qvis", "uvis", "vvis")
        sig_fields = ("sigma", "qsigma", "usigma", "vsigma")
    else:
        out_dtype = ehc.DTAMP
        amp_fields = ("vis",)
        sig_fields = ("sigma",)

    if len(data) == 0:
        return np.empty(0, dtype=out_dtype)

    # Group on baseline + bin only. See coh_avg_vis / module docstring on
    # why tau1, tau2 are carried through but not grouping keys.
    keys = (data["t1"], data["t2"], bin_id)
    gids, n_groups = _group_ids(*keys)
    first = _first_index_per_group(gids, n_groups)

    out = np.empty(n_groups, dtype=out_dtype)
    out["t1"] = data["t1"][first]
    out["t2"] = data["t2"][first]
    out["tint"] = np.bincount(gids, weights=data["tint"], minlength=n_groups)
    counts = np.bincount(gids, minlength=n_groups).astype(np.float64)
    # (u, v) are averaged within each bin.
    out["u"] = np.bincount(gids, weights=data["u"], minlength=n_groups) / counts
    out["v"] = np.bincount(gids, weights=data["v"], minlength=n_groups) / counts

    # Output time: bin midpoint for fixed dt, earliest sample in the scan.
    if scan_avg:
        time_min = np.full(n_groups, np.inf)
        np.minimum.at(time_min, gids, data["time"])
        out["time"] = time_min
    else:
        bin_id_per_group = bin_id[first]
        out["time"] = (bin_id_per_group * dt + dt / 2.0) / 3600.0

    if "tau1" in out.dtype.names:
        out["tau1"] = data["tau1"][first]
        out["tau2"] = data["tau2"][first]

    # Per-group amplitude + sigma. invvar_avg gates BOTH halves so the
    # output is internally consistent on each branch:
    #   - False : legacy mean_incoh_avg (deb_amp + inc_sig). Bit-for-bit
    #     reproduces ehtim.statistics.dataframes.incoh_avg_vis.
    #   - True  : inverse-variance-weighted Rice-debiased amplitude paired
    #     with the inverse-variance sigma 1 / sqrt(sum 1/sigma^2). The
    #     coupling matters: mixing an inv-var amplitude with the Rician-
    #     SNR sigma (or vice versa) is not a coherent estimator.
    # err_type='measured' bootstraps the dispersion independently of
    # invvar_avg.
    for vf, sf in zip(amp_fields, sig_fields):
        if err_type == "predicted" and invvar_avg:
            amp_out, sig_out = _inverse_variance_mean_amplitude_group(
                np.abs(data[vf]), data[sf], gids, n_groups, debias=debias,
            )
        else:
            amp_out, sig_out = _legacy_mean_amplitude_group(
                np.abs(data[vf]), data[sf], gids, n_groups,
                debias=debias, err_type=err_type, num_samples=num_samples,
            )
        if rec_type == "vis":
            out[vf] = amp_out  # cast to complex implicitly via dtype
        else:
            out["amp"] = amp_out
        out[sf] = sig_out

    return out
