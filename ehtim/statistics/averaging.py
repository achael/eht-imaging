"""Visibility time-averaging.

Pure-NumPy implementations of three averaging routines:

- :func:`coh_avg_vis` — coherent (complex) averaging into fixed time bins
  or per scan.
- :func:`coh_moving_avg_vis` — coherent moving-window averaging.
- :func:`incoh_avg_vis` — incoherent (amplitude) averaging with Rice
  debiasing.

Sigma propagation uses the inverse-variance combination

.. math:: \\sigma_{\\rm avg} = 1 / \\sqrt{\\sum_i 1/\\sigma_i^2}.

With ``invvar_avg=True`` (default) visibility values are also combined with
inverse-variance weights, ``<V> = sum_i(V_i/\\sigma_i^2) / sum_i(1/\\sigma_i^2)``.
``invvar_avg=False`` reproduces the legacy direct (unweighted) mean.

Conventions worth flagging for future readers:
  - Fixed-dt bins set the output ``time`` to the bin midpoint; ``scan_avg``
    uses the earliest sample time in each scan (matches the legacy code).
  - Output ``(u, v)`` is the per-bin mean.
  - ``err_type='measured'`` bootstraps the visibility-amplitude dispersion;
    sigma is returned as half the 68% bootstrap interval width. Convention
    inherited from the legacy code, not a standard estimator.
  - For incoherent averaging with ``err_type='predicted'``, the per-bin
    sigma comes from :func:`stats.inc_sig` (eq 9.86 of Thompson et al.,
    Interferometry and Synthesis in Radio Astronomy). It is an analytic
    Rician-SNR estimator that returns the noise level which would produce
    the equivalent SNR penalty after Rice-debiased averaging — neither
    stddev nor pure inverse variance.
  - ``tau1`` / ``tau2`` (per-site opacities) are copied through from the
    first row in each bin. They are not used as grouping keys: they are
    basically always zero in practice, and using them would needlessly
    fragment bins if a site's opacity drifted within a window.
"""

import numpy as np

import ehtim.const_def as ehc
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


def _inverse_variance_sigma(sig_per_row, gids, n_groups):
    """Reduce a per-row sigma array to per-group inverse-variance sigma.

    For each group g, the result is ``1 / sqrt(sum_i 1/sigma_i**2)`` over rows
    i in group g with finite, positive sigma.  Groups with no valid rows get
    ``NaN``.
    """
    finite = np.isfinite(sig_per_row) & (sig_per_row > 0)
    inv_var = np.zeros_like(sig_per_row, dtype=np.float64)
    inv_var[finite] = 1.0 / sig_per_row[finite] ** 2
    sums = np.bincount(gids, weights=inv_var, minlength=n_groups)
    out = np.full(n_groups, np.nan)
    out[sums > 0] = 1.0 / np.sqrt(sums[sums > 0])
    return out


def _legacy_sigma(sig_per_row, gids, n_groups):
    """Reduce per-row sigma to per-group ``sqrt(sum_i sigma_i^2) / N``.

    The legacy formula used by ``dataframes.coh_avg_vis``. NOT inverse
    variance; kept here only so ``invvar_avg=False`` reproduces legacy
    output bit-for-bit. Groups with no finite rows get ``NaN``.
    """
    finite = np.isfinite(sig_per_row) & (sig_per_row > 0)
    sq = np.where(finite, sig_per_row ** 2, 0.0)
    sums = np.bincount(gids, weights=sq, minlength=n_groups)
    counts = np.bincount(gids, weights=finite.astype(np.float64), minlength=n_groups)
    out = np.full(n_groups, np.nan)
    valid = counts > 0
    out[valid] = np.sqrt(sums[valid]) / counts[valid]
    return out


def _amplitude_per_group(amps_per_row, sigs_per_row, gids, n_groups,
                         debias, err_type, num_samples, invvar_avg):
    """Reduce per-row (amplitude, sigma) pairs to per-group (mean_amp, sigma).

    err_type:
      - 'predicted': use the per-row sigmas. mean_incoh_avg applies Rice
        debiasing to the amplitudes and returns the analytic Rician-SNR
        sigma (see module docstring). When invvar_avg=True the per-group
        amplitude mean is inverse-variance weighted; the returned sigma
        still comes from inc_sig (predicted by the per-row sigmas).
      - 'measured': bootstrap the amplitude dispersion within each group.
        The reported sigma is half the 68% bootstrap interval width — a
        convention inherited from the legacy code, not stddev.
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
            if invvar_avg:
                amps_g = np.abs(amps_per_row[mask])
                sigs_g = sigs_per_row[mask]
                finite = (np.isfinite(amps_g) & np.isfinite(sigs_g)
                          & (sigs_g > 0))
                if np.any(finite):
                    w = 1.0 / sigs_g[finite] ** 2
                    a = np.sum(amps_g[finite] * w) / np.sum(w)
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


def _mean_complex(vis_per_row, gids, n_groups):
    """Per-group complex mean (direct, unweighted). NaN-safe.

    Groups with no finite rows get ``NaN + NaN*1j``.
    """
    finite = np.isfinite(vis_per_row.real) & np.isfinite(vis_per_row.imag)
    re = np.where(finite, vis_per_row.real, 0.0)
    im = np.where(finite, vis_per_row.imag, 0.0)
    sums_re = np.bincount(gids, weights=re, minlength=n_groups)
    sums_im = np.bincount(gids, weights=im, minlength=n_groups)
    counts = np.bincount(gids, weights=finite.astype(np.float64), minlength=n_groups)
    out = np.full(n_groups, np.nan + 1j * np.nan, dtype=np.complex128)
    valid = counts > 0
    out.real[valid] = sums_re[valid] / counts[valid]
    out.imag[valid] = sums_im[valid] / counts[valid]
    return out


def _invvar_mean_complex(vis_per_row, sig_per_row, gids, n_groups):
    """Per-group inverse-variance weighted complex mean. NaN-safe.

    ``<V>_g = sum_i (V_i / sigma_i^2) / sum_i (1 / sigma_i^2)`` over rows
    i in group g with finite V and finite positive sigma. Groups with no
    valid rows get ``NaN + NaN*1j``.
    """
    finite = (np.isfinite(vis_per_row.real) & np.isfinite(vis_per_row.imag)
              & np.isfinite(sig_per_row) & (sig_per_row > 0))
    inv_var = np.where(finite, 1.0 / np.maximum(sig_per_row, 1e-300) ** 2, 0.0)
    re_w = np.where(finite, vis_per_row.real * inv_var, 0.0)
    im_w = np.where(finite, vis_per_row.imag * inv_var, 0.0)
    sums_re = np.bincount(gids, weights=re_w, minlength=n_groups)
    sums_im = np.bincount(gids, weights=im_w, minlength=n_groups)
    sums_w = np.bincount(gids, weights=inv_var, minlength=n_groups)
    out = np.full(n_groups, np.nan + 1j * np.nan, dtype=np.complex128)
    valid = sums_w > 0
    out.real[valid] = sums_re[valid] / sums_w[valid]
    out.imag[valid] = sums_im[valid] / sums_w[valid]
    return out


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
            out[vf] = _invvar_mean_complex(data[vf], data[sf], gids, n_groups)
        else:
            out[vf] = _mean_complex(data[vf], gids, n_groups)

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
        complex mean (legacy behavior).

    Returns
    -------
    np.recarray of dtype DTPOL_STOKES or DTPOL_CIRC matching ``obs.polrep``.
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

        for vf, sf in zip(vis_fields, sig_fields):
            vals = data[vf][sorted_rows]
            sigs = data[sf][sorted_rows]
            if invvar_avg:
                finite_v = np.isfinite(vals.real) & np.isfinite(vals.imag)
                finite_s = np.isfinite(sigs) & (sigs > 0)
                finite = finite_v & finite_s
                inv_var = np.where(finite, 1.0 / np.maximum(sigs, 1e-300) ** 2, 0.0)
                re_w = np.where(finite, vals.real * inv_var, 0.0)
                im_w = np.where(finite, vals.imag * inv_var, 0.0)
                re_cs = np.concatenate(([0.0], np.cumsum(re_w)))
                im_cs = np.concatenate(([0.0], np.cumsum(im_w)))
                w_cs = np.concatenate(([0.0], np.cumsum(inv_var)))
                w = w_cs[right] - w_cs[left]
                re_mean = np.where(w > 0, (re_cs[right] - re_cs[left]) / np.maximum(w, 1e-300), np.nan)
                im_mean = np.where(w > 0, (im_cs[right] - im_cs[left]) / np.maximum(w, 1e-300), np.nan)
                out[vf][sorted_rows] = re_mean + 1j * im_mean
            else:
                finite = np.isfinite(vals.real) & np.isfinite(vals.imag)
                re_cs = np.concatenate(([0.0], np.cumsum(np.where(finite, vals.real, 0.0))))
                im_cs = np.concatenate(([0.0], np.cumsum(np.where(finite, vals.imag, 0.0))))
                cnt_cs = np.concatenate(([0.0], np.cumsum(finite.astype(np.float64))))
                cnt = cnt_cs[right] - cnt_cs[left]
                re_mean = np.where(cnt > 0, (re_cs[right] - re_cs[left]) / np.maximum(cnt, 1.0), np.nan)
                im_mean = np.where(cnt > 0, (im_cs[right] - im_cs[left]) / np.maximum(cnt, 1.0), np.nan)
                out[vf][sorted_rows] = re_mean + 1j * im_mean

        for sf in sig_fields:
            vals = data[sf][sorted_rows]
            finite = np.isfinite(vals) & (vals > 0)
            if invvar_avg:
                inv_var = np.where(finite, 1.0 / np.maximum(vals, 1e-300) ** 2, 0.0)
                inv_var_cs = np.concatenate(([0.0], np.cumsum(inv_var)))
                sums = inv_var_cs[right] - inv_var_cs[left]
                out[sf][sorted_rows] = np.where(
                    sums > 0, 1.0 / np.sqrt(np.maximum(sums, 1e-300)), np.nan,
                )
            else:
                # Legacy: sqrt(sum(sig_i^2)) / N within the window.
                sq = np.where(finite, vals ** 2, 0.0)
                sq_cs = np.concatenate(([0.0], np.cumsum(sq)))
                cnt_cs = np.concatenate(([0.0], np.cumsum(finite.astype(np.float64))))
                sums = sq_cs[right] - sq_cs[left]
                cnt = cnt_cs[right] - cnt_cs[left]
                out[sf][sorted_rows] = np.where(
                    cnt > 0, np.sqrt(sums) / np.maximum(cnt, 1.0), np.nan,
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
        When True (default) and ``err_type='predicted'``, the per-bin
        amplitude mean is inverse-variance weighted by the per-row sigmas
        before Rice debiasing reuses the analytic Rician-SNR estimator for
        the output sigma. When False, falls back to the legacy direct mean.

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

    # Per-group amplitude + sigma via mean_incoh_avg / bootstrap.
    for vf, sf in zip(amp_fields, sig_fields):
        amp_out, sig_out = _amplitude_per_group(
            np.abs(data[vf]), data[sf], gids, n_groups,
            debias=debias, err_type=err_type, num_samples=num_samples,
            invvar_avg=invvar_avg,
        )
        if rec_type == "vis":
            out[vf] = amp_out  # cast to complex implicitly via dtype
        else:
            out["amp"] = amp_out
        out[sf] = sig_out

    return out
