"""Visibility time-averaging.

Pure-NumPy implementations of the three averaging routines historically
provided by ``ehtim.statistics.dataframes``:

- :func:`coh_avg_vis` — coherent (complex) averaging into fixed time bins
  or per scan.
- :func:`coh_moving_avg_vis` — coherent moving-window averaging.
- :func:`incoh_avg_vis` — incoherent (amplitude) averaging with Rice
  debiasing.

All three propagate visibility errors via the inverse-variance combination

.. math:: \\sigma_{\\rm avg} = 1 / \\sqrt{\\sum_i 1/\\sigma_i^2}.

This matches the channel/IF averaging in :func:`ehtim.io.load.load_obs_uvfits`
and replaces the historical ``sqrt(mean(sigma_i**2))`` formula in
``dataframes.py`` which underestimated errors and inflated SNRs.
"""

import numpy as np

import ehtim.const_def as ehc
from ehtim.statistics.stats import bootstrap, mean_incoh_avg

__all__ = ["coh_avg_vis", "coh_moving_avg_vis", "incoh_avg_vis"]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _polrep_fields(polrep):
    """Return (vis_fields, sig_fields, out_dtype) for a given polrep."""
    if polrep == "stokes":
        return (
            ("vis", "qvis", "uvis", "vvis"),
            ("sigma", "qsigma", "usigma", "vsigma"),
            ehc.DTPOL_STOKES,
        )
    if polrep == "circ":
        return (
            ("rrvis", "llvis", "rlvis", "lrvis"),
            ("rrsigma", "llsigma", "rlsigma", "lrsigma"),
            ehc.DTPOL_CIRC,
        )
    raise ValueError(f"unsupported polrep {polrep!r}")


def _assign_scan_bin(times_hr, scans):
    """Assign each row in `times_hr` to a scan index (or -1 if outside)."""
    out = np.full(len(times_hr), -1, dtype=np.int64)
    for idx, scan in enumerate(scans):
        tstart, tstop = scan[0], scan[1]
        out[(times_hr >= tstart) & (times_hr <= tstop)] = idx
    return out


def _group_ids(*keys):
    """Map a row's tuple of key values to a contiguous group index.

    Returns ``(gids, n_groups)`` where ``gids[i]`` is the group index of row i.
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
    """Return an array of length n_groups giving one representative row index
    (the first occurrence) for each group."""
    out = np.full(n_groups, -1, dtype=np.int64)
    for i, g in enumerate(gids):
        if out[g] == -1:
            out[g] = i
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


def _mean_complex(vis_per_row, gids, n_groups):
    """Reduce a per-row complex visibility array to per-group complex mean.

    Skips NaN entries.  Groups with no finite rows get ``NaN + NaN*1j``.
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


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def coh_avg_vis(obs, dt=0, scan_avg=False, return_type="rec",
                err_type="predicted", num_samples=int(1e3)):
    """Coherently average visibilities into fixed time bins or per scan.

    Parameters
    ----------
    obs : ehtim.obsdata.Obsdata
    dt : float
        Bin width in seconds.  Ignored when ``scan_avg=True``.
    scan_avg : bool
        If True, average each scan into a single bin.  Requires ``obs.scans``.
    return_type : {'rec'}
        Output format.  Only 'rec' is supported here; the legacy 'df' route
        is gone.
    err_type : {'predicted', 'measured'}
        'predicted' propagates the per-row sigmas via inverse variance.
        'measured' bootstraps the dispersion of the visibility samples.
    num_samples : int
        Bootstrap resample size when ``err_type='measured'``.

    Returns
    -------
    np.recarray of dtype DTPOL_STOKES or DTPOL_CIRC matching ``obs.polrep``.
    """
    if return_type != "rec":
        raise ValueError("only return_type='rec' is supported")
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

    keys = (data["t1"], data["t2"], data["tau1"], data["tau2"], bin_id)
    gids, n_groups = _group_ids(*keys)
    first = _first_index_per_group(gids, n_groups)

    out = np.empty(n_groups, dtype=out_dtype)
    out["t1"] = data["t1"][first]
    out["t2"] = data["t2"][first]
    out["tau1"] = data["tau1"][first]
    out["tau2"] = data["tau2"][first]
    out["tint"] = np.bincount(gids, weights=data["tint"], minlength=n_groups)
    counts = np.bincount(gids, minlength=n_groups).astype(np.float64)
    out["u"] = np.bincount(gids, weights=data["u"], minlength=n_groups) / counts
    out["v"] = np.bincount(gids, weights=data["v"], minlength=n_groups) / counts

    # Time of the bin: midpoint for fixed-dt, scan min for scan_avg.
    if scan_avg:
        time_min = np.full(n_groups, np.inf)
        np.minimum.at(time_min, gids, data["time"])
        out["time"] = time_min
    else:
        bin_id_per_group = bin_id[first]
        out["time"] = (bin_id_per_group * dt + dt / 2.0) / 3600.0

    # Visibilities (complex mean) and sigmas (inverse-variance).
    for vf in vis_fields:
        out[vf] = _mean_complex(data[vf], gids, n_groups)

    if err_type == "predicted":
        for sf in sig_fields:
            out[sf] = _inverse_variance_sigma(data[sf], gids, n_groups)
    else:  # 'measured': bootstrap the visibility-amplitude dispersion per group.
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


def coh_moving_avg_vis(obs, dt=50, return_type="rec"):
    """Coherent moving-window average over time, per baseline.

    For each row at time ``t``, average over rows in the same baseline whose
    times fall in ``[t - dt/2, t + dt/2]``.  Output preserves the input rows
    one-for-one; only the visibility values are replaced.

    Parameters
    ----------
    obs : ehtim.obsdata.Obsdata
    dt : float
        Window full-width in seconds.
    return_type : {'rec'}
        Output format.  Only 'rec' is supported.

    Returns
    -------
    np.recarray of dtype DTPOL_STOKES or DTPOL_CIRC matching ``obs.polrep``.
    """
    if return_type != "rec":
        raise ValueError("only return_type='rec' is supported")
    if dt <= 0:
        raise ValueError("dt must be positive")

    vis_fields, sig_fields, out_dtype = _polrep_fields(obs.polrep)
    data = obs.data
    n = len(data)
    if n == 0:
        return np.empty(0, dtype=out_dtype)

    half = dt / 2.0 / 3600.0  # half-width in hours
    out = data.copy()

    # Group rows by baseline (t1, t2).
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

        for vf in vis_fields:
            vals = data[vf][sorted_rows]
            re_cs = np.concatenate(([0.0], np.cumsum(np.where(np.isfinite(vals), vals.real, 0.0))))
            im_cs = np.concatenate(([0.0], np.cumsum(np.where(np.isfinite(vals), vals.imag, 0.0))))
            cnt_cs = np.concatenate(([0.0], np.cumsum(np.isfinite(vals).astype(np.float64))))
            cnt = cnt_cs[right] - cnt_cs[left]
            re_mean = np.where(cnt > 0, (re_cs[right] - re_cs[left]) / np.maximum(cnt, 1.0), np.nan)
            im_mean = np.where(cnt > 0, (im_cs[right] - im_cs[left]) / np.maximum(cnt, 1.0), np.nan)
            out[vf][sorted_rows] = re_mean + 1j * im_mean

        for sf in sig_fields:
            vals = data[sf][sorted_rows]
            finite = np.isfinite(vals) & (vals > 0)
            inv_var = np.where(finite, 1.0 / np.maximum(vals, 1e-300) ** 2, 0.0)
            inv_var_cs = np.concatenate(([0.0], np.cumsum(inv_var)))
            sums = inv_var_cs[right] - inv_var_cs[left]
            out[sf][sorted_rows] = np.where(sums > 0, 1.0 / np.sqrt(np.maximum(sums, 1e-300)), np.nan)

    return out


def incoh_avg_vis(obs, dt=0, debias=True, scan_avg=False, return_type="rec",
                  rec_type="vis", err_type="predicted", num_samples=int(1e3)):
    """Incoherently (amplitude) average visibilities, with optional Rice debias.

    Parameters
    ----------
    obs : ehtim.obsdata.Obsdata
        Must be polrep='stokes'.
    dt : float
        Bin width in seconds.  Ignored when ``scan_avg=True``.
    debias : bool
        Apply Rice debiasing to per-bin mean amplitude.
    scan_avg : bool
        If True, average each scan into a single bin.
    return_type : {'rec'}
        Output format.  Only 'rec' is supported.
    rec_type : {'vis', 'amp'}
        Output recarray dtype.  ``'vis'`` returns DTPOL_STOKES (the I,Q,U,V
        amplitudes packed into the real parts of the vis fields);
        ``'amp'`` returns DTAMP.
    err_type : {'predicted', 'measured'}
        Sigma propagation method.
    num_samples : int
        Bootstrap resample size when ``err_type='measured'``.

    Returns
    -------
    np.recarray
    """
    if return_type != "rec":
        raise ValueError("only return_type='rec' is supported")
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

    keys = (data["t1"], data["t2"], data["tau1"], data["tau2"], bin_id)
    gids, n_groups = _group_ids(*keys)
    first = _first_index_per_group(gids, n_groups)

    out = np.empty(n_groups, dtype=out_dtype)
    out["t1"] = data["t1"][first]
    out["t2"] = data["t2"][first]
    out["tint"] = np.bincount(gids, weights=data["tint"], minlength=n_groups)
    counts = np.bincount(gids, minlength=n_groups).astype(np.float64)
    out["u"] = np.bincount(gids, weights=data["u"], minlength=n_groups) / counts
    out["v"] = np.bincount(gids, weights=data["v"], minlength=n_groups) / counts

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
        amp_out = np.full(n_groups, np.nan)
        sig_out = np.full(n_groups, np.nan)
        for g in range(n_groups):
            mask = gids == g
            pairs = list(zip(np.abs(data[vf][mask]), data[sf][mask]))
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
                    sig_out[g] = data[sf][mask][0]
        if rec_type == "vis":
            out[vf] = amp_out  # cast to complex implicitly via dtype
        else:
            out["amp"] = amp_out
        out[sf] = sig_out

    return out
