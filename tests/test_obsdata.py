"""Tests for Obsdata methods.

Method-by-method coverage of ehtim/obsdata.py. Test sections mirror the
proposal in obsdata_test_plan.md and the structure of the class itself.

All tests use the session-scoped observation fixtures from conftest.py
(`obs_direct`, `obs_pol_direct`, `obs_noisy`, `obs_gmst`, etc.) so the
expensive `Image.observe` calls are amortised across the whole module.
"""

import numpy as np
import pytest

import ehtim as eh
import ehtim.const_def as ehc

# ---------------------------------------------------------------------------
# Section 1: Construction & basic state (__init__, tarr setter, obsdata_args, copy)
# ---------------------------------------------------------------------------


def test_init_rejects_empty_table(obs_direct):
    empty = np.zeros(0, dtype=ehc.DTPOL_STOKES)
    arglist, argdict = obs_direct.obsdata_args()
    arglist[4] = empty
    with pytest.raises(Exception, match="No data"):
        eh.obsdata.Obsdata(*arglist, **argdict)


def test_init_rejects_bad_polrep(obs_direct):
    arglist, argdict = obs_direct.obsdata_args()
    argdict["polrep"] = "lin"
    with pytest.raises(Exception, match="only 'stokes' and 'circ'"):
        eh.obsdata.Obsdata(*arglist, **argdict)


def test_init_rejects_bad_dtype(obs_direct):
    arglist, argdict = obs_direct.obsdata_args()
    wrong = np.zeros(len(obs_direct.data), dtype=[("time", "f8"), ("u", "f8"), ("v", "f8")])
    arglist[4] = wrong
    with pytest.raises(Exception, match="DTPOL_STOKES or DTPOL_CIRC"):
        eh.obsdata.Obsdata(*arglist, **argdict)


def test_init_rejects_bad_timetype(obs_direct):
    arglist, argdict = obs_direct.obsdata_args()
    argdict["timetype"] = "TAI"
    with pytest.raises(Exception, match="timetype"):
        eh.obsdata.Obsdata(*arglist, **argdict)


def test_init_sets_polrep_attrs_stokes(obs_direct):
    assert obs_direct.polrep == "stokes"
    assert obs_direct.poldict is ehc.POLDICT_STOKES
    assert obs_direct.poltype is ehc.DTPOL_STOKES


def test_init_sets_polrep_attrs_circ(obs_direct):
    obs_circ = obs_direct.switch_polrep("circ")
    assert obs_circ.polrep == "circ"
    assert obs_circ.poldict is ehc.POLDICT_CIRC
    assert obs_circ.poltype is ehc.DTPOL_CIRC


def test_init_tstart_tstop_from_data(obs_direct):
    times = obs_direct.data["time"]
    # tstop may have a +24 wrap if max < min; the unwrap is the "natural" tstop.
    raw_tstop = times[-1] if times[-1] >= times[0] else times[-1] + 24.0
    assert obs_direct.tstart == pytest.approx(times[0])
    assert obs_direct.tstop == pytest.approx(raw_tstop)


def test_tarr_setter_rebuilds_tkey(obs_direct):
    obs = obs_direct.copy()
    obs.tarr = obs.tarr[::-1]
    for site in obs.tarr["site"]:
        assert obs.tarr[obs.tkey[site]]["site"] == site


def test_obsdata_args_roundtrip(obs_direct):
    arglist, argdict = obs_direct.obsdata_args()
    rebuilt = eh.obsdata.Obsdata(*arglist, **argdict)
    assert rebuilt.ra == obs_direct.ra
    assert rebuilt.dec == obs_direct.dec
    assert rebuilt.rf == obs_direct.rf
    assert rebuilt.bw == obs_direct.bw
    assert rebuilt.polrep == obs_direct.polrep
    assert rebuilt.timetype == obs_direct.timetype
    assert len(rebuilt.data) == len(obs_direct.data)
    np.testing.assert_array_equal(rebuilt.data["vis"], obs_direct.data["vis"])


def test_copy_is_deep(obs_direct):
    other = obs_direct.copy()
    other.data["u"][0] += 1.0
    assert other.data["u"][0] != obs_direct.data["u"][0]


# ---------------------------------------------------------------------------
# Section 2: Time / polrep conversion
# ---------------------------------------------------------------------------


def test_switch_timetype_updates_attribute(obs_direct):
    out = obs_direct.switch_timetype("GMST")
    assert out.timetype == "GMST"
    assert obs_direct.timetype == "UTC"  # original unchanged


def test_switch_timetype_shifts_data_time(obs_direct):
    out = obs_direct.switch_timetype("GMST")
    # GMST and UTC differ by station-independent sidereal offset; data['time']
    # must move (not all rows can coincidentally be zero-shifted).
    assert not np.allclose(out.data["time"], obs_direct.data["time"])


@pytest.mark.xfail(
    reason=(
        "obs_helpers.gmst_to_utc uses a constant sidereal-rate approximation "
        "while utc_to_gmst calls astropy per-sample, so the roundtrip drifts "
        "by a few minutes over a 24h obs. See obs_helpers.py:976."
    ),
    strict=True,
)
def test_switch_timetype_roundtrip_utc_gmst(obs_direct):
    rt = obs_direct.switch_timetype("GMST").switch_timetype("UTC")
    diff = (rt.data["time"] - obs_direct.data["time"]) % 24.0
    diff = np.minimum(diff, 24.0 - diff)
    np.testing.assert_allclose(diff, 0.0, atol=1e-9)


def test_switch_timetype_noop(obs_direct):
    out = obs_direct.switch_timetype(obs_direct.timetype)
    assert out is not obs_direct
    np.testing.assert_array_equal(out.data["time"], obs_direct.data["time"])
    assert out.timetype == obs_direct.timetype


def test_switch_timetype_invalid_raises(obs_direct):
    with pytest.raises(Exception, match="timetype_out"):
        obs_direct.switch_timetype("TAI")


def test_switch_polrep_roundtrip_stokes_circ_stokes(obs_pol_direct):
    rt = obs_pol_direct.switch_polrep("circ").switch_polrep("stokes")
    for field in ("vis", "qvis", "uvis", "vvis"):
        np.testing.assert_allclose(rt.data[field], obs_pol_direct.data[field], atol=1e-12)


def test_switch_polrep_circ_to_stokes_formulae(obs_pol_direct):
    obs_circ = obs_pol_direct.switch_polrep("circ")
    rt = obs_circ.switch_polrep("stokes")
    np.testing.assert_allclose(
        rt.data["vis"], 0.5 * (obs_circ.data["rrvis"] + obs_circ.data["llvis"]), atol=1e-12
    )
    np.testing.assert_allclose(
        rt.data["qvis"], 0.5 * (obs_circ.data["lrvis"] + obs_circ.data["rlvis"]), atol=1e-12
    )
    np.testing.assert_allclose(
        rt.data["uvis"], 0.5j * (obs_circ.data["lrvis"] - obs_circ.data["rlvis"]), atol=1e-12
    )
    np.testing.assert_allclose(
        rt.data["vvis"], 0.5 * (obs_circ.data["rrvis"] - obs_circ.data["llvis"]), atol=1e-12
    )


def test_switch_polrep_noop_returns_copy(obs_direct):
    out = obs_direct.switch_polrep("stokes")
    assert out is not obs_direct
    np.testing.assert_array_equal(out.data["vis"], obs_direct.data["vis"])


def test_switch_polrep_invalid_raises(obs_direct):
    with pytest.raises(Exception, match="polrep_out"):
        obs_direct.switch_polrep("lin")


def test_switch_polrep_singlepol_hand_invalid(obs_direct):
    # Force a row's vvis to NaN so the singlepol branch fires on stokes→circ.
    obs = obs_direct.copy()
    obs.data["vvis"][0] = np.nan
    with pytest.raises(Exception, match="singlepol_hand"):
        obs.switch_polrep("circ", allow_singlepol=True, singlepol_hand="X")


def test_switch_polrep_singlepol_rr_nan_fills_from_ll(obs_pol_direct):
    obs_circ = obs_pol_direct.switch_polrep("circ").copy()
    # Drop RR for one row; the singlepol branch should use LL as the Stokes I estimate.
    obs_circ.data["rrvis"][0] = np.nan
    obs_stokes = obs_circ.switch_polrep("stokes", allow_singlepol=True)
    assert obs_stokes.data["vis"][0] == obs_circ.data["llvis"][0]


# ---------------------------------------------------------------------------
# Section 3: Baseline & tarr reordering
# ---------------------------------------------------------------------------


def test_reorder_baselines_idempotent(obs_direct):
    obs = obs_direct.copy()
    snapshot = obs.data.copy()
    obs.reorder_baselines()
    np.testing.assert_array_equal(obs.data, snapshot)


def test_reorder_baselines_t1_le_t2_in_tkey(obs_direct):
    for row in obs_direct.data:
        assert obs_direct.tkey[row["t1"]] <= obs_direct.tkey[row["t2"]]


def test_reorder_baselines_time_sorted(obs_direct):
    assert np.all(np.diff(obs_direct.data["time"]) >= 0)


def test_trial_speedups_matches_default(obs_direct):
    obs_default = obs_direct.copy()
    obs_default.reorder_baselines(trial_speedups=False)
    obs_trial = obs_direct.copy()
    obs_trial.reorder_baselines(trial_speedups=True)
    np.testing.assert_array_equal(
        obs_default.data["time"], obs_trial.data["time"]
    )
    np.testing.assert_array_equal(obs_default.data["t1"], obs_trial.data["t1"])
    np.testing.assert_array_equal(obs_default.data["t2"], obs_trial.data["t2"])
    np.testing.assert_array_equal(obs_default.data["vis"], obs_trial.data["vis"])


def test_reorder_tarr_sefd_monotonic(obs_direct):
    obs = obs_direct.copy()
    obs.reorder_tarr_sefd(reorder_baselines=False)
    norms = np.sqrt(obs.tarr["sefdr"] ** 2 + obs.tarr["sefdl"] ** 2)
    assert np.all(np.diff(norms) >= 0)


def test_reorder_tarr_random_permutes(obs_direct):
    obs = obs_direct.copy()
    sites_before = set(obs.tarr["site"])
    np.random.seed(0)
    obs.reorder_tarr_random(reorder_baselines=False)
    sites_after = set(obs.tarr["site"])
    assert sites_before == sites_after


def test_reorder_tarr_snr_smoke(obs_direct):
    obs = obs_direct.copy()
    obs.reorder_tarr_snr(reorder_baselines=False)
    # The transform preserves the site set; ordering is content-dependent so we
    # just confirm the method ran and tkey is consistent.
    assert set(obs.tarr["site"]) == set(obs_direct.tarr["site"])
    for site in obs.tarr["site"]:
        assert obs.tarr[obs.tkey[site]]["site"] == site


# ---------------------------------------------------------------------------
# Section 4: Data layout helpers (data_conj, tlist, split_obs, getClosestScan, bllist)
# ---------------------------------------------------------------------------


def test_data_conj_doubles_length(obs_direct):
    out = obs_direct.data_conj()
    assert len(out) == 2 * len(obs_direct.data)


def test_data_conj_uv_negated(obs_direct):
    out = obs_direct.data_conj()
    # Every row should have a partner with negated (u,v) at the same time.
    keys = {(row["time"], round(row["u"], 6), round(row["v"], 6)) for row in out}
    for row in obs_direct.data:
        partner = (row["time"], round(-row["u"], 6), round(-row["v"], 6))
        assert partner in keys


def test_data_conj_stokes_vis_conjugated(obs_direct):
    out = obs_direct.data_conj()
    # For each row find a partner at the same time and negated u,v whose vis
    # is the conjugate. Colocated stations (SMA/JCMT in EHT2017) produce
    # multiple partners with identical u,v — match by (t1,t2) reversal.
    for row in obs_direct.data:
        mask = (
            (out["time"] == row["time"])
            & (out["t1"] == row["t2"])
            & (out["t2"] == row["t1"])
        )
        assert mask.sum() == 1
        partner = out[mask][0]
        np.testing.assert_allclose(partner["u"], -row["u"], atol=1e-6)
        np.testing.assert_allclose(partner["v"], -row["v"], atol=1e-6)
        np.testing.assert_allclose(partner["vis"], np.conj(row["vis"]), atol=1e-12)


def test_data_conj_circ_rl_swap(obs_pol_direct):
    obs_circ = obs_pol_direct.switch_polrep("circ")
    out = obs_circ.data_conj()
    for row in obs_circ.data:
        mask = (
            (out["time"] == row["time"])
            & (out["t1"] == row["t2"])
            & (out["t2"] == row["t1"])
        )
        partner = out[mask][0]
        np.testing.assert_allclose(partner["rlvis"], np.conj(row["lrvis"]), atol=1e-12)
        np.testing.assert_allclose(partner["lrvis"], np.conj(row["rlvis"]), atol=1e-12)
        assert partner["rlsigma"] == row["lrsigma"]
        assert partner["lrsigma"] == row["rlsigma"]


def test_tlist_groups_by_time(obs_direct):
    chunks = obs_direct.tlist()
    for chunk in chunks:
        assert len(np.unique(chunk["time"])) == 1


def test_tlist_conj_true_doubles_rows(obs_direct):
    chunks_plain = obs_direct.tlist(conj=False)
    chunks_conj = obs_direct.tlist(conj=True)
    n_plain = sum(len(c) for c in chunks_plain)
    n_conj = sum(len(c) for c in chunks_conj)
    assert n_conj == 2 * n_plain


def test_tlist_t_gather_groups_into_buckets(obs_direct):
    fine = obs_direct.tlist(t_gather=0.0)
    coarse = obs_direct.tlist(t_gather=3600.0)
    assert len(coarse) <= len(fine)


def test_split_obs_count_matches_tlist(obs_direct):
    chunks = obs_direct.tlist()
    splits = obs_direct.split_obs()
    assert len(splits) == len(chunks)


def test_split_obs_preserves_tarr(obs_direct):
    splits = obs_direct.split_obs()
    for sub in splits:
        np.testing.assert_array_equal(sub.tarr, obs_direct.tarr)


def test_get_closest_scan_picks_nearest_tstart(obs_direct):
    splits = obs_direct.split_obs()
    target_idx = len(splits) // 2
    target = splits[target_idx].tstart
    chosen = obs_direct.getClosestScan(target, splitObs=splits)
    assert chosen.tstart == target


def test_bllist_unique_baseline_per_chunk(obs_direct):
    chunks = obs_direct.bllist()
    for chunk in chunks:
        pairs = {frozenset((row["t1"], row["t2"])) for row in chunk}
        assert len(pairs) == 1


# ---------------------------------------------------------------------------
# Section 5: Unpacking (unpack, unpack_dat, unpack_bl)
# ---------------------------------------------------------------------------


def test_unpack_invalid_mode_raises(obs_direct):
    with pytest.raises(Exception, match="mode"):
        obs_direct.unpack(["vis"], mode="banana")


def test_unpack_single_string_field_promoted_to_list(obs_direct):
    # Passing a bare string (not a list) should work.
    out = obs_direct.unpack("u")
    assert out.dtype.names == ("u",)
    assert len(out) == len(obs_direct.data)


def test_unpack_mode_time_returns_list_per_time(obs_direct):
    chunks = obs_direct.unpack(["vis"], mode="time")
    times = obs_direct.tlist(conj=True)
    assert len(chunks) == len(times)


def test_unpack_mode_bl_returns_list_per_baseline(obs_direct):
    chunks = obs_direct.unpack(["vis"], mode="bl")
    bllist = obs_direct.bllist()
    assert len(chunks) == len(bllist)


def test_unpack_conj_true_doubles_rows(obs_direct):
    plain = obs_direct.unpack(["vis"])
    conj = obs_direct.unpack(["vis"], conj=True)
    assert len(conj) == 2 * len(plain)


def test_unpack_uvdist_equals_sqrt_u2_v2(obs_direct):
    out = obs_direct.unpack(["u", "v", "uvdist"])
    np.testing.assert_allclose(out["uvdist"], np.hypot(out["u"], out["v"]), atol=1e-12)


def test_unpack_amp_is_abs_vis(obs_direct):
    out = obs_direct.unpack(["vis", "amp"])
    np.testing.assert_allclose(out["amp"], np.abs(out["vis"]), atol=1e-12)


def test_unpack_phase_radians_vs_degrees(obs_direct):
    deg = obs_direct.unpack(["phase"], ang_unit="deg")["phase"]
    rad = obs_direct.unpack(["phase"], ang_unit="rad")["phase"]
    np.testing.assert_allclose(deg, np.degrees(rad), atol=1e-12)


def test_unpack_snr_equals_amp_over_sigma(obs_direct):
    out = obs_direct.unpack(["amp", "sigma", "snr"])
    np.testing.assert_allclose(out["snr"], out["amp"] / out["sigma"], atol=1e-12)


def test_unpack_polrep_consistency_vis_amp(obs_pol_direct):
    stokes = obs_pol_direct.unpack(["amp"])["amp"]
    circ = obs_pol_direct.switch_polrep("circ").unpack(["amp"])["amp"]
    np.testing.assert_allclose(stokes, circ, atol=1e-12)


@pytest.mark.parametrize("field", ["qamp", "uamp", "vamp"])
def test_unpack_polrep_consistency_q_u_v(obs_pol_direct, field):
    stokes = obs_pol_direct.unpack([field])[field]
    circ = obs_pol_direct.switch_polrep("circ").unpack([field])[field]
    np.testing.assert_allclose(stokes, circ, atol=1e-12)


def test_unpack_pvis_stokes_vs_circ(obs_pol_direct):
    stokes = obs_pol_direct.unpack(["pvis"])["pvis"]
    circ = obs_pol_direct.switch_polrep("circ").unpack(["pvis"])["pvis"]
    np.testing.assert_allclose(stokes, circ, atol=1e-12)


def test_unpack_time_gmst_round_trips(obs_direct):
    utc_times = obs_direct.unpack(["time"], timetype="UTC")["time"]
    gmst_times = obs_direct.unpack(["time"], timetype="GMST")["time"]
    # GMST and UTC should differ on at least some rows.
    assert not np.allclose(utc_times, gmst_times)


def test_unpack_el1_in_range(obs_direct):
    out = obs_direct.unpack(["el1", "el2"])
    # All kept points should have valid elevations (the array has clean data).
    assert np.all(out["el1"] >= -1e-6)
    assert np.all(out["el1"] <= 90.0 + 1e-6)
    assert np.all(out["el2"] >= -1e-6)
    assert np.all(out["el2"] <= 90.0 + 1e-6)


def test_unpack_unknown_field_raises(obs_direct):
    with pytest.raises(Exception, match="not a valid field"):
        obs_direct.unpack(["banana"])


def test_unpack_debias_lowers_amp_at_low_snr(obs_noisy):
    raw = obs_noisy.unpack(["amp"], debias=False)["amp"]
    debiased = obs_noisy.unpack(["amp"], debias=True)["amp"]
    # Debiased amplitude is always <= raw amplitude (Wiener correction).
    assert np.all(debiased <= raw + 1e-12)
    # And at least some rows show a non-trivial correction.
    assert np.any(debiased < raw - 1e-6)


def test_unpack_bl_filters_to_baseline(obs_direct):
    site1, site2 = obs_direct.data["t1"][0], obs_direct.data["t2"][0]
    out = obs_direct.unpack_bl(site1, site2, ["vis"])
    assert len(out) > 0


def test_unpack_bl_invalid_timetype_raises(obs_direct):
    site1, site2 = obs_direct.data["t1"][0], obs_direct.data["t2"][0]
    with pytest.raises(Exception, match="timetype"):
        obs_direct.unpack_bl(site1, site2, ["vis"], timetype="TAI")


# ---------------------------------------------------------------------------
# Section 6: Geometry (sourcevec, res, recompute_uv)
# ---------------------------------------------------------------------------


def test_sourcevec_unit_norm(obs_direct):
    vec = obs_direct.sourcevec()
    np.testing.assert_allclose(np.linalg.norm(vec), 1.0, atol=1e-12)


def test_sourcevec_dec_zero_points_to_x_axis(obs_direct):
    obs = obs_direct.copy()
    obs.dec = 0.0
    vec = obs.sourcevec()
    np.testing.assert_allclose(vec, [1.0, 0.0, 0.0], atol=1e-12)


def test_res_equals_inverse_max_uvdist(obs_direct):
    uvdist = obs_direct.unpack(["uvdist"])["uvdist"]
    assert obs_direct.res() == pytest.approx(1.0 / np.max(uvdist))


def test_recompute_uv_matches_observed(obs_direct):
    rebuilt = obs_direct.recompute_uv()
    np.testing.assert_allclose(rebuilt.data["u"], obs_direct.data["u"], rtol=1e-6, atol=1.0)
    np.testing.assert_allclose(rebuilt.data["v"], obs_direct.data["v"], rtol=1e-6, atol=1.0)


# ---------------------------------------------------------------------------
# Section 7: Chi-squared (chisq, polchisq)
# ---------------------------------------------------------------------------


def test_chisq_invalid_dtype_raises(obs_direct, gauss_im):
    with pytest.raises(Exception, match="not a supported"):
        obs_direct.chisq(gauss_im, dtype="banana", ttype="direct")


def test_chisq_self_consistency(obs_direct, gauss_im):
    chisq = obs_direct.chisq(gauss_im, dtype="vis", ttype="direct")
    assert chisq == pytest.approx(0.0, abs=1e-6)


def test_chisq_amp_nonzero_with_noise(obs_noisy, gauss_im):
    chisq = obs_noisy.chisq(gauss_im, dtype="amp", ttype="direct")
    assert chisq > 0


@pytest.mark.parametrize("dtype", ["vis", "amp", "bs", "cphase", "camp", "logcamp"])
def test_chisq_dtypes_run(obs_direct, gauss_im, dtype):
    chisq = obs_direct.chisq(gauss_im, dtype=dtype, ttype="direct")
    assert np.isfinite(chisq)


def test_polchisq_invalid_dtype_raises(obs_pol_direct, gauss_im_pol):
    with pytest.raises(Exception, match="polarimetric"):
        obs_pol_direct.polchisq(gauss_im_pol, dtype="banana", ttype="direct")


@pytest.mark.xfail(
    reason=(
        "Obsdata.polchisq packs the (I, m, phi, psi) parametrization with "
        "psi = arcsin(V/I), but pol_imager_utils.make_v_image defines "
        "V = I*m*sin(psi), so psi should be arcsin(V/(I*m)). The mismatch "
        "produces an O(1e-2) self-consistency residue. See obsdata.py:1187."
    ),
    strict=True,
)
def test_polchisq_self_consistency(obs_pol_direct, gauss_im_pol):
    chisq = obs_pol_direct.polchisq(gauss_im_pol, dtype="pvis", ttype="direct")
    assert chisq == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Section 8: Averaging (avg_coherent, avg_incoherent)
# ---------------------------------------------------------------------------


def test_avg_coherent_zero_inttime_returns_copy(obs_direct):
    out = obs_direct.avg_coherent(inttime=0.0)
    assert len(out.data) == len(obs_direct.data)


def test_avg_coherent_reduces_npoints(obs_direct):
    # Conftest sets TADV_SEC = 600 (one scan every 10 min). Averaging at 30 min
    # bundles ~3 scans per chunk, definitely fewer rows than the raw obs.
    out = obs_direct.avg_coherent(inttime=1800.0)
    assert len(out.data) < len(obs_direct.data)


def test_avg_coherent_preserves_short_bl_amp(obs_direct):
    short = obs_direct.flag_uvdist(uv_max=5e8)
    if len(short.data) == 0:
        pytest.skip("no short-baseline data")
    raw_mean = np.mean(np.abs(short.unpack(["vis"])["vis"]))
    avg = short.avg_coherent(inttime=1800.0)
    avg_mean = np.mean(np.abs(avg.unpack(["vis"])["vis"]))
    np.testing.assert_allclose(avg_mean, raw_mean, rtol=1e-3)


def test_avg_incoherent_zeroes_phases(obs_direct):
    out = obs_direct.avg_incoherent(inttime=1800.0)
    phases = np.angle(out.data["vis"])
    np.testing.assert_allclose(phases, 0.0, atol=1e-12)


# ---------------------------------------------------------------------------
# Section 9: Scans (add_scans)
# ---------------------------------------------------------------------------


def test_add_scans_self_infers_from_data(obs_direct):
    obs = obs_direct.copy()
    obs.add_scans(info="self")
    assert obs.scans.ndim == 2
    assert obs.scans.shape[1] == 2
    # Every unique data time should sit inside at least one scan window.
    for t in np.unique(obs.data["time"]):
        assert np.any((obs.scans[:, 0] <= t) & (obs.scans[:, 1] >= t))


def test_add_scans_txt_reads_file(obs_direct, tmp_path):
    obs = obs_direct.copy()
    fpath = tmp_path / "scans.txt"
    fpath.write_text("0.0 1.0\n2.0 3.0\n")
    obs.add_scans(info="txt", filepath=str(fpath))
    np.testing.assert_allclose(obs.scans, [[0.0, 1.0], [2.0, 3.0]])


def test_add_scans_invalid_info_leaves_none(obs_direct):
    obs = obs_direct.copy()
    obs.add_scans(info="banana")
    assert obs.scans is None


# ---------------------------------------------------------------------------
# Section 10: Beam / imaging utilities (cleanbeam, fit_beam, dirtybeam, dirtyimage)
# ---------------------------------------------------------------------------


def test_cleanbeam_total_flux_one(obs_direct):
    im = obs_direct.cleanbeam(npix=32, fov=200 * eh.RADPERUAS)
    assert im.total_flux() == pytest.approx(1.0, abs=1e-6)


def test_fit_beam_returns_three_floats(obs_direct):
    params = obs_direct.fit_beam(weighting="uniform", units="rad")
    assert params.shape == (3,)
    fwhm_maj, fwhm_min, theta = params
    assert fwhm_maj > 0
    assert fwhm_min > 0
    assert fwhm_min <= fwhm_maj


def test_fit_beam_natural_units_match_rad(obs_direct):
    rad = obs_direct.fit_beam(weighting="uniform", units="rad")
    nat = obs_direct.fit_beam(weighting="uniform", units="natural")
    # Natural FWHMs are converted from radians to microarcseconds.
    np.testing.assert_allclose(nat[0], rad[0] / eh.RADPERUAS, rtol=1e-6)
    np.testing.assert_allclose(nat[1], rad[1] / eh.RADPERUAS, rtol=1e-6)


def test_dirtybeam_normalized_to_unit_sum(obs_direct):
    im = obs_direct.dirtybeam(npix=32, fov=200 * eh.RADPERUAS)
    assert im.imvec.sum() == pytest.approx(1.0, abs=1e-12)


def test_dirtybeam_natural_weighting_smoke(obs_direct):
    im = obs_direct.dirtybeam(npix=16, fov=200 * eh.RADPERUAS, weighting="natural")
    assert np.all(np.isfinite(im.imvec))


def test_dirtyimage_polrep_consistency(obs_direct):
    # `dirtyimage` Fourier-inverts vis1 of the polrep: 'vis' (Stokes I) for
    # stokes, 'rrvis' (= I+V) for circ. On a V=0 observation these match.
    stokes = obs_direct.dirtyimage(npix=16, fov=200 * eh.RADPERUAS)
    circ = obs_direct.switch_polrep("circ").dirtyimage(npix=16, fov=200 * eh.RADPERUAS)
    np.testing.assert_allclose(stokes.imvec, circ.imvec, atol=1e-12)


# ---------------------------------------------------------------------------
# Section 11: Noise & scale (rescale_zbl, add_*_noise, rescale_noise, ...)
# ---------------------------------------------------------------------------


def test_rescale_zbl_scales_short_baselines(obs_direct):
    out = obs_direct.rescale_zbl(totflux=2.0, uv_max=5e8)
    short_mask = (obs_direct.data["u"] ** 2 + obs_direct.data["v"] ** 2) ** 0.5 < 5e8
    long_mask = ~short_mask
    # Short-baseline vis values change, long-baseline don't.
    if short_mask.any():
        assert not np.allclose(out.data["vis"][short_mask], obs_direct.data["vis"][short_mask])
    if long_mask.any():
        np.testing.assert_array_equal(
            out.data["vis"][long_mask], obs_direct.data["vis"][long_mask]
        )


def test_add_leakage_noise_increases_sigma(obs_pol_direct):
    out = obs_pol_direct.add_leakage_noise(Dterm_amp=0.1)
    assert np.all(out.data["sigma"] >= obs_pol_direct.data["sigma"])
    # And at least some rows must have grown (assuming the test data has any
    # non-trivial cross-hand amplitudes).
    assert np.any(out.data["sigma"] > obs_pol_direct.data["sigma"])


def test_add_fractional_noise_quadrature_form(obs_direct):
    frac = 0.05
    amp = obs_direct.unpack(["amp"])["amp"]
    out = obs_direct.add_fractional_noise(frac)
    expected = np.sqrt(obs_direct.data["sigma"] ** 2 + (frac * amp) ** 2)
    np.testing.assert_allclose(out.data["sigma"], expected, atol=1e-12)


def test_rescale_noise_multiplies_sigmas(obs_direct):
    factor = 1.5
    out = obs_direct.rescale_noise(noise_rescale_factor=factor)
    for sigma in ("sigma", "qsigma", "usigma", "vsigma"):
        np.testing.assert_allclose(
            out.data[sigma], factor * obs_direct.data[sigma], atol=1e-12
        )


def test_find_amt_fractional_noise_runs(obs_noisy, gauss_im):
    # find_amt_fractional_noise calls obs.chisq() with no ttype override, so
    # it inherits the chisq() default of 'nfft' — needs pynfft. Skip when
    # NFFT isn't installed rather than running ttype='direct' (the method
    # doesn't expose the kwarg).
    pytest.importorskip("pynfft")
    out = obs_noisy.find_amt_fractional_noise(gauss_im, dtype="vis", target=1.0, maxiter=5)
    assert np.isfinite(out).all()


def test_estimate_noise_rescale_factor_returns_float(obs_noisy):
    factor = obs_noisy.estimate_noise_rescale_factor()
    assert np.isfinite(factor)
    assert factor > 0


# ---------------------------------------------------------------------------
# Section 12: Flagging
# ---------------------------------------------------------------------------


def test_flag_elev_kept_within_range(obs_direct):
    out = obs_direct.flag_elev(elev_min=10.0, elev_max=80.0)
    el = out.unpack(["el1", "el2"])
    assert np.all(np.minimum(el["el1"], el["el2"]) > 10.0)
    assert np.all(np.maximum(el["el1"], el["el2"]) < 80.0)


def test_flag_elev_output_both(obs_direct):
    result = obs_direct.flag_elev(elev_min=10.0, elev_max=80.0, output="both")
    assert isinstance(result, dict)
    assert set(result.keys()) == {"kept", "flagged"}
    assert len(result["kept"].data) + len(result["flagged"].data) == len(obs_direct.data)


def test_flag_large_fractional_pol_keeps_low_m(obs_pol_direct):
    out = obs_pol_direct.flag_large_fractional_pol(max_fractional_pol=0.5)
    m = np.nan_to_num(out.unpack(["mamp"])["mamp"])
    assert np.all(m < 0.5)


def test_flag_uvdist_max_keeps_short_bl(obs_direct):
    out = obs_direct.flag_uvdist(uv_max=5e9)
    uvdist = out.unpack(["uvdist"])["uvdist"]
    assert np.all(uvdist <= 5e9)


def test_flag_sites_drops_all_baselines_touching_site(obs_direct):
    site = obs_direct.tarr["site"][0]
    out = obs_direct.flag_sites([site])
    assert not np.any(out.data["t1"] == site)
    assert not np.any(out.data["t2"] == site)


def test_flag_bl_drops_only_specific_pair(obs_direct):
    site1 = obs_direct.data["t1"][0]
    site2 = obs_direct.data["t2"][0]
    out = obs_direct.flag_bl([site1, site2])
    # The (site1, site2) baseline is gone but each site appears in other pairs.
    for row in out.data:
        assert not (row["t1"] in (site1, site2) and row["t2"] in (site1, site2))


def test_flag_low_snr_keeps_high_snr(obs_noisy):
    out = obs_noisy.flag_low_snr(snr_cut=2.0)
    snr = out.unpack(["snr"])["snr"]
    assert np.all(snr > 2.0)


def test_flag_high_sigma_keeps_low_sigma(obs_noisy):
    cutoff = float(np.median(obs_noisy.data["sigma"]))
    out = obs_noisy.flag_high_sigma(sigma_cut=cutoff, sigma_type="sigma")
    assert np.all(out.data["sigma"] < cutoff)


def test_flag_UT_range_all(obs_direct):
    # Window the first quarter of the obs only.
    span = obs_direct.tstop - obs_direct.tstart
    out = obs_direct.flag_UT_range(
        UT_start_hour=obs_direct.tstart + 0.25 * span,
        UT_stop_hour=obs_direct.tstop - 0.25 * span,
        flag_type="all",
    )
    t = out.unpack(["time"])["time"]
    # All kept times are outside the flagged window (i.e., in the early or late
    # quartile).
    assert np.all(
        (t <= obs_direct.tstart + 0.25 * span) | (t >= obs_direct.tstop - 0.25 * span)
    )


def test_flag_UT_range_station(obs_direct):
    span = obs_direct.tstop - obs_direct.tstart
    site = obs_direct.tarr["site"][0]
    out = obs_direct.flag_UT_range(
        UT_start_hour=obs_direct.tstart + 0.25 * span,
        UT_stop_hour=obs_direct.tstop - 0.25 * span,
        flag_type="station",
        flag_what=site,
    )
    t = out.unpack(["time"])["time"]
    # The flagged station is absent only during the middle window.
    middle = (t > obs_direct.tstart + 0.25 * span) & (t < obs_direct.tstop - 0.25 * span)
    rows_middle = out.data[middle]
    assert not np.any(rows_middle["t1"] == site)
    assert not np.any(rows_middle["t2"] == site)


def test_flags_from_file_csv(obs_direct, tmp_path):
    fpath = tmp_path / "flags.csv"
    fpath.write_text(
        "mjd_start,mjd_stop,station\n"
        f"{obs_direct.mjd}.0,{obs_direct.mjd}.5,{obs_direct.tarr['site'][0]}\n"
    )
    out = obs_direct.flags_from_file(str(fpath), flag_type="station")
    assert len(out.data) <= len(obs_direct.data)


def test_flag_anomalous_smoke(obs_noisy):
    # flag_anomalous does per-baseline robust-z outlier removal; behaviour is
    # data-dependent so we just confirm it produces a no-larger obs.
    out = obs_noisy.flag_anomalous(field="snr", robust_nsigma_cut=3.0)
    assert len(out.data) <= len(obs_noisy.data)


# ---------------------------------------------------------------------------
# Section 13: Taper / deblur / reweight / fit_gauss
# ---------------------------------------------------------------------------


def test_reverse_taper_inverse_of_taper(obs_direct):
    fwhm = 30.0 * eh.RADPERUAS
    rt = obs_direct.taper(fwhm).reverse_taper(fwhm)
    np.testing.assert_allclose(rt.data["vis"], obs_direct.data["vis"], atol=1e-12)
    np.testing.assert_allclose(rt.data["sigma"], obs_direct.data["sigma"], atol=1e-12)


def test_taper_attenuates_long_baseline_amp(obs_direct):
    fwhm = 30.0 * eh.RADPERUAS
    out = obs_direct.taper(fwhm)
    uvdist = (obs_direct.data["u"] ** 2 + obs_direct.data["v"] ** 2) ** 0.5
    long_mask = uvdist > np.median(uvdist)
    assert np.all(
        np.abs(out.data["vis"][long_mask]) <= np.abs(obs_direct.data["vis"][long_mask]) + 1e-12
    )


def test_deblur_changes_visibilities(obs_direct):
    out = obs_direct.deblur()
    assert not np.allclose(out.data["vis"], obs_direct.data["vis"])
    # All four polarization slots remain finite (regression for the vis3-slot
    # bug fixed in fix/obsdata-latent-bugs).
    for field in ("vis", "qvis", "uvis", "vvis"):
        assert np.all(np.isfinite(out.data[field]))


def test_reweight_preserves_npoints(obs_direct):
    out = obs_direct.reweight(uv_radius=5e8)
    assert len(out.data) == len(obs_direct.data)


def test_fit_gauss_smoke(obs_direct):
    # The fitter is wobbly per its in-code TODO; we just check the call
    # returns finite parameters of the expected shape.
    params = obs_direct.fit_gauss()
    assert params.shape == (3,)
    assert np.all(np.isfinite(params))


# ---------------------------------------------------------------------------
# Section 14: Closure quantities (bispectra, c_phases, c_amplitudes, *_tri/quad,
#                                 *_diag). Idempotence tests are below.
# ---------------------------------------------------------------------------


def test_bispectra_invalid_mode_raises(obs_direct):
    with pytest.raises(Exception, match="mode"):
        obs_direct.bispectra(mode="banana")


def test_bispectra_invalid_count_raises(obs_direct):
    with pytest.raises(Exception, match="count"):
        obs_direct.bispectra(count="banana")


def test_bispectra_invalid_vtype_raises(obs_direct):
    with pytest.raises(Exception, match="vtype"):
        obs_direct.bispectra(vtype="banana")


def test_bispectra_min_subset_of_max(obs_direct):
    bs_max = obs_direct.bispectra(mode="all", count="max")
    bs_min = obs_direct.bispectra(mode="all", count="min")
    max_triangles = {
        (row["time"], frozenset((row["t1"], row["t2"], row["t3"]))) for row in bs_max
    }
    min_triangles = {
        (row["time"], frozenset((row["t1"], row["t2"], row["t3"]))) for row in bs_min
    }
    assert min_triangles <= max_triangles
    assert len(bs_min) <= len(bs_max)


def test_bispectra_uv_min_drops_short_baselines(obs_direct):
    bs = obs_direct.bispectra(mode="all", count="max", uv_min=5e8)
    for row in bs:
        for (u, v) in [(row["u1"], row["v1"]), (row["u2"], row["v2"]), (row["u3"], row["v3"])]:
            assert np.hypot(u, v) >= 5e8


def test_c_phases_in_minus_pi_pi_radians(obs_direct):
    cp = obs_direct.c_phases(mode="all", count="max", ang_unit="rad")
    assert np.all(cp["cphase"] >= -np.pi - 1e-9)
    assert np.all(cp["cphase"] <= np.pi + 1e-9)


def test_c_phases_deg_vs_rad_consistency(obs_direct):
    cp_deg = obs_direct.c_phases(mode="all", count="max", ang_unit="deg")
    cp_rad = obs_direct.c_phases(mode="all", count="max", ang_unit="rad")
    np.testing.assert_allclose(cp_deg["cphase"], np.degrees(cp_rad["cphase"]), atol=1e-9)


def test_c_phases_diag_returns_per_time_tuples(obs_pol_direct):
    diag = obs_pol_direct.c_phases_diag(count="min")
    assert len(diag) > 0
    cparr, tris, us, vs, tform = diag[0]
    assert cparr.dtype.names == ("time", "cphase", "sigmacp")
    assert tform.shape[0] == tform.shape[1]


def test_c_phases_diag_transform_matrix_orthogonal(obs_pol_direct):
    diag = obs_pol_direct.c_phases_diag(count="min")
    for _, _, _, _, tform in diag[:3]:
        mat = tform.astype(np.float64)
        prod = mat @ mat.T
        np.testing.assert_allclose(prod, np.eye(mat.shape[0]), atol=1e-9)


def test_bispectra_tri_from_vis_matches_from_maxset(obs_direct):
    sites = obs_direct.tarr["site"][:3].tolist()
    a = obs_direct.bispectra_tri(*sites, method="from_maxset")
    b = obs_direct.bispectra_tri(*sites, method="from_vis")
    assert len(a) == len(b)
    np.testing.assert_allclose(a["bispec"], b["bispec"], atol=1e-9)


def test_cphase_tri_signs_flip_with_reordering(obs_direct):
    sites = obs_direct.tarr["site"][:3].tolist()
    forward = obs_direct.cphase_tri(*sites, ang_unit="rad", method="from_vis")
    reversed_ = obs_direct.cphase_tri(sites[0], sites[2], sites[1],
                                      ang_unit="rad", method="from_vis")
    assert len(forward) == len(reversed_)
    np.testing.assert_allclose(forward["cphase"], -reversed_["cphase"], atol=1e-9)


def test_c_amplitudes_invalid_ctype_raises(obs_direct):
    with pytest.raises(Exception, match="closure amplitude type"):
        obs_direct.c_amplitudes(ctype="banana")


def test_c_amplitudes_logcamp_equals_log_camp(obs_direct):
    # Without debiasing, logcamp must equal log(camp) on the same quadrangles.
    # (With debiasing, camp and logcamp use different correction formulas.)
    camp = obs_direct.c_amplitudes(mode="all", count="max", ctype="camp", debias=False)
    logcamp = obs_direct.c_amplitudes(mode="all", count="max", ctype="logcamp", debias=False)
    assert len(camp) == len(logcamp)
    np.testing.assert_allclose(logcamp["camp"], np.log(camp["camp"]), atol=1e-9)


def test_c_log_amplitudes_diag_smoke(obs_pol_direct):
    diag = obs_pol_direct.c_log_amplitudes_diag(count="min")
    assert len(diag) > 0
    lcaarr, quads, us, vs, tform = diag[0]
    assert lcaarr.dtype.names == ("time", "camp", "sigmaca")
    assert tform.shape[0] == tform.shape[1]


def test_camp_quad_inverse_relation(obs_direct):
    # camp(A,B,C,D) = |V_AB V_CD| / |V_AD V_BC|; the (A,D,C,B) reordering is
    # the reciprocal. The product must equal 1 exactly — but only with
    # debias=False, since `camp_debias` corrects the numerator and denominator
    # asymmetrically and breaks the algebraic identity.
    s = obs_direct.tarr["site"][:4].tolist()
    forward = obs_direct.camp_quad(*s, ctype="camp", method="from_vis", debias=False)
    inverse = obs_direct.camp_quad(s[0], s[3], s[2], s[1], ctype="camp",
                                   method="from_vis", debias=False)
    if len(forward) == 0 or len(inverse) == 0:
        pytest.skip("no quadrangle data for the chosen four sites")
    np.testing.assert_allclose(forward["camp"] * inverse["camp"], 1.0, atol=1e-9)


# ---------------------------------------------------------------------------
# Section 15: Plot methods — smoke tests only
# ---------------------------------------------------------------------------


def test_plotall_smoke(obs_direct):
    import matplotlib
    matplotlib.use("Agg")
    ax = obs_direct.plotall("uvdist", "amp", show=False)
    assert ax is not None


def test_plot_bl_smoke(obs_direct):
    import matplotlib
    matplotlib.use("Agg")
    site1, site2 = obs_direct.data["t1"][0], obs_direct.data["t2"][0]
    ax = obs_direct.plot_bl(site1, site2, "amp", show=False)
    assert ax is not None


def test_plot_bl_invalid_field_raises(obs_direct):
    # Regression for the `string.join` bug fixed in fix/obsdata-latent-bugs;
    # the error path is now reachable.
    site1, site2 = obs_direct.data["t1"][0], obs_direct.data["t2"][0]
    with pytest.raises(Exception, match="valid fields"):
        obs_direct.plot_bl(site1, site2, "banana", show=False)


def test_plot_cphase_smoke(obs_direct):
    import matplotlib
    matplotlib.use("Agg")
    sites = obs_direct.tarr["site"][:3].tolist()
    obs_direct.plot_cphase(*sites, show=False)


def test_plot_camp_smoke(obs_direct):
    import matplotlib
    matplotlib.use("Agg")
    sites = obs_direct.tarr["site"][:4].tolist()
    obs_direct.plot_camp(*sites, show=False)


# ---------------------------------------------------------------------------
# Closure-quantity idempotence (originally Phase 0)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("vtype", ["vis", "qvis", "uvis", "vvis"])
def test_bispectra_idempotent_stokes(obs_pol_direct, vtype):
    a = obs_pol_direct.bispectra(mode="all", count="max", vtype=vtype)
    b = obs_pol_direct.bispectra(mode="all", count="max", vtype=vtype)
    assert a.dtype == b.dtype
    assert len(a) == len(b)
    for field in a.dtype.names:
        np.testing.assert_array_equal(a[field], b[field], err_msg=f"field={field}")


@pytest.mark.parametrize("vtype", ["rrvis", "llvis", "rlvis", "lrvis"])
def test_bispectra_idempotent_circ(obs_pol_direct, vtype):
    obs_circ = obs_pol_direct.switch_polrep("circ")
    a = obs_circ.bispectra(mode="all", count="max", vtype=vtype)
    b = obs_circ.bispectra(mode="all", count="max", vtype=vtype)
    assert a.dtype == b.dtype
    assert len(a) == len(b)
    for field in a.dtype.names:
        np.testing.assert_array_equal(a[field], b[field], err_msg=f"field={field}")


@pytest.mark.parametrize("vtype", ["vis", "qvis", "uvis", "vvis"])
def test_c_phases_idempotent(obs_pol_direct, vtype):
    a = obs_pol_direct.c_phases(mode="all", count="max", vtype=vtype)
    b = obs_pol_direct.c_phases(mode="all", count="max", vtype=vtype)
    assert a.dtype == b.dtype
    assert len(a) == len(b)
    for field in a.dtype.names:
        np.testing.assert_array_equal(a[field], b[field], err_msg=f"field={field}")


@pytest.mark.parametrize("ctype", ["camp", "logcamp"])
@pytest.mark.parametrize("vtype", ["vis", "qvis", "uvis", "vvis"])
def test_c_amplitudes_idempotent(obs_pol_direct, vtype, ctype):
    a = obs_pol_direct.c_amplitudes(mode="all", count="max", vtype=vtype, ctype=ctype)
    b = obs_pol_direct.c_amplitudes(mode="all", count="max", vtype=vtype, ctype=ctype)
    assert a.dtype == b.dtype
    assert len(a) == len(b)
    for field in a.dtype.names:
        np.testing.assert_array_equal(a[field], b[field], err_msg=f"field={field}")
