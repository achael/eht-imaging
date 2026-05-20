"""Tests for ehtim.observing.obs_simulate.

Covers all seven public functions: make_uvpoints, sample_vis, make_jones,
make_jones_inverse, add_jones_and_noise, apply_jones_inverse, add_noise.
"""
import os

import numpy as np
import pytest

import ehtim as eh
import ehtim.const_def as ehc
from ehtim.observing import obs_helpers as obsh
from ehtim.observing import obs_simulate as os_sim

ARRAY_PATH = os.path.join(os.path.dirname(__file__), "..", "arrays", "EHT2017.txt")

TINT = 60.0
TADV = 600.0
TSTART = 0.0
TSTOP = 24.0
BW = 1e9


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def array():
    return eh.array.load_txt(ARRAY_PATH)


@pytest.fixture(scope="module")
def asymmetric_array(array):
    """Array with sefdr != sefdl and nonzero D-terms (so circ sigmas/dterms differ)."""
    arr = array.copy()
    tarr = np.asarray(arr.tarr).copy()
    tarr["sefdl"] = tarr["sefdr"] * 1.5
    tarr["dr"] = 0.02 + 0.01j
    tarr["dl"] = -0.01 + 0.015j
    arr.tarr = tarr
    return arr


@pytest.fixture(scope="module")
def asymmetric_image():
    """Off-centre elongated Gaussian: distinguishes axes and tests centering."""
    im = eh.image.make_empty(64, 200 * eh.RADPERUAS, 17.761, -29.0, rf=230e9)
    im = im.add_gauss(1.0, (60 * eh.RADPERUAS, 20 * eh.RADPERUAS,
                            np.pi / 6, 30 * eh.RADPERUAS, -15 * eh.RADPERUAS))
    return im


@pytest.fixture(scope="module")
def asymmetric_image_pol(asymmetric_image):
    im = asymmetric_image.copy()
    im.add_qu(0.10 * im.imarr(), 0.05 * im.imarr())
    im.add_v(0.02 * im.imarr())
    return im


def _observe(im, array, ttype="direct"):
    return im.observe(
        array, TINT, TADV, TSTART, TSTOP, BW,
        ampcal=True, phasecal=True, ttype=ttype, add_th_noise=False,
    )


@pytest.fixture(scope="module")
def obs(asymmetric_image, array):
    """Noise-free Stokes observation used by most Jones tests."""
    return _observe(asymmetric_image, array)


@pytest.fixture(scope="module")
def obs_pol(asymmetric_image_pol, array):
    return _observe(asymmetric_image_pol, array)


@pytest.fixture(scope="module")
def obs_asym(asymmetric_image, asymmetric_array):
    """Observation against the asymmetric array (sefdr != sefdl, nonzero D)."""
    return _observe(asymmetric_image, asymmetric_array)


# ---------------------------------------------------------------------------
# make_uvpoints
# ---------------------------------------------------------------------------


class TestMakeUvpoints:
    """Coverage for obs_simulate.make_uvpoints (uv generation + sigma layout)."""

    def test_stokes_dtype(self, array):
        out = os_sim.make_uvpoints(array, 17.761, -29.0, 230e9, BW,
                                   TINT, TADV, TSTART, TSTOP, polrep="stokes")
        assert out.dtype == np.dtype(ehc.DTPOL_STOKES)
        assert set(out.dtype.names) >= {"vis", "qvis", "uvis", "vvis",
                                        "sigma", "qsigma", "usigma", "vsigma"}

    def test_circ_dtype(self, array):
        out = os_sim.make_uvpoints(array, 17.761, -29.0, 230e9, BW,
                                   TINT, TADV, TSTART, TSTOP, polrep="circ")
        assert out.dtype == np.dtype(ehc.DTPOL_CIRC)
        assert set(out.dtype.names) >= {"rrvis", "llvis", "rlvis", "lrvis",
                                        "rrsigma", "llsigma", "rlsigma", "lrsigma"}

    def test_invalid_polrep_raises(self, array):
        with pytest.raises(Exception, match="only 'stokes' and 'circ'"):
            os_sim.make_uvpoints(array, 17.761, -29.0, 230e9, BW,
                                 TINT, TADV, TSTART, TSTOP, polrep="linear")

    def test_baseline_ordering(self, array):
        out = os_sim.make_uvpoints(array, 17.761, -29.0, 230e9, BW,
                                   TINT, TADV, TSTART, TSTOP)
        order = {site: i for i, site in enumerate(array.tarr["site"])}
        for row in out:
            assert order[row["t1"]] < order[row["t2"]]

    def test_stokes_sigma_layout(self, asymmetric_array):
        """For circ: sig_iv = 0.5*sqrt(sig_rr^2 + sig_ll^2); sig_qu = 0.5*sqrt(sig_rl^2 + sig_lr^2)."""
        out = os_sim.make_uvpoints(asymmetric_array, 17.761, -29.0, 230e9, BW,
                                   TINT, TADV, TSTART, TSTOP, polrep="stokes")
        row = out[0]
        t1, t2 = row["t1"], row["t2"]
        i1 = list(asymmetric_array.tarr["site"]).index(t1)
        i2 = list(asymmetric_array.tarr["site"]).index(t2)
        sefdr1 = asymmetric_array.tarr[i1]["sefdr"]
        sefdl1 = asymmetric_array.tarr[i1]["sefdl"]
        sefdr2 = asymmetric_array.tarr[i2]["sefdr"]
        sefdl2 = asymmetric_array.tarr[i2]["sefdl"]
        sig_rr = obsh.blnoise(sefdr1, sefdr2, TINT, BW)
        sig_ll = obsh.blnoise(sefdl1, sefdl2, TINT, BW)
        sig_rl = obsh.blnoise(sefdr1, sefdl2, TINT, BW)
        sig_lr = obsh.blnoise(sefdl1, sefdr2, TINT, BW)
        sig_iv = 0.5 * np.sqrt(sig_rr ** 2 + sig_ll ** 2)
        sig_qu = 0.5 * np.sqrt(sig_rl ** 2 + sig_lr ** 2)
        np.testing.assert_allclose(row["sigma"], sig_iv)
        np.testing.assert_allclose(row["vsigma"], sig_iv)
        np.testing.assert_allclose(row["qsigma"], sig_qu)
        np.testing.assert_allclose(row["usigma"], sig_qu)

    def test_circ_sigma_layout(self, asymmetric_array):
        """For circ: sig1..4 = sig_rr, sig_ll, sig_rl, sig_lr (no mixing)."""
        out = os_sim.make_uvpoints(asymmetric_array, 17.761, -29.0, 230e9, BW,
                                   TINT, TADV, TSTART, TSTOP, polrep="circ")
        row = out[0]
        i1 = list(asymmetric_array.tarr["site"]).index(row["t1"])
        i2 = list(asymmetric_array.tarr["site"]).index(row["t2"])
        sefdr1 = asymmetric_array.tarr[i1]["sefdr"]
        sefdl1 = asymmetric_array.tarr[i1]["sefdl"]
        sefdr2 = asymmetric_array.tarr[i2]["sefdr"]
        sefdl2 = asymmetric_array.tarr[i2]["sefdl"]
        np.testing.assert_allclose(row["rrsigma"], obsh.blnoise(sefdr1, sefdr2, TINT, BW))
        np.testing.assert_allclose(row["llsigma"], obsh.blnoise(sefdl1, sefdl2, TINT, BW))
        np.testing.assert_allclose(row["rlsigma"], obsh.blnoise(sefdr1, sefdl2, TINT, BW))
        np.testing.assert_allclose(row["lrsigma"], obsh.blnoise(sefdl1, sefdr2, TINT, BW))

    def test_scalar_tau_applied(self, array):
        out = os_sim.make_uvpoints(array, 17.761, -29.0, 230e9, BW,
                                   TINT, TADV, TSTART, TSTOP, tau=0.25)
        assert np.all(out["tau1"] == 0.25)
        assert np.all(out["tau2"] == 0.25)

    def test_dict_tau_full_dict(self, array):
        sites = list(array.tarr["site"])
        tau = {s: 0.05 + 0.01 * i for i, s in enumerate(sites)}
        out = os_sim.make_uvpoints(array, 17.761, -29.0, 230e9, BW,
                                   TINT, TADV, TSTART, TSTOP, tau=tau)
        for row in out:
            np.testing.assert_allclose(row["tau1"], tau[row["t1"]])
            np.testing.assert_allclose(row["tau2"], tau[row["t2"]])

    def test_dict_tau_partial_falls_back_per_baseline(self, array):
        """Documented behavior: KeyError on either site falls both back to TAUDEF.

        This is per the source: `try: tau1 = tau[site1]; tau2 = tau[site2] except KeyError:`.
        Slightly surprising (one-missing-site doesn't preserve the other side), but
        deliberate fallback. Pin it so any future change is intentional.
        """
        sites = list(array.tarr["site"])
        tau = {s: 0.05 + 0.01 * i for i, s in enumerate(sites)}
        del tau[sites[3]]   # drop LMT
        out = os_sim.make_uvpoints(array, 17.761, -29.0, 230e9, BW,
                                   TINT, TADV, TSTART, TSTOP, tau=tau)
        for row in out:
            if sites[3] in (row["t1"], row["t2"]):
                assert row["tau1"] == ehc.TAUDEF
                assert row["tau2"] == ehc.TAUDEF
            else:
                np.testing.assert_allclose(row["tau1"], tau[row["t1"]])
                np.testing.assert_allclose(row["tau2"], tau[row["t2"]])

    def test_space_site_tau_zero(self, array):
        """Space-based sites (xyz=0) get tau=0 regardless of input."""
        arr = array.copy()
        tarr = np.asarray(arr.tarr).copy()
        tarr[0]["x"] = tarr[0]["y"] = tarr[0]["z"] = 0.0
        arr.tarr = tarr
        space_site = tarr[0]["site"]
        out = os_sim.make_uvpoints(arr, 17.761, -29.0, 230e9, BW,
                                   TINT, TADV, TSTART, TSTOP,
                                   tau=0.25, no_elevcut_space=True)
        space_rows = out[(out["t1"] == space_site) | (out["t2"] == space_site)]
        assert len(space_rows) > 0
        for row in space_rows:
            if row["t1"] == space_site:
                assert row["tau1"] == 0.0
            if row["t2"] == space_site:
                assert row["tau2"] == 0.0

    def test_tstop_lt_tstart_wraps(self, array):
        out = os_sim.make_uvpoints(array, 17.761, -29.0, 230e9, BW,
                                   TINT, TADV, 22.0, 2.0)
        assert len(out) > 0
        assert out["time"].min() >= 22.0
        assert out["time"].max() < 26.0

    def test_invalid_timetype_falls_back_to_utc(self, array, capsys):
        out_bad = os_sim.make_uvpoints(array, 17.761, -29.0, 230e9, BW,
                                       TINT, TADV, TSTART, TSTOP, timetype="WAT")
        captured = capsys.readouterr()
        assert "Time Type Not Recognized" in captured.out
        out_utc = os_sim.make_uvpoints(array, 17.761, -29.0, 230e9, BW,
                                       TINT, TADV, TSTART, TSTOP, timetype="UTC")
        assert len(out_bad) == len(out_utc)
        np.testing.assert_allclose(out_bad["u"], out_utc["u"])
        np.testing.assert_allclose(out_bad["v"], out_utc["v"])

    def test_no_mutual_visibilities_raises(self, array):
        """Single-element 'array' produces no baselines -> Exception."""
        single = array.copy()
        single.tarr = np.asarray(array.tarr)[:1].copy()
        with pytest.raises(Exception, match="No mutual visibilities"):
            os_sim.make_uvpoints(single, 17.761, -29.0, 230e9, BW,
                                 TINT, TADV, TSTART, TSTOP)

    def test_warns_on_nonpositive_sefd(self, array, capsys):
        bad = array.copy()
        tarr = np.asarray(array.tarr).copy()
        tarr[0]["sefdr"] = 0.0
        bad.tarr = tarr
        os_sim.make_uvpoints(bad, 17.761, -29.0, 230e9, BW,
                             TINT, TADV, TSTART, TSTOP)
        captured = capsys.readouterr()
        assert "SEFDs are <= 0" in captured.out


# ---------------------------------------------------------------------------
# sample_vis  (extends original direct-vs-nfft parity tests)
# ---------------------------------------------------------------------------


class TestSampleVisTtypeParity:
    """sample_vis output agrees across ttype='direct' and ttype='nfft' (existing test)."""

    def test_stokes_i(self, asymmetric_image, array):
        obs_direct = _observe(asymmetric_image, array, "direct")
        obs_nfft = _observe(asymmetric_image, array, "nfft")
        np.testing.assert_allclose(
            obs_nfft.data["vis"], obs_direct.data["vis"],
            rtol=1e-6, atol=1e-9,
        )

    def test_polarimetric_all_stokes(self, asymmetric_image_pol, array):
        obs_direct = _observe(asymmetric_image_pol, array, "direct")
        obs_nfft = _observe(asymmetric_image_pol, array, "nfft")
        for field in ("vis", "qvis", "uvis", "vvis"):
            np.testing.assert_allclose(
                obs_nfft.data[field], obs_direct.data[field],
                rtol=1e-6, atol=1e-9,
                err_msg=f"direct vs nfft mismatch on {field}",
            )

    def test_fast_matches_direct(self, asymmetric_image, array):
        """ttype='fast' (FFT + interpolation) matches direct to a looser tolerance."""
        obs_direct = _observe(asymmetric_image, array, "direct")
        obs_fast = _observe(asymmetric_image, array, "fast")
        np.testing.assert_allclose(
            obs_fast.data["vis"], obs_direct.data["vis"],
            rtol=1e-2, atol=5e-3,
        )


class TestSampleVisBranches:
    """Direct coverage of sample_vis flags and error paths."""

    def test_invalid_uv_shape_raises(self, asymmetric_image):
        with pytest.raises(Exception, match="list of pairs"):
            os_sim.sample_vis(asymmetric_image, np.zeros((10, 3)), ttype="direct")

    def test_invalid_polrep_raises(self, asymmetric_image, obs):
        uv = np.column_stack([obs.data["u"], obs.data["v"]])
        with pytest.raises(Exception, match="only 'stokes' and 'circ'"):
            os_sim.sample_vis(asymmetric_image, uv, polrep_obs="linear", ttype="direct")

    def test_nfft_rejects_odd_dims(self, array):
        odd = eh.image.make_empty(31, 200 * eh.RADPERUAS, 17.761, -29.0, rf=230e9)
        odd = odd.add_gauss(1.0, (50 * eh.RADPERUAS, 50 * eh.RADPERUAS, 0, 0, 0))
        with pytest.raises(Exception, match="odd image dimensions"):
            os_sim.sample_vis(odd, np.array([[1e9, 1e9]]), ttype="nfft")

    def test_polrep_obs_circ(self, asymmetric_image_pol, obs):
        uv = np.column_stack([obs.data["u"], obs.data["v"]])
        out_circ = os_sim.sample_vis(asymmetric_image_pol, uv,
                                     polrep_obs="circ", ttype="direct", verbose=False)
        out_stokes = os_sim.sample_vis(asymmetric_image_pol, uv,
                                       polrep_obs="stokes", ttype="direct", verbose=False)
        # circ order is RR, LL, RL, LR.  Relations:
        #   RR = I + V,  LL = I - V,  RL = Q + iU,  LR = Q - iU
        I, Q, U, V = out_stokes
        np.testing.assert_allclose(out_circ[0], I + V, atol=1e-10)
        np.testing.assert_allclose(out_circ[1], I - V, atol=1e-10)
        np.testing.assert_allclose(out_circ[2], Q + 1j * U, atol=1e-10)
        np.testing.assert_allclose(out_circ[3], Q - 1j * U, atol=1e-10)

    def test_zero_empty_pol_true(self, asymmetric_image, obs):
        """Stokes-I-only image: Q/U/V slots should be zero arrays with zero_empty_pol=True."""
        uv = np.column_stack([obs.data["u"], obs.data["v"]])
        out = os_sim.sample_vis(asymmetric_image, uv, ttype="direct",
                                zero_empty_pol=True, verbose=False)
        assert all(d is not None for d in out)
        for d in out[1:]:
            np.testing.assert_array_equal(d, 0.0)
            assert d.shape == (len(uv),)

    def test_zero_empty_pol_false(self, asymmetric_image, obs):
        uv = np.column_stack([obs.data["u"], obs.data["v"]])
        out = os_sim.sample_vis(asymmetric_image, uv, ttype="direct",
                                zero_empty_pol=False, verbose=False)
        assert out[0] is not None
        for d in out[1:]:
            assert d is None

    def test_sgrscat_multiplies_by_kernel(self, asymmetric_image, obs):
        uv = np.column_stack([obs.data["u"], obs.data["v"]])
        clean = os_sim.sample_vis(asymmetric_image, uv,
                                  ttype="direct", sgrscat=False, verbose=False)
        scat = os_sim.sample_vis(asymmetric_image, uv,
                                 ttype="direct", sgrscat=True, verbose=False)
        ker = obsh.sgra_kernel_uv(asymmetric_image.rf, uv[:, 0], uv[:, 1])
        np.testing.assert_allclose(scat[0], clean[0] * ker, atol=1e-12)

    def test_image_pa_rotates_uv(self, asymmetric_image, obs):
        """im.pa rotates the input uv before sampling: vis(uv; pa) == vis(R(pa)@uv; pa=0)."""
        uv = np.column_stack([obs.data["u"], obs.data["v"]])
        theta = np.pi / 5

        im_pa = asymmetric_image.copy()
        im_pa.pa = theta
        out_pa = os_sim.sample_vis(im_pa, uv, ttype="direct", verbose=False)[0]

        c, s = np.cos(theta), np.sin(theta)
        uv_rot = np.column_stack([c * uv[:, 0] - s * uv[:, 1],
                                  s * uv[:, 0] + c * uv[:, 1]])
        out_pa0 = os_sim.sample_vis(asymmetric_image, uv_rot, ttype="direct", verbose=False)[0]
        np.testing.assert_allclose(out_pa, out_pa0, atol=1e-12)


# ---------------------------------------------------------------------------
# make_jones
# ---------------------------------------------------------------------------


def _all_jones(jm):
    """Flatten {site: {time: 2x2}} into an (Nsite*Ntime, 2, 2) array."""
    out = []
    for site_dict in jm.values():
        for mat in site_dict.values():
            out.append(mat)
    return np.array(out)


class TestMakeJones:
    """Coverage for make_jones.  Defaults give identity; each flag perturbs it."""

    def test_default_all_identity(self, obs):
        jm = os_sim.make_jones(obs)
        mats = _all_jones(jm)
        expected = np.broadcast_to(np.eye(2), mats.shape)
        np.testing.assert_allclose(mats, expected, atol=1e-12)

    def test_default_keys_one_per_site(self, obs):
        jm = os_sim.make_jones(obs)
        assert set(jm.keys()) == set(obs.tarr["site"])
        for site_dict in jm.values():
            assert len(site_dict) > 0

    def test_seed_reproducible(self, obs):
        jm1 = os_sim.make_jones(obs, ampcal=False, phasecal=False, seed=42)
        jm2 = os_sim.make_jones(obs, ampcal=False, phasecal=False, seed=42)
        np.testing.assert_array_equal(_all_jones(jm1), _all_jones(jm2))

    def test_seed_differs(self, obs):
        jm1 = os_sim.make_jones(obs, ampcal=False, phasecal=False, seed=42)
        jm2 = os_sim.make_jones(obs, ampcal=False, phasecal=False, seed=43)
        assert not np.allclose(_all_jones(jm1), _all_jones(jm2))

    def test_frcal_false_diagonal_with_phase(self, obs):
        """frcal=False: diagonal unit-modulus phases; off-diagonals stay zero."""
        jm = os_sim.make_jones(obs, frcal=False, seed=42)
        mats = _all_jones(jm)
        # off-diagonals still zero
        np.testing.assert_allclose(mats[:, 0, 1], 0, atol=1e-12)
        np.testing.assert_allclose(mats[:, 1, 0], 0, atol=1e-12)
        # diagonals are unit modulus (no amp/opacity corruption)
        np.testing.assert_allclose(np.abs(mats[:, 0, 0]), 1.0, atol=1e-12)
        np.testing.assert_allclose(np.abs(mats[:, 1, 1]), 1.0, atol=1e-12)
        # not identity for at least some entries (fr_elev varies in EHT2017)
        assert not np.allclose(mats, np.broadcast_to(np.eye(2), mats.shape))

    def test_ampcal_false_perturbs_diagonal(self, obs):
        jm = os_sim.make_jones(obs, ampcal=False, seed=42)
        mats = _all_jones(jm)
        # off-diagonals still zero, but diagonal magnitudes != 1
        np.testing.assert_allclose(mats[:, 0, 1], 0, atol=1e-12)
        np.testing.assert_allclose(mats[:, 1, 0], 0, atol=1e-12)
        assert not np.allclose(np.abs(mats[:, 0, 0]), 1.0)
        assert not np.allclose(np.abs(mats[:, 1, 1]), 1.0)

    def test_ampcal_false_neggains(self, obs):
        jm = os_sim.make_jones(obs, ampcal=False, neggains=True, seed=42)
        mats = _all_jones(jm)
        assert np.all(np.abs(mats[:, 0, 0]) <= 1.0 + 1e-12)
        assert np.all(np.abs(mats[:, 1, 1]) <= 1.0 + 1e-12)

    def test_phasecal_false_unit_modulus_phases(self, obs):
        jm = os_sim.make_jones(obs, phasecal=False, seed=42)
        mats = _all_jones(jm)
        # diagonals still unit modulus (no amp corruption)
        np.testing.assert_allclose(np.abs(mats[:, 0, 0]), 1.0, atol=1e-12)
        np.testing.assert_allclose(np.abs(mats[:, 1, 1]), 1.0, atol=1e-12)
        # but not real-valued (phases are nontrivial)
        assert np.max(np.abs(np.imag(mats[:, 0, 0]))) > 0.01

    def test_phasecal_false_gaussian_branch(self, obs):
        """phase_std>=0 selects the Gaussian branch (tighter spread)."""
        jm_uniform = os_sim.make_jones(obs, phasecal=False, phase_std=-1, seed=42)
        jm_narrow = os_sim.make_jones(obs, phasecal=False, phase_std=0.01, seed=42)
        ph_u = np.angle(_all_jones(jm_uniform)[:, 0, 0])
        ph_n = np.angle(_all_jones(jm_narrow)[:, 0, 0])
        # Gaussian phase_std=0.01 should be far tighter than uniform [-pi,pi]
        assert np.std(ph_n) < np.std(ph_u) / 10

    def test_dcal_false_nonzero_offdiag(self, obs_asym):
        """dcal=False: off-diagonals = dR*gainR, dL*gainL (here gain=1, fr=0, so = dR, dL)."""
        jm = os_sim.make_jones(obs_asym, dcal=False, frcal=True, seed=42)
        mats = _all_jones(jm)
        # diagonals are still 1 (ampcal=True, phasecal=True)
        np.testing.assert_allclose(np.abs(mats[:, 0, 0]), 1.0, atol=1e-12)
        np.testing.assert_allclose(np.abs(mats[:, 1, 1]), 1.0, atol=1e-12)
        # off-diagonals nonzero, since tarr['dr'] != 0 in asymmetric array
        assert np.max(np.abs(mats[:, 0, 1])) > 0.001
        assert np.max(np.abs(mats[:, 1, 0])) > 0.001

    def test_rlgaincal_false_breaks_rl_symmetry(self, obs):
        jm = os_sim.make_jones(obs, ampcal=False, rlgaincal=False,
                               rlratio_std=0.1, seed=42)
        mats = _all_jones(jm)
        # gain_R != gain_L on every entry that ampcal=False touches
        assert not np.allclose(np.abs(mats[:, 0, 0]), np.abs(mats[:, 1, 1]))

    def test_opacitycal_false_attenuates(self, obs):
        jm = os_sim.make_jones(obs, opacitycal=False, taup=0.0, seed=42)
        mats = _all_jones(jm)
        # opacity is amplitude-only, no phase: imaginary part 0
        np.testing.assert_allclose(np.imag(mats[:, 0, 0]), 0, atol=1e-12)
        # all amplitudes attenuated below 1
        assert np.all(np.abs(mats[:, 0, 0]) <= 1.0 + 1e-12)
        # at least some are noticeably attenuated
        assert np.min(np.abs(mats[:, 0, 0])) < 0.99

    def test_stabilize_scan_amp_constant_within_scan(self, obs):
        """With stabilize_scan_amp=True, the gain is constant for all times in the same scan."""
        jm = os_sim.make_jones(obs, ampcal=False, stabilize_scan_amp=True, seed=42)
        scans = obs.scans
        for site, site_dict in jm.items():
            times = np.array(sorted(site_dict.keys()))
            for sc_start, sc_end in scans:
                mask = (times >= sc_start) & (times <= sc_end)
                if mask.sum() < 2:
                    continue
                gains = np.array([site_dict[t][0, 0] for t in times[mask]])
                np.testing.assert_allclose(gains, gains[0], atol=1e-12,
                                           err_msg=f"site {site} scan {sc_start}-{sc_end}")
                break   # one scan-with-multiple-times check per site is enough

    def test_dict_kwargs_honored(self, obs):
        """Per-site gainp dict is honored: site-specific noise level."""
        sites = list(obs.tarr["site"])
        gainp = {s: 0.1 for s in sites}
        gainp[sites[0]] = 0.0   # one site has zero gain noise
        jm = os_sim.make_jones(obs, ampcal=False, gainp=gainp, seed=42)
        # The zero-noise site should still have constant log-gain (from gain_offset).
        # Compare its gain variance to other sites' variance.
        var_zero = np.var(np.abs([m[0, 0] for m in jm[sites[0]].values()]))
        var_other = np.var(np.abs([m[0, 0] for m in jm[sites[1]].values()]))
        assert var_zero < var_other

    def test_gmst_timetype(self, obs):
        """GMST path: should run without error and produce same shape."""
        obs_gmst = obs.switch_timetype("GMST")
        jm = os_sim.make_jones(obs_gmst, frcal=False, seed=42)
        mats = _all_jones(jm)
        assert mats.shape[1:] == (2, 2)
        assert mats.shape[0] > 0

    def test_caltable_path_writes_file(self, obs, tmp_path):
        prefix = tmp_path / "cal"
        os_sim.make_jones(obs, ampcal=False, dcal=False,
                          caltable_path=str(prefix), seed=42)
        # Caltable.save_txt creates a directory of per-site files
        caldir = tmp_path / "cal_simdata_caltable"
        assert caldir.exists()
        files = list(caldir.iterdir())
        assert len(files) > 0


# ---------------------------------------------------------------------------
# make_jones_inverse
# ---------------------------------------------------------------------------


class TestMakeJonesInverse:

    def test_default_all_identity(self, obs):
        jm = os_sim.make_jones_inverse(obs)
        mats = _all_jones(jm)
        expected = np.broadcast_to(np.eye(2), mats.shape)
        np.testing.assert_allclose(mats, expected, atol=1e-12)

    def test_frcal_false_phases_reverse_jones(self, obs):
        """Forward and inverse have opposite-sign feed-rotation phases on each diagonal slot.

        Forward [0,0] = exp(-i*fr); inverse [0,0] = exp(+i*fr).  Same slot, opposite phase.
        """
        jm = os_sim.make_jones(obs, frcal=False, seed=42)
        jm_inv = os_sim.make_jones_inverse(obs, frcal=False)
        for site in jm:
            for time in jm[site]:
                ph_fwd_00 = np.angle(jm[site][time][0, 0])
                ph_inv_00 = np.angle(jm_inv[site][time][0, 0])
                np.testing.assert_allclose(ph_inv_00, -ph_fwd_00, atol=1e-12)
                ph_fwd_11 = np.angle(jm[site][time][1, 1])
                ph_inv_11 = np.angle(jm_inv[site][time][1, 1])
                np.testing.assert_allclose(ph_inv_11, -ph_fwd_11, atol=1e-12)

    def test_inverse_opacity_amplitude(self, obs):
        """opacitycal=False: amplitudes scale as 1/sqrt(exp(tau_eff)*exp(tau_eff)) = 1/exp(tau_eff)."""
        jm_inv = os_sim.make_jones_inverse(obs, opacitycal=False)
        mats = _all_jones(jm_inv)
        np.testing.assert_allclose(np.imag(mats[:, 0, 0]), 0, atol=1e-12)
        # off-diagonals still zero
        np.testing.assert_allclose(mats[:, 0, 1], 0, atol=1e-12)
        # amplitudes >= 1 (inverse of attenuation)
        assert np.all(np.abs(mats[:, 0, 0]) >= 1.0 - 1e-12)
        assert np.max(np.abs(mats[:, 0, 0])) > 1.01

    def test_inverse_dcal_offdiagonal(self, obs_asym):
        """dcal=False: |off-diagonals| = |dR/(1-dL*dR)|, |dL/(1-dL*dR)|.

        Phases also pick up feed-rotation factors (the `frcal and not dcal` branch
        applies a 2*fr correction even when frcal=True), so we test magnitudes.
        """
        jm_inv = os_sim.make_jones_inverse(obs_asym, dcal=False)
        for site, site_dict in jm_inv.items():
            i = list(obs_asym.tarr["site"]).index(site)
            dR = obs_asym.tarr[i]["dr"]
            dL = obs_asym.tarr[i]["dl"]
            pref = 1.0 / (1.0 - dL * dR)
            for mat in site_dict.values():
                np.testing.assert_allclose(np.abs(mat[0, 1]), np.abs(pref * dR), atol=1e-12)
                np.testing.assert_allclose(np.abs(mat[1, 0]), np.abs(pref * dL), atol=1e-12)

    def test_inverse_actually_inverts(self, obs_asym):
        """J^-1 @ J should be identity at every (site, time), within tolerance.

        Test with frcal=False (the most-likely-to-have-sign-bug term)."""
        jm = os_sim.make_jones(obs_asym, frcal=False, seed=42)
        jm_inv = os_sim.make_jones_inverse(obs_asym, frcal=False)
        for site in jm:
            for time in jm[site]:
                prod = jm_inv[site][time] @ jm[site][time]
                np.testing.assert_allclose(prod, np.eye(2), atol=1e-10,
                                           err_msg=f"site {site} time {time}")


# ---------------------------------------------------------------------------
# add_jones_and_noise
# ---------------------------------------------------------------------------


class TestAddJonesAndNoise:

    def test_identity_preserves_vis(self, obs):
        """All cal True, no thermal noise -> visibilities and sigmas unchanged."""
        obsdat = os_sim.add_jones_and_noise(obs, add_th_noise=False,
                                            verbose=False, seed=42)
        np.testing.assert_allclose(obsdat["vis"], obs.data["vis"], atol=1e-12)
        np.testing.assert_allclose(obsdat["sigma"], obs.data["sigma"], atol=1e-12)

    def test_seed_reproducible_no_noise(self, obs):
        """Without thermal noise, same seed -> identical output."""
        o1 = os_sim.add_jones_and_noise(obs, add_th_noise=False, ampcal=False,
                                        verbose=False, seed=42)
        o2 = os_sim.add_jones_and_noise(obs, add_th_noise=False, ampcal=False,
                                        verbose=False, seed=42)
        np.testing.assert_array_equal(o1["vis"], o2["vis"])

    def test_seed_differs_no_noise(self, obs):
        o1 = os_sim.add_jones_and_noise(obs, add_th_noise=False, ampcal=False,
                                        verbose=False, seed=42)
        o2 = os_sim.add_jones_and_noise(obs, add_th_noise=False, ampcal=False,
                                        verbose=False, seed=43)
        assert not np.allclose(o1["vis"], o2["vis"])

    def test_thermal_noise_residual_matches_sigma(self, obs):
        """With add_th_noise, residual std should be ~sigma."""
        obsdat = os_sim.add_jones_and_noise(obs, add_th_noise=True,
                                            verbose=False, seed=42)
        residual = obsdat["vis"] - obs.data["vis"]
        ratio = np.std(residual.real) / np.mean(obsdat["sigma"])
        assert 0.5 < ratio < 2.0

    def test_ampcal_false_deviates_amplitude(self, obs):
        out = os_sim.add_jones_and_noise(obs, add_th_noise=False, ampcal=False,
                                         verbose=False, seed=42)
        # phases preserved, amplitudes shifted
        np.testing.assert_allclose(np.angle(out["vis"]),
                                   np.angle(obs.data["vis"]), atol=1e-10)
        assert not np.allclose(np.abs(out["vis"]), np.abs(obs.data["vis"]))

    def test_phasecal_false_deviates_phase(self, obs):
        out = os_sim.add_jones_and_noise(obs, add_th_noise=False, phasecal=False,
                                         verbose=False, seed=42)
        # amplitudes preserved, phases shifted
        np.testing.assert_allclose(np.abs(out["vis"]),
                                   np.abs(obs.data["vis"]), atol=1e-10)
        assert not np.allclose(np.angle(out["vis"]), np.angle(obs.data["vis"]))

    def test_sefd_nonpositive_falls_back_to_sigmas(self, obs, capsys):
        """SEFD<=0: code uses obs.data sigmas rather than recomputing.  Warning printed."""
        obs_bad = obs.copy()
        tarr = np.asarray(obs_bad.tarr).copy()
        tarr[0]["sefdr"] = 0.0
        obs_bad.tarr = tarr
        out = os_sim.add_jones_and_noise(obs_bad, add_th_noise=False, verbose=True, seed=42)
        captured = capsys.readouterr()
        assert "SEFDs are <= 0" in captured.out
        # Sigmas should equal the original obs sigmas (no recomputation)
        np.testing.assert_allclose(out["sigma"], obs_bad.data["sigma"])

    def test_caltable_path_propagates(self, obs, tmp_path):
        prefix = tmp_path / "cal"
        os_sim.add_jones_and_noise(obs, add_th_noise=False, ampcal=False,
                                   caltable_path=str(prefix),
                                   verbose=False, seed=42)
        caldir = tmp_path / "cal_simdata_caltable"
        assert caldir.exists()


# ---------------------------------------------------------------------------
# apply_jones_inverse
# ---------------------------------------------------------------------------


class TestApplyJonesInverse:

    def test_identity_preserves_vis(self, obs):
        out = os_sim.apply_jones_inverse(obs, verbose=False)
        np.testing.assert_allclose(out["vis"], obs.data["vis"], atol=1e-12)
        np.testing.assert_allclose(out["sigma"], obs.data["sigma"], atol=1e-12)

    def test_roundtrip_with_frcal_only(self, obs):
        """Corrupt with frcal=False (deterministic), then uncorrupt with frcal=False: identity."""
        obs_corr = obs.copy()
        obs_corr.data = os_sim.add_jones_and_noise(obs, add_th_noise=False, frcal=False,
                                                   verbose=False, seed=42)
        obs_back = obs_corr.copy()
        obs_back.data = os_sim.apply_jones_inverse(obs_corr, frcal=False, verbose=False)
        np.testing.assert_allclose(obs_back.data["vis"], obs.data["vis"], atol=1e-10)

    def test_roundtrip_with_opacity_only(self, obs):
        """taup=0 makes opacity attenuation deterministic; apply_jones_inverse inverts exactly."""
        obs_corr = obs.copy()
        obs_corr.data = os_sim.add_jones_and_noise(
            obs, add_th_noise=False, opacitycal=False, taup=0.0,
            verbose=False, seed=42,
        )
        obs_back = obs_corr.copy()
        obs_back.data = os_sim.apply_jones_inverse(obs_corr, opacitycal=False, verbose=False)
        np.testing.assert_allclose(obs_back.data["vis"], obs.data["vis"], atol=1e-10)


# ---------------------------------------------------------------------------
# add_noise  (legacy path)
# ---------------------------------------------------------------------------


class TestAddNoiseLegacy:

    def test_identity_preserves_vis(self, obs):
        out = os_sim.add_noise(obs, add_th_noise=False, verbose=False, seed=42)
        np.testing.assert_allclose(out["vis"], obs.data["vis"], atol=1e-12)
        np.testing.assert_allclose(out["sigma"], obs.data["sigma"], atol=1e-12)

    def test_seed_reproducible(self, obs):
        o1 = os_sim.add_noise(obs, add_th_noise=False, ampcal=False,
                              verbose=False, seed=42)
        o2 = os_sim.add_noise(obs, add_th_noise=False, ampcal=False,
                              verbose=False, seed=42)
        np.testing.assert_array_equal(o1["vis"], o2["vis"])

    def test_sigmat_raises(self, obs):
        with pytest.raises(Exception, match="correlated gains not supported"):
            os_sim.add_noise(obs, ampcal=False, sigmat=10.0, verbose=False, seed=42)

    def test_caltable_path_prints_unsupported(self, obs, capsys, tmp_path):
        os_sim.add_noise(obs, add_th_noise=False,
                         caltable_path=str(tmp_path / "cal"),
                         verbose=False, seed=42)
        captured = capsys.readouterr()
        assert "caltable saving not implemented" in captured.out

    def test_ampcal_false_deviates_amplitude(self, obs):
        out = os_sim.add_noise(obs, add_th_noise=False, ampcal=False,
                               verbose=False, seed=42)
        # legacy add_noise multiplies vis by a real gain factor: phases unchanged
        np.testing.assert_allclose(np.angle(out["vis"]),
                                   np.angle(obs.data["vis"]), atol=1e-10)
        assert not np.allclose(np.abs(out["vis"]), np.abs(obs.data["vis"]))

    def test_phasecal_false_deviates_phase(self, obs):
        out = os_sim.add_noise(obs, add_th_noise=False, phasecal=False,
                               verbose=False, seed=42)
        np.testing.assert_allclose(np.abs(out["vis"]),
                                   np.abs(obs.data["vis"]), atol=1e-10)
        assert not np.allclose(np.angle(out["vis"]), np.angle(obs.data["vis"]))

    def test_dict_gainp(self, obs):
        sites = list(obs.tarr["site"])
        gainp_dict = {s: 0.1 for s in sites}
        gainp_dict[sites[0]] = 0.0
        out_dict = os_sim.add_noise(obs, add_th_noise=False, ampcal=False,
                                    gainp=gainp_dict, verbose=False, seed=42)
        out_scalar = os_sim.add_noise(obs, add_th_noise=False, ampcal=False,
                                      gainp=0.1, verbose=False, seed=42)
        # The first site has zero gain noise in the dict -> at least some baselines differ.
        assert not np.allclose(out_dict["vis"], out_scalar["vis"])

    def test_stabilize_scan_amp_constant_within_scan(self, obs):
        """With stabilize_scan_amp=True, identical times-in-scan yield identical gain factors,
        so the |vis| ratio (corrupt/original) is constant within a scan on a given baseline."""
        out = os_sim.add_noise(obs, add_th_noise=False, ampcal=False,
                               stabilize_scan_amp=True, verbose=False, seed=42)
        # Pick the first baseline (t1, t2)
        d = obs.data
        t1_pick = d["t1"][0]
        t2_pick = d["t2"][0]
        mask = (d["t1"] == t1_pick) & (d["t2"] == t2_pick)
        ratios = np.abs(out["vis"][mask]) / np.abs(d["vis"][mask])
        # Identify times that fall in the same scan
        scans = obs.scans
        for sc_start, sc_end in scans:
            in_scan = (d["time"][mask] >= sc_start) & (d["time"][mask] <= sc_end)
            if in_scan.sum() < 2:
                continue
            np.testing.assert_allclose(ratios[in_scan], ratios[in_scan][0],
                                       rtol=1e-10)
            break

    def test_thermal_noise_residual_matches_sigma(self, obs):
        residual = out["vis"] - obs.data["vis"]
        ratio = np.std(residual.real) / np.mean(out["sigma"])
        assert 0.5 < ratio < 2.0
