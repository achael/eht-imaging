"""Tests for Array methods.

Method-by-method coverage of ehtim/array.py. Sections mirror the class
itself and reuse the session-scoped ``eht_array`` fixture from conftest.py
so the I/O cost is amortised.

Designed to survive the Phase-2 mixed-pol changes landing in PR #260
(branch ``feature/mixpol-phase2-array`` on ``dev-backend-mixpol``):

* Legacy field names ('sefdr', 'sefdl', 'dr', 'dl') will become title
  aliases on RL-feed arrays after #260, so tests using them on
  homogeneous-RL arrays continue to pass.
* DTARR will gain a ``feed_type`` column. Tests avoid asserting the full
  ``dtype.names`` tuple — they only check that the legacy columns are
  present.
* ``Array.tarr`` will return a ``TarrView`` wrapper instead of a raw
  recarray. Tests interact with ``tarr`` only via behaviour that the
  wrapper preserves (column access on homogeneous-RL arrays, iteration,
  ``len``, ``copy``, row indexing, numpy interop).
* ``add_site`` / ``add_satellite_*`` gain extra kwargs; this suite uses
  the legacy positional/keyword signatures only.
* ``Array.obsdata(polrep=...)`` will gain ``'lin'`` and ``'mixed'``
  options with stricter validation; tests here use only ``'stokes'`` and
  ``'circ'``, both of which remain valid on RL-feed arrays.
"""

import os

import numpy as np
import pytest

import ehtim as eh
import ehtim.const_def as ehc

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_DTARR_DTYPE = np.dtype(ehc.DTARR)


def _site_row(arr, site):
    """Fetch a row by site name. ``arr.tarr`` may be a recarray today or a
    TarrView wrapper after PR #260 — both support row indexing."""
    return arr.tarr[arr.tkey[site]]


# ---------------------------------------------------------------------------
# Section 1: Construction, tarr setter, copy
# ---------------------------------------------------------------------------


def test_init_builds_tkey_from_sites(eht_array):
    for site in eht_array.tarr["site"]:
        assert eht_array.tkey[str(site)] >= 0
        assert str(_site_row(eht_array, site)["site"]) == str(site)


def test_init_tarr_dtype_has_legacy_columns(eht_array):
    # PR #260 added a feed_type column and promoted sefdr/sefdl/dr/dl to title
    # aliases over sefd_p1/sefd_p2/d_p1/d_p2. Title aliases live in
    # dtype.fields (a dict including aliases), not dtype.names (primary only).
    fields = _DTARR_DTYPE.fields
    for col in ("site", "x", "y", "z", "sefdr", "sefdl",
                "dr", "dl", "fr_par", "fr_elev", "fr_off"):
        assert col in fields


def test_init_empty_array_has_empty_tkey():
    arr = eh.array.Array(np.zeros(0, dtype=ehc.DTARR))
    assert arr.tkey == {}
    assert len(arr.tarr) == 0


def test_init_nan_coords_without_ephemeris_raises():
    tarr = np.zeros(1, dtype=ehc.DTARR)
    tarr["site"] = ["SAT_BAD"]
    tarr["x"] = [np.nan]
    tarr["y"] = [np.nan]
    tarr["z"] = [np.nan]
    # Today the missing-ephem path leaks a KeyError; the message
    # "no ephemeris" inside Array.__init__ would be raised if the except
    # clause caught KeyError. We assert "something raised" to stay
    # robust if the except-clause is later fixed.
    with pytest.raises(Exception):
        eh.array.Array(tarr)


def test_init_nan_coords_with_ephemeris_succeeds():
    tarr = np.zeros(1, dtype=ehc.DTARR)
    tarr["site"] = ["SAT_OK"]
    tarr["x"] = [np.nan]
    tarr["y"] = [np.nan]
    tarr["z"] = [np.nan]
    arr = eh.array.Array(tarr, ephem={"SAT_OK": ["name", "line1", "line2"]})
    assert len(arr.tarr) == 1


def test_init_wrong_length_ephemeris_raises():
    tarr = np.zeros(1, dtype=ehc.DTARR)
    tarr["site"] = ["SAT_BAD"]
    tarr["x"] = [np.nan]
    tarr["y"] = [np.nan]
    tarr["z"] = [np.nan]
    with pytest.raises(Exception, match="wrong ephemeris"):
        eh.array.Array(tarr, ephem={"SAT_BAD": ["only_one"]})


def test_tarr_setter_rebuilds_tkey(eht_array):
    arr = eht_array.copy()
    # Pull the underlying recarray out so we can reverse-index safely
    # whether tarr is a raw ndarray (today) or a TarrView (post-#260).
    reversed_tarr = np.asarray(arr.tarr).copy()[::-1]
    arr.tarr = reversed_tarr
    for site in arr.tarr["site"]:
        assert str(_site_row(arr, site)["site"]) == str(site)


def test_copy_is_deep(eht_array):
    other = eht_array.copy()
    assert other is not eht_array
    # Mutating the copy's underlying ndarray must not touch the fixture.
    raw_other = np.asarray(other.tarr)
    raw_other["x"][0] += 1.0
    np.testing.assert_array_equal(
        np.asarray(other.tarr)["x"][0],
        raw_other["x"][0],
    )
    assert np.asarray(eht_array.tarr)["x"][0] != raw_other["x"][0]


# ---------------------------------------------------------------------------
# Section 2: listbls
# ---------------------------------------------------------------------------


def test_listbls_count_matches_combinations(eht_array):
    n = len(eht_array.tarr)
    assert eht_array.listbls().shape == (n * (n - 1) // 2, 2)


def test_listbls_no_self_baselines(eht_array):
    bls = eht_array.listbls()
    for t1, t2 in bls:
        assert t1 != t2


def test_listbls_baselines_unique(eht_array):
    bls = eht_array.listbls()
    pairs = {frozenset((t1, t2)) for t1, t2 in bls}
    assert len(pairs) == len(bls)


def test_listbls_sorted_lexicographically(eht_array):
    # listbls iterates over sorted(tarr['site']) so the output is
    # lexicographic by t1, then t2.
    bls = eht_array.listbls()
    sites = sorted(str(s) for s in eht_array.tarr["site"])
    expected = []
    for i, s1 in enumerate(sites):
        for s2 in sites[i + 1:]:
            expected.append([s1, s2])
    np.testing.assert_array_equal(bls, np.array(expected))


def test_listbls_empty_array_returns_empty():
    arr = eh.array.Array(np.zeros(0, dtype=ehc.DTARR))
    bls = arr.listbls()
    assert bls.shape == (0,)


# ---------------------------------------------------------------------------
# Section 3: make_subarray
# ---------------------------------------------------------------------------


def test_make_subarray_keeps_requested_sites(eht_array):
    requested = ["ALMA", "SMA"]
    sub = eht_array.make_subarray(requested)
    assert set(str(s) for s in sub.tarr["site"]) == set(requested)


def test_make_subarray_preserves_metadata(eht_array):
    sub = eht_array.make_subarray(["ALMA", "SMA"])
    alma_row = _site_row(eht_array, "ALMA")
    alma_sub = _site_row(sub, "ALMA")
    for col in ("x", "y", "z", "sefdr", "sefdl", "fr_par"):
        assert alma_sub[col] == alma_row[col]


def test_make_subarray_with_unknown_site_returns_empty(eht_array):
    sub = eht_array.make_subarray(["NOTASITE"])
    assert len(sub.tarr) == 0


def test_make_subarray_preserves_ephem(eht_array):
    sub = eht_array.make_subarray(["ALMA"])
    assert sub.ephem == eht_array.ephem


# ---------------------------------------------------------------------------
# Section 4: add_site
# ---------------------------------------------------------------------------


def test_add_site_appends_row(eht_array):
    n0 = len(eht_array.tarr)
    out = eht_array.add_site("NEW", (1.0, 2.0, 3.0))
    assert len(out.tarr) == n0 + 1
    assert "NEW" in out.tkey


def test_add_site_does_not_mutate_original(eht_array):
    n0 = len(eht_array.tarr)
    eht_array.add_site("TMPSITE", (1.0, 2.0, 3.0))
    assert len(eht_array.tarr) == n0
    assert "TMPSITE" not in eht_array.tkey


def test_add_site_writes_coords(eht_array):
    out = eht_array.add_site("NEW", (1.0, 2.0, 3.0))
    row = _site_row(out, "NEW")
    assert row["x"] == 1.0
    assert row["y"] == 2.0
    assert row["z"] == 3.0


def test_add_site_default_sefd_is_10000(eht_array):
    out = eht_array.add_site("NEW", (1.0, 2.0, 3.0))
    row = _site_row(out, "NEW")
    assert row["sefdr"] == 10000.0
    assert row["sefdl"] == 10000.0


def test_add_site_sefd_kwarg_applies_to_both_feeds(eht_array):
    out = eht_array.add_site("NEW", (1.0, 2.0, 3.0), sefd=5000)
    row = _site_row(out, "NEW")
    assert row["sefdr"] == 5000.0
    assert row["sefdl"] == 5000.0


def test_add_site_dr_dl_propagate(eht_array):
    out = eht_array.add_site("NEW", (1.0, 2.0, 3.0),
                             dr=0.1 + 0.2j, dl=0.3 + 0.4j)
    row = _site_row(out, "NEW")
    assert row["dr"] == 0.1 + 0.2j
    assert row["dl"] == 0.3 + 0.4j


def test_add_site_fr_params_propagate(eht_array):
    out = eht_array.add_site("NEW", (1.0, 2.0, 3.0),
                             fr_par=1.5, fr_elev=-1.0, fr_off=0.25)
    row = _site_row(out, "NEW")
    assert row["fr_par"] == 1.5
    assert row["fr_elev"] == -1.0
    assert row["fr_off"] == 0.25


# ---------------------------------------------------------------------------
# Section 5: remove_site
# ---------------------------------------------------------------------------


def test_remove_site_drops_row(eht_array):
    added = eht_array.add_site("TMP", (1.0, 2.0, 3.0))
    rem = added.remove_site("TMP")
    assert len(rem.tarr) == len(eht_array.tarr)
    assert "TMP" not in rem.tkey


def test_remove_site_unknown_raises(eht_array):
    with pytest.raises(Exception, match="could not find site"):
        eht_array.remove_site("NOTASITE")


def test_remove_site_does_not_mutate_original(eht_array):
    added = eht_array.add_site("TMP", (1.0, 2.0, 3.0))
    n_before = len(added.tarr)
    added.remove_site("TMP")
    assert len(added.tarr) == n_before


def test_remove_site_also_removes_ephem_entry():
    tarr = np.zeros(1, dtype=ehc.DTARR)
    tarr["site"] = ["SAT"]
    tarr["x"] = [np.nan]
    tarr["y"] = [np.nan]
    tarr["z"] = [np.nan]
    arr = eh.array.Array(tarr, ephem={"SAT": ["n", "l1", "l2"]})
    out = arr.remove_site("SAT")
    assert "SAT" not in out.ephem


# ---------------------------------------------------------------------------
# Section 6: add_satellite_tle
# ---------------------------------------------------------------------------


@pytest.mark.xfail(reason="Bug in Array.add_satellite_tle (array.py:264): "
                          "body references `tlearr` but the kwarg is `tlelist`. "
                          "Fixed by PR #260 which renames the kwarg to `tlearr`.")
def test_add_satellite_tle_appends_row(eht_array):
    tle = ["SAT1", "tle_line_1", "tle_line_2"]
    out = eht_array.add_satellite_tle(tle, sefd=8000)
    assert "SAT1" in out.tkey
    row = _site_row(out, "SAT1")
    assert row["sefdr"] == 8000.0
    assert row["sefdl"] == 8000.0
    assert out.ephem["SAT1"] == tle


# ---------------------------------------------------------------------------
# Section 7: add_satellite_elements
# ---------------------------------------------------------------------------


def test_add_satellite_elements_appends_row(eht_array):
    out = eht_array.add_satellite_elements("SAT_KEP", sefd=8000)
    assert "SAT_KEP" in out.tkey
    row = _site_row(out, "SAT_KEP")
    assert row["sefdr"] == 8000.0
    assert row["sefdl"] == 8000.0


def test_add_satellite_elements_coords_zero(eht_array):
    out = eht_array.add_satellite_elements("SAT_KEP")
    row = _site_row(out, "SAT_KEP")
    assert row["x"] == 0.0
    assert row["y"] == 0.0
    assert row["z"] == 0.0


def test_add_satellite_elements_ephem_has_six_kepler_elements(eht_array):
    out = eht_array.add_satellite_elements(
        "SAT_KEP",
        perigee_mjd=58000.0, period_days=1.0, eccentricity=0.1,
        inclination=45.0, arg_perigee=30.0, long_ascending=15.0,
    )
    assert len(out.ephem["SAT_KEP"]) == 6
    assert out.ephem["SAT_KEP"][0] == 58000.0
    assert out.ephem["SAT_KEP"][1] == 1.0


# ---------------------------------------------------------------------------
# Section 8: obsdata
# ---------------------------------------------------------------------------


def test_obsdata_returns_obsdata_with_stokes(eht_array):
    obs = eht_array.obsdata(17.761, -29.0, 230e9, 4e9,
                             tint=5, tadv=600, tstart=0, tstop=24)
    assert isinstance(obs, eh.obsdata.Obsdata)
    assert obs.polrep == "stokes"
    assert len(obs.data) > 0


def test_obsdata_polrep_circ(eht_array):
    obs = eht_array.obsdata(17.761, -29.0, 230e9, 4e9,
                             tint=5, tadv=600, tstart=0, tstop=24,
                             polrep="circ")
    assert obs.polrep == "circ"


def test_obsdata_propagates_ra_dec_rf_bw(eht_array):
    obs = eht_array.obsdata(17.761, -29.0, 230e9, 4e9,
                             tint=5, tadv=600, tstart=0, tstop=24)
    assert obs.ra == pytest.approx(17.761)
    assert obs.dec == pytest.approx(-29.0)
    assert obs.rf == pytest.approx(230e9)
    assert obs.bw == pytest.approx(4e9)


def test_obsdata_default_timetype_utc(eht_array):
    obs = eht_array.obsdata(17.761, -29.0, 230e9, 4e9,
                             tint=5, tadv=600, tstart=0, tstop=24)
    assert obs.timetype == "UTC"


def test_obsdata_timetype_gmst(eht_array):
    obs = eht_array.obsdata(17.761, -29.0, 230e9, 4e9,
                             tint=5, tadv=600, tstart=0, tstop=24,
                             timetype="GMST")
    assert obs.timetype == "GMST"


def test_obsdata_calibration_flags_all_true(eht_array):
    obs = eht_array.obsdata(17.761, -29.0, 230e9, 4e9,
                             tint=5, tadv=600, tstart=0, tstop=24)
    # Array.obsdata always builds a perfect-calibration empty obs.
    assert obs.ampcal is True
    assert obs.phasecal is True
    assert obs.opacitycal is True
    assert obs.dcal is True
    assert obs.frcal is True


# ---------------------------------------------------------------------------
# Section 9: save_txt / load_txt round-trip
# ---------------------------------------------------------------------------


def test_save_load_txt_roundtrip_sites(eht_array, tmp_path):
    fname = str(tmp_path / "arr.txt")
    eht_array.save_txt(fname)
    loaded = eh.array.load_txt(fname)
    assert set(str(s) for s in loaded.tarr["site"]) == \
           set(str(s) for s in eht_array.tarr["site"])


def test_save_load_txt_roundtrip_coords(eht_array, tmp_path):
    fname = str(tmp_path / "arr.txt")
    eht_array.save_txt(fname)
    loaded = eh.array.load_txt(fname)
    # Sites may be reordered, so compare via tkey lookups.
    for site in eht_array.tarr["site"]:
        site = str(site)
        a = _site_row(eht_array, site)
        b = _site_row(loaded, site)
        assert a["x"] == pytest.approx(b["x"], rel=1e-6)
        assert a["y"] == pytest.approx(b["y"], rel=1e-6)
        assert a["z"] == pytest.approx(b["z"], rel=1e-6)


def test_save_load_txt_roundtrip_sefds(eht_array, tmp_path):
    fname = str(tmp_path / "arr.txt")
    eht_array.save_txt(fname)
    loaded = eh.array.load_txt(fname)
    for site in eht_array.tarr["site"]:
        site = str(site)
        a = _site_row(eht_array, site)
        b = _site_row(loaded, site)
        # Save uses %.2f for SEFD, so allow a small absolute tolerance.
        assert a["sefdr"] == pytest.approx(b["sefdr"], abs=1e-2)
        assert a["sefdl"] == pytest.approx(b["sefdl"], abs=1e-2)


def test_load_txt_known_array_has_eht2017_sites():
    arr = eh.array.load_txt(
        os.path.join(os.path.dirname(__file__), "..", "arrays", "EHT2017.txt"))
    sites = set(str(s) for s in arr.tarr["site"])
    # Pinning the EHT 2017 site list — any change to the canonical file
    # should surface in tests.
    assert sites == {"ALMA", "APEX", "JCMT", "LMT", "PV", "SMA", "SMT", "SPT"}


# ---------------------------------------------------------------------------
# Section 10: Pickle / interop smoke tests
# ---------------------------------------------------------------------------


def test_array_deepcopy_independent(eht_array):
    import copy
    other = copy.deepcopy(eht_array)
    raw_other = np.asarray(other.tarr)
    raw_other["x"][0] += 1.0
    assert np.asarray(eht_array.tarr)["x"][0] != raw_other["x"][0]


def test_array_pickle_roundtrip(eht_array):
    import pickle
    rt = pickle.loads(pickle.dumps(eht_array))
    assert set(rt.tkey.keys()) == set(eht_array.tkey.keys())
    for site in eht_array.tarr["site"]:
        a = _site_row(eht_array, site)
        b = _site_row(rt, site)
        assert a["x"] == b["x"]
        assert a["sefdr"] == b["sefdr"]
