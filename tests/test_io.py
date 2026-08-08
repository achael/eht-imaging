"""Tests for ehtim I/O functions."""

import glob
import os

import numpy as np
import pytest
from astropy.io import fits

import ehtim as eh
from ehtim.io import load

_ROOT = os.path.join(os.path.dirname(__file__), "..")
DATA_DIR = os.path.join(_ROOT, "data")
ARRAY_DIR = os.path.join(_ROOT, "arrays")
MODEL_DIR = os.path.join(_ROOT, "models")

UVFITS_FILES = sorted(glob.glob(os.path.join(DATA_DIR, "*.uvfits")))


def test_load_obs_uvfits():
    """Test that load_obs_uvfits can read a UVFITS file and return an Obsdata."""
    obs = load.load_obs_uvfits(os.path.join(DATA_DIR, "sample.uvfits"))
    assert isinstance(obs, eh.obsdata.Obsdata)


@pytest.mark.parametrize("path", UVFITS_FILES,
                         ids=[os.path.basename(p) for p in UVFITS_FILES])
@pytest.mark.parametrize("polrep", ["stokes", "circ"])
def test_load_obs_uvfits_speedups_matches_default(path, polrep):
    """The vectorized `speedups` paths in load_obs_uvfits (vectorized site lookup,
    alternate datatable assembly) must produce data and tarr identical to the
    legacy paths, across every uvfits file in the data directory and both
    polreps. This is the verification the 'speedups' default relies on."""
    obs_legacy = load.load_obs_uvfits(path, polrep=polrep, speedups=False)
    obs_fast = load.load_obs_uvfits(path, polrep=polrep, speedups=True)
    np.testing.assert_array_equal(obs_legacy.data, obs_fast.data)
    np.testing.assert_array_equal(obs_legacy.tarr, obs_fast.tarr)


def test_load_array_txt():
    """Test that load_array_txt can read an array file and return an Array."""
    arr = eh.array.load_txt(os.path.join(ARRAY_DIR, "EHT2017.txt"))
    assert isinstance(arr, eh.array.Array)


def test_load_im_txt():
    """Test that load_im_txt can read a text image and return an Image."""
    im = eh.image.load_txt(os.path.join(MODEL_DIR, "avery_sgra_eofn.txt"))
    assert isinstance(im, eh.image.Image)


def test_fill_nan_sigmas_order_dependent():
    """_fill_nan_sigmas fills rr from ll first, then ll/rl/lr from the filled rr."""
    nan = np.nan
    rr = np.array([nan, 2.0, 1.0, nan])
    ll = np.array([5.0, nan, 1.0, nan])
    rl = np.array([nan, nan, 1.0, 3.0])
    lr = np.array([9.0, nan, 1.0, nan])
    rrf, llf, rlf, lrf = load._fill_nan_sigmas(rr, ll, rl, lr)
    # row0: rr<-ll=5, ll=5, rl<-rr_filled=5, lr=9
    # row1: rr=2, ll<-rr=2, rl<-rr=2, lr<-rr=2
    # row2: all 1 (untouched)
    # row3: rr,ll both nan -> rr_filled=nan, ll=nan, rl=3 (set), lr=nan
    np.testing.assert_array_equal(rrf, [5.0, 2.0, 1.0, nan])
    np.testing.assert_array_equal(llf, [5.0, 2.0, 1.0, nan])
    np.testing.assert_array_equal(rlf, [5.0, 2.0, 1.0, 3.0])
    np.testing.assert_array_equal(lrf, [9.0, 2.0, 1.0, nan])
    # inputs not mutated
    np.testing.assert_array_equal(rr, [nan, 2.0, 1.0, nan])


# ---------------------------------------------------------------------------
# Non-averaging load path: load_uvfits(average_if=..., average_channel=...)
#
# The default averages all channels/IFs into one Obsdata; setting either flag
# False keeps that axis resolved and returns a list of Obsdata. Real multi-channel
# data here is synthetic -- a saved single-channel obs with its FREQ axis tiled --
# so these run in CI; two guarded tests use the real HL Tau files when present.
# ---------------------------------------------------------------------------

# Large reference dataset, not in the repo. Point EHTIM_HLTAU_DIR at a local copy to
# run these; they skip otherwise, so a fresh clone is green without it.
_HLTAU = os.environ.get("EHTIM_HLTAU_DIR", "")
_HLTAU_SINGLE = os.path.join(_HLTAU, "hltau_mf_223.8.uvfits")
_HLTAU_LINEAR = os.path.join(_HLTAU, "ALMA_data/Band7/HLTau_Band7_spw02_polswap.uvfits")

_FACTORS = [1.0, 2.0, 0.5]
_CDELT = 2.0e8


def _expand_channels(inpath, outpath, factors=_FACTORS, cdelt=_CDELT, flag_channel=None):
    """Tile a single-channel uvfits's FREQ axis into len(factors) channels.

    Channel c's visibilities (not weights) are scaled by factors[c] so each channel
    is distinguishable, and CDELT4 is set to `cdelt` to give the channels distinct
    frequencies. If flag_channel is given, that channel's weights are zeroed.
    """
    h = fits.open(inpath)
    gd = h[0].data
    dat = np.asarray(gd["DATA"])
    new = np.repeat(dat, len(factors), axis=4)                 # tile the FREQ axis
    for c, f in enumerate(factors):
        new[:, :, :, :, c, :, 0:2] *= f                        # scale re + im only
    if flag_channel is not None:
        new[:, :, :, :, flag_channel, :, 2] = 0.0              # zero that channel's weights
    pars = list(gd.parnames)
    pardata = [gd.par(i) for i in range(len(pars))]
    h[0].data = fits.GroupData(new.astype("float32"), parnames=pars, pardata=pardata, bitpix=-32)
    h[0].header["CDELT4"] = cdelt
    h.writeto(outpath, overwrite=True)
    h.close()


def _set_linear_feeds(inpath, outpath):
    """Rewrite the antenna table's POLTYA/POLTYB feed columns to linear (X/Y)."""
    h = fits.open(inpath)
    an = h["AIPS AN"]
    for col, val in (("POLTYA", "X"), ("POLTYB", "Y")):
        if col in an.data.columns.names:
            an.data[col] = np.array([val] * len(an.data))
    h.writeto(outpath, overwrite=True)
    h.close()


def _keyed(obs):
    """Return obs.data sorted by (time, t1, t2) so two loads line up row-for-row."""
    d = obs.data
    return d[np.lexsort((d["t2"], d["t1"], d["time"]))]


@pytest.fixture
def single_uvfits(obs_direct, tmp_path):
    """A single-channel circular uvfits saved from the noise-free Gaussian obs."""
    path = str(tmp_path / "single.uvfits")
    obs_direct.save_uvfits(path)
    return path


@pytest.fixture
def multichan_uvfits(single_uvfits, tmp_path):
    """The single-channel file expanded to 3 channels scaled by _FACTORS."""
    path = str(tmp_path / "multi.uvfits")
    _expand_channels(single_uvfits, path)
    return path


def test_load_uvfits_default_returns_single_obsdata(single_uvfits):
    """Default (average everything) still returns one Obsdata, not a list."""
    assert isinstance(eh.obsdata.load_uvfits(single_uvfits), eh.obsdata.Obsdata)


def test_load_uvfits_average_channel_false_returns_list(single_uvfits):
    """Keeping the channel axis returns a list, even for a single channel."""
    out = eh.obsdata.load_uvfits(single_uvfits, average_channel=False)
    assert isinstance(out, list) and len(out) == 1
    assert isinstance(out[0], eh.obsdata.Obsdata)


def test_load_uvfits_single_channel_spectral_matches_average(single_uvfits):
    """On a single-channel file the resolved and averaged loads read the same data."""
    ref = eh.obsdata.load_uvfits(single_uvfits, polrep="circ")
    spec = eh.obsdata.load_uvfits(single_uvfits, polrep="circ", average_channel=False)[0]
    a, b = _keyed(ref), _keyed(spec)
    assert len(a) == len(b)
    np.testing.assert_allclose(a["rrvis"], b["rrvis"], rtol=1e-5, atol=1e-6, equal_nan=True)
    np.testing.assert_allclose(a["rrsigma"], b["rrsigma"], rtol=1e-5, atol=1e-8, equal_nan=True)
    # rf and u,v also agree (the FREQ-axis CRVAL4 equals the AN-table FREQ on a well-formed file)
    assert np.isclose(ref.rf, spec.rf, rtol=1e-12)
    np.testing.assert_allclose(a["u"], b["u"], rtol=1e-6)
    np.testing.assert_allclose(a["v"], b["v"], rtol=1e-6)


def test_load_uvfits_channel_selection(multichan_uvfits):
    """channel=[...] with average_channel=False loads only the requested channels."""
    obslist = eh.obsdata.load_uvfits(multichan_uvfits, polrep="circ",
                                     channel=[0, 2], average_if=False, average_channel=False)
    assert len(obslist) == 2
    crval4 = fits.open(multichan_uvfits)[0].header["CRVAL4"]
    rfs = sorted(o.rf for o in obslist)
    np.testing.assert_allclose(rfs, [crval4, crval4 + 2 * _CDELT], rtol=1e-12)


def test_load_uvfits_multichannel_split(single_uvfits, multichan_uvfits):
    """Both axes resolved -> one Obsdata per channel, with per-channel freq and u,v."""
    obslist = eh.obsdata.load_uvfits(multichan_uvfits, polrep="circ",
                                     average_if=False, average_channel=False)
    assert len(obslist) == len(_FACTORS)

    crval4 = fits.open(multichan_uvfits)[0].header["CRVAL4"]
    rfs = np.array([o.rf for o in obslist])
    np.testing.assert_allclose(rfs, crval4 + np.arange(len(_FACTORS)) * _CDELT, rtol=1e-12)

    base_obs = eh.obsdata.load_uvfits(single_uvfits, polrep="circ")
    base = _keyed(base_obs)
    for c, f in enumerate(_FACTORS):
        ch = _keyed(obslist[c])
        # channel c's visibilities are the base scaled by factors[c]
        np.testing.assert_allclose(ch["rrvis"], f * base["rrvis"], rtol=1e-5, atol=1e-6,
                                   equal_nan=True)
        # u,v are scaled by that channel's frequency, so u/rf recovers the shared UU
        np.testing.assert_allclose(ch["u"] / obslist[c].rf, base["u"] / base_obs.rf, rtol=1e-6)

    # the per-channel u,v really differ (the frequency scaling was applied)
    assert not np.allclose(_keyed(obslist[0])["u"], _keyed(obslist[-1])["u"])


def test_load_uvfits_multichannel_default_averages(single_uvfits, multichan_uvfits):
    """Default load of the multichannel file is the inverse-variance channel average."""
    avg = eh.obsdata.load_uvfits(multichan_uvfits, polrep="circ")
    assert isinstance(avg, eh.obsdata.Obsdata)
    base = _keyed(eh.obsdata.load_uvfits(single_uvfits, polrep="circ"))
    # equal per-channel weights, so the average is the mean of the channel scalings
    expected = (sum(_FACTORS) / len(_FACTORS)) * base["rrvis"]
    np.testing.assert_allclose(_keyed(avg)["rrvis"], expected, rtol=1e-4, atol=1e-6,
                               equal_nan=True)


def test_load_uvfits_average_if_false_single_if_averages_channels(multichan_uvfits):
    """With one IF, average_if=False still averages the channels -> a one-element list."""
    both = eh.obsdata.load_uvfits(multichan_uvfits, polrep="circ")            # both averaged
    out = eh.obsdata.load_uvfits(multichan_uvfits, polrep="circ", average_if=False)
    assert isinstance(out, list) and len(out) == 1
    np.testing.assert_allclose(_keyed(out[0])["rrvis"], _keyed(both)["rrvis"],
                               rtol=1e-6, equal_nan=True)


def test_load_uvfits_empty_channel_skipped(single_uvfits, tmp_path):
    """A fully-flagged channel is dropped from the returned list."""
    path = str(tmp_path / "flagged.uvfits")
    _expand_channels(single_uvfits, path, factors=[1.0, 1.0, 1.0], flag_channel=1)
    obslist = eh.obsdata.load_uvfits(path, polrep="circ",
                                     average_if=False, average_channel=False)
    assert len(obslist) == 2


def test_load_uvfits_linear_feed_raises(single_uvfits, tmp_path):
    """Linear/mixed feeds are out of scope for this loader and must raise."""
    path = str(tmp_path / "linear.uvfits")
    _set_linear_feeds(single_uvfits, path)
    with pytest.raises(NotImplementedError):
        eh.obsdata.load_uvfits(path, average_channel=False)


def test_load_uvfits_trial_speedups_kwarg_removed(single_uvfits):
    """The old experimental `trial_speedups` name is gone (renamed to `speedups`)."""
    with pytest.raises(TypeError):
        eh.obsdata.load_uvfits(single_uvfits, trial_speedups=True)


@pytest.mark.skipif(not os.path.exists(_HLTAU_SINGLE), reason="HL Tau data not present")
def test_load_uvfits_real_hltau_single_channel():
    """On a real single-channel HL Tau file the resolved load matches the averaged load."""
    ref = eh.obsdata.load_uvfits(_HLTAU_SINGLE, polrep="circ")
    spec = eh.obsdata.load_uvfits(_HLTAU_SINGLE, polrep="circ", average_channel=False)
    assert len(spec) == 1
    a, b = _keyed(ref), _keyed(spec[0])
    np.testing.assert_allclose(a["rrvis"], b["rrvis"], rtol=1e-5, atol=1e-6, equal_nan=True)


@pytest.mark.skipif(not os.path.exists(_HLTAU_LINEAR), reason="HL Tau linear data not present")
def test_load_uvfits_real_linear_feed_raises():
    """The real linear-feed ALMA file raises (confirms the feed guard on genuine data)."""
    with pytest.raises(NotImplementedError):
        eh.obsdata.load_uvfits(_HLTAU_LINEAR, average_channel=False)
