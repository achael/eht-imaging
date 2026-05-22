"""Tests for ehtim I/O functions."""

import os

import numpy as np

import ehtim as eh
from ehtim.io import load

_ROOT = os.path.join(os.path.dirname(__file__), "..")
DATA_DIR = os.path.join(_ROOT, "data")
ARRAY_DIR = os.path.join(_ROOT, "arrays")
MODEL_DIR = os.path.join(_ROOT, "models")


def test_load_obs_uvfits():
    """Test that load_obs_uvfits can read a UVFITS file and return an Obsdata."""
    obs = load.load_obs_uvfits(os.path.join(DATA_DIR, "sample.uvfits"))
    assert isinstance(obs, eh.obsdata.Obsdata)


def test_load_obs_uvfits_trial_speedups_matches_default():
    """The trial_speedups paths in load_obs_uvfits (vectorized site lookup,
    alternate datatable assembly) must produce data and tarr identical to the
    default paths."""
    path = os.path.join(DATA_DIR, "sample.uvfits")
    obs_default = load.load_obs_uvfits(path, trial_speedups=False)
    obs_trial = load.load_obs_uvfits(path, trial_speedups=True)
    np.testing.assert_array_equal(obs_default.data, obs_trial.data)
    np.testing.assert_array_equal(obs_default.tarr, obs_trial.tarr)


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
