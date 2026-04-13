"""Tests for ehtim I/O functions."""

import os

import ehtim as eh
from ehtim.io import load

_ROOT = os.path.join(os.path.dirname(__file__), "..")
DATA_DIR = os.path.join(_ROOT, "data")


def test_load_obs_uvfits():
    """Test that load_obs_uvfits can read a UVFITS file and return an Obsdata."""
    obs = load.load_obs_uvfits(os.path.join(DATA_DIR, "sample.uvfits"))
    assert isinstance(obs, eh.obsdata.Obsdata)
