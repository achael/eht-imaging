"""Tests for ehtim.modeling.modeling_utils.modeler_func."""

import numpy as np
import pytest

import ehtim as eh
import ehtim.const_def as ehc


@pytest.fixture(scope="module")
def gauss_model():
    mod = eh.model.Model()
    mod = mod.add_circ_gauss(1.0, 40.0 * eh.RADPERUAS)
    return mod


@pytest.fixture(scope="module")
def model_obs(gauss_model, eht_array):
    return gauss_model.observe(eht_array, 5, 600, 0, 4, 1e9,
                               ampcal=False, phasecal=True, seed=42)


def test_modeler_func_fit_gains_caltable(gauss_model, model_obs):
    """fit_gains=True must build a valid 5-field DTCAL caltable.

    Regression for the DTCAL write-site that appended a 3-tuple gain row,
    which fails to cast against the widened (5-field) DTCAL dtype.
    """
    res = eh.modeler_func(model_obs, gauss_model, gauss_model.default_prior(),
                          d1='amp', fit_model=False, fit_gains=True, quiet=True)

    ct = res['caltable']
    assert isinstance(ct, eh.caltable.Caltable)

    # every per-site gain table must carry the full 5-field dtype
    assert len(ct.data) > 0
    for site, rows in ct.data.items():
        assert rows.dtype.names == np.dtype(ehc.DTCAL).names
        assert len(rows) > 0
