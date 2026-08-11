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
    """fit_gains=True must build a caltable whose rows match the DTCAL dtype.

    Regression for the modeling gain write-site, which builds its rows as
    tuple literals: the literal's width and the dtype's field count have to
    agree, so the write-site breaks whenever DTCAL gains or loses a column.
    The assertion compares against np.dtype(ehc.DTCAL).names rather than a
    fixed field count, so it holds on either side of such a change.
    """
    res = eh.modeler_func(model_obs, gauss_model, gauss_model.default_prior(),
                          d1='amp', fit_model=False, fit_gains=True, quiet=True)

    ct = res['caltable']
    assert isinstance(ct, eh.caltable.Caltable)

    # every per-site gain table must carry the current DTCAL fields
    assert len(ct.data) > 0
    for site, rows in ct.data.items():
        assert rows.dtype.names == np.dtype(ehc.DTCAL).names
        assert len(rows) > 0
