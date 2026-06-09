"""Smoke tests for ehtim.model geometric models.

Builds every mring variant (each stores complex beta-coefficient arrays) and
checks the model-parameter dispatch, including the unrecognized-type branch.
"""

import numpy as np
import pytest

import ehtim as eh

RADPERUAS = eh.RADPERUAS

MRING_VARIANTS = [
    ("add_mring", {}),
    ("add_stretched_mring", {"stretch": 1.2, "stretch_PA": 0.3}),
    ("add_thick_mring", {"alpha": 10 * RADPERUAS}),
    ("add_thick_mring_floor", {"alpha": 10 * RADPERUAS, "ff": 0.1}),
    ("add_thick_mring_Gfloor", {"alpha": 10 * RADPERUAS, "ff": 0.1, "FWHM": 40 * RADPERUAS}),
    ("add_stretched_thick_mring", {"alpha": 10 * RADPERUAS, "stretch": 1.2, "stretch_PA": 0.3}),
    ("add_stretched_thick_mring_floor",
     {"alpha": 10 * RADPERUAS, "ff": 0.1, "stretch": 1.2, "stretch_PA": 0.3}),
]


@pytest.mark.parametrize("builder,kwargs", MRING_VARIANTS)
def test_mring_variant_builds_and_images(builder, kwargs):
    mod = eh.model.Model()
    mod = getattr(mod, builder)(F0=1.5, d=40 * RADPERUAS,
                                beta_list=[0.0, 0.2 + 0.1j], **kwargs)
    im = mod.make_image(120 * RADPERUAS, 32)
    assert np.all(np.isfinite(im.imvec))
    assert im.total_flux() > 0


def test_model_params_unknown_returns_empty():
    # the dispatch fall-through branch (unrecognized model type)
    assert eh.model.model_params("not_a_real_model") == []


def test_model_params_mring_includes_beta_terms():
    params = eh.model.model_params("mring", {"beta_list": [0.0, 0.1j]})
    assert "F0" in params
    assert "d" in params
    assert any(p.startswith("beta1") for p in params)
