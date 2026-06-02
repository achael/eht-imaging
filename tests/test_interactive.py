"""Smoke tests for ehtim.plotting.interactive."""

import os

import numpy as np
import pytest

import ehtim as eh

# Module-level guard - skip the whole file when plotly is missing.
go = pytest.importorskip("plotly.graph_objects")

from ehtim.plotting import interactive  # noqa: E402  (importorskip above)

# The dashboard smoke test runs self_cal; tag the whole file as slow so
# `pytest -m "not slow"` skips it (consistent with the cal-cluster tests).
pytestmark = pytest.mark.slow

_ROOT = os.path.join(os.path.dirname(__file__), "..")
SAMPLE_UVFITS = os.path.join(_ROOT, "data", "sample.uvfits")
SAMPLE_IMAGE = os.path.join(_ROOT, "models", "avery_sgra_eofn.txt")

POL_FRAC_Q = 0.10
POL_FRAC_U = 0.05


@pytest.fixture(scope="module")
def obs():
    return eh.obsdata.load_uvfits(SAMPLE_UVFITS)


@pytest.fixture(scope="module")
def im():
    return eh.image.load_txt(SAMPLE_IMAGE)


@pytest.fixture(scope="module")
def im_pol(im):
    out = im.copy()
    I2d = im.imvec.reshape(im.ydim, im.xdim)
    out.add_qu(POL_FRAC_Q * I2d, POL_FRAC_U * I2d)
    return out


def test_plot_bl_returns_figure(obs):
    fig = interactive.plot_bl(obs, "ALMA", "SMT", "amp")
    assert isinstance(fig, go.Figure)


def test_plotall_returns_figure(obs):
    fig = interactive.plotall(obs, "uvdist", "amp")
    assert isinstance(fig, go.Figure)


def test_plotall_uv_coverage_is_square(obs):
    """u-vs-v gets symmetric range, scaleanchor lock, and square canvas."""
    fig = interactive.plotall(obs, "u", "v")
    assert fig.layout.xaxis.scaleanchor == "y"
    assert fig.layout.xaxis.scaleratio == 1.0
    # Symmetric range driven by max(|u|,|v|).
    rx, ry = fig.layout.xaxis.range, fig.layout.yaxis.range
    assert rx is not None and ry is not None
    np.testing.assert_allclose(rx, ry)
    np.testing.assert_allclose(rx[0], -rx[1])


def test_plotall_labels_have_no_latex_dollars(obs):
    """Plotly axis titles should not contain $-wrapped LaTeX (matplotlib-only)."""
    fig = interactive.plotall(obs, "uvdist", "sigma")
    assert "$" not in fig.layout.xaxis.title.text
    assert "$" not in fig.layout.yaxis.title.text


def test_dashboard_returns_figure_with_pol_toggle(obs, im_pol):
    """Dashboard builds, contains a pol-toggle button + intensity colorbar."""
    from ehtim.calibrating.self_cal import self_cal
    caltable = self_cal(
        obs, im_pol, method="both", caltable=True,
        processes=2, show_solution=False,
    )
    fig = interactive.dashboard(im_pol, obs, caltable, show=False)
    assert isinstance(fig, go.Figure)
    button_labels = [b.label for um in fig.layout.updatemenus for b in (um.buttons or [])]
    assert any("Toggle polarization" in s for s in button_labels)


def test_uv_area_triangle_quadrangle():
    """Sanity-check the new obs_helpers uv-area utilities."""
    from ehtim.observing.obs_helpers import uv_area_quadrangle, uv_area_triangle
    # Unit right-triangle area.
    assert uv_area_triangle(1, 0, 0, 1) == pytest.approx(0.5)
    # Unit square area via three corners + origin.
    assert uv_area_quadrangle(1, 0, 1, 1, 0, 1) == pytest.approx(1.0)


def test_plain_fig_write_html_carries_toolbar_js(obs, tmp_path):
    """The instance-patched fig.write_html should inject the toolbar JS even
    when the user calls plotly's native method directly (not the wrapper)."""
    fig = interactive.plotall(obs, "uvdist", "amp")
    path = tmp_path / "plain.html"
    fig.write_html(str(path), include_plotlyjs="cdn")
    body = path.read_text()
    assert "initToolbar" in body
    assert "Color all" in body
    assert "data-ehtim-toolbar-for" in body
