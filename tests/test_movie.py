"""Tests for ehtim.movie.Movie.

Focused on Phase 3 mixed-pol additions:
- polrep='lin' construction and round-trips
- harmonized cross-polrep frame accessors (computed on demand for any source)
- xxframes/yyframes/xyframes/yxframes accessors
- rlvec/lrvec deprecation shims (renamed to rlframes/lrframes)
- setter polrep guards remain (only the storage basis is writable)
"""

import numpy as np
import pytest

import ehtim as eh
from ehtim.observing import pol_conventions as pc


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_N = 8
_T = 4


def _gaussian_2d(scale=1.0):
    coords = np.arange(_N * _N)
    return scale * np.exp(-((coords - _N * _N / 2.0) ** 2) / 100.0).reshape(_N, _N)


@pytest.fixture
def times():
    return np.linspace(0.0, 1.0, _T)


@pytest.fixture
def frames_I():
    return [_gaussian_2d(1 + 0.1 * t) for t in range(_T)]


@pytest.fixture
def stokes_movie_pol(times, frames_I):
    """Stokes movie with all four pols populated."""
    mov = eh.movie.Movie(frames_I, times, 1e-10, 0.0, 0.0, polrep="stokes")
    mov.add_pol_movie([0.10 * f for f in frames_I], "Q")
    mov.add_pol_movie([0.05 * f for f in frames_I], "U")
    mov.add_pol_movie([0.02 * f for f in frames_I], "V")
    return mov


@pytest.fixture(autouse=True)
def _reset_warning():
    pc._reset_convention_warning()
    yield


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_init_stokes_default(times, frames_I):
    mov = eh.movie.Movie(frames_I, times, 1e-10, 0.0, 0.0)
    assert mov.polrep == "stokes"
    assert mov.pol_prim == "I"


def test_init_lin_default_pol_prim_is_xx(times, frames_I):
    mov = eh.movie.Movie(frames_I, times, 1e-10, 0.0, 0.0, polrep="lin")
    assert mov.polrep == "lin"
    assert mov.pol_prim == "XX"


def test_init_lin_pol_prim_yy(times, frames_I):
    mov = eh.movie.Movie(frames_I, times, 1e-10, 0.0, 0.0,
                         polrep="lin", pol_prim="YY")
    assert mov.pol_prim == "YY"


def test_init_lin_rejects_bad_pol_prim(times, frames_I):
    with pytest.raises(Exception, match="pol_prim"):
        eh.movie.Movie(frames_I, times, 1e-10, 0.0, 0.0,
                       polrep="lin", pol_prim="XY")


def test_init_unknown_polrep_raises(times, frames_I):
    with pytest.raises(Exception, match="polrep"):
        eh.movie.Movie(frames_I, times, 1e-10, 0.0, 0.0, polrep="bogus")


# ---------------------------------------------------------------------------
# Harmonized cross-polrep frame accessors (the big behavior change)
# ---------------------------------------------------------------------------


def test_stokes_movie_rrframes_computes_from_iv(stokes_movie_pol):
    """Pre-Phase-3 this would raise; now it computes I+V."""
    rr = stokes_movie_pol.rrframes
    expected = stokes_movie_pol.iframes + stokes_movie_pol.vframes
    np.testing.assert_allclose(rr, expected, atol=1e-12)


def test_stokes_movie_xxframes_computes_from_iq(stokes_movie_pol):
    xx = stokes_movie_pol.xxframes
    expected = stokes_movie_pol.iframes + stokes_movie_pol.qframes
    np.testing.assert_allclose(xx, expected, atol=1e-12)


def test_stokes_movie_xyframes_complex_from_uv(stokes_movie_pol):
    xy = stokes_movie_pol.xyframes
    expected = stokes_movie_pol.uframes - 1j * stokes_movie_pol.vframes
    np.testing.assert_allclose(xy, expected, atol=1e-12)


def test_circ_movie_iframes_computes(stokes_movie_pol):
    mov_c = stokes_movie_pol.switch_polrep("circ")
    np.testing.assert_allclose(mov_c.iframes,
                               0.5 * (mov_c.rrframes + mov_c.llframes),
                               atol=1e-12)


def test_lin_movie_qframes_computes(stokes_movie_pol):
    mov_l = stokes_movie_pol.switch_polrep("lin")
    np.testing.assert_allclose(mov_l.qframes,
                               0.5 * (mov_l.xxframes - mov_l.yyframes),
                               atol=1e-12)


def test_cross_polrep_read_does_not_populate_fundict(stokes_movie_pol):
    """Cross-polrep reads compute on demand; they don't materialize _fundict."""
    _ = stokes_movie_pol.rrframes  # cross-polrep computation
    assert stokes_movie_pol._fundict["I"] is not None  # storage interp lives
    # circ-side _fundict keys don't even exist on a stokes movie
    assert "RR" not in stokes_movie_pol._fundict


# ---------------------------------------------------------------------------
# Setters retain their polrep guard
# ---------------------------------------------------------------------------


def test_iframes_setter_rejects_non_stokes(stokes_movie_pol):
    mov_c = stokes_movie_pol.switch_polrep("circ")
    with pytest.raises(Exception, match="stokes"):
        mov_c.iframes = mov_c._movdict["RR"]


def test_xxframes_setter_rejects_non_lin(stokes_movie_pol):
    with pytest.raises(Exception, match="lin"):
        stokes_movie_pol.xxframes = stokes_movie_pol._movdict["I"]


def test_rrframes_setter_rejects_non_circ(stokes_movie_pol):
    with pytest.raises(Exception, match="circ"):
        stokes_movie_pol.rrframes = stokes_movie_pol._movdict["I"]


# ---------------------------------------------------------------------------
# switch_polrep round-trips through all three polreps
# ---------------------------------------------------------------------------


def test_stokes_to_circ_to_stokes_roundtrip(stokes_movie_pol):
    rt = stokes_movie_pol.switch_polrep("circ").switch_polrep("stokes")
    for attr in ("iframes", "qframes", "uframes", "vframes"):
        np.testing.assert_allclose(getattr(rt, attr),
                                   getattr(stokes_movie_pol, attr), atol=1e-12)


def test_stokes_to_lin_to_stokes_roundtrip(stokes_movie_pol):
    rt = stokes_movie_pol.switch_polrep("lin").switch_polrep("stokes")
    for attr in ("iframes", "qframes", "uframes", "vframes"):
        np.testing.assert_allclose(getattr(rt, attr),
                                   getattr(stokes_movie_pol, attr), atol=1e-12)


def test_circ_to_lin_via_stokes_composition(stokes_movie_pol):
    mov_c = stokes_movie_pol.switch_polrep("circ")
    direct = mov_c.switch_polrep("lin")
    two_step = mov_c.switch_polrep("stokes").switch_polrep("lin")
    for attr in ("xxframes", "yyframes", "xyframes", "yxframes"):
        np.testing.assert_allclose(getattr(direct, attr),
                                   getattr(two_step, attr), atol=1e-12)


def test_lin_movie_xxframes_consistent_with_circ_movie(stokes_movie_pol):
    """Cross-polrep accessors give the same physics from circ and lin sources."""
    mov_c = stokes_movie_pol.switch_polrep("circ")
    mov_l = stokes_movie_pol.switch_polrep("lin")
    np.testing.assert_allclose(mov_c.xxframes, mov_l.xxframes, atol=1e-12)


# ---------------------------------------------------------------------------
# rlvec / lrvec deprecation shims
# ---------------------------------------------------------------------------


def test_rlvec_getter_raises_with_migration_message(stokes_movie_pol):
    mov_c = stokes_movie_pol.switch_polrep("circ")
    with pytest.raises(AttributeError, match="rlframes"):
        _ = mov_c.rlvec


def test_rlvec_setter_raises_with_migration_message(stokes_movie_pol):
    mov_c = stokes_movie_pol.switch_polrep("circ")
    with pytest.raises(AttributeError, match="rlframes"):
        mov_c.rlvec = []


def test_lrvec_getter_raises_with_migration_message(stokes_movie_pol):
    mov_c = stokes_movie_pol.switch_polrep("circ")
    with pytest.raises(AttributeError, match="lrframes"):
        _ = mov_c.lrvec


def test_rlframes_works_after_rename(stokes_movie_pol):
    """rlframes is the new canonical name (was rlvec)."""
    mov_c = stokes_movie_pol.switch_polrep("circ")
    rl = mov_c.rlframes
    expected = mov_c.qframes + 1j * mov_c.uframes
    np.testing.assert_allclose(rl, expected, atol=1e-12)


# ---------------------------------------------------------------------------
# add_pol_movie for lin pols
# ---------------------------------------------------------------------------


def test_add_pol_movie_lin_yy(times, frames_I):
    mov = eh.movie.Movie(frames_I, times, 1e-10, 0.0, 0.0, polrep="lin")
    yy_frames = [0.5 * f for f in frames_I]
    mov.add_pol_movie(yy_frames, "YY")
    np.testing.assert_allclose(
        mov.yyframes,
        np.array([f.flatten() for f in yy_frames]),
        atol=1e-12,
    )


def test_add_pol_movie_lin_xy_complex(times, frames_I):
    mov = eh.movie.Movie(frames_I, times, 1e-10, 0.0, 0.0, polrep="lin")
    xy_frames = [0.3 * f * 1j for f in frames_I]
    mov.add_pol_movie(xy_frames, "XY")
    np.testing.assert_allclose(
        mov.xyframes,
        np.array([f.flatten() for f in xy_frames]),
        atol=1e-12,
    )
