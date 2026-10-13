"""Tests for multi-GPU sharding (ehtim.imaging.sharding).

The sharded objective must equal the single-device jax objective and the numpy
analytic gradient bit-for-bit -- the visibility-axis padding + correction is exact,
not approximate, even when Nvis does not divide the device count. Requires >= 2
local GPUs. The value_and_grad is jitted: eager execution of a sharded graph stalls
on per-op collectives (the optimizer loop jits the whole iteration, as here).
"""
import numpy as np
import pytest

import ehtim as eh
from ehtim.imaging.imager_backend import make_value_and_grad_jax

# `jax` at module level; `gpu` rides on requires_2gpu instead of the module, so the
# CPU-runnable stacking tests below can be collected while `-m gpu` still selects
# exactly the tests that need hardware.
pytestmark = pytest.mark.jax

jax = pytest.importorskip("jax")
pytest.importorskip("optax")

try:
    _N_GPU = len(jax.devices("gpu"))
except RuntimeError:
    _N_GPU = 0
def requires_2gpu(fn):
    """Mark a test as needing >= 2 GPUs: `gpu` for selection, skipif for safety."""
    return pytest.mark.gpu(
        pytest.mark.skipif(_N_GPU < 2, reason="needs >= 2 GPUs")(fn))

VALUE_RTOL = 1e-9
GRAD_RTOL = 1e-9
NXCORR_FLOOR = 0.8
EPSILON_TV = 1e-10


def _nxcorr(a, b):
    a = a - a.mean()
    b = b - b.mean()
    d = np.sqrt(np.sum(a * a) * np.sum(b * b))
    return float(np.sum(a * b) / d) if d > 0 else 0.0


def _build_imager(obs, gauss_im, gauss_prior):
    return eh.imager.Imager(
        obs, gauss_prior, prior_im=gauss_prior, flux=gauss_im.total_flux(),
        data_term={"vis": 1}, reg_term={"simple": 1, "tv": 1},
        ttype="direct", maxit=100, epsilon_tv=EPSILON_TV)


def _backend_args(imgr):
    return (imgr._init_arr, imgr._config, imgr._which_solve, imgr._data_tuples,
            imgr._logfreqratio_list, len(imgr.obslist_next), imgr.dat_term_next,
            imgr.reg_term_next, imgr._prior_arr, imgr.norm_reg, imgr._regparams(),
            imgr._embed_mask)


def _perturbed_x0(imgr):
    rng = np.random.default_rng(4)
    base = np.asarray(imgr._init_vec, dtype=np.float64)
    return base + 0.1 * rng.standard_normal(base.size)


@requires_2gpu
@pytest.mark.parametrize("ttype", ["direct", "nfft"])
@pytest.mark.parametrize("data_term", [{"vis": 1}, {"amp": 1, "cphase": 1, "logcamp": 1}],
                         ids=["vis", "closures"])
def test_baseline_sharded_matches(obs_direct, gauss_im, gauss_prior, ttype, data_term):
    # baseline (visibility-axis) sharding for both transforms and both linear (vis) and
    # closure (cphase/logcamp) terms must match single-device + numpy bit-for-bit. nfft
    # routes its gradient through a custom_vjp; closures rely on the finite-sample padding.
    from ehtim.imaging.sharding import build_mesh, make_sharded_value_and_grad
    # tight nfft_eps so the jax/numpy nfft accuracy gap stays under GRAD_RTOL for the
    # sharded-vs-numpy check (closures amplify it); sharded-vs-single is exact regardless.
    imgr = eh.imager.Imager(
        obs_direct, gauss_prior, prior_im=gauss_prior, flux=gauss_im.total_flux(),
        data_term=data_term, reg_term={"simple": 1, "tv": 1},
        ttype=ttype, maxit=100, epsilon_tv=EPSILON_TV, nfft_eps=1e-12)
    imgr.init_imager()
    x = _perturbed_x0(imgr)

    vg0, _, to0 = make_value_and_grad_jax(*_backend_args(imgr))
    v0, g0 = jax.jit(vg0)(to0(x))
    v0, g0 = float(v0), np.asarray(g0)

    gnp = np.asarray(imgr.objgrad(np.asarray(x)))

    mesh = build_mesh()
    vg1, _, to1, aux1 = make_sharded_value_and_grad(*_backend_args(imgr), mesh=mesh, shard_axis="baseline")
    v1, g1 = jax.jit(vg1)(to1(x), aux1)
    v1, g1 = float(v1), np.asarray(g1)

    # exact: padding contributes zero and the correction restores the normalization
    assert np.allclose(v1, v0, rtol=VALUE_RTOL)
    assert np.linalg.norm(g1 - g0) / np.linalg.norm(g0) < GRAD_RTOL
    assert np.linalg.norm(g1 - gnp) / np.linalg.norm(gnp) < GRAD_RTOL


@requires_2gpu
@pytest.mark.slow
def test_sharded_make_image_recovers(obs_direct, gauss_im, gauss_prior):
    out = _build_imager(obs_direct, gauss_im, gauss_prior).make_image(shard=True, show_updates=False)
    assert _nxcorr(out.imvec, gauss_im.imvec) > NXCORR_FLOOR


# nchan=2 on a 2-device mesh pads to 2, i.e. NOT AT ALL, so it never exercises the
# padding fill or the validity mask. nchan=3 pads to 4 and does. Without the padded
# case, flipping the sigma fill from 1 to 0 turns the sharded objective into NaN with
# the whole suite still green.
@requires_2gpu
@pytest.mark.parametrize("freqs", [(220e9, 240e9), (220e9, 230e9, 240e9)],
                         ids=["nchan2-unpadded", "nchan3-padded"])
def test_frequency_sharded_matches_single_and_numpy(eht_array, gauss_im, freqs):
    # multifrequency: shard the channel axis (channel count padded to the mesh
    # size with a validity mask). Must match single-device + numpy bit-for-bit.
    from ehtim.imaging.sharding import build_mesh, make_sharded_value_and_grad
    im = gauss_im.copy().add_const_mf(1.0, 0)
    prior = im.blur_circ(40 * eh.RADPERUAS)
    obslist = [im.get_image_mf(nu).observe(eht_array, 5, 600, 0, 24, 4e9, ampcal=True,
                                           phasecal=True, ttype="direct", add_th_noise=True, seed=42)
               for nu in freqs]
    imgr = eh.imager.Imager(obslist, prior, prior_im=prior, flux=im.total_flux(),
                            data_term={"vis": 1}, reg_term={"simple": 1, "tv": 1},
                            ttype="direct", pol="I", mf=True, mf_order=1, maxit=100, epsilon_tv=EPSILON_TV)
    imgr.init_imager()
    x = _perturbed_x0(imgr)

    vg0, _, to0 = make_value_and_grad_jax(*_backend_args(imgr))
    v0, g0 = jax.jit(vg0)(to0(x))
    v0, g0 = float(v0), np.asarray(g0)

    gnp = np.asarray(imgr.objgrad(np.asarray(x)))

    mesh = build_mesh()
    vg1, _, to1, aux1 = make_sharded_value_and_grad(*_backend_args(imgr), mesh=mesh, shard_axis="frequency")
    v1, g1 = jax.jit(vg1)(to1(x), aux1)
    v1, g1 = float(v1), np.asarray(g1)

    assert np.allclose(v1, v0, rtol=VALUE_RTOL)
    assert np.linalg.norm(g1 - g0) / np.linalg.norm(g0) < GRAD_RTOL
    assert np.linalg.norm(g1 - gnp) / np.linalg.norm(gnp) < GRAD_RTOL


# ---------------------------------------------------------------------------
# Channel stacking (CPU-runnable: no GPU, no mesh)
#
# The per-channel operators used to be copied into one np.zeros((nf_pad, Nvis,
# Npix)) on the host and then device_put, while the originals were still alive
# on the Imager. Measured at 5 channels padded to 8, 500 vis, 4096 pixels:
# 156 MiB of real operators -> 250 MiB host staging + 250 MiB device, 4.2x.
# ---------------------------------------------------------------------------


def _naive_host_stack(per, nf_pad, fill):
    """The pre-existing build, kept as the value reference."""
    first = np.asarray(per[0])
    out = np.full((nf_pad,) + first.shape, fill, dtype=first.dtype)
    for i, a in enumerate(per):
        out[i] = np.asarray(a)
    return out


@pytest.mark.parametrize("fill", [0, 1])
@pytest.mark.parametrize("n_real,nf_pad", [(3, 4), (5, 8), (4, 4)])
def test_stack_channels_matches_the_host_stack(n_real, nf_pad, fill):
    from jax.sharding import SingleDeviceSharding

    from ehtim.imaging.sharding import stack_channels_on_device
    rng = np.random.default_rng(0)
    per = [rng.standard_normal((7, 5)) for _ in range(n_real)]
    got = stack_channels_on_device(per, nf_pad, SingleDeviceSharding(jax.devices()[0]), fill)
    assert np.array_equal(np.asarray(got), _naive_host_stack(per, nf_pad, fill))


def test_stack_channels_preserves_dtype_and_shape():
    from jax.sharding import SingleDeviceSharding

    from ehtim.imaging.sharding import stack_channels_on_device
    per = [np.ones((3, 2), dtype=np.complex128) for _ in range(2)]
    got = stack_channels_on_device(per, 4, SingleDeviceSharding(jax.devices()[0]), 0)
    assert got.shape == (4, 3, 2)
    assert np.asarray(got).dtype == np.complex128


@pytest.mark.parametrize("ndev,n_real,nf_pad", [(4, 4, 4), (4, 3, 4), (4, 6, 8), (2, 3, 4)])
def test_stack_channels_matches_the_host_stack_across_a_real_mesh(ndev, n_real, nf_pad):
    """Values over a genuinely partitioned mesh, not SingleDeviceSharding.

    SingleDeviceSharding hands the callback the whole axis, so the shard loop
    degenerates and index-arithmetic bugs (an i/j swap, an off-by-one in the
    real-vs-padded test) cannot show up. Forcing CPU devices exercises the real
    branch without needing a GPU.
    """
    import subprocess
    import sys
    import textwrap
    code = textwrap.dedent(f"""
        import os
        os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count={ndev}"
        import numpy as np, jax
        jax.config.update("jax_enable_x64", True)
        from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
        from ehtim.imaging.sharding import stack_channels_on_device
        rng = np.random.default_rng(0)
        per = [rng.standard_normal((3, 5)) + 1j*rng.standard_normal((3, 5))
               for _ in range({n_real})]
        mesh = Mesh(np.array(jax.devices("cpu")[:{ndev}]), ("shard",))
        sh = NamedSharding(mesh, P("shard", None, None))
        for fill in (0, 1):
            got = np.asarray(stack_channels_on_device(per, {nf_pad}, sh, fill))
            ref = np.full(({nf_pad}, 3, 5), fill, dtype=per[0].dtype)
            for i, a in enumerate(per):
                ref[i] = a
            assert np.array_equal(got, ref), f"mismatch at fill={{fill}}"
        print("RESULT ok", len(jax.devices("cpu")))
    """)
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    line = [x for x in r.stdout.splitlines() if x.startswith("RESULT")]
    assert line, f"subprocess failed:\n{r.stderr[-1200:]}"
    assert int(line[0].split()[-1]) >= ndev


@requires_2gpu
def test_stack_channels_does_not_spike_one_device():
    """The saving is on device, not host.

    The previous build did jnp.asarray(padded_stack) before applying the
    sharding, which lands the whole stack on the default device and then
    reshards: device 0 peaks at several times its share, and the factor grows
    with the device count. Host cost is unchanged either way, because jax
    materializes every addressable shard before transferring -- an earlier
    version of this test asserted a host saving and only passed because its
    subprocess ran in complex64.
    """
    import jax.numpy as jnp
    from jax.sharding import Mesh, NamedSharding
    from jax.sharding import PartitionSpec as P

    from ehtim.imaging.sharding import stack_channels_on_device
    devs = jax.devices("gpu")[:2]
    mesh = Mesh(np.array(devs), ("shard",))
    sh = NamedSharding(mesh, P("shard", None, None))
    n, nvis, npix = 4, 1500, 4096
    per = [np.ones((nvis, npix), dtype=np.complex128) for _ in range(n)]
    share = n * nvis * npix * 16 / len(devs)

    def peak_after(build):
        for d in devs:
            d.memory_stats()  # touch before measuring
        out = build()
        out.block_until_ready()
        pk = devs[0].memory_stats()["peak_bytes_in_use"]
        del out
        return pk

    new = peak_after(lambda: stack_channels_on_device(per, n, sh, 0))
    naive_stack = np.stack(per)
    old = peak_after(lambda: jax.device_put(jnp.asarray(naive_stack), sh))
    assert old > 1.5 * share, (
        f"expected the naive build to overshoot device 0's {share/2**20:.0f} MiB "
        f"share; got {old/2**20:.0f} MiB -- the premise no longer holds")
    assert new < 0.7 * old, (
        f"device 0 peak: naive {old/2**20:.0f} MiB, sharded build "
        f"{new/2**20:.0f} MiB ({new/old:.2f}x)")
