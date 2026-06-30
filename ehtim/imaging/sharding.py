"""Multi-GPU sharding for the imaging objective.

``make_sharded_value_and_grad`` returns a device value_and_grad whose data products are
split across a mesh of GPUs while the image and regularizers stay replicated -- only the
data-fidelity (chi^2) term is distributed. Two axes are supported:

- ``shard_axis="baseline"``: split each data term over its visibility/baseline axis. Good
  for a single large image or many visibilities.
- ``shard_axis="frequency"``: give each GPU a subset of frequency channels. Good for
  multifrequency data; needs more than one channel.

jax requires the sharded axis to divide evenly across the mesh, so each term is padded to a
multiple of the device count with zero-weight rows: the operator is padded with ones (keeping
the padded sample finite, so closures don't hit log(0)/angle(0)) and sigma with infinity (so
the row adds exactly 0 to chi^2). Padding still inflates the 1/len(data) normalization, so each
chi^2 is multiplied by pad_len/true_len -- the sharded objective is then bit-for-bit the
single-device one, for linear (vis/amp) and closure (cphase/logcamp) terms alike.

The ``direct`` transform shards its dense Fourier matrix and reduces with shard_map + pmean.
The ``nfft`` transform can't be differentiated through shard_map (jax_finufft's nufft2
transpose is wrong under SPMD), so its grid->samples map is a custom_vjp with a forward-nufft1
backward (see ``_make_sharded_nufft2``). Either way the returned value_and_grad has the same
(x) -> (value, grad) shape as the single-device factory and drops straight into the optax loop
in ``imaging.optimizers``. jax / sharding imports are lazy so ``import ehtim`` stays jax-free.
"""
import numpy as np

import ehtim.imaging.multifreq_imager_utils as mfutils
from ehtim.imaging.imager_backend import (
    compute_chisq_dict,
    compute_chisq_term,
    compute_reg_dict,
    transform_imarr,
    unpack_imarr,
)


def build_mesh(devices=None, axis="shard"):
    """A 1-D device mesh over `devices` (default: all local GPUs).

    The sharded objective places its data on this mesh; the only requirement is that the
    sharded axis -- visibilities or channels, after padding -- divides evenly across it.
    """
    import jax
    devices = devices if devices is not None else jax.devices("gpu")
    return jax.sharding.Mesh(np.asarray(devices), (axis,))


def _pad_rows(arr, pad, fill=0.0):
    """Pad `pad` rows of `fill` onto axis 0 (no-op when pad == 0)."""
    if pad == 0:
        return arr
    width = [(0, pad)] + [(0, 0)] * (arr.ndim - 1)
    return np.pad(np.asarray(arr), width, constant_values=fill)


class _NFFTView:
    """Minimal NFFTInfo stand-in for the sharded jax path: nufft2_backend reads
    xdim/ydim/uv_finufft/eps and the nfft chi2 kernels read pulsefac. Rebuilt per
    device inside shard_map from the sharded uv_finufft + pulsefac, so each GPU runs
    jax_finufft.nufft2 on its own slice of visibilities. The stateful numpy finufft
    plan is not needed (the sharded path is jax-only)."""

    def __init__(self, uv_finufft, pulsefac, eps, xdim, ydim):
        self.uv_finufft = uv_finufft
        self.pulsefac = pulsefac
        self.eps = eps
        self.xdim = xdim
        self.ydim = ydim


def _make_sharded_nufft2(mesh, axis, uv_sharded, eps, shape):
    """custom_vjp for a replicated image grid -> sharded visibility samples on `mesh`.

    Forward runs jax_finufft.nufft2 per device on its uv slice; backward applies the type-1
    adjoint as an explicit FORWARD nufft1 + psum. jax_finufft's registered nufft2 transpose
    is incorrect under shard_map, but a forward nufft1 is correct and type-1 is additive over
    points, so psum(nufft1(local)) == nufft1(global). `shape` is the (xdim, ydim) grid.
    """
    import jax
    from jax.sharding import PartitionSpec as P
    from jax_finufft import nufft1, nufft2
    try:
        from jax import shard_map
    except ImportError:
        from jax.experimental.shard_map import shard_map

    fwd = shard_map(lambda g, u: nufft2(g, u[:, 0], u[:, 1], iflag=-1, eps=eps),
                    mesh=mesh, in_specs=(P(), P(axis, None)), out_specs=P(axis))
    bwd = shard_map(lambda c, u: jax.lax.psum(
                        nufft1(shape, c, u[:, 0], u[:, 1], iflag=-1, eps=eps), axis),
                    mesh=mesh, in_specs=(P(axis), P(axis, None)), out_specs=P())

    @jax.custom_vjp
    def transform(f_hat):
        return fwd(f_hat, uv_sharded)

    transform.defvjp(lambda f_hat: (fwd(f_hat, uv_sharded), None),
                     lambda _res, c: (bwd(c, uv_sharded),))
    return transform


def make_sharded_value_and_grad(initvec, config, which_solve, data_tuples,
                                logfreqratio_list, n_obs, dat_term, reg_term,
                                priorvec, norm_reg, reg_params, embed_mask,
                                mesh, shard_axis="baseline"):
    """Return (value_and_grad, loss, to_device) with data sharded across `mesh`.

    Same arguments as make_value_and_grad_jax plus the device `mesh`. The data
    tuples are padded + row-sharded; the solver vector / prior / init stay
    replicated. value_and_grad has signature (x, aux) -> (value, grad): the sharded
    device arrays are bundled into `aux` and passed as a JIT ARGUMENT, not closed
    over -- closing over sharded arrays makes jax mis-partition them (it materializes
    them from the wrong/uninitialized device buffers), giving non-deterministic NaN
    results. Returns (value_and_grad, loss, to_device, aux).
    """
    import jax
    import jax.numpy as jnp
    from jax.sharding import NamedSharding
    from jax.sharding import PartitionSpec as P
    try:
        from jax import shard_map
    except ImportError:
        from jax.experimental.shard_map import shard_map

    k = mesh.size
    axis = mesh.axis_names[0]
    replicated = NamedSharding(mesh, P())

    def to_device(a):
        return jax.device_put(jnp.asarray(a), replicated)

    init_d, prior_d = to_device(initvec), to_device(priorvec)
    dat_keys = sorted(dat_term.keys())
    reg_keys = sorted(reg_term.keys())

    def regterm_of(imcur, prior):
        reg = compute_reg_dict(imcur, reg_keys, config, logfreqratio_list, n_obs,
                               prior, norm_reg, reg_params, embed_mask)
        return sum(reg_term[rn] * reg[rn] for rn in reg_keys)

    if shard_axis == "baseline":
        rows = NamedSharding(mesh, P(axis))          # (N,) data / sigma
        rows2d = NamedSharding(mesh, P(axis, None))  # (N, Npix) Fourier matrix

        def shard_rows(a, sharding, fill=0.0):
            true_n = np.asarray(a).shape[0]
            pad = (-true_n) % k
            return jax.device_put(jnp.asarray(_pad_rows(a, pad, fill)), sharding), true_n, pad

        # Pad + shard each data term on its data-point (visibility/closure) axis. The
        # operator is padded with ones and sigma with infinity, so padded rows keep finite
        # samples (closures stay clear of log(0)/angle(0)) yet add exactly 0 to chi^2;
        # correction[key] = pad_len/true_len then fixes the inflated 1/len(data) normalization.
        # The direct and nfft branches below differ in how they shard the operator and reduce.
        data_d, correction, data_specs = {}, {}, {}
        aux = {"init": init_d, "prior": prior_d, "data": data_d}

        if config.ttype == "nfft":
            # nfft: the operator is a list of NFFTInfo. jax_finufft's nufft2 forward shards
            # cleanly but its transpose is wrong under shard_map, so each NFFTInfo's grid ->
            # samples map is wrapped as a custom_vjp whose backward is an explicit forward
            # nufft1 + psum (_make_sharded_nufft2, injected via nufft2_backend). The reused
            # chi^2 kernels then run at top level and GSPMD all-reduces the chi^2 sums over the
            # sharded visibility axis. Sharded uv/pulsefac/data/sigma are passed as jit args;
            # the views (which carry the transform closure) are rebuilt inside the loss so
            # nothing closes over a sharded array as a compile-time constant.
            nfft_static = {}
            for key, (data, sigma, A) in data_tuples.items():
                data_s, true_n, pad = shard_rows(data, rows)
                sigma_s, _, _ = shard_rows(sigma, rows, fill=np.inf)
                infos = list(A) if isinstance(A, (tuple, list)) else [A]
                uvs = tuple(shard_rows(info.uv_finufft, rows2d)[0] for info in infos)
                pfs = tuple(shard_rows(info.pulsefac, rows, fill=1.0)[0] for info in infos)
                data_d[key] = (data_s, sigma_s, uvs, pfs)
                nfft_static[key] = [(info.eps, info.xdim, info.ydim) for info in infos]
                correction[key] = (true_n + pad) / true_n

            def loss(x, aux):
                imcur = transform_imarr(unpack_imarr(x, aux["init"], which_solve),
                                        config.transforms, which_solve)
                rebuilt = {}
                for key, (data, sigma, uvs, pfs) in aux["data"].items():
                    views = []
                    for i, (uv_s, pf_s) in enumerate(zip(uvs, pfs)):
                        eps, xdim, ydim = nfft_static[key][i]
                        v = _NFFTView(uv_s, pf_s, eps, xdim, ydim)
                        v._sharded_transform = _make_sharded_nufft2(
                            mesh, axis, uv_s, eps, (xdim, ydim))
                        views.append(v)
                    rebuilt[key] = (data, sigma, views)
                chi2 = compute_chisq_dict(imcur, dat_keys, config, rebuilt,
                                          logfreqratio_list, n_obs, embed_mask)
                datterm = 0.0
                for dname in dat_keys:
                    for i in range(n_obs):
                        key = dname if n_obs == 1 else f"{dname}_{i}"
                        datterm = datterm + dat_term[dname] * (chi2[key] * correction[key] - 1.0)
                return datterm + regterm_of(imcur, aux["prior"])
        else:
            # direct: the operator is a dense (Nvis, Npix) matrix (or a list of them for
            # closure terms). Shard its rows; differentiating the dense matmul through
            # shard_map is correct, so the default jax.value_and_grad(loss) is used.
            for key, (data, sigma, A) in data_tuples.items():
                data_s, true_n, pad = shard_rows(data, rows)
                sigma_s, _, _ = shard_rows(sigma, rows, fill=np.inf)
                if isinstance(A, (tuple, list)):
                    A_s = tuple(shard_rows(a, rows2d, fill=1.0)[0] for a in A)
                    a_spec = tuple(P(axis, None) for _ in A)
                elif np.ndim(A) == 2:
                    A_s = shard_rows(A, rows2d, fill=1.0)[0]
                    a_spec = P(axis, None)
                else:
                    raise NotImplementedError(
                        f"baseline sharding does not support ttype={config.ttype!r}")
                data_d[key] = (data_s, sigma_s, A_s)
                data_specs[key] = (P(axis), P(axis), a_spec)
                correction[key] = (true_n + pad) / true_n

            def _local_chisq(imcur, data_shard):
                local = compute_chisq_dict(imcur, dat_keys, config, data_shard,
                                           logfreqratio_list, n_obs, embed_mask)
                return {kk: jax.lax.pmean(vv, axis) for kk, vv in local.items()}

            sharded_chisq = shard_map(_local_chisq, mesh=mesh,
                                      in_specs=(P(), data_specs),
                                      out_specs={kk: P() for kk in data_d})

            def loss(x, aux):
                imcur = transform_imarr(unpack_imarr(x, aux["init"], which_solve),
                                        config.transforms, which_solve)
                chi2 = sharded_chisq(imcur, aux["data"])
                datterm = 0.0
                for dname in dat_keys:
                    for i in range(n_obs):
                        key = dname if n_obs == 1 else f"{dname}_{i}"
                        datterm = datterm + dat_term[dname] * (chi2[key] * correction[key] - 1.0)
                return datterm + regterm_of(imcur, aux["prior"])

    elif shard_axis == "frequency":
        if n_obs < 2:
            raise ValueError("frequency sharding needs n_obs > 1 (multifrequency)")
        nf_pad = n_obs + (-n_obs) % k                # pad channel count to a mesh multiple
        valid = np.zeros(nf_pad)
        valid[:n_obs] = 1.0                          # 0 for padded channels (drop the -1 offset)
        logfreq = np.zeros(nf_pad)
        logfreq[:n_obs] = np.asarray(logfreqratio_list)[:n_obs]

        ch = NamedSharding(mesh, P(axis))              # (nf,) valid / logfreq
        ch2d = NamedSharding(mesh, P(axis, None))       # (nf, Nvis)
        ch3d = NamedSharding(mesh, P(axis, None, None))  # (nf, Nvis, Npix)

        # restack each data term over the channel axis; padded channels are dummy
        # (data 0, sigma 1, matrix 0) and zeroed by the validity mask.
        stacks = {}
        for dname in dat_keys:
            per = [data_tuples[f"{dname}_{i}"] for i in range(n_obs)]
            A0 = per[0][2]
            if isinstance(A0, (tuple, list)) or np.ndim(A0) != 2:
                raise NotImplementedError(
                    "frequency sharding supports dense single-matrix data terms "
                    "(vis/amp); closure / nfft terms are not yet wired")
            nvis = np.asarray(per[0][0]).shape[0]
            if any(np.asarray(d).shape[0] != nvis for d, _, _ in per):
                raise NotImplementedError("frequency sharding assumes equal Nvis per channel")
            npix = np.asarray(A0).shape[1]
            data_st = np.zeros((nf_pad, nvis), dtype=np.asarray(per[0][0]).dtype)
            sigma_st = np.ones((nf_pad, nvis), dtype=np.asarray(per[0][1]).dtype)
            A_st = np.zeros((nf_pad, nvis, npix), dtype=np.asarray(A0).dtype)
            for i, (d, s, A) in enumerate(per):
                data_st[i] = np.asarray(d)
                sigma_st[i] = np.asarray(s)
                A_st[i] = np.asarray(A)
            stacks[dname] = (jax.device_put(jnp.asarray(data_st), ch2d),
                             jax.device_put(jnp.asarray(sigma_st), ch2d),
                             jax.device_put(jnp.asarray(A_st), ch3d))
        aux = {"init": init_d, "prior": prior_d, "stacks": stacks,
               "valid": jax.device_put(jnp.asarray(valid), ch),
               "logfreq": jax.device_put(jnp.asarray(logfreq), ch)}

        def loss(x, aux):
            imcur = transform_imarr(unpack_imarr(x, aux["init"], which_solve),
                                    config.transforms, which_solve)

            def per_freq(slices, logfreq_i, valid_i):
                imcur_nu = mfutils.image_at_freq(imcur, logfreq_i) if config.mf else imcur
                dt = 0.0
                for dname in dat_keys:
                    d_i, s_i, A_i = slices[dname]
                    chisq = compute_chisq_term(imcur_nu, dname, A_i, d_i, s_i,
                                               ttype=config.ttype, mask=embed_mask)
                    dt = dt + dat_term[dname] * (chisq - 1.0)
                return valid_i * dt

            datterm = jnp.sum(jax.vmap(per_freq)(aux["stacks"], aux["logfreq"], aux["valid"]))
            return datterm + regterm_of(imcur, aux["prior"])

    else:
        raise NotImplementedError(f"shard_axis={shard_axis!r} not recognized")

    return jax.value_and_grad(loss, argnums=0), loss, to_device, aux
