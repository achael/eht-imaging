"""Spread the imaging objective across several GPUs.

Only the data term gets divided up. The image and the regularizers are small, so every
GPU keeps its own copy of those; what we split is the chi^2 sum over the data, which is
where the time actually goes. `make_sharded_value_and_grad` hands back a (value, grad)
function with the same signature as the single-device one, so the optax loop in
`imaging.optimizers` never needs to know which of the two it is driving.

There are two ways to divide the work:

- `shard_axis="baseline"` gives each GPU a slice of the visibilities. This is the general
  case and works for any dataset.
- `shard_axis="frequency"` gives each GPU a few frequency channels. Only useful for
  multifrequency data, and you need at least as many channels as GPUs.

Padding. jax insists that the split axis divide evenly across the GPUs, and real datasets
rarely oblige, so each data term is padded up to a multiple of the device count. The padded
rows have to be inert. Sigma is set to infinity, so the row contributes exactly zero to
chi^2, and the operator is set to one rather than zero, so the padded sample stays finite
(a zero would put us at log(0) or angle(0) in the closure terms). Padding still inflates
the 1/len(data) normalization, so each chi^2 is scaled back by pad_len/true_len. With that
in place the sharded objective agrees with the single-device one to the last bit, for the
linear terms (vis, amp) and the closure terms alike.

Fourier transforms. The `direct` transform simply shards its dense matrix and adds up the
pieces with shard_map and pmean. The `nfft` transform needs more care: jax_finufft's nufft2
has an incorrect transpose rule under shard_map, so we cannot differentiate through it as
it stands. Instead its grid->samples step is wrapped in a custom_vjp whose backward pass is
an explicit forward nufft1 (see `_make_sharded_nufft2`).

jax and the sharding machinery are imported lazily, so `import ehtim` still works for
people who do not have jax installed.
"""
import numpy as np

import ehtim.imaging.multifreq_imager_utils as mfutils
from ehtim.imaging.imager_backend import (
    check_jax_supported,
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

    Parameters
    ----------
    devices : sequence of jax.Device, optional
        Devices to build the mesh over. Defaults to every local GPU.
    axis : str, optional
        Name of the mesh axis the data is sharded along.

    Returns
    -------
    jax.sharding.Mesh
        A 1-D mesh over `devices`.

    Raises
    ------
    ValueError
        If no devices are given and no GPU is visible.
    """
    import jax
    if devices is None:
        try:
            devices = jax.devices("gpu")
        except RuntimeError as e:
            # chained deliberately: this also fires when the CUDA plugin fails to load, and
            # that message is the only way to tell that apart from having no GPU at all
            raise ValueError(
                "sharding defaults to the local GPUs and none are visible; pass "
                "mesh=build_mesh(devices=...) to shard over specific devices.") from e
    return jax.sharding.Mesh(np.asarray(devices), (axis,))


def _pad_rows(arr, pad, fill=0.0):
    """Pad `pad` rows of `fill` onto axis 0 (no-op when pad == 0)."""
    if pad == 0:
        return arr
    width = [(0, pad)] + [(0, 0)] * (arr.ndim - 1)
    return np.pad(np.asarray(arr), width, constant_values=fill)


class _NFFTView:
    """A stand-in for NFFTInfo carrying just the fields the sharded jax path reads.

    Each GPU builds one of these inside shard_map from its own slice of uv_finufft and
    pulsefac, so it runs jax_finufft.nufft2 on its own visibilities. The real NFFTInfo
    also holds a stateful numpy finufft plan, which is no use here because this path is
    jax-only.
    """

    def __init__(self, uv_finufft, pulsefac, eps, xdim, ydim):
        self.uv_finufft = uv_finufft
        self.pulsefac = pulsefac
        self.eps = eps
        self.xdim = xdim
        self.ydim = ydim


def _make_sharded_nufft2(mesh, axis, uv_sharded, eps, shape):
    """Map a replicated image grid to sharded visibility samples, with a hand-written gradient.

    The forward pass is just nufft2 on each GPU's slice of uv. The backward pass is the part
    that needs explaining: jax_finufft registers a transpose rule for nufft2 that is wrong
    under shard_map, so rather than rely on it we write the adjoint ourselves as a forward
    nufft1 followed by a psum. That is legitimate because a type-1 transform is a sum over
    points, so adding up each device's nufft1 gives the same answer as one nufft1 over all
    of them.

    `shape` is the (xdim, ydim) image grid.
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
    """Return (value_and_grad, loss, to_device, aux) with the data sharded across `mesh`.

    Takes the same arguments as make_value_and_grad_jax plus the device `mesh`. The data
    tuples are padded and split by row; the solver vector, prior and init stay replicated
    on every device.

    value_and_grad has signature (x, aux) -> (value, grad). The sharded arrays ride along
    in `aux` and are passed as a jit argument rather than closed over, and that part is not
    stylistic: closing over sharded arrays makes jax partition them wrongly, reading from
    uninitialized device buffers, and you get NaNs that come and go between runs.
    """
    check_jax_supported(config.ttype, dat_term)
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

        # Pad each data term out to a multiple of the device count, then split it by row.
        # Sigma is padded with infinity so the extra rows add nothing to chi^2, and the
        # operator with ones so those rows stay finite (see the padding note at the top of
        # the file). correction[key] undoes the 1/len(data) that the padding inflated.
        # direct and nfft differ below in how they shard the operator and reduce.
        data_d, correction, data_specs = {}, {}, {}
        aux = {"init": init_d, "prior": prior_d, "data": data_d}

        if config.ttype == "nfft":
            # nfft: the operator is a list of NFFTInfo. The forward nufft2 shards fine, but
            # its transpose is wrong under shard_map, so we swap in _make_sharded_nufft2 --
            # a custom_vjp whose backward is a forward nufft1 plus a psum. The chi^2 kernels
            # themselves are untouched and run at the top level, where GSPMD all-reduces the
            # sums over the sharded visibility axis. uv/pulsefac/data/sigma go in as jit
            # arguments, and the views holding the transform are rebuilt inside the loss, so
            # no sharded array ends up baked in as a compile-time constant.
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
            # only 'direct' reaches here: check_jax_supported above rejects 'fast', whose
            # gridded operator is a tuple and used to reach _pad_rows and raise IndexError.
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
                        f"cannot shard the operator for data term {key!r}: expected a dense "
                        f"matrix or a tuple of them, got {type(A).__name__}")
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
