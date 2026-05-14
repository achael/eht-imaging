# imager_utils_jax.py
# General imager functions mirroring imager_utils.py with jax.numpy.
#
#    Copyright (C) 2018 Andrew Chael
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.

import jax.numpy as jnp

import ehtim.const_def as ehc
import ehtim.observing.obs_helpers_jax as obsh


def embed(imvec, mask, clipfloor=0., randomfloor=False):
    """Embeds a 1d image vector into the size of boolean embed mask."""
    imvec = jnp.asarray(imvec)
    mask = jnp.asarray(mask, dtype=bool)

    if randomfloor and clipfloor != 0.:
        raise NotImplementedError(
            "imager_utils_jax.embed does not support randomfloor=True with "
            "nonzero clipfloor because the mirrored signature has no JAX PRNG key."
        )

    source_index = jnp.cumsum(mask.astype(jnp.int32)) - 1
    source_index = jnp.maximum(source_index, 0)
    active_values = jnp.take(imvec, source_index, mode="clip")
    return jnp.where(mask, active_values, jnp.asarray(clipfloor, dtype=imvec.dtype))


def chisq_vis(imvec, Amatrix, vis, sigma):
    vis = jnp.asarray(vis)
    sigma = jnp.asarray(sigma)
    samples = jnp.dot(jnp.asarray(Amatrix), imvec)
    return jnp.sum(jnp.abs((samples - vis) / sigma) ** 2) / (2 * len(vis))


def chisq_amp(imvec, A, amp, sigma):
    amp = jnp.asarray(amp)
    sigma = jnp.asarray(sigma)
    amp_samples = jnp.abs(jnp.dot(jnp.asarray(A), imvec))
    return jnp.sum(jnp.abs((amp - amp_samples) / sigma) ** 2) / len(amp)


def chisq_logamp(imvec, A, amp, sigma):
    amp = jnp.asarray(amp)
    sigma = jnp.asarray(sigma)
    logsigma = sigma / amp
    amp_samples = jnp.abs(jnp.dot(jnp.asarray(A), imvec))
    return jnp.sum(jnp.abs((jnp.log(amp) - jnp.log(amp_samples)) / logsigma) ** 2) / len(amp)


def chisq_bs(imvec, Amatrices, bis, sigma):
    bis = jnp.asarray(bis)
    sigma = jnp.asarray(sigma)
    bisamples = (jnp.dot(jnp.asarray(Amatrices[0]), imvec) *
                 jnp.dot(jnp.asarray(Amatrices[1]), imvec) *
                 jnp.dot(jnp.asarray(Amatrices[2]), imvec))
    return jnp.sum(jnp.abs((bis - bisamples) / sigma) ** 2) / (2 * len(bis))


def chisq_cphase(imvec, Amatrices, clphase, sigma):
    clphase = jnp.asarray(clphase) * ehc.DEGREE
    sigma = jnp.asarray(sigma) * ehc.DEGREE

    clphase_samples = jnp.angle(jnp.dot(jnp.asarray(Amatrices[0]), imvec) *
                                jnp.dot(jnp.asarray(Amatrices[1]), imvec) *
                                jnp.dot(jnp.asarray(Amatrices[2]), imvec))

    chisq = jnp.sum((1.0 - jnp.cos(clphase - clphase_samples)) / (sigma ** 2))
    return chisq * (2.0 / len(clphase))


def chisq_cphase_diag(imvec, Amatrices, clphase_diag, sigma):
    clphase_diag = jnp.concatenate([jnp.asarray(arr) for arr in clphase_diag]) * ehc.DEGREE
    sigma = jnp.concatenate([jnp.asarray(arr) for arr in sigma]) * ehc.DEGREE

    A3_diag = Amatrices[0]
    tform_mats = Amatrices[1]

    clphase_diag_samples = []
    for iA, A3 in enumerate(A3_diag):
        clphase_samples = jnp.angle(jnp.dot(jnp.asarray(A3[0]), imvec) *
                                    jnp.dot(jnp.asarray(A3[1]), imvec) *
                                    jnp.dot(jnp.asarray(A3[2]), imvec))
        clphase_diag_samples.append(
            jnp.dot(jnp.asarray(tform_mats[iA]), clphase_samples)
        )
    clphase_diag_samples = jnp.concatenate(clphase_diag_samples)

    chisq = jnp.sum((1.0 - jnp.cos(clphase_diag - clphase_diag_samples)) / (sigma ** 2))
    return chisq * (2.0 / len(clphase_diag))


def chisq_camp(imvec, Amatrices, clamp, sigma):
    clamp = jnp.asarray(clamp)
    sigma = jnp.asarray(sigma)
    i1 = jnp.dot(jnp.asarray(Amatrices[0]), imvec)
    i2 = jnp.dot(jnp.asarray(Amatrices[1]), imvec)
    i3 = jnp.dot(jnp.asarray(Amatrices[2]), imvec)
    i4 = jnp.dot(jnp.asarray(Amatrices[3]), imvec)
    clamp_samples = jnp.abs((i1 * i2) / (i3 * i4))

    return jnp.sum(jnp.abs((clamp - clamp_samples) / sigma) ** 2) / len(clamp)


def chisq_logcamp(imvec, Amatrices, log_clamp, sigma):
    log_clamp = jnp.asarray(log_clamp)
    sigma = jnp.asarray(sigma)
    a1 = jnp.abs(jnp.dot(jnp.asarray(Amatrices[0]), imvec))
    a2 = jnp.abs(jnp.dot(jnp.asarray(Amatrices[1]), imvec))
    a3 = jnp.abs(jnp.dot(jnp.asarray(Amatrices[2]), imvec))
    a4 = jnp.abs(jnp.dot(jnp.asarray(Amatrices[3]), imvec))

    samples = jnp.log(a1) + jnp.log(a2) - jnp.log(a3) - jnp.log(a4)
    return jnp.sum(jnp.abs((log_clamp - samples) / sigma) ** 2) / len(log_clamp)


def chisq_logcamp_diag(imvec, Amatrices, log_clamp_diag, sigma):
    log_clamp_diag = jnp.concatenate([jnp.asarray(arr) for arr in log_clamp_diag])
    sigma = jnp.concatenate([jnp.asarray(arr) for arr in sigma])

    A4_diag = Amatrices[0]
    tform_mats = Amatrices[1]

    log_clamp_diag_samples = []
    for iA, A4 in enumerate(A4_diag):
        a1 = jnp.abs(jnp.dot(jnp.asarray(A4[0]), imvec))
        a2 = jnp.abs(jnp.dot(jnp.asarray(A4[1]), imvec))
        a3 = jnp.abs(jnp.dot(jnp.asarray(A4[2]), imvec))
        a4 = jnp.abs(jnp.dot(jnp.asarray(A4[3]), imvec))

        log_clamp_samples = jnp.log(a1) + jnp.log(a2) - jnp.log(a3) - jnp.log(a4)
        log_clamp_diag_samples.append(
            jnp.dot(jnp.asarray(tform_mats[iA]), log_clamp_samples)
        )

    log_clamp_diag_samples = jnp.concatenate(log_clamp_diag_samples)
    chisq = jnp.sum(jnp.abs((log_clamp_diag - log_clamp_diag_samples) / sigma) ** 2)
    return chisq / len(log_clamp_diag)


def chisq_vis_fft(vis_arr, A, vis, sigma):
    vis = jnp.asarray(vis)
    sigma = jnp.asarray(sigma)
    samples = obsh.sampler(vis_arr, A[1], sample_type="vis")
    return jnp.sum(jnp.abs((samples - vis) / sigma) ** 2) / (2 * len(vis))


def chisq_amp_fft(vis_arr, A, amp, sigma):
    amp = jnp.asarray(amp)
    sigma = jnp.asarray(sigma)
    amp_samples = jnp.abs(obsh.sampler(vis_arr, A[1], sample_type="vis"))
    return jnp.sum(jnp.abs((amp_samples - amp) / sigma) ** 2) / len(amp)


def chisq_logamp_fft(vis_arr, A, amp, sigma):
    amp = jnp.asarray(amp)
    sigma = jnp.asarray(sigma)
    logsigma = sigma / amp
    amp_samples = jnp.abs(obsh.sampler(vis_arr, A[1], sample_type="vis"))
    chisq = jnp.sum(jnp.abs((jnp.log(amp_samples) - jnp.log(amp)) / logsigma) ** 2)
    return chisq / len(amp)


def chisq_bs_fft(vis_arr, A, bis, sigma):
    bis = jnp.asarray(bis)
    sigma = jnp.asarray(sigma)
    bisamples = obsh.sampler(vis_arr, A[1], sample_type="bs")
    return jnp.sum(jnp.abs((bis - bisamples) / sigma) ** 2) / (2 * len(bis))


def chisq_cphase_fft(vis_arr, A, clphase, sigma):
    clphase = jnp.asarray(clphase) * ehc.DEGREE
    sigma = jnp.asarray(sigma) * ehc.DEGREE
    clphase_samples = jnp.angle(obsh.sampler(vis_arr, A[1], sample_type="bs"))
    chisq = jnp.sum((1.0 - jnp.cos(clphase - clphase_samples)) / (sigma ** 2))
    return chisq * (2.0 / len(clphase))


def chisq_cphase_diag_fft(imvec, A, clphase_diag, sigma):
    clphase_diag = jnp.concatenate([jnp.asarray(arr) for arr in clphase_diag]) * ehc.DEGREE
    sigma = jnp.concatenate([jnp.asarray(arr) for arr in sigma]) * ehc.DEGREE

    A3 = A[0]
    tform_mats = A[1]
    vis_arr = obsh.fft_imvec(imvec, A3[0])
    clphase_samples = jnp.angle(obsh.sampler(vis_arr, A3[1], sample_type="bs"))

    count = 0
    clphase_diag_samples = []
    for tform_mat in tform_mats:
        clphase_samples_here = clphase_samples[count:count + len(tform_mat)]
        clphase_diag_samples.append(
            jnp.dot(jnp.asarray(tform_mat), clphase_samples_here)
        )
        count += len(tform_mat)

    clphase_diag_samples = jnp.concatenate(clphase_diag_samples)
    chisq = jnp.sum((1.0 - jnp.cos(clphase_diag - clphase_diag_samples)) / (sigma ** 2))
    return chisq * (2.0 / len(clphase_diag))


def chisq_camp_fft(vis_arr, A, clamp, sigma):
    clamp = jnp.asarray(clamp)
    sigma = jnp.asarray(sigma)
    clamp_samples = obsh.sampler(vis_arr, A[1], sample_type="camp")
    return jnp.sum(jnp.abs((clamp - clamp_samples) / sigma) ** 2) / len(clamp)


def chisq_logcamp_fft(vis_arr, A, log_clamp, sigma):
    log_clamp = jnp.asarray(log_clamp)
    sigma = jnp.asarray(sigma)
    log_clamp_samples = jnp.log(obsh.sampler(vis_arr, A[1], sample_type="camp"))
    return jnp.sum(jnp.abs((log_clamp - log_clamp_samples) / sigma) ** 2) / len(log_clamp)


def chisq_logcamp_diag_fft(imvec, A, log_clamp_diag, sigma):
    log_clamp_diag = jnp.concatenate([jnp.asarray(arr) for arr in log_clamp_diag])
    sigma = jnp.concatenate([jnp.asarray(arr) for arr in sigma])

    A4 = A[0]
    tform_mats = A[1]
    vis_arr = obsh.fft_imvec(imvec, A4[0])
    log_clamp_samples = jnp.log(obsh.sampler(vis_arr, A4[1], sample_type="camp"))

    count = 0
    log_clamp_diag_samples = []
    for tform_mat in tform_mats:
        log_clamp_samples_here = log_clamp_samples[count:count + len(tform_mat)]
        log_clamp_diag_samples.append(
            jnp.dot(jnp.asarray(tform_mat), log_clamp_samples_here)
        )
        count += len(tform_mat)

    log_clamp_diag_samples = jnp.concatenate(log_clamp_diag_samples)
    chisq = jnp.sum(jnp.abs((log_clamp_diag - log_clamp_diag_samples) / sigma) ** 2)
    return chisq / len(log_clamp_diag)
