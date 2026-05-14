# pol_imager_utils_jax.py
# Polarimetric imager functions mirroring pol_imager_utils.py with jax.numpy.
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


def make_i_image(imarr):
    return imarr[0]


def make_p_image(imarr):
    return imarr[0] * imarr[1] * jnp.exp(1j * imarr[2]) * jnp.cos(imarr[3])


def make_v_image(imarr):
    return imarr[0] * imarr[1] * jnp.sin(imarr[3])


def chisq_p(imarr, Amatrix, p, sigmap):
    p = jnp.asarray(p)
    sigmap = jnp.asarray(sigmap)
    psamples = jnp.dot(jnp.asarray(Amatrix), make_p_image(imarr))
    return jnp.sum(jnp.abs(p - psamples) ** 2 / (sigmap ** 2)) / (2 * len(p))


def chisq_m(imarr, Amatrix, m, sigmam):
    m = jnp.asarray(m)
    sigmam = jnp.asarray(sigmam)
    psamples = jnp.dot(jnp.asarray(Amatrix), make_p_image(imarr))
    isamples = jnp.dot(jnp.asarray(Amatrix), make_i_image(imarr))
    msamples = psamples / isamples
    return jnp.sum(jnp.abs(m - msamples) ** 2 / (sigmam ** 2)) / (2 * len(m))


def chisq_vvis(imarr, Amatrix, v, sigmav):
    v = jnp.asarray(v)
    sigmav = jnp.asarray(sigmav)
    vsamples = jnp.dot(jnp.asarray(Amatrix), make_v_image(imarr))
    return jnp.sum(jnp.abs(v - vsamples) ** 2 / (sigmav ** 2)) / (2 * len(v))
