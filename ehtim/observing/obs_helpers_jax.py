# obs_helpers_jax.py
# JAX helper functions for simulating and manipulating observations.
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
import jax.scipy.ndimage as jnd


def fft_imvec(imvec, im_info):
    """Returns fft of imvec on grid."""
    imarr = jnp.asarray(imvec).reshape(im_info.ydim, im_info.xdim)
    imarr = jnp.pad(
        imarr,
        ((im_info.padvaly1, im_info.padvaly2),
         (im_info.padvalx1, im_info.padvalx2)),
        mode="constant",
        constant_values=0.0,
    )

    if imarr.shape[0] != imarr.shape[1]:
        raise Exception("FFT padding did not return a square image!")

    return jnp.fft.fftshift(jnp.fft.fft2(jnp.fft.ifftshift(imarr)))


def sampler(griddata, sampler_info_list, sample_type="vis"):
    """Samples gridded FFT data at uv points."""
    if sample_type not in ["vis", "bs", "camp"]:
        raise Exception("sampler sample_type should be either 'vis','bs',or 'camp'!")
    if griddata.shape[0] != griddata.shape[1]:
        raise Exception("griddata should be a square array!")

    dataset = []
    for sampler_info in sampler_info_list:
        if sampler_info.order > 1:
            raise NotImplementedError(
                "obs_helpers_jax.sampler supports interpolation order <= 1; "
                f"got order={sampler_info.order}"
            )

        uv = jnp.asarray(sampler_info.uv)
        pulsefac = jnp.asarray(sampler_info.pulsefac)
        datare = jnd.map_coordinates(jnp.real(griddata), uv, order=sampler_info.order)
        dataim = jnd.map_coordinates(jnp.imag(griddata), uv, order=sampler_info.order)
        dataset.append((datare + 1j * dataim) * pulsefac)

    if sample_type == "vis":
        return dataset[0]
    if sample_type == "bs":
        return dataset[0] * dataset[1] * dataset[2]
    return jnp.abs((dataset[0] * dataset[1]) / (dataset[2] * dataset[3]))
