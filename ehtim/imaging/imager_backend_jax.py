# imager_backend_jax.py
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


# Data term and polarization-mode names recognized by the chi^2 dispatchers.
DATATERMS = ['vis', 'bs', 'amp', 'cphase', 'cphase_diag', 'camp', 'logcamp', 'logcamp_diag']
DATATERMS_POL = ['pvis', 'm', 'vvis']
POLARIZATION_MODES = ['P', 'QU', 'IP', 'IQU', 'V', 'IV', 'IQUV', 'IPV']

# Regularizer term names recognized by the regularizer dispatchers.
REGULARIZERS = ['gs', 'tv', 'tvlog', 'tv2', 'tv2log', 'l1', 'l1w', 'lA', 'patch',
                'flux', 'cm', 'simple', 'compact', 'compact2', 'rgauss']
REGULARIZERS_POL = ['msimple', 'hw', 'ptv', 'l1v', 'l2v', 'vtv', 'vtv2', 'vflux']

REGULARIZERS_ALLFREQS_I = ['flux_mf']
REGULARIZERS += REGULARIZERS_ALLFREQS_I

REGULARIZERS_SPECIND = ['l2_alpha', 'tv_alpha']
REGULARIZERS_CURV = ['l2_beta', 'tv_beta']
REGULARIZERS_SPECIND_P = ['l2_alphap', 'tv_alphap']
REGULARIZERS_CURV_P = ['l2_betap', 'tv_betap']
REGULARIZERS_RM = ['l2_rm', 'tv_rm']
REGULARIZERS_CM = ['l2_cm', 'tv_cm']
REGULARIZERS_ISPECTRAL = REGULARIZERS_SPECIND + REGULARIZERS_CURV
REGULARIZERS_POLSPECTRAL = REGULARIZERS_SPECIND_P + REGULARIZERS_CURV_P + REGULARIZERS_RM + REGULARIZERS_CM
REGULARIZERS_SPECTRAL = REGULARIZERS_ISPECTRAL + REGULARIZERS_POLSPECTRAL


def pack_imarr(imarr, which_solve):
    """pack image array imarr into 1D array vec for minimizaiton
       ignore quantities not solved for
    """
    imarr = jnp.asarray(imarr)
    which_solve = jnp.asarray(which_solve)

    imarrdim = len(imarr.shape)
    if imarrdim == 2:
        nsolve = imarr.shape[0]
    elif imarrdim == 1:
        nsolve = 1
        imarr = imarr.reshape((nsolve, imarr.shape[0]))
    else:
        raise Exception("in pack_imarr, imarr should have one or two dimensions!")

    if nsolve != len(which_solve):
        raise Exception("in pack_imarr, imarr has inconsistent shape with which_solve!")

    rows = [imarr[kk] for kk in range(nsolve) if bool(which_solve[kk] != 0)]
    if not rows:
        return jnp.array([], dtype=imarr.dtype)
    return jnp.concatenate(rows)


def compute_embed(imvec, xdim, ydim, psize, clipfloor):
    """Compute embedding mask and coordinate matrix from a prior image vector."""

    imvec = jnp.asarray(imvec)
    embed_mask = imvec > clipfloor
    if not bool(jnp.any(embed_mask)):
        raise Exception("clipfloor too large: all prior pixels have been clipped!")

    xmax = xdim // 2
    ymax = ydim // 2

    if xdim % 2:
        xmin = -xmax - 1
    else:
        xmin = -xmax

    if ydim % 2:
        ymin = -ymax - 1
    else:
        ymin = -ymax

    xcoord = jnp.arange(xmax, xmin, -1)
    ycoord = jnp.arange(ymax, ymin, -1)
    coord_x, coord_y = jnp.meshgrid(xcoord, ycoord, indexing='xy')
    coord = jnp.stack((coord_x, coord_y), axis=-1)

    coord = coord.reshape(ydim * xdim, 2)
    coord = coord * psize

    coord_matrix = coord[embed_mask]

    return embed_mask, coord_matrix
