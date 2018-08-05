# diagnostics.py
# useful diagnostic tests on images
#
#    Copyright (C) 2018 Katie Bouman
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


from __future__ import division
from __future__ import print_function
from builtins import range
from builtins import object

import numpy as np

def sumdown_lin(y, n=16):
    """Sum segments of a line together to reduce its size
    """
    nold = y.shape[0]
    nnew = n

    xold = np.linspace(0, 1, num=nold, endpoint=False)
    xnew = np.linspace(0, 1, num=nnew, endpoint=False)

    csum  = np.zeros(nnew+1)
    for i, x in enumerate(xnew):
        u = np.argmax(xold > x)
        l = u - 1
        csum[i] = (np.sum(y[0:l]) +
                   y[l] * (x - xold[l]) / (xold[u] - xold[l]))
    csum[nnew] = sum(y)

    return np.diff(csum)

def sumdown_img(img, n=16):
    """Summing patches of an image together to reduce its size

       For simplicity, we just pad each side of an image to an
       integer multiple of n and then down sample but summing up the
       intensity of the subcells.  This is not perfect but is good enough
       for a proof of concept.
    """
    nx = img.shape[0]
    ny = img.shape[1]
    mx = (nx - 1)//n + 1
    my = (ny - 1)//n + 1
    Nx = mx * n
    Ny = my * n
    px = (Nx - nx)//2
    py = (Ny - ny)//2

    img = np.pad(img, ((px, Nx-nx-px), (py, Ny-ny-py)), 'constant')
    return img.reshape(n, mx, n, my).sum(axis=(1,3))

def onedimize(imgs, n=16, gt=None):
    """One-dimensionalize an image by sorting in terms of pixel intensity

    Args:
        imgs: a python array of two-dimensional numpy arrays
        n:    the number of pixel in both dimensions of the output images

    Return:
        oneds: a python array of one-dimensional numpy arrays
        mean:  the one-dimensionalized mean image
    """
    imgs = [sumdown_img(img, n=n) for img in imgs]

    if gt is None:
        gt = np.dstack(imgs).mean(axis=2)
    else:
        gt = sumdown_img(gt, n=n)

    idxs = np.argsort(-gt.reshape(n*n))
    return [img.reshape(n*n)[idxs] for img in imgs], gt.reshape(n*n)[idxs]
