from __future__ import division
from __future__ import print_function
from builtins import range
from builtins import object

import numpy as np

def resample(y, n=16):
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

def resample2(img, n=16):
    """Resampling an image

    For simplicity, we will just pad each side of an image to an
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
