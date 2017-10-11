from __future__ import division
from __future__ import print_function
from builtins import range
from builtins import object

import numpy as np

def sumdown_lin(y, n=16):
    """Summing segments of a line together to reduce its size
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

def onedimize(imgs, n=16):
    imgs = [sumdown_img(img, n=n) for img in imgs]
    mean = np.dstack(imgs).mean(axis=2)
    idxs = np.argsort(-mean.reshape(n*n))
    return [img.reshape(n*n)[idxs] for img in imgs], mean.reshape(n*n)[idxs]
