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
