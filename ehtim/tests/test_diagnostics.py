from .. import diagnostics as ds

import numpy as np
import matplotlib.pyplot as plt

def gauss(x0, sx):
    n = 128
    X = 0.5 * (n - 1)
    x = np.linspace(-X, X, n) / n - x0
    return np.exp(-0.5 * (x * x) / (sx * sx)) / np.sqrt(2 * np.pi * sx * sx)

def gauss2(x0, y0, sx, sy):
    return np.sqrt(np.outer(gauss(x0, sx), gauss(y0, sy)))

def test_diagnostics_onedimize():
    """Test if onedimize() produces expected results
    """
    img1 = gauss2(0.0,0.0,0.1,0.1)
    img2 = 0.5 * (gauss2(0.0,-0.1,0.1,0.1) + gauss2(0.0,0.1,0.1,0.1))
    img3 = 0.5 * (gauss2(0.0,-0.2,0.1,0.1) + gauss2(0.0,0.2,0.1,0.1))
    imgs = [img1, img2, img3]
    labels = ['One Gaussian', 'Two nearby Gaussians', 'Two far-away Gaussians']

    n = 32
    x = np.arange(n * n)
    oneds, ref = ds.onedimize(imgs, n=n)

    f, axs = plt.subplots(2, sharex=True)

    axs[0].step(x, ref, linewidth=3, color='k', label='Mean')
    for i, oned in enumerate(oneds):
        axs[0].step(x, oned, alpha=0.5, label=labels[i])
    axs[0].legend()
    axs[0].set_ylabel('Intensity')

    axs[1].step(x, np.log(ref), linewidth=3, color='k', label='Mean')
    for i, oned in enumerate(oneds):
        axs[1].step(x, np.log(oned), alpha=0.5, label=labels[i])
    axs[1].legend()
    axs[1].set_xlabel('Sorting index')
    axs[1].set_ylabel('ln(Intensity)')

    plt.show()
