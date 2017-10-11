#!/usr/bin/env python

import ehtim.diagnostics as ed

import numpy as np
import matplotlib.pyplot as plt
from skimage import io

def gauss(x0, sx):
    n = 128
    X = 0.5 * (n - 1)
    x = np.linspace(-X, X, n) / n - x0
    return np.exp(-0.5 * (x * x) / (sx * sx)) / np.sqrt(2 * np.pi * sx * sx)

def gauss2(x0, y0, sx, sy):
    return np.sqrt(np.outer(gauss(x0, sx), gauss(y0, sy)))


urltemplate = "http://vlbiimaging.csail.mit.edu/static/testData/uploads/{}/pngimages/test_02.png"
submissions = ["2617818", "7039776", "8649732", "6288084", "5869777", "6356229"]
labels      = ["CHRIP - Natural", "SQUEEZE - Total Variation", "BSMEM", "Bispectrum MEM L-BFGS",
               "Bispectral MEM L-BFGS Total Variation", "CHIRP - Celestial"]

urls = [urltemplate.format(s) for s in submissions]
imgs = [io.imread(url) for url in urls]
imgs = [img / np.sum(img.astype(float)) for img in imgs]

n = 8
x = np.arange(n * n)
oneds, ref = ed.onedimize(imgs, n=n)

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
