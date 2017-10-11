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
labels      = ["CHRIP\nNatural", "SQUEEZE\nTotal Variation", "BSMEM", "Bispectrum MEM\nL-BFGS",
               "Bispectral MEM\nL-BFGS Total Variation", "CHIRP\nCelestial"]

urls = [urltemplate.format(s) for s in submissions]
imgs = [io.imread(url) for url in urls]
imgs = [img / np.sum(img.astype(float)) for img in imgs]

n = 8
x = np.arange(n * n) + 1
oneds, ref = ed.onedimize(imgs, n=n)

f, axs = plt.subplots(1, len(submissions), sharex=True, sharey=True)

for i, oned in enumerate(oneds):
    axs[i].step(np.log(x), np.log(ref),  linewidth=3, color='k')
    axs[i].step(np.log(x), np.log(oned), alpha=0.5)
    axs[i].set_title(labels[i])
    axs[i].set_xlabel('Sorting index')
    if i == 0:
        axs[i].set_ylabel('ln(Intensity)')

axs[0].set_ylim(-8,-2)
f.subplots_adjust(hspace=0,wspace=0)
plt.show()
