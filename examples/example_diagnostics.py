#!/usr/bin/env python

import ehtim.diagnostics as ed

import numpy as np
import matplotlib.pyplot as plt
from skimage import io

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
