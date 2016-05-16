#!/usr/bin/python

import vlbi_imaging_utils as vb
import numpy as np
import sys

im = vb.load_im_fits(sys.argv[1], punit="deg")
arr = vb.load_array(sys.argv[2])

out = sys.argv[3]
tint = float(sys.argv[4])
tadv = float(sys.argv[5])
bw = float(sys.argv[6])

obs = im.observe(arr, tint, tadv, 0, 24.0, bw, ampcal=False, phasecal=False, sgrscat=False)
obs.save_txt(out)
