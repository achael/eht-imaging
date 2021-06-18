"""
.. module:: ehtim
    :platform: Unix
    :synopsis: EHT Imaging Utilities

.. moduleauthor:: Andrew Chael (achael@cfa.harvard.edu)

"""
from __future__ import division
from __future__ import print_function

from builtins import str
from builtins import range
from builtins import object

import ehtim.observing
from ehtim.const_def import *
from ehtim.imaging.imager_utils import imager_func
from ehtim.modeling.modeling_utils import modeler_func
import ehtim.imaging
from ehtim.features import rex
import ehtim.features
from ehtim.plotting.summary_plots import *
from ehtim.plotting.comparisons import *
from ehtim.plotting.comp_plots import *
from ehtim.plotting import comparisons
from ehtim.plotting import comp_plots
import ehtim.plotting
from ehtim.calibrating.network_cal import network_cal as netcal
from ehtim.calibrating.self_cal import self_cal as selfcal
from ehtim.calibrating.pol_cal import *
from ehtim.calibrating import pol_cal
from ehtim.calibrating import network_cal
from ehtim.calibrating import self_cal
import ehtim.calibrating
import ehtim.parloop
import ehtim.caltable
import ehtim.vex
import ehtim.imager
import ehtim.obsdata
import ehtim.array
import ehtim.movie
import ehtim.image
import ehtim.model


import warnings
warnings.filterwarnings(
    "ignore", message="numpy.dtype size changed, may indicate binary incompatibility.")

# necessary to prevent hangs from astropy iers bug in astropy v 2.0.8
#from astropy.utils import iers
#iers.conf.auto_download = False

try:
    import pkg_resources
    version = pkg_resources.get_distribution("ehtim").version
    print("Welcome to eht-imaging! v", version,'\n')
except:
    print("Welcome to eht-imaging!\n")


def logo():
    for line in BHIMAGE:
        print(line)


def eht():
    for line in EHTIMAGE:
        print(line)
