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

import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed, may indicate binary incompatibility.")

import ehtim.image
import ehtim.movie
import ehtim.array
import ehtim.obsdata
import ehtim.imager
import ehtim.vex
import ehtim.caltable
import ehtim.parloop

import ehtim.calibrating
from ehtim.calibrating import self_cal
from ehtim.calibrating import network_cal
from ehtim.calibrating import pol_cal
from ehtim.calibrating.pol_cal import *
from ehtim.calibrating.self_cal import self_cal as selfcal
from ehtim.calibrating.network_cal import network_cal as netcal

import ehtim.plotting
from ehtim.plotting import comp_plots
from ehtim.plotting import comparisons
from ehtim.plotting.comp_plots import *
from ehtim.plotting.comparisons import *
from ehtim.plotting.summary_plots import *

import ehtim.features
from ehtim.features import rex

import ehtim.imaging
from ehtim.imaging.imager_utils import imager_func

from ehtim.const_def import *

# necessary to prevent hangs from astropy iers bug in astropy v 2.0.8
#from astropy.utils import iers
#iers.conf.auto_download = False 

try:
    import pkg_resources
    version = pkg_resources.get_distribution("ehtim").version
    print("Welcome to eht-imaging! v ",version)
except:
    print("Welcome to eht-imaging!")


def logo():
    for line in BHIMAGE:
        print(line)

def eht():
    for line in EHTIMAGE:    
        print(line)
