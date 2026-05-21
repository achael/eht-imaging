"""
.. module:: ehtim
    :platform: Unix
    :synopsis: EHT Imaging Utilities

.. moduleauthor:: Andrew Chael (achael@outlook.com)

"""

# Use a non-colliding local name: the bare `warnings` name shadows the
# `ehtim.warnings` submodule attribute and breaks `import ehtim.warnings as ehw`.
import warnings as _stdlib_warnings
from builtins import object, range, str

import ehtim.array
import ehtim.calibrating
import ehtim.caltable
import ehtim.features
import ehtim.image
import ehtim.imager
import ehtim.imaging
import ehtim.model
import ehtim.movie
import ehtim.obsdata
import ehtim.observing
import ehtim.parloop
import ehtim.plotting
import ehtim.survey
import ehtim.vex
from ehtim.calibrating import network_cal, pol_cal, self_cal
from ehtim.calibrating.network_cal import network_cal as netcal
from ehtim.calibrating.pol_cal import *
from ehtim.calibrating.pol_cal_new import *
from ehtim.calibrating.self_cal import self_cal as selfcal
from ehtim.const_def import *
from ehtim.features import rex
from ehtim.modeling.modeling_utils import modeler_func
from ehtim.plotting import comp_plots, comparisons
from ehtim.plotting.comp_plots import *
from ehtim.plotting.comparisons import *
from ehtim.plotting.summary_plots import *

_stdlib_warnings.filterwarnings(
    "ignore", message="numpy.dtype size changed, may indicate binary incompatibility.")

try:
    from importlib.metadata import PackageNotFoundError, metadata
    _meta = metadata("ehtim")
    __version__ = _meta["Version"]
    __author__  = _meta["Author-email"]
    __license__ = _meta["License"]
    __summary__ = _meta["Summary"]
    print("Welcome to eht-imaging! v", __version__, '\n')
except PackageNotFoundError:
    __version__ = "unknown"
    __author__  = "unknown"
    __license__ = "unknown"
    __summary__ = "unknown"
    print("Welcome to eht-imaging!\n")


def logo():
    for line in BHIMAGE:
        print(line)


def eht():
    for line in EHTIMAGE:
        print(line)
