"""
.. module:: ehtim
    :platform: Unix
    :synopsis: EHT Imaging Utilities

.. moduleauthor:: Andrew Chael (achael@outlook.com)

"""
# The import order below is constrained by circular dependencies between ehtim
# submodules; alphabetical sorting breaks `import ehtim`, so isort is disabled.
# The const_def star import re-exports its public constants (const_def.__all__)
# into the top-level namespace, so new constants propagate automatically.
# isort: off
import ehtim.observing as observing
from ehtim.const_def import *
import ehtim.const_def as _const_def
from ehtim.modeling.modeling_utils import modeler_func
import ehtim.imaging as imaging
from ehtim.backends import backend, get_backend, set_backend
from ehtim.features import rex
import ehtim.features as features
from ehtim.plotting.summary_plots import (
    imgsum,
    imgsum_pol,
)
from ehtim.plotting.comparisons import (
    change_cut_off,
    generate_consistency_plot,
    get_psize_fov,
    image_agreements,
    image_consistency,
)
from ehtim.plotting.comp_plots import (
    merge_obs,
    plot_bl_compare,
    plot_bl_obs_compare,
    plot_bl_obs_im_compare,
    plot_camp_compare,
    plot_camp_obs_compare,
    plot_camp_obs_im_compare,
    plot_cphase_compare,
    plot_cphase_obs_compare,
    plot_cphase_obs_im_compare,
    plotall_compare,
    plotall_obs_compare,
    plotall_obs_im_compare,
    plotall_obs_im_cphases,
    prep_plot_lists,
)
from ehtim.plotting import comparisons
from ehtim.plotting import comp_plots
import ehtim.plotting as plotting
from ehtim.calibrating.network_cal import network_cal as netcal
from ehtim.calibrating.self_cal import self_cal as selfcal
# TODO: replace pol_cal with pol_cal_new (validated D-term cal) in the near future.
from ehtim.calibrating.pol_cal import (
    leakage_cal,
    plot_leakage,
)
from ehtim.calibrating.pol_cal_new import (
    leakage_cal_new,
)
from ehtim.calibrating import pol_cal
from ehtim.calibrating import network_cal
from ehtim.calibrating import self_cal
import ehtim.calibrating as calibrating
import ehtim.parloop as parloop
import ehtim.caltable as caltable
import ehtim.vex as vex
import ehtim.imager as imager
import ehtim.obsdata as obsdata
import ehtim.array as array
import ehtim.movie as movie
import ehtim.image as image
import ehtim.model as model
import ehtim.survey as survey


# Use a non-colliding local name: the bare `warnings` name shadows the
# `ehtim.warnings` submodule attribute and breaks `import ehtim.warnings as ehw`.
import warnings as _stdlib_warnings
# isort: on

# const_def constants are appended from const_def.__all__ so they stay in one place.
__all__ = [
    "backend", "get_backend", "set_backend",
    "modeler_func", "rex", "logo", "eht",
    "imgsum", "imgsum_pol",
    "change_cut_off", "generate_consistency_plot", "get_psize_fov",
    "image_agreements", "image_consistency",
    "merge_obs",
    "plot_bl_compare", "plot_bl_obs_compare", "plot_bl_obs_im_compare",
    "plot_camp_compare", "plot_camp_obs_compare", "plot_camp_obs_im_compare",
    "plot_cphase_compare", "plot_cphase_obs_compare", "plot_cphase_obs_im_compare",
    "plotall_compare", "plotall_obs_compare", "plotall_obs_im_compare",
    "plotall_obs_im_cphases", "prep_plot_lists",
    "netcal", "selfcal", "network_cal", "self_cal",
    "leakage_cal", "plot_leakage", "leakage_cal_new", "pol_cal",
    "array", "calibrating", "caltable", "comp_plots", "comparisons",
    "features", "image", "imager", "imaging", "model", "movie",
    "obsdata", "observing", "parloop", "plotting", "survey", "vex",
] + list(_const_def.__all__)

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
