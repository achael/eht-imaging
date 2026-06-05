"""
.. module:: ehtim
    :platform: Unix
    :synopsis: EHT Imaging Utilities

.. moduleauthor:: Andrew Chael (achael@outlook.com)

"""
# The import order below is constrained by circular dependencies between ehtim
# submodules; alphabetical sorting breaks `import ehtim`, so isort is disabled.
# isort: off
import ehtim.observing as observing
from ehtim.const_def import (
    BHIMAGE,
    BOUNDS_ERROR,
    C,
    DEC_DEFAULT,
    DEC_M87,
    DEC_SGRA,
    DEGREE,
    DTAMP,
    DTARR,
    DTBIS,
    DTCAL,
    DTCAMP,
    DTCPHASE,
    DTCPHASEDIAG,
    DTERMPDEF,
    DTLOGCAMPDIAG,
    DTPOL_CIRC,
    DTPOL_STOKES,
    DTSCANS,
    EHTIMAGE,
    ELEV_HIGH,
    ELEV_LOW,
    EP,
    FFT_INTERP_DEFAULT,
    FFT_PAD_DEFAULT,
    FIELDS,
    FIELDS_AMPS,
    FIELDS_PHASE,
    FIELDS_SIGPHASE,
    FIELDS_SIGS,
    FIELDS_SNRS,
    FIELD_LABELS,
    FWHM_MAJ,
    FWHM_MIN,
    GAINPDEF,
    GRIDDER_CONV_FUNC_DEFAULT,
    GRIDDER_P_RAD_DEFAULT,
    HOUR,
    INTERP_DEFAULT,
    MJD_DEFAULT,
    NFFT_EPS_DEFAULT,
    NFFT_KERSIZE_DEFAULT,
    POLDICT_CIRC,
    POLDICT_STOKES,
    POS_ANG,
    PULSE_DEFAULT,
    RADPERAS,
    RADPERUAS,
    RA_DEFAULT,
    RA_M87,
    RA_SGRA,
    RF_DEFAULT,
    SCOLORS,
    SOURCE_DEFAULT,
    TAUDEF,
    amp_poldict,
    show_noblock,
    sig_poldict,
    trianglePulse2D,
    vis_poldict,
)
from ehtim.modeling.modeling_utils import modeler_func
import ehtim.imaging as imaging
from ehtim.backends import backend, get_backend, set_backend
from ehtim.features import rex
import ehtim.features as features
from ehtim.plotting.summary_plots import (
    FONTSIZE,
    HSPACE,
    MARGINS,
    MARKERSIZE,
    PROCESSES,
    WSPACE,
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
    COLORLIST,
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
from ehtim.calibrating.pol_cal import (
    leakage_cal,
    plot_leakage,
)
from ehtim.calibrating.pol_cal_new import (
    MAXIT,
    MAXLS,
    NHIST,
    STOP,
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


import warnings
# isort: on

__all__ = [
    "BHIMAGE",
    "BOUNDS_ERROR",
    "C",
    "COLORLIST",
    "DEC_DEFAULT",
    "DEC_M87",
    "DEC_SGRA",
    "DEGREE",
    "DTAMP",
    "DTARR",
    "DTBIS",
    "DTCAL",
    "DTCAMP",
    "DTCPHASE",
    "DTCPHASEDIAG",
    "DTERMPDEF",
    "DTLOGCAMPDIAG",
    "DTPOL_CIRC",
    "DTPOL_STOKES",
    "DTSCANS",
    "EHTIMAGE",
    "ELEV_HIGH",
    "ELEV_LOW",
    "EP",
    "FFT_INTERP_DEFAULT",
    "FFT_PAD_DEFAULT",
    "FIELDS",
    "FIELDS_AMPS",
    "FIELDS_PHASE",
    "FIELDS_SIGPHASE",
    "FIELDS_SIGS",
    "FIELDS_SNRS",
    "FIELD_LABELS",
    "FONTSIZE",
    "FWHM_MAJ",
    "FWHM_MIN",
    "GAINPDEF",
    "GRIDDER_CONV_FUNC_DEFAULT",
    "GRIDDER_P_RAD_DEFAULT",
    "HOUR",
    "HSPACE",
    "INTERP_DEFAULT",
    "MARGINS",
    "MARKERSIZE",
    "MAXIT",
    "MAXLS",
    "MJD_DEFAULT",
    "NFFT_EPS_DEFAULT",
    "NFFT_KERSIZE_DEFAULT",
    "NHIST",
    "POLDICT_CIRC",
    "POLDICT_STOKES",
    "POS_ANG",
    "PROCESSES",
    "PULSE_DEFAULT",
    "RADPERAS",
    "RADPERUAS",
    "RA_DEFAULT",
    "RA_M87",
    "RA_SGRA",
    "RF_DEFAULT",
    "SCOLORS",
    "SOURCE_DEFAULT",
    "STOP",
    "TAUDEF",
    "WSPACE",
    "amp_poldict",
    "array",
    "backend",
    "calibrating",
    "caltable",
    "change_cut_off",
    "comp_plots",
    "comparisons",
    "eht",
    "features",
    "generate_consistency_plot",
    "get_backend",
    "get_psize_fov",
    "image",
    "image_agreements",
    "image_consistency",
    "imager",
    "imaging",
    "imgsum",
    "imgsum_pol",
    "leakage_cal",
    "leakage_cal_new",
    "logo",
    "merge_obs",
    "model",
    "modeler_func",
    "movie",
    "netcal",
    "network_cal",
    "obsdata",
    "observing",
    "parloop",
    "plot_bl_compare",
    "plot_bl_obs_compare",
    "plot_bl_obs_im_compare",
    "plot_camp_compare",
    "plot_camp_obs_compare",
    "plot_camp_obs_im_compare",
    "plot_cphase_compare",
    "plot_cphase_obs_compare",
    "plot_cphase_obs_im_compare",
    "plot_leakage",
    "plotall_compare",
    "plotall_obs_compare",
    "plotall_obs_im_compare",
    "plotall_obs_im_cphases",
    "plotting",
    "pol_cal",
    "prep_plot_lists",
    "rex",
    "self_cal",
    "selfcal",
    "set_backend",
    "show_noblock",
    "sig_poldict",
    "survey",
    "trianglePulse2D",
    "vex",
    "vis_poldict",
]

warnings.filterwarnings(
    "ignore", message="numpy.dtype size changed, may indicate binary incompatibility.")

# necessary to prevent hangs from astropy iers bug in astropy v 2.0.8
#from astropy.utils import iers
#iers.conf.auto_download = False

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
