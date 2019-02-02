# comp_plots.py
# Make data plots with multiple observations,images etc.
#
#    Copyright (C) 2018 Andrew Chael
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.

from builtins import range
import numpy as np
import numpy.matlib as matlib
import matplotlib as mpl
import matplotlib.pyplot as plt
import itertools as it
import copy

from ehtim.const_def import *
from ehtim.observing.obs_helpers import *
from ehtim.obsdata import merge_obs

COLORLIST = SCOLORS

##################################################################################################
# Plotters
##################################################################################################


def plotall_compare(obslist, imlist, field1, field2,
                    conj=False, debias=True, sgrscat=False,
                    ang_unit='deg', timetype='UTC', ttype='nfft',
                    axis=False, rangex=False, rangey=False, snrcut=0., 
                    clist=COLORLIST, legendlabels=None, markersize=MARKERSIZE,
                    export_pdf="", grid=False, ebar=True,
                    axislabels=True, legend=True, show=True):

    """Plot data from observations compared to ground truth from an image on the same axes.

       Args:
           obslist (list): list of observations to plot
           imlist (list): list of ground truth images to compare to
           field1 (str): x-axis field (from FIELDS)
           field2 (str): y-axis field (from FIELDS)

           conj (bool): Plot conjuage baseline data points if True
           debias (bool): If True, debias amplitudes.
           sgrscat (bool): if True, the visibilites will be blurred by the Sgr A* scattering kernel

           ang_unit (str): phase unit 'deg' or 'rad'
           timetype (str): 'GMST' or 'UTC'
           ttype (str): if "fast" or "nfft" use FFT to produce visibilities. Else "direct" for DTFT

           axis (matplotlib.axes.Axes): add plot to this axis
           rangex (list): [xmin, xmax] x-axis limits
           rangey (list): [ymin, ymax] y-axis limits
           clist (list): list of colors scatterplot points
           legendlabels (list): list of labels of the same length of obslist or imlist
           markersize (int): size of plot markers
           export_pdf (str): path to pdf file to save figure
           grid (bool): Plot gridlines if True
           ebar (bool): Plot error bars if True
           axislabels (bool): Show axis labels if True
           legend (bool): Show legend if True
           show (bool): Display the plot if true

       Returns:
           (matplotlib.axes.Axes): Axes object with data plot
    """
    (obslist_plot, clist_plot, legendlabels_plot, markers) = prep_plot_lists(obslist, imlist, clist=clist,
                                                                             legendlabels=legendlabels, sgrscat=sgrscat, ttype=ttype)

    for i in range(len(obslist_plot)):
        obs = obslist_plot[i]
        axis = obs.plotall(field1, field2,
                           conj=conj, debias=debias,
                           ang_unit=ang_unit, timetype=timetype,
                           axis=axis, rangex=rangex, rangey=rangey,
                           grid=grid,ebar=ebar,axislabels=axislabels,
                           show=False, tag_bl=False, legend=False, snrcut=snrcut,
                           label=legendlabels_plot[i], color=clist_plot[i%len(clist_plot)],
                           marker=markers[i], markersize=markersize)

    if legend:
        plt.legend()
    if grid:
        axis.grid()
    if show:
        plt.show(block=False)

    if export_pdf != "":
        plt.savefig(export_pdf, bbox_inches='tight', pad_inches = 0)

    return axis

def plot_bl_compare(obslist, imlist, site1, site2, field,
                    debias=True, sgrscat=False,
                    ang_unit='deg', timetype='UTC', ttype='nfft',
                    axis=False, rangex=False, rangey=False, snrcut=0.,
                    clist=COLORLIST, legendlabels=None, markersize=MARKERSIZE,
                    export_pdf="", grid=False, ebar=True,
                    axislabels=True, legend=True, show=True):

    """Plot data from multiple observations vs time on a single baseline on the same axes.

       Args:
           obslist (list): list of observations to plot
           imlist (list): list of ground truth images  to compare to
           site1 (str): station 1 name
           site2 (str): station 2 name
           field (str): y-axis field (from FIELDS)

           debias (bool): If True, debias amplitudes.
           sgrscat (bool): if True, the visibilites will be blurred by the Sgr A* scattering kernel

           ang_unit (str): phase unit 'deg' or 'rad'
           timetype (str): 'GMST' or 'UTC'
           ttype (str): if "fast" or "nfft" use FFT to produce visibilities. Else "direct" for DTFT
           snrcut (float): a  snr cutoff

           axis (matplotlib.axes.Axes): add plot to this axis
           rangex (list): [xmin, xmax] x-axis limits
           rangey (list): [ymin, ymax] y-axis limits
           clist (list): list of colors scatterplot points
           legendlabels (list): list of labels of the same length of obslist or imlist
           markersize (int): size of plot markers
           export_pdf (str): path to pdf file to save figure
           grid (bool): Plot gridlines if True
           ebar (bool): Plot error bars if True
           axislabels (bool): Show axis labels if True
           legend (bool): Show legend if True
           show (bool): Display the plot if true

       Returns:
           (matplotlib.axes.Axes): Axes object with data plot

    """
    (obslist_plot, clist_plot, legendlabels_plot, markers) = prep_plot_lists(obslist, imlist, clist=clist,
                                                                             legendlabels=legendlabels, sgrscat=sgrscat, ttype=ttype)

    for i in range(len(obslist_plot)):
        obs = obslist_plot[i]
        axis = obs.plot_bl(site1, site2, field,
                           debias=debias, ang_unit=ang_unit, timetype=timetype,
                           axis=axis, rangex=rangex, rangey=rangey,
                           grid=grid,ebar=ebar,axislabels=axislabels,
                           show=False, legend=False, snrcut=snrcut,
                           label=legendlabels_plot[i], color=clist_plot[i%len(clist_plot)],
                           marker=markers[i], markersize=markersize)
    if legend:
        plt.legend()
    if grid:
        axis.grid()
    if show:
        plt.show(block=False)

    if export_pdf != "":
        plt.savefig(export_pdf, bbox_inches='tight', pad_inches = 0)

    return axis

def plot_cphase_compare(obslist, imlist, site1, site2, site3,
                        vtype='vis', cphases=[],force_recompute=False,
                        ang_unit='deg', timetype='UTC', ttype='nfft',
                        axis=False, rangex=False, rangey=False, snrcut=0.,
                        clist=COLORLIST, legendlabels=None, markersize=MARKERSIZE,
                        export_pdf="", grid=False, ebar=True,
                        axislabels=True, legend=True, show=True):


    """Plot closure phase on a triangle compared to ground truth from an image on the same axes.

       Args:
           obslist (list): list of observations to plot
           imlist (list): list of ground truth images to compare to
           site1 (str): station 1 name
           site2 (str): station 2 name
           site3 (str): station 3 name

           vtype (str): The visibilty type ('vis','qvis','uvis','vvis','pvis') from which to assemble bispectra
           cphases (list): optionally pass in a list of cphases so they don't have to be recomputed
           force_recompute (bool): if True, recompute closure phases instead of using stored data

           ang_unit (str): phase unit 'deg' or 'rad'
           timetype (str): 'GMST' or 'UTC'
           ttype (str): if "fast" or "nfft" use FFT to produce visibilities. Else "direct" for DTFT
           snrcut (float): a  snr cutoff

           axis (matplotlib.axes.Axes): add plot to this axis
           rangex (list): [xmin, xmax] x-axis limits
           rangey (list): [ymin, ymax] y-axis limits
           clist (list): list of colors scatterplot points
           legendlabels (list): list of labels of the same length of obslist or imlist
           markersize (int): size of plot markers
           export_pdf (str): path to pdf file to save figure
           grid (bool): Plot gridlines if True
           ebar (bool): Plot error bars if True
           axislabels (bool): Show axis labels if True
           legend (bool): Show legend if True
           show (bool): Display the plot if true

       Returns:
           (matplotlib.axes.Axes): Axes object with data plot

    """
    try: len(obslist)
    except TypeError: obslist = [obslist]

    if len(cphases)==0:
        cphases = matlib.repmat([],len(obslist),1)

    if len(cphases) != len(obslist):
        raise Exception("cphases list must be same length as obslist!")

    cphases_back = []
    for i in range(len(obslist)):
        cphases_back.append(obslist[i].cphase)
        obslist[i].cphase=cphases[i]

    (obslist_plot, clist_plot, legendlabels_plot, markers) = prep_plot_lists(obslist, imlist, clist=clist,
                                                                             legendlabels=legendlabels, sgrscat=False, ttype=ttype)

    for i in range(len(obslist_plot)):
        obs = obslist_plot[i]
        axis = obs.plot_cphase(site1, site2, site3,
                               vtype=vtype, force_recompute=force_recompute,
                               ang_unit=ang_unit, timetype=timetype,
                               axis=axis, rangex=rangex, rangey=rangey,
                               grid=grid,ebar=ebar,axislabels=axislabels,
                               show=False, legend=False, snrcut=snrcut,
                               label=legendlabels_plot[i], color=clist_plot[i%len(clist_plot)],
                               marker=markers[i], markersize=markersize)

   # return to original cphase attribute
    for i in range(len(obslist)):
        obslist[i].cphase=cphases_back[i]

    if legend:
        plt.legend()
    if grid:
        axis.grid()
    if show:
        plt.show(block=False)
    if export_pdf != "":
        plt.savefig(export_pdf, bbox_inches='tight', pad_inches = 0)

    return axis

def plot_camp_compare(obslist, imlist, site1, site2, site3, site4,
                      vtype='vis', ctype='camp', camps=[], force_recompute=False,
                      debias=True, sgrscat=False, timetype='UTC', ttype='nfft',
                      axis=False, rangex=False, rangey=False, snrcut=0.,
                      clist=COLORLIST, legendlabels=None, markersize=MARKERSIZE,
                      export_pdf="", grid=False, ebar=True,
                      axislabels=True, legend=True, show=True):

    """Plot closure amplitude on a triangle vs time from multiple observations on the same axes.

       Args:
           obslist (list): list of observations to plot
           imlist (list): list of  ground truth images to compare to
           site1 (str): station 1 name
           site2 (str): station 2 name
           site3 (str): station 3 name
           site4 (str): station 4 name

           vtype (str): The visibilty type ('vis','qvis','uvis','vvis','pvis') from which to assemble bispectra
           ctype (str): The closure amplitude type ('camp' or 'logcamp')
           camps (list): optionally pass in a list of camp so they don't have to be recomputed
           force_recompute (bool): if True, recompute closure phases instead of using stored data

           debias (bool): If True, debias amplitudes.
           sgrscat (bool): if True, the visibilites will be blurred by the Sgr A* scattering kernel

           timetype (str): 'GMST' or 'UTC'
           ttype (str): if "fast" or "nfft" use FFT to produce visibilities. Else "direct" for DTFT
           snrcut (float): a  snr cutoff

           axis (matplotlib.axes.Axes): add plot to this axis
           rangex (list): [xmin, xmax] x-axis limits
           rangey (list): [ymin, ymax] y-axis limits
           clist (list): list of colors scatterplot points
           legendlabels (list): list of labels of the same length of obslist or imlist
           markersize (int): size of plot markers
           export_pdf (str): path to pdf file to save figure
           grid (bool): Plot gridlines if True
           ebar (bool): Plot error bars if True
           axislabels (bool): Show axis labels if True
           legend (bool): Show legend if True
           show (bool): Display the plot if true

       Returns:
           (matplotlib.axes.Axes): Axes object with data plot

    """

    try: len(obslist)
    except TypeError: obslist = [obslist]

    if len(camps)==0:
        camps = matlib.repmat([],len(obslist),1)

    if len(camps) != len(obslist):
        raise Exception("camps list must be same length as obslist!")

    camps_back = []
    for i in range(len(obslist)):
        if ctype=='camp':
            camps_back.append(obslist[i].camp)
            obslist[i].camp=camps[i]
        elif ctype=='logcamp':
            camps_back.append(obslist[i].logcamp)
            obslist[i].logcamp=camps[i]


    (obslist_plot, clist_plot, legendlabels_plot, markers) = prep_plot_lists(obslist, imlist, clist=clist,
                                                                             legendlabels=legendlabels, sgrscat=sgrscat, ttype=ttype)

    for i in range(len(obslist_plot)):
        obs = obslist_plot[i]
        axis = obs.plot_camp(site1, site2, site3, site4,
                               vtype=vtype, ctype=ctype, force_recompute=force_recompute,
                               debias=debias, timetype=timetype,
                               axis=axis, rangex=rangex, rangey=rangey,
                               grid=grid,ebar=ebar,axislabels=axislabels,
                               show=False, legend=False,snrcut=0.,
                               label=legendlabels_plot[i], color=clist_plot[i%len(clist_plot)],
                               marker=markers[i], markersize=markersize)

    for i in range(len(obslist)):
        if ctype=='camp':
            obslist[i].camp=camps_back[i]
        elif ctype=='logcamp':
            obslist[i].logcamp=camps_back[i]

    if legend:
        plt.legend()
    if grid:
        axis.grid()
    if show:
        plt.show(block=False)
    if export_pdf != "":
        plt.savefig(export_pdf, bbox_inches='tight', pad_inches = 0)

    return axis


##################################################################################################
# Aliases
##################################################################################################
def plotall_obs_compare(obslist, field1, field2, **kwargs):
    """Plot data from observations compared to ground truth from an image on the same axes.

       Args:
           obslist (list): list of observations to plot
           field1 (str): x-axis field (from FIELDS)
           field2 (str): y-axis field (from FIELDS)

           conj (bool): Plot conjuage baseline data points if True
           debias (bool): If True, debias amplitudes.
           ang_unit (str): phase unit 'deg' or 'rad'
           sgrscat (bool): if True, the visibilites will be blurred by the Sgr A* scattering kernel
           ttype (str): if "fast" or "nfft" use FFT to produce visibilities. Else "direct" for DTFT

           rangex (list): [xmin, xmax] x-axis limits
           rangey (list): [ymin, ymax] y-axis limits
           legendlabels (str): should be a list of labels of the same length of obslist or imlist
           snrcut (float): a  snr cutoff

           grid (bool): Plot gridlines if True
           ebar (bool): Plot error bars if True
           axislabels (bool): Show axis labels if True
           legend (bool): Show legend if True
           show (bool): Display the plot if true
           axis (matplotlib.axes.Axes): add plot to this axis

           clist (list): list of colors scatterplot points
           markersize (int): size of plot markers
           export_pdf (str): path to pdf file to save figure

       Returns:
           (matplotlib.axes.Axes): Axes object with data plot
    """
    axis = plotall_compare(obslist, [], field1, field2, **kwargs)
    return axis

def plotall_obs_im_compare(obslist, imlist, field1, field2, **kwargs):

    """Plot data from observations compared to ground truth from an image on the same axes.

       Args:
           obslist (list): list of observations to plot
           imlist (list): list of images to plot
           field1 (str): x-axis field (from FIELDS)
           field2 (str): y-axis field (from FIELDS)

           conj (bool): Plot conjuage baseline data points if True
           debias (bool): If True, debias amplitudes.
           ang_unit (str): phase unit 'deg' or 'rad'
           sgrscat (bool): if True, the visibilites will be blurred by the Sgr A* scattering kernel
           ttype (str): if "fast" or "nfft" use FFT to produce visibilities. Else "direct" for DTFT

           rangex (list): [xmin, xmax] x-axis limits
           rangey (list): [ymin, ymax] y-axis limits
           legendlabels (str): should be a list of labels of the same length of obslist
           snrcut (float): a  snr cutoff

           grid (bool): Plot gridlines if True
           ebar (bool): Plot error bars if True
           axislabels (bool): Show axis labels if True
           legend (bool): Show legend if True
           show (bool): Display the plot if true
           axis (matplotlib.axes.Axes): add plot to this axis

           clist (list): list of colors scatterplot points
           markersize (int): size of plot markers
           export_pdf (str): path to pdf file to save figure

       Returns:
           (matplotlib.axes.Axes): Axes object with data plot
    """
    axis = plotall_compare(obslist, imlist, field1, field2, **kwargs)
    return axis


def plot_bl_obs_compare(obslist,  site1, site2, field, **kwargs):

    """Plot data from multiple observations vs time on a single baseline on the same axes.

       Args:
           obslist (list): list of observations to plot
           site1 (str): station 1 name
           site2 (str): station 2 name
           field (str): y-axis field (from FIELDS)

           debias (bool): If True and plotting vis amplitudes, debias them
           axislabels (bool): Show axis labels if True
           legendlabels (str): should be a list of labels of the same length of obslist or imlist
           ang_unit (str): phase unit 'deg' or 'rad'
           timetype (str): 'GMST' or 'UTC'
           sgrscat (bool): if True, the visibilites will be blurred by the Sgr A* scattering kernel
           snrcut (float): a  snr cutoff

           rangex (list): [xmin, xmax] x-axis (time) limits
           rangey (list): [ymin, ymax] y-axis limits

           legend (bool): Show legend if True
           grid (bool): Plot gridlines if True
           ebar (bool): Plot error bars if True
           show (bool): Display the plot if true
           axis (matplotlib.axes.Axes): add plot to this axis
           clist (list): list of color strings of scatterplot points
           export_pdf (str): path to pdf file to save figure

       Returns:
           (matplotlib.axes.Axes): Axes object with data plot

    """
    axis = plot_bl_compare(obslist, [], site1, site2, field, **kwargs)
    return axis


def plot_bl_obs_im_compare(obslist, imlist, site1, site2, field, **kwargs):

    """Plot data from multiple observations vs time on a single baseline on the same axes.

       Args:
           obslist (list): list of observations to plot
           imlist (list): list of ground truth images  to compare to
           site1 (str): station 1 name
           site2 (str): station 2 name
           field (str): y-axis field (from FIELDS)

           debias (bool): If True and plotting vis amplitudes, debias them
           axislabels (bool): Show axis labels if True
           legendlabels (str): should be a list of labels of the same length of obslist or imlist
           ang_unit (str): phase unit 'deg' or 'rad'
           timetype (str): 'GMST' or 'UTC'
           sgrscat (bool): if True, the visibilites will be blurred by the Sgr A* scattering kernel
           snrcut (float): a  snr cutoff

           rangex (list): [xmin, xmax] x-axis (time) limits
           rangey (list): [ymin, ymax] y-axis limits

           legend (bool): Show legend if True
           grid (bool): Plot gridlines if True
           ebar (bool): Plot error bars if True
           show (bool): Display the plot if true
           axis (matplotlib.axes.Axes): add plot to this axis
           clist (list): list of color strings of scatterplot points
           export_pdf (str): path to pdf file to save figure

       Returns:
           (matplotlib.axes.Axes): Axes object with data plot

    """
    axis = plot_bl_compare(obslist, imlist, site1, site2, field, **kwargs)
    return axis

def plot_cphase_obs_compare(obslist,  site1, site2, site3, **kwargs):

    """Plot closure phase on a triangle vs time from multiple observations on the same axes.

       Args:
           obslist (list): list of observations to plot
           site1 (str): station 1 name
           site2 (str): station 2 name
           site3 (str): station 3 name

           vtype (str): The visibilty type ('vis','qvis','uvis','vvis','pvis') from which to assemble bispectra
           cphases (list): optionally pass in a list of cphases so they don't have to be recomputed
           force_recompute (bool): if True, recompute closure phases instead of using stored data

           ang_unit (str): phase unit 'deg' or 'rad'
           timetype (str): 'GMST' or 'UTC'
           ttype (str): if "fast" or "nfft" use FFT to produce visibilities. Else "direct" for DTFT
           snrcut (float): a  snr cutoff

           axis (matplotlib.axes.Axes): add plot to this axis
           rangex (list): [xmin, xmax] x-axis limits
           rangey (list): [ymin, ymax] y-axis limits
           clist (list): list of colors scatterplot points
           legendlabels (list): list of labels of the same length of obslist or imlist
           markersize (int): size of plot markers
           export_pdf (str): path to pdf file to save figure
           grid (bool): Plot gridlines if True
           ebar (bool): Plot error bars if True
           axislabels (bool): Show axis labels if True
           legend (bool): Show legend if True
           show (bool): Display the plot if true

       Returns:
           (matplotlib.axes.Axes): Axes object with data plot

    """
    axis = plot_cphase_compare(obslist, [], site1, site2, site3, **kwargs)
    return axis

def plot_cphase_obs_im_compare(obslist, imlist, site1, site2, site3, **kwargs):

    """Plot closure phase on a triangle vs time from multiple observations on the same axes.

       Args:
           obslist (list): list of observations to plot
           imlist (list): list of ground truth images to compare to
           site1 (str): station 1 name
           site2 (str): station 2 name
           site3 (str): station 3 name

           vtype (str): The visibilty type ('vis','qvis','uvis','vvis','pvis') from which to assemble bispectra
           cphases (list): optionally pass in a list of cphases so they don't have to be recomputed
           force_recompute (bool): if True, recompute closure phases instead of using stored data

           ang_unit (str): phase unit 'deg' or 'rad'
           timetype (str): 'GMST' or 'UTC'
           ttype (str): if "fast" or "nfft" use FFT to produce visibilities. Else "direct" for DTFT
           snrcut (float): a  snr cutoff

           axis (matplotlib.axes.Axes): add plot to this axis
           rangex (list): [xmin, xmax] x-axis limits
           rangey (list): [ymin, ymax] y-axis limits
           clist (list): list of colors scatterplot points
           legendlabels (list): list of labels of the same length of obslist or imlist
           markersize (int): size of plot markers
           export_pdf (str): path to pdf file to save figure
           grid (bool): Plot gridlines if True
           ebar (bool): Plot error bars if True
           axislabels (bool): Show axis labels if True
           legend (bool): Show legend if True
           show (bool): Display the plot if true

       Returns:
           (matplotlib.axes.Axes): Axes object with data plot

    """
    axis = plot_cphase_compare(obslist, imlist, site1, site2, site3, **kwargs)
    return axis


def plot_camp_obs_compare(obslist,  site1, site2, site3, site4, **kwargs):

    """Plot closure amplitude on a triangle vs time from multiple observations on the same axes.

       Args:
           obslist (list): list of observations to plot
           site1 (str): station 1 name
           site2 (str): station 2 name
           site3 (str): station 3 name
           site4 (str): station 4 name

           vtype (str): The visibilty type ('vis','qvis','uvis','vvis','pvis') from which to assemble bispectra
           ctype (str): The closure amplitude type ('camp' or 'logcamp')
           camps (list): optionally pass in a list of camps so they don't have to be recomputed
           force_recompute (bool): if True, recompute closure phases instead of using stored data

           debias (bool): If True, debias amplitudes.
           sgrscat (bool): if True, the visibilites will be blurred by the Sgr A* scattering kernel

           timetype (str): 'GMST' or 'UTC'
           ttype (str): if "fast" or "nfft" use FFT to produce visibilities. Else "direct" for DTFT
           snrcut (float): a  snr cutoff

           axis (matplotlib.axes.Axes): add plot to this axis
           rangex (list): [xmin, xmax] x-axis limits
           rangey (list): [ymin, ymax] y-axis limits
           clist (list): list of colors scatterplot points
           legendlabels (list): list of labels of the same length of obslist or imlist
           markersize (int): size of plot markers
           export_pdf (str): path to pdf file to save figure
           grid (bool): Plot gridlines if True
           ebar (bool): Plot error bars if True
           axislabels (bool): Show axis labels if True
           legend (bool): Show legend if True
           show (bool): Display the plot if true

       Returns:
           (matplotlib.axes.Axes): Axes object with data plot

    """
    axis = plot_camp_compare(obslist, [], site1, site2, site3, site4, **kwargs)
    return axis

def plot_camp_obs_im_compare(obslist,  imlist, site1, site2, site3, site4, **kwargs):

    """Plot closure amplitude on a triangle vs time from multiple observations on the same axes.

       Args:
           obslist (list): list of observations to plot
           image (Image): ground truth image to compare to
           site1 (str): station 1 name
           site2 (str): station 2 name
           site3 (str): station 3 name
           site4 (str): station 4 name

           vtype (str): The visibilty type ('vis','qvis','uvis','vvis','pvis') from which to assemble bispectra
           ctype (str): The closure amplitude type ('camp' or 'logcamp')
           camps (list): optionally pass in a list of camps so they don't have to be recomputed
           force_recompute (bool): if True, recompute closure phases instead of using stored data

           debias (bool): If True, debias amplitudes.
           sgrscat (bool): if True, the visibilites will be blurred by the Sgr A* scattering kernel

           timetype (str): 'GMST' or 'UTC'
           ttype (str): if "fast" or "nfft" use FFT to produce visibilities. Else "direct" for DTFT
           snrcut (float): a  snr cutoff

           axis (matplotlib.axes.Axes): add plot to this axis
           rangex (list): [xmin, xmax] x-axis limits
           rangey (list): [ymin, ymax] y-axis limits
           clist (list): list of colors scatterplot points
           legendlabels (list): list of labels of the same length of obslist or imlist
           markersize (int): size of plot markers
           export_pdf (str): path to pdf file to save figure
           grid (bool): Plot gridlines if True
           ebar (bool): Plot error bars if True
           axislabels (bool): Show axis labels if True
           legend (bool): Show legend if True
           show (bool): Display the plot if true

       Returns:
           (matplotlib.axes.Axes): Axes object with data plot

    """
    axis = plot_camp_compare(obslist, imlist, site1, site2, site3, site4, **kwargs)
    return axis

##################################################################################################
# Plotters: Compare Observations to Image
##################################################################################################
def plotall_obs_im_cphases(obs, imlist,
                           vtype='vis', ang_unit='deg', timetype='UTC',
                           ttype='nfft', sgrscat=False,
                           rangex=False, rangey=[-180,180],legend=False,legendlabels=None,
                           show=True, ebar=True,axislabels=False,print_chisqs=True,
                           display_mode='all'):
    """Plot all observation closure phases on  top of image ground truth values. Works with ONE obs and MULTIPLE images.

       Args:
           obs (Obsdata): observation to plot
           imlist (list): list of ground truth images to compare to

           vtype (str): The visibilty type ('vis','qvis','uvis','vvis','pvis') from which to assemble bispectra
           ang_unit (str): phase unit 'deg' or 'rad'
           timetype (str): 'GMST' or 'UTC'
           ttype (str): if "fast" or "nfft" use FFT to produce visibilities. Else "direct" for DTFT
           sgrscat (bool): if True, the visibilites will be blurred by the Sgr A* scattering kernel

           rangex (list): [xmin, xmax] x-axis (time) limits
           rangey (list): [ymin, ymax] y-axis (phase) limits

           show (bool): Display the plot if True
           ebar (bool): Plot error bars if True
           axislabels (bool): Show axis labels if True
           print_chisqs (bool): print individual chisqs if True

           display_mode (str): 'all' or 'individual' to plot a giant single plot or multiple small Jones

       Returns:
           (matplotlib.axes.Axes): Axes object with data plot

    """

    try: len(imlist)
    except TypeError: imlist = [imlist]

    # get closure triangle combinations
    sites = []
    for i in range(0, len(obs.tarr)):
        sites.append(obs.tarr[i][0])
    uniqueclosure_tri = list(it.combinations(sites,3))

    # generate data
    cphases_obs = obs.c_phases(mode='all', count='max', vtype=vtype)
    obs_all = [obs]
    cphases_all = [cphases_obs]
    for image in imlist:
        obs_model = image.observe_same(obs, sgrscat=sgrscat, add_th_noise=False, ttype=ttype)
        cphases_model = obs_model.c_phases(mode='all', count='max', vtype=vtype)
        obs_all.append(obs_model)
        cphases_all.append(cphases_model)

    # display  as individual plots or as a huge sheet
    if display_mode=='individual':
        show=True
    else:
        nplots = len(uniqueclosure_tri)
        ncols = 4
        nrows = nplots / ncols
        show=False
        fig = plt.figure(figsize=(nrows*20, 40))

    # plot closure phases
    print("\n")

    nplot = 0
    for c in range(0, len(uniqueclosure_tri)):
        cphases_obs_tri = obs.cphase_tri(uniqueclosure_tri[c][0], uniqueclosure_tri[c][1], uniqueclosure_tri[c][2],
                                     vtype=vtype, ang_unit='deg', cphases=cphases_obs)

        if len(cphases_obs_tri)>0:
            if print_chisqs:
                printstr = "%s %s %s :" % (uniqueclosure_tri[c][0], uniqueclosure_tri[c][1], uniqueclosure_tri[c][2])
                for i in range(1,len(obs_all)):
                    cphases_model_tri = obs_all[i].cphase_tri(uniqueclosure_tri[c][0], uniqueclosure_tri[c][1], uniqueclosure_tri[c][2],
                                                 vtype=vtype, ang_unit='deg', cphases=cphases_all[i])
                    chisq_tri= np.sum((1.0 - np.cos(cphases_obs_tri['cphase']*DEGREE-cphases_model_tri['cphase']*DEGREE))/
                                      ((cphases_obs_tri['sigmacp']*DEGREE)**2))
                    chisq_tri *= (2.0/len(cphases_obs_tri))
                    printstr += " chisq%i: %0.2f" % (i, chisq_tri)
                print(printstr)

            if display_mode=='individual':
                ax=False
                axislabels=axislabels
            else:
                ax = plt.subplot2grid((nrows,ncols), (nplot/ncols, nplot%ncols), fig=fig)
                axislabels=False

            f = plot_cphase_obs_compare(obs_all,
                                        uniqueclosure_tri[c][0], uniqueclosure_tri[c][1], uniqueclosure_tri[c][2],
                                        vtype=vtype, rangex=rangex, rangey=rangey, ebar=ebar, show=show, legend=legend, legendlabels=legendlabels,
                                        cphases=cphases_all, axis=ax, axislabels=axislabels)
            nplot += 1

    if display_mode!='individual':
        plt.ion()
        f = fig
        f.subplots_adjust(wspace=0.1, hspace=0.5)
        f.show()

    return f

##################################################################################################
# Misc
##################################################################################################

def prep_plot_lists(obslist, imlist, clist=SCOLORS, legendlabels=None, sgrscat=False, ttype='nfft'):
    """Return observation, color, marker, legend lists for comp plots"""

    if imlist is None or imlist==False:
        imlist = []

    try: len(obslist)
    except TypeError: obslist = [obslist]

    try: len(imlist)
    except TypeError: imlist = [imlist]

    if not((len(imlist)==len(obslist)) or len(imlist)<=1 or len(obslist)<=1):
        raise Exception("imlist and obslist must be the same length, or either must have length 1")

    if not (legendlabels is None) and (len(legendlabels)!=max(len(imlist),len(obslist))):
        raise Exception("legendlabels should be the same length of the longer of imlist, obslist!")

    if legendlabels is None:
        legendlabels = [str(i+1) for i in range(max(len(imlist),len(obslist)))]

    obslist_plot = []
    clist_plot = copy.copy(clist)
    legendlabels_plot = copy.copy(legendlabels)

    #one image, multiple observations
    if len(imlist)==0:
        markers =  []
        for i in range(len(obslist)):
            obslist_plot.append(obslist[i])
            markers.append('o')

    elif len(imlist)==1 and len(obslist)>1:
        obslist_true=[]
        markers =  ['s']
        clist_plot = ['k']
        for i in range(len(obslist)):
            obslist_plot.append(obslist[i])
            obstrue = imlist[0].observe_same(obslist[i], sgrscat=sgrscat, add_th_noise=False, ttype=ttype)
            for sigma_type in obstrue.data.dtype.names[-4:]:
                obstrue.data[sigma_type] *= 0
            obslist_true.append(obstrue)
            markers.append('o')
            clist_plot.append(clist[i])

        obstrue = merge_obs(obslist_true)
        obslist_plot.insert(0, obstrue)
        legendlabels_plot.insert(0,'Image')

    #one observation, multiple images
    elif len(obslist)==1 and len(imlist)>1:
        obslist_plot.append(obslist[0])
        markers =  ['o']
        for i in range(len(imlist)):
            obstrue = imlist[i].observe_same(obslist[0], sgrscat=sgrscat, add_th_noise=False, ttype=ttype)
            for sigma_type in obstrue.data.dtype.names[-4:]:
                obstrue.data[sigma_type] *= 0
            obslist_plot.append(obstrue)
            markers.append('s')

        clist_plot.insert(0,'k')
        legendlabels_plot.insert(0,'Observation')

    #same number of images and observations
    elif len(obslist)==1 and len(imlist)==1:
        obslist_plot.append(obslist[0])

        obstrue = imlist[0].observe_same(obslist[0], sgrscat=sgrscat, add_th_noise=False, ttype=ttype)
        for sigma_type in obstrue.data.dtype.names[-4:]:
            obstrue.data[sigma_type] *= 0
        obslist_plot.append(obstrue)

        markers =  ['o','s']
        clist_plot = ['k',clist[0]]
        legendlabels_plot = [legendlabels[0]+'_obs', legendlabels[0]+'_im']

    else:
        markers = []
        legendlabels_plot = []
        clist_plot = []
        for i in range(len(obslist)):
            obstrue = imlist[i].observe_same(obslist[i], sgrscat=sgrscat, add_th_noise=False, ttype=ttype)
            for sigma_type in obstrue.data.dtype.names[-4:]:
                obstrue.data[sigma_type] *= 0
            obslist_plot.append(obstrue)
            clist_plot.append(clist[i])
            legendlabels_plot.append(legendlabels[i]+'_im')
            markers.append('s')

            obslist_plot.append(obslist[i])
            clist_plot.append(clist[i])
            legendlabels_plot.append(legendlabels[i]+'_obs')
            markers.append('o')

    if len(obslist_plot) > len(clist):
        Exception("More observations than colors -- Add more colors to clist!")

    return (obslist_plot, clist_plot, legendlabels_plot, markers)
