# Andrew Chael, 10/15/2016

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
import matplotlib.pyplot as plt
import itertools as it

from ehtim.const_def import *
from ehtim.observing.obs_helpers import *

COLORLIST = ['b','m','g','c','y','k','r']

##################################################################################################
# Plotters: Compare Observations
##################################################################################################
def plotall_obs_compare(obslist, field1, field2, rangex=False, rangey=False, conj=False,
                        clist=COLORLIST, ebar=True, debias=True, ang_unit='deg', export_pdf="", axis=False, show=True):
        """Make a scatter plot for multiple observations of 2 real baseline observation fields in (FIELDS) with error bars.

           Args:
               obslist (list): list of observations to plot
               field1 (str): x-axis field (from FIELDS)
               field2 (str): y-axis field (from FIELDS)

               conj (bool): Plot conjuage baseline data points if True
               debias (bool): If True, debias amplitudes.
               ang_unit (str): phase unit 'deg' or 'rad'
               sgrscat (bool): if True, the visibilites will be blurred by the Sgr A* scattering kernel

               rangex (list): [xmin, xmax] x-axis limits
               rangey (list): [ymin, ymax] y-axis limits

               ebar (bool): Plot error bars if True
               show (bool): Display the plot if true
               axis (matplotlib.axes.Axes): add plot to this axis
               clist (list): list of color strings of scatterplot points
               export_pdf (str): path to pdf file to save figure

           Returns:
               (matplotlib.axes.Axes): Axes object with data plot
    """

    try: len(obslist)
    except TypeError: obslist = [obslist]

    if len(obslist) > len(clist):
        Exception("More observations than colors -- Add more colors to clist!")

    for i in range(len(obslist)):
        obs = obslist[i]
        axis = obs.plotall(field1, field2, rangex=rangex, rangey=rangey, debias=debias, ang_unit=ang_unit,
                           conj=conj, show=False, axis=axis, color=clist[i%len(clist)], ebar=ebar)

    if show:
        plt.show(block=False)

    if export_pdf != "":
        plt.savefig(export_pdf, bbox_inches='tight', pad_inches = 0)

    return axis

def plot_bl_obs_compare(obslist,  site1, site2, field,
                        rangex=False, rangey=False, show=True, clist=COLORLIST,
                        timetype=False, ebar=True, debias=True, export_pdf="", axis=False):
    """Plot data from multiple observations vs time on a single baseline on the same axes.

           Args:
               obslist (list): list of observations to plot
               site1 (str): station 1 name
               site2 (str): station 2 name
               field (str): y-axis field (from FIELDS)

               debias (bool): If True and plotting vis amplitudes, debias them
               ang_unit (str): phase unit 'deg' or 'rad'
               timetype (str): 'GMST' or 'UTC'
               sgrscat (bool): if True, the visibilites will be blurred by the Sgr A* scattering kernel

               rangex (list): [xmin, xmax] x-axis (time) limits
               rangey (list): [ymin, ymax] y-axis limits

               ebar (bool): Plot error bars if True
               show (bool): Display the plot if true
               axis (matplotlib.axes.Axes): add plot to this axis
               clist (list): list of color strings of scatterplot points
               export_pdf (str): path to pdf file to save figure

           Returns:
               (matplotlib.axes.Axes): Axes object with data plot

        """

    try: len(obslist)
    except TypeError: obslist = [obslist]

    if len(obslist) > len(clist):
        Exception("More observations than colors -- Add more colors to clist!")

    for i in range(len(obslist)):
        obs = obslist[i]
        axis = obs.plot_bl(site1, site2, field, rangex=rangex, rangey=rangey, show=False, axis=axis, color=clist[i%len(clist)], timetype=timetype, ebar=ebar, debias=debias)


    if show:
        plt.show(block=False)

    if export_pdf != "":
        plt.savefig(export_pdf, bbox_inches='tight', pad_inches = 0)

    return axis



def plot_cphase_obs_compare(obslist,  site1, site2, site3, rangex=False, rangey=False, show=True,
                            clist=COLORLIST, ang_unit='deg', vtype='vis', timetype=False,
                            ebar=True, cphases=[], export_pdf="",axis=False, labels=True,
                            force_recompute=False):

    """Plot closure phase on a triangle vs time from multiple observations on the same axes.

           Args:
               obslist (list): list of observations to plot
               site1 (str): station 1 name
               site2 (str): station 2 name
               site3 (str): station 3 name

               vtype (str): The visibilty type ('vis','qvis','uvis','vvis','pvis') from which to assemble bispectra
               ang_unit (str): phase unit 'deg' or 'rad'
               timetype (str): 'GMST' or 'UTC'
               sgrscat (bool): if True, the visibilites will be blurred by the Sgr A* scattering kernel

               rangex (list): [xmin, xmax] x-axis (time) limits
               rangey (list): [ymin, ymax] y-axis (phase) limits

               ebar (bool): Plot error bars if True
               labels (bool): Show axis labels if True
               show (bool): Display the plot if true
               axis (matplotlib.axes.Axes): add plot to this axis
               clist (list): List of color strings of scatterplot points
               export_pdf (str): path to pdf file to save figure

               cphases (list): optionally pass in the time-sorted cphases so they don't have to be recomputed
               force_recompute (bool): if True, recompute closure phases instead of using stored data

           Returns:
               (matplotlib.axes.Axes): Axes object with data plot

        """

    try: len(obslist)
    except TypeError: obslist = [obslist]

    if len(obslist) > len(clist):
        Exception("More observations than colors -- Add more colors to clist!")

    if len(cphases)==0:
        cphases = np.matlib.repmat([],len(obslist),1)

    for i in range(len(obslist)):
        obs = obslist[i]

        axis = obs.plot_cphase(site1, site2, site3, rangex=rangex, rangey=rangey, show=False, axis=axis, color=clist[i%len(clist)],
                               ang_unit=ang_unit, timetype=timetype, vtype=vtype, ebar=ebar, cphases=cphases[i],labels=labels)


    if show:
        plt.show(block=False)

    if export_pdf != "":
        plt.savefig(export_pdf, bbox_inches='tight', pad_inches = 0)

    return axis



def plot_camp_obs_compare(obslist,  site1, site2, site3, site4, rangex=False, rangey=False,
                          show=True, clist=COLORLIST, vtype='vis', ctype='camp', debias=True,
                          timetype=False, ebar=True, camps=[], export_pdf="", axis=False,
                          force_recompute=False):

    """Plot closure amplitude on a triangle vs time from multiple observations on the same axes.

           Args:
               obslist (list): list of observations to plot
               site1 (str): station 1 name
               site2 (str): station 2 name
               site3 (str): station 3 name
               site4 (str): station 4 name

               vtype (str): The visibilty type ('vis','qvis','uvis','vvis','pvis') from which to assemble closure amplitudes
               ctype (str): The closure amplitude type ('camp' or 'logcamp')
               debias (bool): If True, debias the closure amplitude - the individual visibility amplitudes are always debiased.
               timetype (str): 'GMST' or 'UTC'
               sgrscat (bool): if True, the visibilites will be blurred by the Sgr A* scattering kernel

               rangex (list): [xmin, xmax] x-axis (time) limits
               rangey (list): [ymin, ymax] y-axis (phase) limits

               ebar (bool): Plot error bars if True
               show (bool): Display the plot if true
               axis (matplotlib.axes.Axes): add plot to this axis
               clist (list): List of color strings of scatterplot points
               export_pdf (str): path to pdf file to save figure

               camps (list): optionally pass in the time-sorted camps so they don't have to be recomputed
               force_recompute (bool): if True, recompute closure amplitudes instead of using stored  data
           Returns:
               (matplotlib.axes.Axes): Axes object with data plot

        """

    try: len(obslist)
    except TypeError: obslist = [obslist]

    if len(obslist) > len(clist):
        Exception("More observations than colors -- Add more colors to clist!")

    if len(camps)==0:
        cphases = np.matlib.repmat([],len(obslist),1)

    for i in range(len(obslist)):
        obs = obslist[i]

        axis = obs.plot_camp(site1, site2, site3, site4, rangex=rangex, rangey=rangey, show=False, axis=axis, color=clist[i%len(clist)],
                               timetype=timetype, vtype=vtype, ctype=ctype, debias=debias, ebar=ebar, camps=camps[i])



    if show:
        plt.show(block=False)

    if export_pdf != "":
        plt.savefig(export_pdf, bbox_inches='tight', pad_inches = 0)

    return axis

##################################################################################################
# Plotters: Compare Observations to Image
##################################################################################################
def plotall_obs_im_compare(obslist, image, field1, field2,
                           ttype='direct', sgrscat=False,
                           rangex=False, rangey=False, conj=False, clist=COLORLIST, ebar=True,
                           axis=False,show=True, export_pdf=""):

    """Plot data from observations compared to ground truth from an image on the same axes.

           Args:
               obslist (list): list of observations to plot
               image (Image): ground truth image to compare to
               field1 (str): x-axis field (from FIELDS)
               field2 (str): y-axis field (from FIELDS)

               conj (bool): Plot conjuage baseline data points if True
               debias (bool): If True, debias amplitudes.
               ang_unit (str): phase unit 'deg' or 'rad'
               sgrscat (bool): if True, the visibilites will be blurred by the Sgr A* scattering kernel
               ttype (str): if "fast" or "nfft" use FFT to produce visibilities. Else "direct" for DTFT

               rangex (list): [xmin, xmax] x-axis limits
               rangey (list): [ymin, ymax] y-axis limits

               ebar (bool): Plot error bars if True
               show (bool): Display the plot if true
               axis (matplotlib.axes.Axes): add plot to this axis
               clist (list): list of color strings of scatterplot points
               export_pdf (str): path to pdf file to save figure

           Returns:
               (matplotlib.axes.Axes): Axes object with data plot
    """

    try: len(obslist)
    except TypeError: obslist = [obslist]

    for i in range(len(obslist)):
        obstrue = image.observe_same(obslist[i], sgrscat=sgrscat, add_th_noise=False, ttype=ttype)
        obstrue.data['sigma'] *= 0
        obslist.append(obstrue)

    if len(obslist) > len(clist):
        Exception("More observations than colors -- Add more colors to clist!")

    for i in range(len(obslist)):
        obs = obslist[i]
        axis = obs.plotall(field1, field2, rangex=rangex, rangey=rangey, conj=conj, show=False, axis=axis, color=clist[i%len(clist)], ebar=ebar)


    if show:
        plt.show(block=False)

    if export_pdf != "":
        plt.savefig(export_pdf, bbox_inches='tight', pad_inches = 0)

    return axis

def plot_bl_obs_im_compare(obslist, image, site1, site2, field, ttype='direct', sgrscat=False,
                           rangex=False, rangey=False, show=True, clist=COLORLIST,
                           timetype=False, ebar=True, debias=True, export_pdf="", axis=False):
    """Plot data vs time on a single baseline compared to ground truth from an image on the same axes.
           Args:
               obslist (list): list of observations to plot
               image (Image): ground truth image to compare to
               site1 (str): station 1 name
               site2 (str): station 2 name
               field (str): y-axis field (from FIELDS)

               debias (bool): If True and plotting vis amplitudes, debias them
               ang_unit (str): phase unit 'deg' or 'rad'
               timetype (str): 'GMST' or 'UTC'
               sgrscat (bool): if True, the visibilites will be blurred by the Sgr A* scattering kernel
               ttype (str): if "fast" or "nfft" use FFT to produce visibilities. Else "direct" for DTFT

               rangex (list): [xmin, xmax] x-axis (time) limits
               rangey (list): [ymin, ymax] y-axis limits

               ebar (bool): Plot error bars if True
               show (bool): Display the plot if true
               axis (matplotlib.axes.Axes): add plot to this axis
               clist (list): list of color strings of scatterplot points
               export_pdf (str): path to pdf file to save figure

           Returns:
               (matplotlib.axes.Axes): Axes object with data plot

        """


    try: len(obslist)

    except TypeError: obslist = [obslist]

    for i in range(len(obslist)):
        obstrue = image.observe_same(obslist[i], sgrscat=sgrscat, add_th_noise=False, ttype=ttype)
        obstrue.data['sigma'] *= 0
        obslist.append(obstrue)

    if len(obslist) > len(clist):
        Exception("More observations than colors -- Add more colors to clist!")

    for i in range(len(obslist)):
        obs = obslist[i]

        axis = obs.plot_bl(site1, site2, field, rangex=rangex, rangey=rangey, show=False, axis=axis, color=clist[i%len(clist)], timetype=timetype, ebar=ebar, debias=debias)


    if show:
        plt.show(block=False)

    if export_pdf != "":
        plt.savefig(export_pdf, bbox_inches='tight', pad_inches = 0)

    return axis



def plot_cphase_obs_im_compare(obslist, image, site1, site2, site3, ttype='direct', sgrscat=False,
                               rangex=False, rangey=False, show=True, clist=COLORLIST, ang_unit='deg',
                               vtype='vis', timetype=False, ebar=True, axis=False, labels=True, export_pdf=""):

    """Plot closure phase on a triangle compared to ground truth from an image on the same axes.

           Args:
               obslist (list): list of observations to plot
               image (Image): ground truth image to compare to
               site1 (str): station 1 name
               site2 (str): station 2 name
               site3 (str): station 3 name

               vtype (str): The visibilty type ('vis','qvis','uvis','vvis','pvis') from which to assemble bispectra
               ang_unit (str): phase unit 'deg' or 'rad'
               timetype (str): 'GMST' or 'UTC'
               sgrscat (bool): if True, the visibilites will be blurred by the Sgr A* scattering kernel
               ttype (str): if "fast" or "nfft" use FFT to produce visibilities. Else "direct" for DTFT

               rangex (list): [xmin, xmax] x-axis (time) limits
               rangey (list): [ymin, ymax] y-axis (phase) limits

               ebar (bool): Plot error bars if True
               labels (bool): Show axis labels if True
               show (bool): Display the plot if true
               axis (matplotlib.axes.Axes): add plot to this axis
               clist (list): List of color strings of scatterplot points
               export_pdf (str): path to pdf file to save figure

               cphases (list): optionally pass in the time-sorted cphases so they don't have to be recomputed
               force_recompute (bool): if True, recompute closure phases instead of using stored data

           Returns:
               (matplotlib.axes.Axes): Axes object with data plot

        """

    try: len(obslist)
    except TypeError: obslist = [obslist]

    for i in range(len(obslist)):
        obstrue = image.observe_same(obslist[i], sgrscat=sgrscat,add_th_noise=False, ttype=ttype)
        obstrue.data['sigma'] *= 0
        obslist.append(obstrue)

    if len(obslist) > len(clist):
        Exception("More observations than colors -- Add more colors to clist!")


    for i in range(len(obslist)):
        obs = obslist[i]

        axis = obs.plot_cphase(site1, site2, site3, rangex=rangex, rangey=rangey, show=False, axis=axis, color=clist[i%len(clist)],
                               ang_unit=ang_unit, timetype=timetype, vtype=vtype, ebar=ebar, labels=labels)

    if show:
        plt.show(block=False)

    if export_pdf != "":
        plt.savefig(export_pdf, bbox_inches='tight', pad_inches=0)

    return axis



def plot_camp_obs_im_compare(obslist, image, site1, site2, site3, site4, ttype='nfft', sgrscat=False,
                             rangex=False, rangey=False, show=True, clist=COLORLIST, vtype='vis', ctype='camp',
                              debias=True, timetype=False, axis=False, ebar=True, export_pdf=""):



    """Plot closure amplitude on a triangle vs time from multiple observations on the same axes.

           Args:
               obslist (list): list of observations to plot
               image (Image): ground truth image to compare to
               site1 (str): station 1 name
               site2 (str): station 2 name
               site3 (str): station 3 name
               site4 (str): station 4 name

               vtype (str): The visibilty type ('vis','qvis','uvis','vvis','pvis') from which to assemble closure amplitudes
               ctype (str): The closure amplitude type ('camp' or 'logcamp')
               debias (bool): If True, debias the closure amplitude - the individual visibility amplitudes are always debiased.
               timetype (str): 'GMST' or 'UTC'
               sgrscat (bool): if True, the visibilites will be blurred by the Sgr A* scattering kernel
               ttype (str): if "fast" or "nfft" use FFT to produce visibilities. Else "direct" for DTFT

               rangex (list): [xmin, xmax] x-axis (time) limits
               rangey (list): [ymin, ymax] y-axis (phase) limits

               ebar (bool): Plot error bars if True
               show (bool): Display the plot if true
               axis (matplotlib.axes.Axes): add plot to this axis
               clist (list): List of color strings of scatterplot points
               export_pdf (str): path to pdf file to save figure

               camps (list): optionally pass in the time-sorted camps so they don't have to be recomputed
               force_recompute (bool): if True, recompute closure amplitudes instead of using stored  data
           Returns:
               (matplotlib.axes.Axes): Axes object with data plot

        """

    try: len(obslist)
    except TypeError: obslist = [obslist]

    for i in range(len(obslist)):
        obstrue = image.observe_same(obslist[i], sgrscat=sgrscat, add_th_noise=False, ttype=ttype)
        obstrue.data['sigma'] *= 0
        obslist.append(obstrue)

    if len(obslist) > len(clist):
        Exception("More observations than colors -- Add more colors to clist!")

    for i in range(len(obslist)):
        obs = obslist[i]

        axis = obs.plot_camp(site1, site2, site3, site4, rangex=rangex, rangey=rangey, show=False, axis=axis, color=clist[i%len(clist)],
                               timetype=timetype, vtype=vtype, ctype=ctype, debias=debias, ebar=ebar)

    if show:
        plt.show(block=False)

    if export_pdf != "":
        plt.savefig(export_pdf, bbox_inches='tight', pad_inches = 0)

    return axis


def plotall_obs_im_cphases(obs, image, ttype='nfft', sgrscat=False,
                           rangex=False, rangey=[-180,180], show=True, ebar=True,
                           vtype='vis',ang_unit='deg', timetype='UTC',
                           display_mode='all', labels=False):
    """Plot all observation closure phases on  top of image ground truth values.

           Args:
               obs (Obsdata):  observation to plot
               image (Image): ground truth image to compare to

               vtype (str): The visibilty type ('vis','qvis','uvis','vvis','pvis') from which to assemble bispectra
               ang_unit (str): phase unit 'deg' or 'rad'
               timetype (str): 'GMST' or 'UTC'
               sgrscat (bool): if True, the visibilites will be blurred by the Sgr A* scattering kernel
               ttype (str): if "fast" or "nfft" use FFT to produce visibilities. Else "direct" for DTFT

               rangex (list): [xmin, xmax] x-axis (time) limits
               rangey (list): [ymin, ymax] y-axis (phase) limits

               ebar (bool): Plot error bars if True
               labels (bool): Show axis labels if True
               show (bool): Display the plot if true
               display_mode (str): 'all' or 'individual' to plot a giant single plot or multiple small Jones

               cphases (list): optionally pass in the time-sorted cphases so they don't have to be recomputed
               force_recompute (bool): if True, recompute closure phases instead of using stored data

           Returns:
               (matplotlib.axes.Axes): Axes object with data plot

        """

    # get closure triangle combinations
    sites = []
    for i in range(0, len(obs.tarr)):
        sites.append(obs.tarr[i][0])
    uniqueclosure_tri = list(it.combinations(sites,3))

    # generate data
    obs_model = image.observe_same(obs, sgrscat=sgrscat, add_th_noise=False, ttype=ttype)
    cphases_obs = obs.c_phases(mode='time', count='max', vtype=vtype)
    cphases_model = obs_model.c_phases(mode='time', count='max', vtype=vtype)

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
    print()

    nplot = 0
    for c in range(0, len(uniqueclosure_tri)):
        cphases_obs_tri = obs.cphase_tri(uniqueclosure_tri[c][0], uniqueclosure_tri[c][1], uniqueclosure_tri[c][2],
                                     vtype=vtype, ang_unit='deg', cphases=cphases_obs)

        if len(cphases_obs_tri)>0:
            cphases_model_tri = obs_model.cphase_tri(uniqueclosure_tri[c][0], uniqueclosure_tri[c][1], uniqueclosure_tri[c][2],
                                         vtype=vtype, ang_unit='deg', cphases=cphases_model)
            chisq_tri= np.sum((1.0 - np.cos(cphases_obs_tri['cphase']*DEGREE-cphases_model_tri['cphase']*DEGREE))/
                              ((cphases_obs_tri['sigmacp']*DEGREE)**2))
            chisq_tri *= (2.0/len(cphases_obs_tri))
            print ("%s %s %s : cphase_chisq: %0.2f" % (uniqueclosure_tri[c][0], uniqueclosure_tri[c][1], uniqueclosure_tri[c][2],chisq_tri))

            if display_mode=='individual':
                ax=False
                labels=True
            else:
                ax = plt.subplot2grid((nrows,ncols), (nplot/ncols, nplot%ncols), fig=fig)
                labels=False

            f = plot_cphase_obs_compare([obs, obs_model],
                                        uniqueclosure_tri[c][0], uniqueclosure_tri[c][1], uniqueclosure_tri[c][2],
                                        vtype=vtype, rangex=rangex, rangey=rangey, ebar=ebar, show=show,
                                        cphases=[cphases_obs, cphases_model], axis=ax, labels=labels)
            nplot += 1

    if display_mode!='individual':
        plt.ion()
        f = fig
        f.subplots_adjust(wspace=0.1, hspace=0.5)
        f.show()

    return f
