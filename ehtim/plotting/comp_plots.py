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
def plotall_obs_compare(obslist, field1, field2, rangex=False, rangey=False, conj=False, show=True, clist=COLORLIST, ebar=True, export_pdf=""):
    """Plot data from multiple observations on the same axes.
        """
    
    try: len(obslist)
    except TypeError: obslist = [obslist]
    
    if len(obslist) > len(clist):
        Exception("More observations than colors -- Add more colors to clist!")
    
    axis = False
    for i in range(len(obslist)):
        obs = obslist[i]

        axis = obs.plotall(field1, field2, rangex=rangex, rangey=rangey, conj=conj, show=False, axis=axis, color=clist[i%len(clist)], ebar=ebar)

    if show:
        plt.show(block=False)

    if export_pdf != "":
        plt.savefig(export_pdf, bbox_inches='tight', pad_inches = 0)

    return axis

def plot_bl_obs_compare(obslist,  site1, site2, field, rangex=False, rangey=False, show=True, clist=COLORLIST, timetype=False, ebar=True, debias=True, export_pdf=""):
    """Plot data from multiple observations vs time on a single baseline on the same axes.
        """
    
    try: len(obslist)
    except TypeError: obslist = [obslist]
    
    if len(obslist) > len(clist):
        Exception("More observations than colors -- Add more colors to clist!")
    
    axis = False
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
                             ebar=True, cphases=[], export_pdf="",axis=False, labels=True):

    """Plot closure phase on a triangle vs time from multiple observations on the same axes.
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



def plot_camp_obs_compare(obslist,  site1, site2, site3, site4, rangex=False, rangey=False, show=True, clist=COLORLIST, vtype='vis', ctype='camp', debias=True, timetype=False, ebar=True, camps=[], export_pdf=""):

    """Plot closure amplitude on a triangle vs time from multiple observations on the same axes.
        """
    
    try: len(obslist)
    except TypeError: obslist = [obslist]
    
    if len(obslist) > len(clist):
        Exception("More observations than colors -- Add more colors to clist!")

    if len(camps)==0:
        cphases = np.matlib.repmat([],len(obslist),1)

    axis = False
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
def plotall_obs_im_compare(obslist, image, field1, field2, ttype='direct', sgrscat=False, rangex=False, rangey=False, conj=False, show=True, clist=COLORLIST, ebar=True, export_pdf=""):
    """Plot data from observations compared to ground truth from an image on the same axes.
        """
    
    try: len(obslist)
    except TypeError: obslist = [obslist]
    
    for i in range(len(obslist)):
        obstrue = image.observe_same(obslist[i], sgrscat=sgrscat, add_th_noise=False, ttype=ttype)
        obstrue.data['sigma'] *= 0
        obslist.append(obstrue)
    
    if len(obslist) > len(clist):
        Exception("More observations than colors -- Add more colors to clist!")
    
    axis = False
    for i in range(len(obslist)):
        obs = obslist[i]

        axis = obs.plotall(field1, field2, rangex=rangex, rangey=rangey, conj=conj, show=False, axis=axis, color=clist[i%len(clist)], ebar=ebar)


    if show:
        plt.show(block=False)

    if export_pdf != "":
        plt.savefig(export_pdf, bbox_inches='tight', pad_inches = 0)

    return axis

def plot_bl_obs_im_compare(obslist, image, site1, site2, field, ttype='direct', sgrscat=False,  rangex=False, rangey=False, show=True, clist=COLORLIST, timetype=False, ebar=True, debias=True, export_pdf=""):
    """Plot data vs time on a single baseline compared to ground truth from an image on the same axes.
    """
    
    try: len(obslist)
    
    except TypeError: obslist = [obslist]
    
    for i in range(len(obslist)):
        obstrue = image.observe_same(obslist[i], sgrscat=sgrscat, add_th_noise=False, ttype=ttype)
        obstrue.data['sigma'] *= 0
        obslist.append(obstrue)
    
    if len(obslist) > len(clist):
        Exception("More observations than colors -- Add more colors to clist!")
    
    axis = False
    for i in range(len(obslist)):
        obs = obslist[i]

        axis = obs.plot_bl(site1, site2, field, rangex=rangex, rangey=rangey, show=False, axis=axis, color=clist[i%len(clist)], timetype=timetype, ebar=ebar, debias=debias)


    if show:
        plt.show(block=False)

    if export_pdf != "":
        plt.savefig(export_pdf, bbox_inches='tight', pad_inches = 0)

    return axis



def plot_cphase_obs_im_compare(obslist, image, site1, site2, site3, ttype='direct', sgrscat=False, rangex=False, rangey=False, show=True, clist=COLORLIST, ang_unit='deg', vtype='vis', timetype=False, ebar=True, axis=False, labels=True, export_pdf=""):

    """Plot closure phase on a triangle compared to ground truth from an image on the same axes.
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



def plot_camp_obs_im_compare(obslist, image, site1, site2, site3, site4, ttype='direct', sgrscat=False, rangex=False, rangey=False, show=True, clist=COLORLIST, vtype='vis', ctype='camp', debias=True, timetype=False, ebar=True, export_pdf=""):


    """Plot closure amplitude on a quadrangle compared to ground truth from an image on the same axes.
    """
    
    try: len(obslist)
    except TypeError: obslist = [obslist]
    
    for i in range(len(obslist)):
        obstrue = image.observe_same(obslist[i], sgrscat=sgrscat, add_th_noise=False, ttype=ttype)
        obstrue.data['sigma'] *= 0
        obslist.append(obstrue)
    
    if len(obslist) > len(clist):
        Exception("More observations than colors -- Add more colors to clist!")
    
    axis = False
    for i in range(len(obslist)):
        obs = obslist[i]

        axis = obs.plot_camp(site1, site2, site3, site4, rangex=rangex, rangey=rangey, show=False, axis=axis, color=clist[i%len(clist)],
                               timetype=timetype, vtype=vtype, ctype=ctype, debias=debias, ebar=ebar)

    if show:
        plt.show(block=False)

    if export_pdf != "":
        plt.savefig(export_pdf, bbox_inches='tight', pad_inches = 0)

    return axis


def plotall_obs_im_cphases(obs, image, ttype='direct', sgrscat=False, 
                           rangex=False, rangey=[-180,180], show=True, ebar=True, 
                           vtype='vis',display_mode='all'):
    """plot image on top of all cphases
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
        fig = plt.figure(figsize=(40, nrows*20))

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

            f = plot_cphase_obs_compare([obs, obs_model], uniqueclosure_tri[c][0], uniqueclosure_tri[c][1], uniqueclosure_tri[c][2], 
                                        vtype=vtype, rangex=rangex, rangey=rangey, ebar=ebar, show=show, 
                                        cphases=[cphases_obs, cphases_model], axis=ax, labels=labels)
            nplot += 1

    if display_mode!='individual': 
        plt.ion()
        f = fig
        f.subplots_adjust(wspace=0.1,hspace=0.5)
        f.show()
    
    return f
