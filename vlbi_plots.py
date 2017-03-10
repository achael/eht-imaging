# vlbi_plots.py
# Andrew Chael, 10/15/2016
# Make data plots with multiple observations, etc.

import string
import numpy as np
import numpy.lib.recfunctions as rec
import matplotlib.pyplot as plt
import scipy.signal
import scipy.optimize
import itertools as it
import astropy.io.fits as fits
import datetime
import writeData
import oifits_new as oifits
import time as ttime
import vlbi_imaging_utils as vb
import pulses

# Observation fields for plotting and retrieving data        
FIELDS = vb.FIELDS
          
COLORLIST = ['b','m','g','c','y','k','r']

##################################################################################################
# Plotters: Compare Observations
##################################################################################################          
def plotall_obs_compare(obslist, field1, field2, rangex=False, rangey=False, conj=False, show=True, clist=COLORLIST):
    """Plot data from multiple observations on the same axes"""
    
    try: len(obslist) 
    except TypeError: obslist = [obslist]
        
    if len(obslist) > len(clist):
        Exception("More observations than colors -- Add more colors to clist!")
        
    axis = False
    for i in range(len(obslist)):
        obs = obslist[i]
        axis = obs.plotall(field1, field2, rangex=rangex, rangey=rangey, conj=conj, show=False, axis=axis, color=clist[i%len(clist)])
    
    if show:
        plt.show(block=False)
    return axis
          
def plot_bl_obs_compare(obslist,  site1, site2, field, rangex=False, rangey=False, show=True, clist=COLORLIST):
    """Plot data from multiple observations vs time on a single baseline on the same axes"""
     
    try: len(obslist) 
    except TypeError: obslist = [obslist]
            
    if len(obslist) > len(clist):
        Exception("More observations than colors -- Add more colors to clist!")
        
    axis = False
    for i in range(len(obslist)):
        obs = obslist[i]
        axis = obs.plot_bl(site1, site2, field, rangex=rangex, rangey=rangey, show=False, axis=axis, color=clist[i%len(clist)])
    
    if show:
        plt.show(block=False)
    return axis    
    
    
def plot_cphase_obs_compare(obslist,  site1, site2, site3, rangex=False, rangey=False, show=True, clist=COLORLIST):
    """Plot closure phase on a triangle vs time from multiple observations on the same axes"""
    
    try: len(obslist) 
    except TypeError: obslist = [obslist]
            
    if len(obslist) > len(clist):
        Exception("More observations than colors -- Add more colors to clist!")
        
    axis = False
    for i in range(len(obslist)):
        obs = obslist[i]
        axis = obs.plot_cphase(site1, site2, site3, rangex=rangex, rangey=rangey, show=False, axis=axis, color=clist[i%len(clist)])
    
    if show:
        plt.show(block=False)
    return axis                
          
          
def plot_camp_obs_compare(obslist,  site1, site2, site3, site4, rangex=False, rangey=False, show=True, clist=COLORLIST):
    """Plot closure amplitude on a triangle vs time from multiple observations on the same axes"""
    
    try: len(obslist) 
    except TypeError: obslist = [obslist]
            
    if len(obslist) > len(clist):
        Exception("More observations than colors -- Add more colors to clist!")
        
    axis = False
    for i in range(len(obslist)):
        obs = obslist[i]
        axis = obs.plot_camp(site1, site2, site3, site4, rangex=rangex, rangey=rangey, show=False, axis=axis, color=clist[i%len(clist)])
    
    if show:
        plt.show(block=False)
    return axis                
    
##################################################################################################
# Plotters: Compare Observations to Image
##################################################################################################          
def plotall_obs_im_compare(obslist, image, field1, field2, sgrscat=False, rangex=False, rangey=False, conj=False, show=True, clist=COLORLIST):
    """Plot data from observations compared to ground truth from an image on the same axes"""
    
    try: len(obslist) 
    except TypeError: obslist = [obslist]
           
    for i in range(len(obslist)):
        obstrue = image.observe_same(obslist[i], sgrscat=sgrscat, add_th_noise=False)
        obstrue.data['sigma'] *= 0
        obslist.append(obstrue)
    
    if len(obslist) > len(clist):
        Exception("More observations than colors -- Add more colors to clist!")  
         
    axis = False
    for i in range(len(obslist)):
        obs = obslist[i]
        axis = obs.plotall(field1, field2, rangex=rangex, rangey=rangey, conj=conj, show=False, axis=axis, color=clist[i%len(clist)])
    
    if show:
        plt.show(block=False)
    return axis
          
def plot_bl_obs_im_compare(obslist, image, site1, site2, field, sgrscat=False,  rangex=False, rangey=False, show=True, clist=COLORLIST):
    """Plot data vs time on a single baseline compared to ground truth from an image on the same axes"""
    
    try: len(obslist) 
   
    except TypeError: obslist = [obslist]
           
    for i in range(len(obslist)):
        obstrue = image.observe_same(obslist[i], sgrscat=sgrscat, add_th_noise=False)
        obstrue.data['sigma'] *= 0
        obslist.append(obstrue)
    
    if len(obslist) > len(clist):
        Exception("More observations than colors -- Add more colors to clist!")  
        
    axis = False
    for i in range(len(obslist)):
        obs = obslist[i]
        axis = obs.plot_bl(site1, site2, field, rangex=rangex, rangey=rangey, sgrscat=False, show=False, axis=axis, color=clist[i%len(clist)])
    
    if show:
        plt.show(block=False)
    return axis    
    
    
def plot_cphase_obs_im_compare(obslist, image, site1, site2, site3, sgrscat=False, rangex=False, rangey=False, show=True, clist=COLORLIST):
    """Plot closure phase on a triangle compared to ground truth from an image on the same axes"""
    
    try: len(obslist) 
    except TypeError: obslist = [obslist]
           
    for i in range(len(obslist)):
        obstrue = image.observe_same(obslist[i], sgrscat=sgrscat,add_th_noise=False)
        obstrue.data['sigma'] *= 0
        obslist.append(obstrue)
    
    if len(obslist) > len(clist):
        Exception("More observations than colors -- Add more colors to clist!")  
        
    axis = False
    for i in range(len(obslist)):
        obs = obslist[i]
        axis = obs.plot_cphase(site1, site2, site3, rangex=rangex, rangey=rangey, show=False, axis=axis, color=clist[i%len(clist)])
    
    if show:
        plt.show(block=False)
    return axis                
          
          
def plot_camp_obs_im_compare(obslist, image, site1, site2, site3, site4, sgrscat=False, rangex=False, rangey=False, show=True, clist=COLORLIST):
    """Plot closure amplitude on a quadrangle compared to ground truth from an image on the same axes"""
    
    try: len(obslist) 
    except TypeError: obslist = [obslist]
           
    for i in range(len(obslist)):
        obstrue = image.observe_same(obslist[i], sgrscat=sgrscat, add_th_noise=False)
        obstrue.data['sigma'] *= 0
        obslist.append(obstrue)
    
    if len(obslist) > len(clist):
        Exception("More observations than colors -- Add more colors to clist!")  
        
    axis = False
    for i in range(len(obslist)):
        obs = obslist[i]
        axis = obs.plot_camp(site1, site2, site3, site4, rangex=rangex, rangey=rangey, show=False, axis=axis, color=clist[i%len(clist)])
    
    if show:
        plt.show(block=False)
    return axis                
