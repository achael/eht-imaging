# dynamical_imaging.py
# imaging movies with interferometric data
#
#    Copyright (C) 2018 Michael Johnson
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


# Note: This library is still under very active development and is likely to change considerably
# Contact Michael Johnson (mjohnson@cfa.harvard.edu) with any questions
# The methods/techniques used in this are described in http://adsabs.harvard.edu/abs/2017ApJ...850..172J

import time
import numpy as np

import scipy.optimize as opt
import scipy.ndimage.filters as filt
import scipy.signal

import matplotlib.pyplot as plt

import itertools as it

from ehtim.const_def import * #Note: C is m/s rather than cm/s.
from ehtim.observing.obs_helpers import *
import ehtim.obsdata as obsdata
import ehtim.image as image
from ehtim.imaging.imager_utils import *

import ehtim.scattering as so

from IPython import display

from multiprocessing import Pool
from functools import partial

#imports from the blazarFileDownloader
import calendar
import requests
import os
from html.parser import HTMLParser
#from HTMLParser import HTMLParser

Fast_Convolve = True # This option will not wrap the convolution around the image

# These parameters are only global to allow parallelizing the chi^2 calculation without huge memory overhead. It would be nice to do this locally, using the parallel array capabilities.
# Fourier matrices:
A1_List = [None,]
A2_List = [None,]
A3_List = [None,]
# Data used:
data1_List = [None,]
data2_List = [None,]
data3_List = [None,]
# Standard deviation of data used:
sigma1_List = [None,]
sigma2_List = [None,]
sigma3_List = [None,]

##################################################################################################
# Constants
##################################################################################################
#NHIST = 25 # number of steps to store for hessian approx
nit = 0 # global variable to track the iteration number in the plotting callback

##################################################################################################
# Tools for AGN images
##################################################################################################

def get_KLS(im1, im2, shift = [0,0], blur_size_uas=100, dynamic_range=200):
    # Symmetrized Kullback-Liebler Divergence, with optional blurring and max dynamic range
    ep = np.max(im1.imvec)/dynamic_range
    A = im1.blur_circ(blur_size_uas*RADPERUAS).imvec.reshape((im1.ydim,im1.xdim)) + ep
    B = im2.blur_circ(blur_size_uas*RADPERUAS).imvec.reshape((im2.ydim,im2.xdim)) + ep

    B = np.roll(B, shift, (0,1))
    return np.sum( (B - A)*np.log(B/A) )

def get_core_position(im, blur_size_uas=100):
    #Estimate the core position (i.e., the brightest region)
    #Convolve the input image with the beam, then find the brightest pixel
    im_blur = im.blur_circ(blur_size_uas*RADPERUAS).imvec
    return np.array(np.unravel_index(im_blur.argmax(),(im.ydim,im.xdim)))

def center_core(im, blur_size_uas=100):
    im_rotate = im.copy()
    core_pos = get_core_position(im,blur_size_uas)
    center = np.array((int((im.ydim-1)/2),int((im.xdim-1)/2)))
    print ("Rotating By",center - core_pos)
    im_rotate.imvec = np.roll(im.imvec.reshape((im.ydim,im.xdim)), center - core_pos, (0,1)).flatten()
    if len(im.qvec):
        im_rotate.qvec = np.roll(im.qvec.reshape((im.ydim,im.xdim)), center - core_pos, (0,1)).flatten()
    if len(im.uvec):
        im_rotate.uvec = np.roll(im.uvec.reshape((im.ydim,im.xdim)), center - core_pos, (0,1)).flatten()
    if len(im.vvec):
        im_rotate.vvec = np.roll(im.vvec.reshape((im.ydim,im.xdim)), center - core_pos, (0,1)).flatten()
    
    return im_rotate

def align_left(im,min_frac=0.1,opposite_frac_thresh=0.05):
    #Aligns the core at the middle, assuming that there is no appreciable flux to the left of the core
    im_rotate = im.copy()
    center = np.array((int((im.ydim-1)/2),int((im.xdim-1)/2)))
    projected_flux = np.sum(im.imvec.reshape((im.ydim,im.xdim)),axis=0)
    thresh = np.max(projected_flux)*min_frac
    #opposite_thresh = np.max(projected_flux[-int(opposite_frac_thresh*im.xdim):])

    for j in range(im.xdim):
        if projected_flux[j] > thresh:  #or projected_flux[j] > opposite_thresh
            break

    im_rotate.imvec = np.roll(im.imvec.reshape((im.ydim,im.xdim)), center[0] - j, (1)).flatten()
    if len(im.qvec):
        im_rotate.qvec = np.roll(im.qvec.reshape((im.ydim,im.xdim)), center[0] - j, (1)).flatten()
    if len(im.uvec):
        im_rotate.uvec = np.roll(im.uvec.reshape((im.ydim,im.xdim)), center[0] - j, (1)).flatten()
    if len(im.vvec):
        im_rotate.vvec = np.roll(im.vvec.reshape((im.ydim,im.xdim)), center[0] - j, (1)).flatten()

    return im_rotate


##################################################################################################
# Movie Export Tools
##################################################################################################


def export_multipanel_movie(im_List_Set, out='movie.mp4', fps=10, dpi=120, scale='linear', dynamic_range=1000.0, pad_factor=1, verbose=False, xlim = None, ylim = None, titles = [], size=8.0):
    # Example: di.export_multipanel_movie([im_List,im_List_2],scale='log',xlim=[1000,-1000],ylim=[-3000,500],dynamic_range=[1000,5000], titles = ['43 GHz (BU)','15 GHz (MOJAVE)'])
    import matplotlib
    matplotlib.use('agg')
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    fig = plt.figure()

    mjd_step = (im_List_Set[0][0].mjd - im_List_Set[0][-1].mjd)/len(im_List_Set[0])

    #if len(im_List_Set)%2 == 1:
    #    verticalalignment = 'bottom'
    #else:
    #    verticalalignment = 'top'

    N_set = len(im_List_Set)
    extent = [np.array((1,-1,-1,1))]*N_set
    maxi   = np.zeros(N_set)
    plt_im = [None,]*N_set

    if type(dynamic_range) == float or type(dynamic_range) == int:
        dynamic_range = np.zeros(N_set) + dynamic_range

    for j in range(N_set):
        extent[j] = im_List_Set[j][0].psize/RADPERUAS*im_List_Set[j][0].xdim*np.array((1,-1,-1,1)) / 2.
        maxi[j]   = np.max(np.concatenate([im.imvec for im in im_List_Set[j]]))

    def im_data(i_set, n):
        n_data = (n-n%pad_factor)//pad_factor
        if scale == 'linear':
            return im_List_Set[i_set][n_data].imvec.reshape((im_List_Set[i_set][n_data].ydim,im_List_Set[i_set][n_data].xdim))
        else:
            return np.log(im_List_Set[i_set][n_data].imvec.reshape((im_List_Set[i_set][n_data].ydim,im_List_Set[i_set][n_data].xdim)) + maxi[i_set]/dynamic_range[i_set])

    for j in range(N_set):
        ax = plt.subplot(1, N_set, j+1)
        plt_im[j] = plt.imshow(im_data(j, 0), extent=extent[j], cmap=plt.get_cmap('afmhot'), interpolation='gaussian')

        if xlim != None:
            ax.set_xlim(xlim)
        if ylim != None:
            ax.set_ylim(ylim)

        if scale == 'linear':
            plt_im[j].set_clim([0,maxi[j]])
        else:
            plt_im[j].set_clim([np.log(maxi[j]/dynamic_range[j]),np.log(maxi[j])])

        if j == 0:
            plt.xlabel('Relative RA ($\mu$as)')
            plt.ylabel('Relative Dec ($\mu$as)')

        if len(titles) > 0:
            ax.set_title(titles[j])

    fig.set_size_inches([size,size/len(im_List_Set)])
    plt.tight_layout()

    def update_img(n):
        if verbose:
            print ("processing frame {0} of {1}".format(n, len(im_List)*pad_factor))
        for j in range(N_set):
            plt_im[j].set_data(im_data(j, n))

        if mjd_step > 0.1:
            fig.suptitle('MJD: ' + str(im_List_Set[0][int((n-n%pad_factor)//pad_factor)].mjd), verticalalignment = verticalalignment)
        else:
            time = im_List_Set[0][int((n-n%pad_factor)//pad_factor)].time
            time_str = ("%d:%02d.%02d" % (int(time), (time*60) % 60, (time*3600) % 60))
            fig.suptitle(time_str)

        return plt_im

    ani = animation.FuncAnimation(fig,update_img,len(im_List_Set[0])*pad_factor,interval=1e3/fps)
    writer = animation.writers['ffmpeg'](fps=fps, bitrate=1e6)
    ani.save(out,writer=writer,dpi=dpi)

def export_movie(im_List, out='movie.mp4', fps=10, dpi=120, scale='linear', cbar_unit = 'Jy', gamma=0.5, dynamic_range=1000.0, pad_factor=1, verbose=False):
    import matplotlib
    matplotlib.use('agg')
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    mjd_range = im_List[-1].mjd - im_List[0].mjd    

    fig = plt.figure()

    extent = im_List[0].psize/RADPERUAS*im_List[0].xdim*np.array((1,-1,-1,1)) / 2.
    maxi = np.max(np.concatenate([im.imvec for im in im_List]))

# TODO: fix this
#    if cbar_unit == 'mJy':
#        imvec = imvec * 1.e3
#        qvec = qvec * 1.e3
#        uvec = uvec * 1.e3
#    elif cbar_unit == '$\mu$Jy':
#        imvec = imvec * 1.e6
#        qvec = qvec * 1.e6
#        uvec = uvec * 1.e6

    unit = cbar_unit + '/pixel'

    if scale=='log':
        unit = 'log(' + cbar_unit + '/pixel)'

    if scale=='gamma':
        unit = '(' + cbar_unit + '/pixel)^gamma'

    def im_data(n):
        n_data = (n-n%pad_factor)//pad_factor
        if scale == 'linear':
            return im_List[n_data].imvec.reshape((im_List[n_data].ydim,im_List[n_data].xdim))
        elif scale == 'log':
            return np.log(im_List[n_data].imvec.reshape((im_List[n_data].ydim,im_List[n_data].xdim)) + maxi/dynamic_range)
        elif scale == 'gamma':
            return (im_List[n_data].imvec.reshape((im_List[n_data].ydim,im_List[n_data].xdim)) + maxi/dynamic_range)**(gamma)

    plt_im = plt.imshow(im_data(0), extent=extent, cmap=plt.get_cmap('afmhot'), interpolation='gaussian')
    if scale == 'linear':
        plt_im.set_clim([0,maxi])
    elif scale == 'log':
        plt_im.set_clim([np.log(maxi/dynamic_range),np.log(maxi)])
    elif scale == 'gamma':
        plt_im.set_clim([(maxi/dynamic_range)**gamma,(maxi)**(gamma)])

    plt.xlabel('Relative RA ($\mu$as)')
    plt.ylabel('Relative Dec ($\mu$as)')

    fig.set_size_inches([5,5])
    plt.tight_layout()

    def update_img(n):
        if verbose:
            print ("processing frame {0} of {1}".format(n, len(im_List)*pad_factor))
        plt_im.set_data(im_data(n))
        if mjd_range != 0:
            fig.suptitle('MJD: ' + str(im_List[int((n-n%pad_factor)//pad_factor)].mjd))
        else:
            time = im_List[int((n-n%pad_factor)//pad_factor)].time
            time_str = ("%d:%02d.%02d" % (int(time), (time*60) % 60, (time*3600) % 60))
            fig.suptitle(time_str)

        return plt_im

    ani = animation.FuncAnimation(fig,update_img,len(im_List)*pad_factor,interval=1e3/fps)
    writer = animation.writers['ffmpeg'](fps=fps, bitrate=1e6)
    ani.save(out,writer=writer,dpi=dpi)


##################################################################################################
# Convenience Functions for Data Processing
##################################################################################################

def split_obs(obs):
    """Split single observation into multiple observation files, one per scan
    """

    print ("Splitting Observation File into " + str(len(obs.tlist())) + " scans")
    return [ obsdata.Obsdata(obs.ra, obs.dec, obs.rf, obs.bw, tdata, obs.tarr, source=obs.source, mjd=obs.mjd, ampcal=obs.ampcal, phasecal=obs.phasecal) for tdata in obs.tlist() ]

def merge_obs(obs_List):
    """Merge a list of observations into a single observation file
    """

    if len(set([obs.ra for obs in obs_List])) > 1 or len(set([obs.dec for obs in obs_List])) > 1 or len(set([obs.rf for obs in obs_List])) > 1 or len(set([obs.bw for obs in obs_List])) > 1 or len(set([obs.source for obs in obs_List])) > 1:
        print ("All observations must have the same parameters!")
        return

    #The important things to merge are the mjd and the data
    data_merge = np.hstack([obs.data for obs in obs_List])

    return obsdata.Obsdata(obs_List[0].ra, obs_List[0].dec, obs_List[0].rf, obs_List[0].bw, data_merge, obs_List[0].tarr, source=obs_List[0].source, mjd=obs_List[0].mjd, ampcal=obs_List[0].ampcal, phasecal=obs_List[0].phasecal)

def average_im_list(im_List):
    """Return the average of a list of images
    """
    avg_im = im_List[0].copy()
    avg_im.imvec = np.mean([im.imvec for im in im_List],axis=0)
    if len(im_List[0].qvec):
        avg_im.qvec = np.mean([im.qvec for im in im_List],axis=0)
    if len(im_List[0].uvec):
        avg_im.uvec = np.mean([im.uvec for im in im_List],axis=0)
    if len(im_List[0].vvec):
        avg_im.vvec = np.mean([im.vvec for im in im_List],axis=0)

    return avg_im

def blur_im_list(im_List, fwhm_x, fwhm_t):
    """Apply a gaussian filter to a list of images, with fwhm_x in radians and fwhm_t in frames. Currently only for Stokes I.

       Args:
           fwhm_x (float): circular beam size for spatial blurring in radians
           fwhm_t (float): temporal blurring in frames
       Returns:
           (Image): output image list
    """

    # Blur Stokes I
    sigma_x = fwhm_x / im_List[0].psize / (2. * np.sqrt(2. * np.log(2.)))
    sigma_t = fwhm_t / (2. * np.sqrt(2. * np.log(2.)))

    arr = np.array([im.imvec.reshape(im.ydim, im.xdim) for im in im_List])
    arr = filt.gaussian_filter(arr, (sigma_t, sigma_x, sigma_x))

    ret = []
    for j in range(len(im_List)):
        ret.append(image.Image(arr[j], im_List[0].psize, im_List[0].ra, im_List[0].dec, rf=im_List[0].rf, source=im_List[0].source, mjd=im_List[j].mjd))

    return ret

##################################################################################################
# Convenience Functions for Analytical Work
##################################################################################################

def Wrapped_Convolve(sig,ker):
    if np.sum(ker) == 0.0:
        return sig

    N = sig.shape[0]

    if Fast_Convolve == False:
        return scipy.signal.fftconvolve(np.pad(sig,((N, N), (N, N)), 'wrap'), np.pad(ker,((N, N), (N, N)), 'constant'),mode='same')[N:(2*N),N:(2*N)]
    else:
        return scipy.signal.fftconvolve(sig, ker,mode='same')

def Wrapped_Gradient(M):
    G = np.gradient(np.pad(M,((1, 1), (1, 1)), 'wrap'))
    Gx = G[0][1:-1,1:-1]
    Gy = G[1][1:-1,1:-1]
    return (Gx, Gy)

def Wrapped_Gradient_Reorder(M):
    G = np.gradient(np.pad(M,((1, 1), (1, 1)), 'wrap'))
    Gx = G[0][1:-1,1:-1]
    Gy = G[1][1:-1,1:-1]
    return np.transpose(np.array([Gx, Gy]),axes=[1,2,0])

def Wrapped_Divergence( vectorfield ):
    Gx = np.gradient(np.pad(vectorfield[:,:,0],((1, 1), (1, 1)), 'wrap'), axis=0)[1:-1,1:-1]
    Gy = np.gradient(np.pad(vectorfield[:,:,1],((1, 1), (1, 1)), 'wrap'), axis=1)[1:-1,1:-1]
    return Gx+Gy

def Wrapped_Weighted_Divergence( weight, M ):  #(weight \cdot \nabla) M
    grad = Wrapped_Gradient(M)
    return weight[:,:,0]*grad[0] + weight[:,:,1]*grad[1]

##################################################################################################
# Dynamic Regularizers and their Gradients
##################################################################################################

#RdF Regularizer (continuity of total flux density from frame to frame)
def RdF_clip(Frame_List, embed_mask_List):
    F_List = [np.sum(Frame_List[j].ravel()[embed_mask_List[j]]) for j in range(len(Frame_List))]
    return np.sum(np.diff(F_List)**2)

def RdF_gradient_clip(Frame_List, embed_mask_List):
    N_frame = Frame_List.shape[0]
    F_List = [np.sum(Frame_List[j].ravel()[embed_mask_List[j]]) for j in range(len(Frame_List))]
    F_grad_List = [1.0 + 0.0*np.copy(Frame_List[j].ravel()[embed_mask_List[j]]) for j in range(len(Frame_List))]
    grad = np.copy(F_grad_List)*0.0

    for i in range(1,N_frame):
        grad[i] = grad[i] + 2.0*(F_List[i] - F_List[i-1])*F_grad_List[i]

    for i in range(N_frame-1):
        grad[i] = grad[i] + 2.0*(F_List[i] - F_List[i+1])*F_grad_List[i]

    return np.concatenate([grad[i]*(Frame_List[i].ravel()[embed_mask_List[i]]) for i in range(N_frame)])

#RdS Regularizer (continuity of entropy from frame to frame)
def RdS(Frame_List, Prior_List, embed_mask_List, entropy="simple", norm_reg=True, **kwargs):
    S_List = [static_regularizer(np.array([Frame_List[j]]), np.array([Prior_List[j]]), np.array([embed_mask_List[j]]), Prior_List[0].total_flux(), Prior_List[0].psize, entropy=entropy, norm_reg=norm_reg, **kwargs) for j in range(len(Frame_List))]
    return np.sum(np.diff(S_List)**2)

def RdS_gradient(Frame_List, Prior_List, embed_mask_List, entropy="simple", norm_reg=True, **kwargs):
    #The Jacobian_Factor is already part of the entropy gradient that this function calls
    N_frame = Frame_List.shape[0]
    S_List = [static_regularizer(np.array([Frame_List[j]]), np.array([Prior_List[j]]), np.array([embed_mask_List[j]]), Prior_List[0].total_flux(), Prior_List[0].psize, entropy=entropy, norm_reg=norm_reg, **kwargs) for j in range(len(Frame_List))]
    S_grad_List = np.array([static_regularizer_gradient(np.array([Frame_List[j]]), np.array([Prior_List[j]]), np.array([embed_mask_List[j]]), Prior_List[0].total_flux(), Prior_List[0].psize, entropy=entropy, norm_reg=norm_reg, **kwargs) for j in range(len(Frame_List))])

    grad = np.copy(S_grad_List)*0.0

    for i in range(1,N_frame):
        grad[i] = grad[i] + 2.0*(S_List[i] - S_List[i-1])*S_grad_List[i]

    for i in range(N_frame-1):
        grad[i] = grad[i] + 2.0*(S_List[i] - S_List[i+1])*S_grad_List[i]

    return np.concatenate([grad[i] for i in range(N_frame)])


######## Rdt, RdI, and Rflow master functions ########
def Rdt(Frames, ker, metric='SymKL', p=2.0, **kwargs):
    if metric == 'KL':
        return Rdt_KL(Frames, ker)
    elif metric == 'SymKL':
        return Rdt_SymKL(Frames, ker)
    elif metric == 'D2':
        return Rdt_Dp(Frames, ker, p=2.0)
    elif metric == 'Dp':
        return Rdt_Dp(Frames, ker, p=p)
    else:
        return 0.0

def Rdt_gradient(Frames, ker, metric='SymKL', p=2.0, **kwargs):
    if metric == 'KL':
        return Rdt_KL_gradient(Frames, ker)
    elif metric == 'SymKL':
        return Rdt_SymKL_gradient(Frames, ker)
    elif metric == 'D2':
        return Rdt_Dp_gradient(Frames, ker, p=2.0)
    elif metric == 'Dp':
        return Rdt_Dp_gradient(Frames, ker, p=p)
    else:
        return 0.0

def RdI(Frames, metric='SymKL', p=2.0, **kwargs):
    if metric == 'KL':
        return RdI_KL(Frames)
    elif metric == 'SymKL':
        return RdI_SymKL(Frames)
    elif metric == 'D2':
        return RdI_Dp(Frames, p=2.0)
    elif metric == 'Dp':
        return RdI_Dp(Frames, p=p)
    else:
        return 0.0

def RdI_gradient(Frames, metric='SymKL', p=2.0, **kwargs):
    if metric == 'KL':
        return RdI_KL_gradient(Frames)
    elif metric == 'SymKL':
        return RdI_SymKL_gradient(Frames)
    elif metric == 'D2':
        return RdI_Dp_gradient(Frames, p=2.0)
    elif metric == 'Dp':
        return RdI_Dp_gradient(Frames, p=p)
    else:
        return 0.0

def Rflow(Frames, Flow, metric='D2', p=2.0, **kwargs):
    if metric == 'KL':
        return Rflow_KL(Frames, Flow)
    elif metric == 'SymKL':
        return Rflow_SymKL(Frames, Flow)
    elif metric == 'D2':
        return Rflow_D2(Frames, Flow)
    elif metric == 'Dp':
        return Rflow_Dp(Frames, Flow, p=p)
    else:
        return 0.0

def Rflow_gradient_I(Frames, Flow, metric='D2', p=2.0, **kwargs):
    if metric == 'KL':
        return Rflow_KL_gradient_I(Frames, Flow)
    elif metric == 'SymKL':
        return Rflow_SymKL_gradient_I(Frames, Flow)
    elif metric == 'D2':
        return Rflow_D2_gradient_I(Frames, Flow)
    elif metric == 'Dp':
        return Rflow_Dp_gradient_I(Frames, Flow, p=p)
    else:
        return 0.0

def Rflow_gradient_m(Frames, Flow, metric='D2', p=2.0, **kwargs):
    if metric == 'KL':
        return Rflow_KL_gradient_m(Frames, Flow)
    elif metric == 'SymKL':
        return Rflow_SymKL_gradient_m(Frames, Flow)
    elif metric == 'D2':
        return Rflow_Dp_gradient_m(Frames, Flow)
    elif metric == 'Dp':
        return Rflow_Dp_gradient_m(Frames, Flow, p=p)
    else:
        return 0.0

####################################

#Rdt Regularizer with relative entropy (Kullback-Leibler Divergence)
def Rdt_KL(Frames, ker):
    ep=1e-10
    N_frame = Frames.shape[0]
    Blur_Frames = np.array([Wrapped_Convolve(f, ker) for f in Frames])

    R = 0.0
    for i in range(1,N_frame):
        R += np.sum( Blur_Frames[i]*np.log((Blur_Frames[i]+ep)/(Blur_Frames[i-1]+ep)) )

    return R/N_frame

def Rdt_KL_gradient(Frames, ker):
    #The Jacobian_Factor accounts for the frames being written as log(frame) in the imaging algorithm
    ep=1e-10
    N_frame = Frames.shape[0]
    Blur_Frames = np.array([Wrapped_Convolve(f, ker) for f in Frames])

    grad = np.copy(Frames)*0.0

    for i in range(1,len(Frames)):
        grad[i] = grad[i] + np.log((Blur_Frames[i]+ep)/(Blur_Frames[i-1]+ep)) + 1.0

    for i in range(len(Frames)-1):
        grad[i] = grad[i] - (Blur_Frames[i+1]+ep)/(Blur_Frames[i]+ep)

    return np.array([Wrapped_Convolve(grad[i],ker)/N_frame*Frames[i] for i in range(N_frame)]).flatten()

#Rdt Regularizer with symmetrized relative entropy
def Rdt_SymKL(Frames, ker):
    ep=1e-10
    N_frame = Frames.shape[0]
    Blur_Frames = np.array([Wrapped_Convolve(f, ker) for f in Frames])

    R = 0.0
    for i in range(1,N_frame):
        R += np.sum( (Blur_Frames[i] - Blur_Frames[i-1])*np.log((Blur_Frames[i]+ep)/(Blur_Frames[i-1]+ep)) )

    return 0.5*R/N_frame

def Rdt_SymKL_gradient(Frames, ker):
    #The Jacobian_Factor accounts for the frames being written as log(frame) in the imaging algorithm
    ep=1e-10
    N_frame = Frames.shape[0]
    Blur_Frames = np.array([Wrapped_Convolve(f, ker) for f in Frames])

    grad = np.copy(Frames)*0.0

    for i in range(1,len(Frames)):
        grad[i] = grad[i] + 1.0 - (Blur_Frames[i-1]+ep)/(Blur_Frames[i]+ep) + np.log((Blur_Frames[i]+ep)/(Blur_Frames[i-1]+ep))

    for i in range(len(Frames)-1):
        grad[i] = grad[i] + 1.0 - (Blur_Frames[i+1]+ep)/(Blur_Frames[i]+ep) - np.log((Blur_Frames[i+1]+ep)/(Blur_Frames[i]+ep))

    return 0.5*np.array([Wrapped_Convolve(grad[i],ker)/N_frame*Frames[i] for i in range(N_frame)]).flatten()

#Rdt Regularizer with MSE (or l_p norm)
def Rdt_Dp(Frames, ker, p=2.0):
    N_frame = Frames.shape[0]
    Blur_Frames = np.array([Wrapped_Convolve(f, ker) for f in Frames])
    return np.sum(np.abs(np.diff(Blur_Frames,axis=0))**p)/N_frame

def Rdt_Dp_gradient(Frames, ker, p=2.0):
    N_frame = Frames.shape[0]

    grad = np.copy(Frames)*0.0

    if p==2.0:
        for i in range(1,len(Frames)):
            grad[i] = grad[i] + 2.0*(Frames[i] - Frames[i-1])

        for i in range(len(Frames)-1):
            grad[i] = grad[i] + 2.0*(Frames[i] - Frames[i+1])

        return np.array([Wrapped_Convolve(Wrapped_Convolve(grad[i],ker),ker)/N_frame*Frames[i] for i in range(len(Frames))]).flatten()
    else:
        for i in range(1,len(Frames)):
            grad_temp = Wrapped_Convolve(Frames[i] - Frames[i-1],ker)
            grad[i] = grad[i] + p*(np.abs(grad_temp)**(p-1.0)*np.sign(grad_temp))

        for i in range(len(Frames)-1):
            grad_temp = Wrapped_Convolve(Frames[i] - Frames[i+1],ker)
            grad[i] = grad[i] + p*(np.abs(grad_temp)**(p-1.0)*np.sign(grad_temp))

        return np.array([Wrapped_Convolve(grad[i],ker)/N_frame*Frames[i] for i in range(N_frame)]).flatten()

#RdI Regularizer
def RdI_KL(Frames):
    N_frame = Frames.shape[0]
    ep=1e-10
    avg_Image = np.mean(Frames,axis=0)

    return np.sum(Frames * np.log((Frames + ep)/(avg_Image + ep)))/N_frame

def RdI_KL_gradient(Frames):
    N_frame = Frames.shape[0]
    ep=1e-10
    avg_Image = np.mean(Frames,axis=0)
    return np.concatenate([(np.log((Frames[i]+ep)/(avg_Image+ep))/N_frame*Frames[i]).ravel() for i in range(N_frame)])

def RdI_SymKL(Frames):
    N_frame = Frames.shape[0]
    ep=1e-10
    avg_Image = np.mean(Frames,axis=0)

    return np.sum(0.5 * (Frames - avg_Image) * np.log((Frames + ep)/(avg_Image + ep)))/N_frame

def RdI_SymKL_gradient(Frames):
    N_frame = Frames.shape[0]
    ep=1e-10
    avg_Image = np.mean(Frames,axis=0)
    term2 = 1.0/N_frame * np.sum(np.log((Frames + ep)/(avg_Image + ep)), axis=0)

    return np.concatenate([(0.5*( (Frames[i]-avg_Image)/(Frames[i]+ep) + np.log((Frames[i]+ep)/(avg_Image+ep)) - term2)/N_frame*Frames[i]).ravel() for i in range(N_frame)])

def RdI_D2(Frames):
    N_frame = Frames.shape[0]
    avg_Image = np.mean(Frames,axis=0)
    return np.sum((Frames - avg_Image)**2)/N_frame

def RdI_D2_gradient(Frames):
    N_frame = Frames.shape[0]
    avg_Image = np.mean(Frames,axis=0)
    return np.concatenate([(2.0*(Frames[i] - avg_Image)/N_frame*Frames[i]).ravel() for i in range(Frames.shape[0])])

def RdI_Dp(Frames, p=2.0):
    N_frame = Frames.shape[0]
    avg_Image = np.mean(Frames,axis=0)
    return np.sum(np.abs(Frames - avg_Image)**p)/N_frame

def RdI_Dp_gradient(Frames, p=2.0):
    N_frame = Frames.shape[0]
    avg_Image = np.mean(Frames,axis=0)
    term2 = -p/N_frame * np.sum([np.abs(Frames[i] - avg_Image)**(p-1.0)*np.sign(Frames[i] - avg_Image) for i in range(N_frame)],axis=0)
    return np.concatenate([((p*np.abs(Frames[i] - avg_Image)**(p-1.0)*np.sign(Frames[i] - avg_Image) + term2)/N_frame*Frames[i]).ravel() for i in range(N_frame)])

#Rflow Regularizer
def Rflow_D2(Frames, Flow):
    N_frame = Frames.shape[0]
    val = 0.0

    for j in range(len(Frames)-1):
        #dI_dt = -Wrapped_Divergence( Frames[j,:,:,None]*Flow )    #this is not the same as the expanded version (for discrete derivatives)
        dI_dt = -(Wrapped_Weighted_Divergence(Flow,Frames[j]) + Frames[j]*Wrapped_Divergence(Flow))
        val = val + np.sum( (Frames[j+1] - (Frames[j] + dI_dt))**2 )

    return val/N_frame

def Rflow_D2_gradient_I(Frames, Flow):
    N_frame = Frames.shape[0]
    grad = 0.0*np.copy(Frames)

    for j in range(1,len(Frames)):
        #dI_dt = -Wrapped_Divergence( Frames[j-1,:,:,None]*Flow  )
        dI_dt = -(Wrapped_Weighted_Divergence(Flow,Frames[j-1]) + Frames[j-1]*Wrapped_Divergence(Flow))
        deltaI = Frames[j] - (Frames[j-1] + dI_dt)
        grad[j] = grad[j] + 2.0*deltaI

    for j in range(0,len(Frames)-1):
        #dI_dt = -Wrapped_Divergence( Frames[j,:,:,None]*Flow )
        dI_dt = -(Wrapped_Weighted_Divergence(Flow,Frames[j]) + Frames[j]*Wrapped_Divergence(Flow))
        deltaI = Frames[j+1] - (Frames[j] + dI_dt)
        grad[j] = grad[j] - 2.0*(deltaI + Wrapped_Weighted_Divergence(Flow, deltaI))

    for j in range(len(Frames)):
        grad[j] = grad[j]*Frames[j]

    return np.concatenate([g.flatten() for g in grad])/N_frame

def Rflow_D2_gradient_m(Frames, Flow):
    N_frame = Frames.shape[0]
    grad = 0.0*np.copy(Flow)

    for j in range(len(Frames)-1):
        #dI_dt = -Wrapped_Divergence( Frames[j,:,:,None]*Flow )
        dI_dt = -(Wrapped_Weighted_Divergence(Flow,Frames[j]) + Frames[j]*Wrapped_Divergence(Flow))
        deltaI = Frames[j+1] - (Frames[j] + dI_dt)
        grad = grad + 2.0 * deltaI[:,:,None] * Wrapped_Gradient_Reorder( Frames[j] )
        grad = grad - 2.0 * Wrapped_Gradient_Reorder( deltaI*Frames[j] )

    return np.array(grad).flatten()/N_frame

def Rflow_D2_gradient_m_alt(Frames, Flow): #not sure if this is correct
    N_frame = Frames.shape[0]
    grad = 0.0*np.copy(Flow)

    for j in range(len(Frames)-1):
        #dI_dt = -Wrapped_Divergence( Frames[j,:,:,None]*Flow )
        dI_dt = -(Wrapped_Weighted_Divergence(Flow,Frames[j]) + Frames[j]*Wrapped_Divergence(Flow))
        deltaI = Frames[j+1] - (Frames[j] + dI_dt)
        grad = -2.0 * Frames[j][:,:,None] * Wrapped_Gradient_Reorder( deltaI )

    return np.array(grad).flatten()/N_frame

#### Helper functions for the flow ####

def squared_gradient_flow(flow):
    """Total squared gradient of flow"""

    return np.sum(np.array(Wrapped_Gradient(flow[:,:,0]))**2 + np.array(Wrapped_Gradient(flow[:,:,1]))**2)

def squared_gradient_flow_grad(flow):
    """Total squared gradient of flow gradient wrt flow"""

    grad_x = -2.0*Wrapped_Divergence(Wrapped_Gradient_Reorder(flow[:,:,0]))
    grad_y = -2.0*Wrapped_Divergence(Wrapped_Gradient_Reorder(flow[:,:,1]))

    return np.transpose([grad_x.ravel(),grad_y.ravel()]).ravel()

###### Static Regularizer Master Functions #######

def static_regularizer(Frame_List, Prior_List, embed_mask_List, flux, psize, stype="simple", norm_reg=True, **kwargs):
    N_frame = Frame_List.shape[0]
    xdim = int(len(Frame_List[0].ravel())**0.5)

    s = np.sum( regularizer(Frame_List[i].ravel()[embed_mask_List[i]], Prior_List[i].ravel()[embed_mask_List[i]], embed_mask_List[i], flux=flux, xdim=xdim, ydim=xdim, psize=psize, stype=stype, norm_reg=norm_reg, **kwargs) for i in range(N_frame))

    return s/N_frame

def static_regularizer_gradient(Frame_List, Prior_List, embed_mask_List, flux, psize, stype="simple", norm_reg=True, **kwargs):
    # Note: this function includes Jacobian factor to account for the frames being written as log(frame)
    N_frame = Frame_List.shape[0]
    xdim = int(len(Frame_List[0].ravel())**0.5)

    s = np.concatenate([regularizergrad((Frame_List[i].ravel())[embed_mask_List[i]], Prior_List[i].ravel()[embed_mask_List[i]], embed_mask_List[i], flux=flux, xdim=xdim, ydim=xdim, psize=psize, stype=stype, norm_reg=norm_reg, **kwargs)*(Frame_List[i].ravel())[embed_mask_List[i]] for i in range(N_frame)])

    return s/N_frame

##################################################################################################
# Other Regularization Functions
##################################################################################################

def centroid(Frame_List, coord):
    return np.sum(np.sum(im.ravel() * coord[:,0])**2 + np.sum(im.ravel() * coord[:,1])**2 for im in Frame_List)/len(Frame_List)

def centroid_gradient(Frame_List, coord): #Includes Jacobian factor to account for the frames being written as log(frame)
    return 2.0 * np.concatenate([(np.sum(im.ravel() * coord[:,0])*coord[:,0] + np.sum(im.ravel() * coord[:,1])*coord[:,1])*im.ravel() for im in Frame_List])/len(Frame_List)

def movie_flux_constraint(Frame_List, flux_List):
    # This is the mean squared *fractional* difference in image total flux density
    # Negative means ignore
    norm = float(np.sum([f > 0.0 for f in flux_List]))
    return np.sum([(np.sum(Frame_List[j]) - flux_List[j])**2/flux_List[j]**2/norm*(flux_List[j] >= 0.0) for j in range(len(Frame_List))])

def movie_flux_constraint_grad(Frame_List, flux_List): #Includes Jacobian factor to account for the frames being written as log(frame)
    norm = float(np.sum([f > 0.0 for f in flux_List]))
    return np.concatenate([2.0*(np.sum(Frame_List[j]) - flux_List[j])/flux_List[j]**2/norm*Frame_List[j].ravel()*(flux_List[j] >= 0.0) for j in range(len(Frame_List))])

##################################################################################################
# chi^2 estimation routines
##################################################################################################


def get_chisq(i, imvec_embed, d1, d2, d3, ttype, mask):
    global A1_List, A2_List, A3_List, data1_List, data2_List, data3_List, sigma1_List, sigma2_List, sigma3_List
    chisq1 = chisq2 = chisq3 = 1.0

    if d1 != False and len(data1_List[i])>0:

        chisq1 = chisq(imvec_embed, A1_List[i], data1_List[i], sigma1_List[i], d1, ttype=ttype, mask=mask)

    if d2 != False and len(data2_List[i])>0:
        chisq2 = chisq(imvec_embed, A2_List[i], data2_List[i], sigma2_List[i], d2, ttype=ttype, mask=mask)

    if d3 != False and len(data3_List[i])>0:
        chisq3 = chisq(imvec_embed, A3_List[i], data3_List[i], sigma3_List[i], d3, ttype=ttype, mask=mask)

    return [chisq1, chisq2, chisq3]

def get_chisq_wrap(args):
    return get_chisq(*args)


def get_chisqgrad(i, imvec_embed, d1, d2, d3, ttype, mask):
    global A1_List, A2_List, A3_List, data1_List, data2_List, data3_List, sigma1_List, sigma2_List, sigma3_List
    chisqgrad1 = chisqgrad2 = chisqgrad3 = 0.0*imvec_embed

    if d1 != False and len(data1_List[i])>0:

        chisqgrad1 = chisqgrad(imvec_embed, A1_List[i], data1_List[i], sigma1_List[i], d1, ttype=ttype, mask=mask) #This *does not* include the Jacobian factor

    if d2 != False and len(data2_List[i])>0:
        chisqgrad2 = chisqgrad(imvec_embed, A2_List[i], data2_List[i], sigma2_List[i], d2, ttype=ttype, mask=mask) #This *does not* include the Jacobian factor

    if d3 != False and len(data3_List[i])>0:
        chisqgrad3 = chisqgrad(imvec_embed, A3_List[i], data3_List[i], sigma3_List[i], d3, ttype=ttype, mask=mask) #This *does not* include the Jacobian factor

    return [chisqgrad1, chisqgrad2, chisqgrad3]

def get_chisqgrad_wrap(args):
    return get_chisqgrad(*args)

##################################################################################################
# Imagers
##################################################################################################


def dynamical_imaging_minimal(Obsdata_List, InitIm_List, Prior, flux_List = [],
d1='vis', d2=False, d3=False,
alpha_d1=10, alpha_d2=10, alpha_d3=10,
systematic_noise1=0.0, systematic_noise2=0.0, systematic_noise3=0.0,
entropy1="tv2", entropy2="l1",
alpha_s1=1.0, alpha_s2=1.0, norm_reg=True, alpha_A=1.0,
R_dt  ={'alpha':0.0, 'metric':'SymKL', 'p':2.0},
maxit=200, J_factor = 0.001, stop=1.0e-10, ipynb=False, refresh_interval = 1000, 
minimizer_method = 'L-BFGS-B', NHIST = 25, update_interval = 1, clipfloor=0., 
ttype = 'nfft', fft_pad_factor=2):

    global A1_List, A2_List, A3_List, data1_List, data2_List, data3_List, sigma1_List, sigma2_List, sigma3_List

    N_frame = len(Obsdata_List)
    N_pixel = Prior.xdim #pixel dimension

    # Determine the appropriate final resolution
    all_res = []
    for obs in Obsdata_List:
        if len(obs.data) > 0:
            all_res.append(obs.res())

    beam_size = np.min(np.array(all_res))
    print("Maximal Resolution:",beam_size)

    # Find an observation with data
    for j in range(N_frame):
        if len(Obsdata_List[j].data) > 0:
            first_obs = Obsdata_List[j]
            break

    # Catch problem if uvrange < largest baseline
    if 1./Prior.psize < np.max(first_obs.unpack(['uvdist'])['uvdist']):
        raise Exception("pixel spacing is larger than smallest spatial wavelength!")

    if alpha_flux > 0.0 and len(flux_List) != N_frame:
        raise Exception("Number of elements in the list of total flux densities does not match the number of frames!")

    # Make the blurring kernel for R_dt
    # Note: There are odd problems when sigma_dt is too small. I can't figure out why it causes the convolution to crash.
    # However, having sigma_dt -> 0 is not a problem in theory. So we'll just set the kernel to be zero in that case and then ignore it later in the convolution.

    B_dt = np.zeros((Prior.ydim,Prior.xdim))

    embed_mask_List = [Prior.imvec > clipfloor for j in range(N_frame)]
    embed_mask_All = np.array(embed_mask_List).flatten()

    embed_totals = [np.sum(embed_mask) for embed_mask in embed_mask_List]

    logprior_List = [None,] * N_frame
    loginit_List = [None,] * N_frame

    nprior_embed_List = [None,] * N_frame
    nprior_List = [None,] * N_frame

    ninit_embed_List = [InitIm_List[i].imvec for i in range(N_frame)]
    ninit_List = [ninit_embed_List[i][embed_mask_List[i]] for i in range(N_frame)]

    print ("Calculating lists/matrices for chi-squared terms...")
    A1_List = [None,] * N_frame
    A2_List = [None,] * N_frame
    A3_List = [None,] * N_frame
    data1_List = [[],] * N_frame
    data2_List = [[],] * N_frame
    data3_List = [[],] * N_frame
    sigma1_List = [None,] * N_frame
    sigma2_List = [None,] * N_frame
    sigma3_List = [None,] * N_frame

    # Get data and Fourier matrices for the data terms
    for i in range(N_frame):
        pixel_max = np.max(InitIm_List[i].imvec)
        prior_flux_rescale = 1.0
        if len(flux_List) > 0:
            prior_flux_rescale = flux_List[i]/Prior.total_flux()

        nprior_embed_List[i] = Prior.imvec * prior_flux_rescale

        nprior_List[i] = nprior_embed_List[i][embed_mask_List[i]]
        logprior_List[i] = np.log(nprior_List[i])
        loginit_List[i] = np.log(ninit_List[i] + pixel_max/Target_Dynamic_Range/1.e6)  #add the dynamic range floor here

        if len(Obsdata_List[i].data) == 0:  #This allows the algorithm to create frames for periods with no data
            continue

        (data1_List[i], sigma1_List[i], A1_List[i]) = chisqdata(Obsdata_List[i], Prior, embed_mask_List[i], d1, ttype=ttype, fft_pad_factor=fft_pad_factor, systematic_noise=systematic_noise1)
        (data2_List[i], sigma2_List[i], A2_List[i]) = chisqdata(Obsdata_List[i], Prior, embed_mask_List[i], d2, ttype=ttype, fft_pad_factor=fft_pad_factor, systematic_noise=systematic_noise2)
        (data3_List[i], sigma3_List[i], A3_List[i]) = chisqdata(Obsdata_List[i], Prior, embed_mask_List[i], d3, ttype=ttype, fft_pad_factor=fft_pad_factor, systematic_noise=systematic_noise3)

    # Define the objective function and gradient
    def objfunc(x):
        # Frames is a list of the *unscattered* frames
        Frames = np.zeros((N_frame, N_pixel, N_pixel))
        log_Frames = np.zeros((N_frame, N_pixel, N_pixel))

        init_i = 0
        for i in range(N_frame):
            cur_len = np.sum(embed_mask_List[i])
            log_Frames[i] = embed(x[init_i:(init_i+cur_len)], embed_mask_List[i]).reshape((N_pixel, N_pixel))
            Frames[i] = np.exp(log_Frames[i])*(embed_mask_List[i].reshape((N_pixel, N_pixel)))
            init_i += cur_len

        s1 = s2 = 0.0

        if alpha_s1 != 0.0:
            s1 = static_regularizer(Frames, nprior_embed_List, embed_mask_List, Prior.total_flux(), Prior.psize, entropy1, norm_reg=norm_reg, beam_size=beam_size, alpha_A=alpha_A)*alpha_s1
        if alpha_s2 != 0.0:
            s2 = static_regularizer(Frames, nprior_embed_List, embed_mask_List, Prior.total_flux(), Prior.psize, entropy2, norm_reg=norm_reg, beam_size=beam_size, alpha_A=alpha_A)*alpha_s2

        s_dynamic = 0.0

        if R_dt['alpha'] != 0.0: s_dynamic += Rdt(Frames, B_dt, **R_dt)*R_dt['alpha']

        chisq = np.array([get_chisq(j, Frames[j].ravel()[embed_mask_List[j]], d1, d2, d3, ttype, embed_mask_List[j]) for j in range(N_frame)])

        chisq = ((np.sum(chisq[:,0])/N_frame - 1.0)*alpha_d1 +
                 (np.sum(chisq[:,1])/N_frame - 1.0)*alpha_d2 +
                 (np.sum(chisq[:,2])/N_frame - 1.0)*alpha_d3)

        return (s1 + s2 + s_dynamic + chisq)*J_factor

    def objgrad(x):
        Frames = np.zeros((N_frame, N_pixel, N_pixel))
        log_Frames = np.zeros((N_frame, N_pixel, N_pixel))

        init_i = 0
        for i in range(N_frame):
            cur_len = np.sum(embed_mask_List[i])
            log_Frames[i] = embed(x[init_i:(init_i+cur_len)], embed_mask_List[i]).reshape((N_pixel, N_pixel))
            Frames[i] = np.exp(log_Frames[i])*(embed_mask_List[i].reshape((N_pixel, N_pixel)))
            init_i += cur_len

        s1 = s2 = 0.0

        if alpha_s1 != 0.0:
            s1 = static_regularizer_gradient(Frames, nprior_embed_List, embed_mask_List, Prior.total_flux(), Prior.psize, entropy1, norm_reg=norm_reg, beam_size=beam_size, alpha_A=alpha_A)*alpha_s1
        if alpha_s2 != 0.0:
            s2 = static_regularizer_gradient(Frames, nprior_embed_List, embed_mask_List, Prior.total_flux(), Prior.psize, entropy2, norm_reg=norm_reg, beam_size=beam_size, alpha_A=alpha_A)*alpha_s2

        s_dynamic_grad = 0.0
        if R_dt['alpha'] != 0.0: s_dynamic_grad += Rdt_gradient(Frames, B_dt, **R_dt)*R_dt['alpha']


        chisq_grad = np.array([get_chisqgrad(j, Frames[j].ravel()[embed_mask_List[j]], d1, d2, d3, ttype, embed_mask_List[j]) for j in range(N_frame)])

        # Now add the Jacobian factor and concatenate
        for j in range(N_frame):
            chisq_grad[j,0] = chisq_grad[j,0]*Frames[j].ravel()[embed_mask_List[j]]
            chisq_grad[j,1] = chisq_grad[j,1]*Frames[j].ravel()[embed_mask_List[j]]
            chisq_grad[j,2] = chisq_grad[j,2]*Frames[j].ravel()[embed_mask_List[j]]

        chisq_grad = (np.concatenate([embed(chisq_grad[i,0], embed_mask_List[i]) for i in range(N_frame)])/N_frame*alpha_d1
                    + np.concatenate([embed(chisq_grad[i,1], embed_mask_List[i]) for i in range(N_frame)])/N_frame*alpha_d2
                    + np.concatenate([embed(chisq_grad[i,2], embed_mask_List[i]) for i in range(N_frame)])/N_frame*alpha_d3)

        return (np.concatenate((s1 + s2 + (s_dynamic_grad + chisq_grad)[embed_mask_All]))*J_factor)

    # Plotting function for each iteration
    global nit
    nit = 0
    def plotcur(x, final=False):
        global nit
        nit += 1

        if nit%update_interval == 0 or final == True:
            print ("iteration %d" % nit)

            Frames = np.zeros((N_frame, N_pixel, N_pixel))
            log_Frames = np.zeros((N_frame, N_pixel, N_pixel))

            init_i = 0
            for i in range(N_frame):
                cur_len = np.sum(embed_mask_List[i])
                log_Frames[i] = embed(x[init_i:(init_i+cur_len)], embed_mask_List[i]).reshape((N_pixel, N_pixel))
                Frames[i] = np.exp(log_Frames[i])*(embed_mask_List[i].reshape((N_pixel, N_pixel)))
                init_i += cur_len

            s1 = s2 = 0.0

            if alpha_s1 != 0.0:

                s1 = static_regularizer(Frames, nprior_embed_List, embed_mask_List, Prior.total_flux(),
                                        Prior.psize, entropy1, norm_reg=norm_reg, beam_size=beam_size, alpha_A=alpha_A)*alpha_s1
            if alpha_s2 != 0.0:
                s2 = static_regularizer(Frames, nprior_embed_List, embed_mask_List, Prior.total_flux(),
                                        Prior.psize, entropy2, norm_reg=norm_reg, beam_size=beam_size, alpha_A=alpha_A)*alpha_s2

            s_dynamic = 0.0

            if R_dt['alpha'] != 0.0: s_dynamic += Rdt(Frames, B_dt, **R_dt)*R_dt['alpha']

            chisq = np.array([get_chisq(j, Frames[j].ravel()[embed_mask_List[j]],
                              d1, d2, d3, ttype, embed_mask_List[j]) for j in range(N_frame)])

            chisq1_List = chisq[:,0]
            chisq2_List = chisq[:,1]
            chisq3_List = chisq[:,2]
            chisq1 = np.sum(chisq1_List)/N_frame
            chisq2 = np.sum(chisq2_List)/N_frame
            chisq3 = np.sum(chisq3_List)/N_frame
            chisq1_max = np.max(chisq1_List)
            chisq2_max = np.max(chisq2_List)
            chisq3_max = np.max(chisq3_List)
            if d1 != False: print ("chi2_1: %f" % chisq1)
            if d2 != False: print ("chi2_2: %f" % chisq2)
            if d3 != False: print ("chi2_3: %f" % chisq3)
            if d1 != False: print ("weighted chi2_1: %f" % (chisq1 * alpha_d1))
            if d2 != False: print ("weighted chi2_2: %f" % (chisq2 * alpha_d2))
            if d3 != False: print ("weighted chi2_3: %f" % (chisq3 * alpha_d3))
            if d1 != False: print ("Max Frame chi2_1: %f" % chisq1_max)
            if d2 != False: print ("Max Frame chi2_2: %f" % chisq2_max)
            if d3 != False: print ("Max Frame chi2_3: %f" % chisq3_max)

            if final == True:
                if d1 != False: print ("All chisq1:",chisq1_List)
                if d2 != False: print ("All chisq2:",chisq2_List)
                if d3 != False: print ("All chisq3:",chisq3_List)

            if s1 != 0.0: print ("weighted s1: %f" % (s1))
            if s2 != 0.0: print ("weighted s2: %f" % (s2))
            print ("weighted s_dynamic: %f" % (s_dynamic))

            if nit%refresh_interval == 0:
                print ("Plotting Functionality Temporarily Disabled...")

    loginit = np.hstack(loginit_List).flatten()
    x0 = loginit

    print ("Total Pixel #: ",(N_pixel*N_pixel*N_frame))
    print ("Clipped Pixel #: ",(len(loginit)))

    print ("Initial Values:")
    plotcur(x0)

    # Minimize
    optdict = {'maxiter':maxit, 'ftol':stop, 'maxcor':NHIST, 'gtol': 1e-10} # minimizer params
    tstart = time.time()
    res = opt.minimize(objfunc, x0, method=minimizer_method, jac=objgrad, options=optdict, callback=plotcur)
    tstop = time.time()

    Frames = np.zeros((N_frame, N_pixel, N_pixel))
    log_Frames = np.zeros((N_frame, N_pixel, N_pixel))

    init_i = 0
    for i in range(N_frame):
        cur_len = np.sum(embed_mask_List[i])
        log_Frames[i] = embed(res.x[init_i:(init_i+cur_len)], embed_mask_List[i]).reshape((N_pixel, N_pixel))
        #Impose the prior mask in linear space for the output
        Frames[i] = np.exp(log_Frames[i])*(embed_mask_List[i].reshape((N_pixel, N_pixel)))
        init_i += cur_len

    plotcur(res.x, final=True)

    # Print stats
    print ("time: %f s" % (tstop - tstart))
    print ("J: %f" % res.fun)
    print (res.message)

    outim = [image.Image(Frames[i].reshape(Prior.ydim, Prior.xdim), Prior.psize,
                         Prior.ra, Prior.dec, rf=Obsdata_List[i].rf, source=Prior.source,
                         mjd=Prior.mjd, pulse=Prior.pulse) for i in range(N_frame)]

    return outim


def dynamical_imaging(Obsdata_List, InitIm_List, Prior, Flow_Init = [], flux_List = [],
d1='vis', d2=False, d3=False,
alpha_d1=10, alpha_d2=10, alpha_d3=10,
systematic_noise1=0.0, systematic_noise2=0.0, systematic_noise3=0.0,
entropy1="tv2", entropy2="l1",
alpha_s1=1.0, alpha_s2=1.0, norm_reg=True, alpha_A=1.0,
R_dI  ={'alpha':0.0, 'metric':'SymKL', 'p':2.0},
R_dt  ={'alpha':0.0, 'metric':'SymKL', 'sigma_dt':0.0, 'p':2.0},
R_flow={'alpha':0.0, 'metric':'SymKL', 'p':2.0, 'alpha_flow_tv':50.0},
alpha_centroid=0.0, alpha_flux=0.0, alpha_dF=0.0, alpha_dS1=0.0, alpha_dS2=0.0, #other regularizers
stochastic_optics=False, scattering_model=False, alpha_phi = 1.e4, #options for scattering
Target_Dynamic_Range = 10000.0,
maxit=200, J_factor = 0.001, stop=1.0e-10, ipynb=False, refresh_interval = 1000, 
minimizer_method = 'L-BFGS-B', NHIST = 25, update_interval = 1, clipfloor=0., processes = -1, 
recalculate_chisqdata = True,  ttype = 'nfft', fft_pad_factor=2, **kwargs):

    """Run dynamical imaging.

       Args:
           Obsdata_List (List): List of Obsdata objects, one per reconstructed frame. Some can have empty data arrays.
           InitIm_List (List): List of initial images, each an Image object, one per reconstructed frame.
           Prior (Image): The Image object with the prior image
           Flow_Init: Optional initialization for imaging with R_flow
           flux_List (List): Optional specification of the total flux density for each frame
           d1 (str): The first data term; options are 'vis', 'bs', 'amp', 'cphase', 'camp', 'logcamp'
           d2 (str): The second data term; options are 'vis', 'bs', 'amp', 'cphase', 'camp', 'logcamp'
           d3 (str): The third data term; options are 'vis', 'bs', 'amp', 'cphase', 'camp', 'logcamp'
           systematic_noise1 (float): Systematic noise on the first data term, as a fraction of the visibility amplitude
           systematic_noise2 (float): Systematic noise on the second data term, as a fraction of the visibility amplitude
           systematic_noise3 (float): Systematic noise on the third data term, as a fraction of the visibility amplitude
           entropy1 (str): The first regularizer; options are 'simple', 'gs', 'tv', 'tv2', 'l1', 'patch','compact','compact2','rgauss'
           entropy2 (str): The second regularizer; options are 'simple', 'gs', 'tv', 'tv2','l1', 'patch','compact','compact2','rgauss'
           alpha_d1 (float): The first data term weighting
           alpha_d2 (float): The second data term weighting
           alpha_s1 (float): The first regularizer term weighting
           alpha_s2 (float): The second regularizer term weighting
           alpha_flux (float): The weighting for the total flux constraint
           alpha_centroid (float): The weighting for the center of mass constraint
           alpha_dF (float): The weighting for temporal continuity of the total flux density.
           alpha_dS1 (float): The weighting for temporal continuity of entropy1.
           alpha_dS2 (float): The weighting for temporal continuity of entropy2.

           maxit (int): Maximum number of minimizer iterations
           stop (float): The convergence criterion
           minimizer_method (str): Minimizer method (e.g., 'L-BFGS-B' or 'CG')
           update_interval (int): Print convergence status every update_interval steps
           norm_reg (bool): If True, normalizes regularizer terms
           ttype (str): The Fourier transform type; options are 'fast', 'direct', 'nfft'

           stochastic_optics (bool): If True, stochastic optics imaging is used. 
           scattering_model (ScatteringModel): Optional specification of the ScatteringModel object. 
           alpha_phi (float): Weighting for screen phase regularization in stochastic optics.
minimizer_method = 'L-BFGS-B', update_interval = 1

       Returns:
           List or Dictionary: A list of Image objects, one per frame, unless a flow or stochastic optics is used in which case it returns a dictionary {'Frames', 'Flow', 'EpsilonList' }.
    """

    global A1_List, A2_List, A3_List, data1_List, data2_List, data3_List, sigma1_List, sigma2_List, sigma3_List

    N_frame = len(Obsdata_List)
    N_pixel = Prior.xdim #pixel dimension

    # Determine the appropriate final resolution
    all_res = []
    for obs in Obsdata_List:
        if len(obs.data) > 0:
            all_res.append(obs.res())

    beam_size = np.min(np.array(all_res))
    print("Maximal Resolution:",beam_size)

    # Find an observation with data
    for j in range(N_frame):
        if len(Obsdata_List[j].data) > 0:
            first_obs = Obsdata_List[j]
            break

    # Catch problem if uvrange < largest baseline
    if 1./Prior.psize < np.max(first_obs.unpack(['uvdist'])['uvdist']):
        raise Exception("pixel spacing is larger than smallest spatial wavelength!")

    if alpha_flux > 0.0 and len(flux_List) != N_frame:
        raise Exception("Number of elements in the list of total flux densities does not match the number of frames!")

    # If using stochastic optics, do some preliminary calculations
    if stochastic_optics == True:
        # Doesn't yet work with clipping
        clipfloor = -1.0

        if scattering_model == False:
            print("No scattering model specified. Assuming the default scattering for Sgr A*.")
            scattering_model = so.ScatteringModel()

        # First some preliminary definitions
        N = InitIm_List[0].xdim
        FOV = InitIm_List[0].psize * N * scattering_model.observer_screen_distance #Field of view, in cm, at the scattering screen

        # The ensemble-average convolution kernel and its gradients
        wavelength_List = np.array([C/obs.rf*100.0 for obs in Obsdata_List]) #Observing wavelength for each frame [cm]
        wavelengthbar_List = wavelength_List/(2.0*np.pi) #lambda/(2pi) [cm]
        rF_List = [scattering_model.rF(wavelength) for wavelength in wavelength_List]

        print("Computing the Ensemble-Average Kernel for Each Frame...")
        ea_ker = [scattering_model.Ensemble_Average_Kernel(InitIm_List[0], wavelength_cm = wavelength_List[j]) for j in range(N_frame)]
        ea_ker_gradient = [so.Wrapped_Gradient(ea_ker[j]/(FOV/N)) for j in range(N_frame)]
        ea_ker_gradient_x = [-ea_ker_gradient[j][1] for j in range(N_frame)]
        ea_ker_gradient_y = [-ea_ker_gradient[j][0] for j in range(N_frame)]

        # The power spectrum (note: rotation is not currently implemented; the gradients would need to be modified slightly)
        sqrtQ = np.real(scattering_model.sqrtQ_Matrix(InitIm_List[0],t_hr=0.0))

    # Make the blurring kernel for R_dt
    # Note: There are odd problems when sigma_dt is too small. I can't figure out why it causes the convolution to crash.
    # However, having sigma_dt -> 0 is not a problem in theory. So we'll just set the kernel to be zero in that case and then ignore it later in the convolution.

    if R_dt['sigma_dt'] > 0.0:
        B_dt = np.abs(np.array([[np.exp(-1.0*(float(i)**2+float(j)**2)/(2.*R_dt['sigma_dt']**2))
                              for i in np.linspace((Prior.xdim-1)/2., -(Prior.xdim-1)/2., num=Prior.xdim)]
                              for j in np.linspace((Prior.ydim-1)/2., -(Prior.ydim-1)/2., num=Prior.ydim)]))
        if np.max(B_dt) == 0.0 or np.sum(B_dt) == 0.0:
            raise Exception("Error with the blurring kernel!")
        B_dt = B_dt / np.sum(B_dt) # normalize to be flux preserving
    else:
        B_dt = np.zeros((Prior.ydim,Prior.xdim))

    embed_mask_List = [Prior.imvec > clipfloor for j in range(N_frame)]
    embed_mask_All = np.array(embed_mask_List).flatten()

    embed_totals = [np.sum(embed_mask) for embed_mask in embed_mask_List]
    if len(set(embed_totals)) > 1 and R_flow['alpha'] > 0.0:
        print ("If a flow is used, then each frame must have the same prior!")
        return

    logprior_List = [None,] * N_frame
    loginit_List = [None,] * N_frame

    nprior_embed_List = [None,] * N_frame
    nprior_List = [None,] * N_frame

    ninit_embed_List = [InitIm_List[i].imvec for i in range(N_frame)]
    ninit_List = [ninit_embed_List[i][embed_mask_List[i]] for i in range(N_frame)]

    if (recalculate_chisqdata == True and ttype == 'direct') or ttype != 'direct':
        print ("Calculating lists/matrices for chi-squared terms...")
        A1_List = [None,] * N_frame
        A2_List = [None,] * N_frame
        A3_List = [None,] * N_frame
        data1_List = [[],] * N_frame
        data2_List = [[],] * N_frame
        data3_List = [[],] * N_frame
        sigma1_List = [None,] * N_frame
        sigma2_List = [None,] * N_frame
        sigma3_List = [None,] * N_frame

    # Get data and Fourier matrices for the data terms
    for i in range(N_frame):
        pixel_max = np.max(InitIm_List[i].imvec)
        prior_flux_rescale = 1.0
        if len(flux_List) > 0:
            prior_flux_rescale = flux_List[i]/Prior.total_flux()

        nprior_embed_List[i] = Prior.imvec * prior_flux_rescale

        nprior_List[i] = nprior_embed_List[i][embed_mask_List[i]]
        logprior_List[i] = np.log(nprior_List[i])
        loginit_List[i] = np.log(ninit_List[i] + pixel_max/Target_Dynamic_Range/1.e6)  #add the dynamic range floor here

        if len(Obsdata_List[i].data) == 0:  #This allows the algorithm to create frames for periods with no data
            continue

        if (recalculate_chisqdata == True and ttype == 'direct') or ttype != 'direct':
            (data1_List[i], sigma1_List[i], A1_List[i]) = chisqdata(Obsdata_List[i], Prior, embed_mask_List[i], d1, ttype=ttype, fft_pad_factor=fft_pad_factor, systematic_noise=systematic_noise1)
            (data2_List[i], sigma2_List[i], A2_List[i]) = chisqdata(Obsdata_List[i], Prior, embed_mask_List[i], d2, ttype=ttype, fft_pad_factor=fft_pad_factor, systematic_noise=systematic_noise2)
            (data3_List[i], sigma3_List[i], A3_List[i]) = chisqdata(Obsdata_List[i], Prior, embed_mask_List[i], d3, ttype=ttype, fft_pad_factor=fft_pad_factor, systematic_noise=systematic_noise3)

    # Coordinate matrix for COM constraint
    coord = np.array([[[x,y] for x in np.linspace(Prior.xdim/2,-Prior.xdim/2,Prior.xdim)]
                             for y in np.linspace(Prior.ydim/2,-Prior.ydim/2,Prior.ydim)])
    coord = coord.reshape(Prior.ydim*Prior.xdim, 2)

    # Make the pool for parallel processing
    if processes > 0:
        print("Using Multiprocessing")
        pool = Pool(processes=processes)
    elif processes == 0:
        processes = int(cpu_count())
        print("Using Multiprocessing with %d Processes" % processes)
        pool = Pool(processes=processes)
    else:
        print("Not Using Multiprocessing")

    # Define the objective function and gradient
    def objfunc(x):
        # Frames is a list of the *unscattered* frames
        Frames = np.zeros((N_frame, N_pixel, N_pixel))
        log_Frames = np.zeros((N_frame, N_pixel, N_pixel))

        init_i = 0
        for i in range(N_frame):
            cur_len = np.sum(embed_mask_List[i])
            log_Frames[i] = embed(x[init_i:(init_i+cur_len)], embed_mask_List[i]).reshape((N_pixel, N_pixel))
            Frames[i] = np.exp(log_Frames[i])*(embed_mask_List[i].reshape((N_pixel, N_pixel)))
            init_i += cur_len

        if R_flow['alpha'] != 0.0:
            cur_len = np.sum(embed_mask_List[0]) #assumes all the priors have the same embedding
            Flow_x = embed(x[init_i:(init_i+2*cur_len-1):2],   embed_mask_List[i]).reshape((N_pixel, N_pixel))
            Flow_y = embed(x[(init_i+1):(init_i+2*cur_len):2], embed_mask_List[i]).reshape((N_pixel, N_pixel))
            Flow = np.transpose([Flow_x.ravel(),Flow_y.ravel()]).reshape((N_pixel, N_pixel,2))
            init_i += 2*cur_len

        if stochastic_optics == True:
            EpsilonList = x[init_i:(init_i + N**2-1)]
            im_List = [image.Image(Frames[j], Prior.psize, Prior.ra, Prior.dec, rf=Obsdata_List[j].rf, source=Prior.source, mjd=Prior.mjd) for j in range(N_frame)]
            #the list of scattered image vectors
            scatt_im_List = [scattering_model.Scatter(im_List[j], Epsilon_Screen=so.MakeEpsilonScreenFromList(EpsilonList, N), ea_ker = ea_ker[j], sqrtQ=sqrtQ, Linearized_Approximation=True).imvec for j in range(N_frame)]
            init_i += len(EpsilonList)

        s1 = s2 = 0.0

        if alpha_s1 != 0.0:
            s1 = static_regularizer(Frames, nprior_embed_List, embed_mask_List, Prior.total_flux(), Prior.psize, entropy1, norm_reg=norm_reg, beam_size=beam_size, alpha_A=alpha_A, **kwargs)*alpha_s1
        if alpha_s2 != 0.0:
            s2 = static_regularizer(Frames, nprior_embed_List, embed_mask_List, Prior.total_flux(), Prior.psize, entropy2, norm_reg=norm_reg, beam_size=beam_size, alpha_A=alpha_A, **kwargs)*alpha_s2

        s_dynamic = cm = flux = s_dS = s_dF = 0.0

        if R_dI['alpha'] != 0.0: s_dynamic += RdI(Frames, **R_dI)*R_dI['alpha']
        if R_dt['alpha'] != 0.0: s_dynamic += Rdt(Frames, B_dt, **R_dt)*R_dt['alpha']

        if alpha_dS1 != 0.0: s_dS += RdS(Frames, nprior_embed_List, embed_mask_List, entropy1, norm_reg, beam_size=beam_size, alpha_A=alpha_A)*alpha_dS1
        if alpha_dS2 != 0.0: s_dS += RdS(Frames, nprior_embed_List, embed_mask_List, entropy2, norm_reg, beam_size=beam_size, alpha_A=alpha_A)*alpha_dS2

        if alpha_dF != 0.0: s_dF += RdF_clip(Frames, embed_mask_List)*alpha_dF

        if alpha_centroid != 0.0: cm = centroid(Frames, coord) * alpha_centroid

        if alpha_flux > 0.0:
            flux = alpha_flux * movie_flux_constraint(Frames, flux_List)

        if stochastic_optics == False:
            if processes > 0:
                chisq = np.array(pool.map(get_chisq_wrap, [[j, Frames[j].ravel()[embed_mask_List[j]], d1, d2, d3, ttype, embed_mask_List[j]] for j in range(N_frame)]))
            else:
                chisq = np.array([get_chisq(j, Frames[j].ravel()[embed_mask_List[j]], d1, d2, d3, ttype, embed_mask_List[j]) for j in range(N_frame)])
        else:
            if processes > 0:
                chisq = np.array(pool.map(get_chisq_wrap, [[j, scatt_im_List[j][embed_mask_List[j]], d1, d2, d3, ttype, embed_mask_List[j]] for j in range(N_frame)]))
            else:
                chisq = np.array([get_chisq(j, scatt_im_List[j][embed_mask_List[j]], d1, d2, d3, ttype, embed_mask_List[j]) for j in range(N_frame)])

        chisq = ((np.sum(chisq[:,0])/N_frame - 1.0)*alpha_d1 +
                 (np.sum(chisq[:,1])/N_frame - 1.0)*alpha_d2 +
                 (np.sum(chisq[:,2])/N_frame - 1.0)*alpha_d3)

        if R_flow['alpha'] != 0.0:
            flow_tv =  squared_gradient_flow(Flow)
            s_dynamic += flow_tv*R_flow['alpha_flow_tv']
            s_dynamic += Rflow(Frames, Flow, **R_flow)*R_flow['alpha']

        # Scattering screen regularization term
        regterm_scattering = 0.0
        if stochastic_optics == True:
            chisq_epsilon = sum(EpsilonList*EpsilonList)/((N*N-1.0)/2.0)
            regterm_scattering = alpha_phi * (chisq_epsilon - 1.0)

        return (s1 + s2 + s_dF + s_dS + s_dynamic + chisq + cm + flux + regterm_scattering)*J_factor

    def objgrad(x):
        Frames = np.zeros((N_frame, N_pixel, N_pixel))
        log_Frames = np.zeros((N_frame, N_pixel, N_pixel))

        init_i = 0
        for i in range(N_frame):
            cur_len = np.sum(embed_mask_List[i])
            log_Frames[i] = embed(x[init_i:(init_i+cur_len)], embed_mask_List[i]).reshape((N_pixel, N_pixel))
            Frames[i] = np.exp(log_Frames[i])*(embed_mask_List[i].reshape((N_pixel, N_pixel)))
            init_i += cur_len

        if R_flow['alpha'] != 0.0:
            cur_len = np.sum(embed_mask_List[0]) #assumes all the priors have the same embedding
            Flow_x = embed(x[init_i:(init_i+2*cur_len-1):2],   embed_mask_List[i]).reshape((N_pixel, N_pixel))
            Flow_y = embed(x[(init_i+1):(init_i+2*cur_len):2], embed_mask_List[i]).reshape((N_pixel, N_pixel))
            Flow = np.transpose([Flow_x.ravel(),Flow_y.ravel()]).reshape((N_pixel, N_pixel,2))
            init_i += 2*cur_len

        if stochastic_optics == True:
            EpsilonList = x[init_i:(init_i + N**2-1)]
            Epsilon_Screen = so.MakeEpsilonScreenFromList(EpsilonList, N)
            im_List = [image.Image(Frames[j], Prior.psize, Prior.ra, Prior.dec, rf=Obsdata_List[j].rf, source=Prior.source, mjd=Prior.mjd) for j in range(N_frame)]
            scatt_im_List = [scattering_model.Scatter(im_List[j], Epsilon_Screen=so.MakeEpsilonScreenFromList(EpsilonList, N), ea_ker = ea_ker[j], sqrtQ=sqrtQ, Linearized_Approximation=True).imvec for j in range(N_frame)] #the list of scattered image vectors
            Epsilon_Screen = so.MakeEpsilonScreenFromList(EpsilonList, N)
            init_i += len(EpsilonList)

        s1 = s2 = 0.0

        if alpha_s1 != 0.0:
            s1 = static_regularizer_gradient(Frames, nprior_embed_List, embed_mask_List, Prior.total_flux(), Prior.psize, entropy1, norm_reg=norm_reg, beam_size=beam_size, alpha_A=alpha_A, **kwargs)*alpha_s1
        if alpha_s2 != 0.0:
            s2 = static_regularizer_gradient(Frames, nprior_embed_List, embed_mask_List, Prior.total_flux(), Prior.psize, entropy2, norm_reg=norm_reg, beam_size=beam_size, alpha_A=alpha_A, **kwargs)*alpha_s2

        s_dynamic_grad = cm_grad = flux_grad = s_dS = s_dF = 0.0
        if R_dI['alpha'] != 0.0: s_dynamic_grad += RdI_gradient(Frames,**R_dI)*R_dI['alpha']
        if R_dt['alpha'] != 0.0: s_dynamic_grad += Rdt_gradient(Frames, B_dt, **R_dt)*R_dt['alpha']

        if alpha_dS1 != 0.0: s_dS += RdS_gradient(Frames, nprior_embed_List, embed_mask_List, entropy1, norm_reg, beam_size=beam_size, alpha_A=alpha_A)*alpha_dS1
        if alpha_dS2 != 0.0: s_dS += RdS_gradient(Frames, nprior_embed_List, embed_mask_List, entropy2, norm_reg, beam_size=beam_size, alpha_A=alpha_A)*alpha_dS2

        if alpha_dF != 0.0: s_dF += RdF_gradient_clip(Frames, embed_mask_List)*alpha_dF

        if alpha_centroid != 0.0: cm_grad = centroid_gradient(Frames, coord) * alpha_centroid

        if alpha_flux > 0.0:
            flux_grad = alpha_flux * movie_flux_constraint_grad(Frames, flux_List)

        dchisq_dIa_List = []


        # Michael -- can we do something about this
        if stochastic_optics == False:
            if processes > 0:
                chisq_grad = np.array(pool.map(get_chisqgrad_wrap, [[j, Frames[j].ravel()[embed_mask_List[j]], d1, d2, d3, ttype, embed_mask_List[j]] for j in range(N_frame)]))
            else:
                chisq_grad = np.array([get_chisqgrad(j, Frames[j].ravel()[embed_mask_List[j]], d1, d2, d3, ttype, embed_mask_List[j]) for j in range(N_frame)])
        else:
            if processes > 0:
                chisq_grad = np.array(pool.map(get_chisqgrad_wrap, [[j, scatt_im_List[j][embed_mask_List[j]], d1, d2, d3, ttype, embed_mask_List[j]] for j in range(N_frame)]))
            else:
                chisq_grad = np.array([get_chisqgrad(j, scatt_im_List[j][embed_mask_List[j]], d1, d2, d3, ttype, embed_mask_List[j]) for j in range(N_frame)])

            # Now, the chi^2 gradient must be modified so that it corresponds to the gradient wrt the unscattered image
            for j in range(N_frame):
                rF = rF_List[j]
                phi = scattering_model.MakePhaseScreen(Epsilon_Screen, im_List[0], obs_frequency_Hz=im_List[j].rf,sqrtQ_init=sqrtQ).imvec.reshape((N, N))
                phi_Gradient = so.Wrapped_Gradient(phi/(FOV/N))
                phi_Gradient_x = -phi_Gradient[1]
                phi_Gradient_y = -phi_Gradient[0]
                dchisq_dIa_List.append( ((chisq_grad[j,0]*alpha_d1 + chisq_grad[j,1]*alpha_d2 + chisq_grad[j,2]*alpha_d3)/N_frame).reshape((N,N)) )

                dchisq_dIa = chisq_grad[j,0].reshape((N,N))
                gx = (rF**2.0 * so.Wrapped_Convolve(ea_ker_gradient_x[j][::-1,::-1], phi_Gradient_x * (dchisq_dIa))).flatten()
                gy = (rF**2.0 * so.Wrapped_Convolve(ea_ker_gradient_y[j][::-1,::-1], phi_Gradient_y * (dchisq_dIa))).flatten()
                chisq_grad[j,0] = so.Wrapped_Convolve(ea_ker[j][::-1,::-1], (dchisq_dIa)).flatten() + gx + gy

                dchisq_dIa = chisq_grad[j,1].reshape((N,N))
                gx = (rF**2.0 * so.Wrapped_Convolve(ea_ker_gradient_x[j][::-1,::-1], phi_Gradient_x * (dchisq_dIa))).flatten()
                gy = (rF**2.0 * so.Wrapped_Convolve(ea_ker_gradient_y[j][::-1,::-1], phi_Gradient_y * (dchisq_dIa))).flatten()
                chisq_grad[j,1] = so.Wrapped_Convolve(ea_ker[j][::-1,::-1], (dchisq_dIa)).flatten() + gx + gy

                dchisq_dIa = chisq_grad[j,2].reshape((N,N))
                gx = (rF**2.0 * so.Wrapped_Convolve(ea_ker_gradient_x[j][::-1,::-1], phi_Gradient_x * (dchisq_dIa))).flatten()
                gy = (rF**2.0 * so.Wrapped_Convolve(ea_ker_gradient_y[j][::-1,::-1], phi_Gradient_y * (dchisq_dIa))).flatten()
                chisq_grad[j,2] = so.Wrapped_Convolve(ea_ker[j][::-1,::-1], (dchisq_dIa)).flatten() + gx + gy

        # Now add the Jacobian factor and concatenate
        for j in range(N_frame):
            chisq_grad[j,0] = chisq_grad[j,0]*Frames[j].ravel()[embed_mask_List[j]]
            chisq_grad[j,1] = chisq_grad[j,1]*Frames[j].ravel()[embed_mask_List[j]]
            chisq_grad[j,2] = chisq_grad[j,2]*Frames[j].ravel()[embed_mask_List[j]]

        chisq_grad = (np.concatenate([embed(chisq_grad[i,0], embed_mask_List[i]) for i in range(N_frame)])/N_frame*alpha_d1
                    + np.concatenate([embed(chisq_grad[i,1], embed_mask_List[i]) for i in range(N_frame)])/N_frame*alpha_d2
                    + np.concatenate([embed(chisq_grad[i,2], embed_mask_List[i]) for i in range(N_frame)])/N_frame*alpha_d3)

        # Gradient of the data chi^2 wrt to the epsilon screen -- this is the really difficult one
        chisq_grad_epsilon = np.array([])
        if stochastic_optics == True:
            #Preliminary Definitions
            chisq_grad_epsilon = np.zeros(N**2-1)
            ell_mat = np.zeros((N,N))
            m_mat   = np.zeros((N,N))
            for ell in range(0, N):
                for m in range(0, N):
                    ell_mat[ell,m] = ell
                    m_mat[ell,m] = m

            for j in range(N_frame):
                rF = rF_List[j]
                dchisq_dIa = dchisq_dIa_List[j]
                EA_Image = scattering_model.Ensemble_Average_Blur(im_List[j], ker = ea_ker[j])
                EA_Gradient = so.Wrapped_Gradient((EA_Image.imvec/(FOV/N)).reshape(N, N))
                #The gradient signs don't actually matter, but let's make them match intuition (i.e., right to left, bottom to top)
                EA_Gradient_x = -EA_Gradient[1]
                EA_Gradient_y = -EA_Gradient[0]

                i_grad = 0
                #Real part; top row
                for t in range(1, (N+1)//2):
                    s=0
                    grad_term = so.Wrapped_Gradient(wavelengthbar_List[j]/FOV*sqrtQ[s][t]*2.0*np.cos(2.0*np.pi/N*(ell_mat*s + m_mat*t))/(FOV/N))
                    grad_term_x = -grad_term[1]
                    grad_term_y = -grad_term[0]
                    chisq_grad_epsilon[i_grad] += np.sum( dchisq_dIa * rF**2 * ( EA_Gradient_x * grad_term_x + EA_Gradient_y * grad_term_y ) )
                    i_grad = i_grad + 1

                #Real part; remainder
                for s in range(1,(N+1)//2):
                    for t in range(N):
                        grad_term = so.Wrapped_Gradient(wavelengthbar_List[j]/FOV*sqrtQ[s][t]*2.0*np.cos(2.0*np.pi/N*(ell_mat*s + m_mat*t))/(FOV/N))
                        grad_term_x = -grad_term[1]
                        grad_term_y = -grad_term[0]
                        chisq_grad_epsilon[i_grad] += np.sum( dchisq_dIa * rF**2 * ( EA_Gradient_x * grad_term_x + EA_Gradient_y * grad_term_y ) )
                        i_grad = i_grad + 1

                #Imaginary part; top row
                for t in range(1, (N+1)//2):
                    s=0
                    grad_term = so.Wrapped_Gradient(-wavelengthbar_List[j]/FOV*sqrtQ[s][t]*2.0*np.sin(2.0*np.pi/N*(ell_mat*s + m_mat*t))/(FOV/N))
                    grad_term_x = -grad_term[1]
                    grad_term_y = -grad_term[0]
                    chisq_grad_epsilon[i_grad] += np.sum( dchisq_dIa * rF**2 * ( EA_Gradient_x * grad_term_x + EA_Gradient_y * grad_term_y ) )
                    i_grad = i_grad + 1

                #Imaginary part; remainder
                for s in range(1,(N+1)//2):
                    for t in range(N):
                        grad_term = so.Wrapped_Gradient(-wavelengthbar_List[j]/FOV*sqrtQ[s][t]*2.0*np.sin(2.0*np.pi/N*(ell_mat*s + m_mat*t))/(FOV/N))
                        grad_term_x = -grad_term[1]
                        grad_term_y = -grad_term[0]
                        chisq_grad_epsilon[i_grad] += np.sum( dchisq_dIa * rF**2 * ( EA_Gradient_x * grad_term_x + EA_Gradient_y * grad_term_y ) )
                        i_grad = i_grad + 1

        # Gradients related to the flow
        flow_grad = np.array([])
        if R_flow['alpha'] != 0.0:
            cur_len = np.sum(embed_mask_List[0])
            Flow_x = embed(x[init_i:(init_i+2*cur_len-1):2],   embed_mask_List[i]).reshape((N_pixel, N_pixel))
            Flow_y = embed(x[(init_i+1):(init_i+2*cur_len):2], embed_mask_List[i]).reshape((N_pixel, N_pixel))
            Flow = np.transpose([Flow_x.ravel(),Flow_y.ravel()]).reshape((N_pixel, N_pixel,2))
            flow_tv_grad = squared_gradient_flow_grad(Flow)
            s_dynamic_grad_Frames = s_dynamic_grad_Flow = 0.0
            s_dynamic_grad_Frames = Rflow_gradient_I(Frames, Flow, R_flow)
            s_dynamic_grad_Flow   = Rflow_gradient_m(Frames, Flow, R_flow)

            s_dynamic_grad += s_dynamic_grad_Frames*R_flow['alpha']
            flow_grad = s_dynamic_grad_Flow*R_flow['alpha'] + flow_tv_grad*R_flow['alpha_flow_tv']
            # now handle the embedding
            flow_grad = np.transpose([flow_grad[::2][embed_mask_List[0]], flow_grad[1::2][embed_mask_List[0]]]).ravel()

        # Gradient of the chi^2 regularization term for the epsilon screen
        chisq_epsilon_grad = np.array([])
        if stochastic_optics == True:
            chisq_epsilon_grad = alpha_phi * 2.0*EpsilonList/((N*N-1)/2.0)

        return (np.concatenate((s1 + s2 + s_dF + s_dS + (s_dynamic_grad + chisq_grad + cm_grad + cm_grad + flux_grad)[embed_mask_All], flow_grad, chisq_grad_epsilon + chisq_epsilon_grad))*J_factor)

    # Plotting function for each iteration
    global nit
    nit = 0
    def plotcur(x, final=False):
        global nit
        nit += 1

        if nit%update_interval == 0 or final == True:
            print ("iteration %d" % nit)

            Frames = np.zeros((N_frame, N_pixel, N_pixel))
            log_Frames = np.zeros((N_frame, N_pixel, N_pixel))

            init_i = 0
            for i in range(N_frame):
                cur_len = np.sum(embed_mask_List[i])
                log_Frames[i] = embed(x[init_i:(init_i+cur_len)], embed_mask_List[i]).reshape((N_pixel, N_pixel))
                Frames[i] = np.exp(log_Frames[i])*(embed_mask_List[i].reshape((N_pixel, N_pixel)))
                init_i += cur_len

            if R_flow['alpha'] != 0.0:
                cur_len = np.sum(embed_mask_List[0]) #assumes all the priors have the same embedding
                Flow_x = embed(x[init_i:(init_i+2*cur_len-1):2],   embed_mask_List[i]).reshape((N_pixel, N_pixel))
                Flow_y = embed(x[(init_i+1):(init_i+2*cur_len):2], embed_mask_List[i]).reshape((N_pixel, N_pixel))
                Flow = np.transpose([Flow_x.ravel(),Flow_y.ravel()]).reshape((N_pixel, N_pixel,2))
                init_i += 2*cur_len

            if stochastic_optics == True:
                EpsilonList = x[init_i:(init_i + N**2-1)]
                im_List = [image.Image(Frames[j], Prior.psize, Prior.ra, Prior.dec, rf=Obsdata_List[j].rf, source=Prior.source, mjd=Prior.mjd) for j in range(N_frame)]

                scatt_im_List = [scattering_model.Scatter(im_List[j], Epsilon_Screen=so.MakeEpsilonScreenFromList(EpsilonList, N), ea_ker = ea_ker[j], sqrtQ=sqrtQ, Linearized_Approximation=True).imvec
                                 for j in range(N_frame)] #the list of scattered image vectors

            s1 = s2 = 0.0

            if alpha_s1 != 0.0:

                s1 = static_regularizer(Frames, nprior_embed_List, embed_mask_List, Prior.total_flux(),
                                        Prior.psize, entropy1, norm_reg=norm_reg, beam_size=beam_size, alpha_A=alpha_A, **kwargs)*alpha_s1
            if alpha_s2 != 0.0:
                s2 = static_regularizer(Frames, nprior_embed_List, embed_mask_List, Prior.total_flux(),
                                        Prior.psize, entropy2, norm_reg=norm_reg, beam_size=beam_size, alpha_A=alpha_A, **kwargs)*alpha_s2

            s_dynamic = cm = s_dS = s_dF = 0.0

            if R_dI['alpha'] != 0.0: s_dynamic += RdI(Frames, **R_dI)*R_dI['alpha']
            if R_dt['alpha'] != 0.0: s_dynamic += Rdt(Frames, B_dt, **R_dt)*R_dt['alpha']

            if alpha_dS1 != 0.0: s_dS += RdS(Frames, nprior_embed_List, embed_mask_List, entropy1, norm_reg, beam_size=beam_size, alpha_A=alpha_A, **kwargs)*alpha_dS1
            if alpha_dS2 != 0.0: s_dS += RdS(Frames, nprior_embed_List, embed_mask_List, entropy2, norm_reg, beam_size=beam_size, alpha_A=alpha_A, **kwargs)*alpha_dS2

            if alpha_dF != 0.0: s_dF += RdF_clip(Frames, embed_mask_List)*alpha_dF

            if alpha_centroid != 0.0: cm = centroid(Frames, coord) * alpha_centroid

            if stochastic_optics == False:
                if processes > 0:

                    chisq = np.array(pool.map(get_chisq_wrap, [[j, Frames[j].ravel()[embed_mask_List[j]],
                                              d1, d2, d3, ttype, embed_mask_List[j]] for j in range(N_frame)]))
                else:
                    chisq = np.array([get_chisq(j, Frames[j].ravel()[embed_mask_List[j]],
                                      d1, d2, d3, ttype, embed_mask_List[j]) for j in range(N_frame)])
            else:
                if processes > 0:
                    chisq = np.array(pool.map(get_chisq_wrap, [[j, scatt_im_List[j][embed_mask_List[j]],
                                              d1, d2, d3, ttype, embed_mask_List[j]] for j in range(N_frame)]))
                else:
                    chisq = np.array([get_chisq(j, scatt_im_List[j][embed_mask_List[j]],
                                      d1, d2, d3, ttype, embed_mask_List[j]) for j in range(N_frame)])

            chisq1_List = chisq[:,0]
            chisq2_List = chisq[:,1]
            chisq3_List = chisq[:,2]
            chisq1 = np.sum(chisq1_List)/N_frame
            chisq2 = np.sum(chisq2_List)/N_frame
            chisq3 = np.sum(chisq3_List)/N_frame
            chisq1_max = np.max(chisq1_List)
            chisq2_max = np.max(chisq2_List)
            chisq3_max = np.max(chisq3_List)
            if d1 != False: print ("chi2_1: %f" % chisq1)
            if d2 != False: print ("chi2_2: %f" % chisq2)
            if d3 != False: print ("chi2_3: %f" % chisq3)
            if d1 != False: print ("weighted chi2_1: %f" % (chisq1 * alpha_d1))
            if d2 != False: print ("weighted chi2_2: %f" % (chisq2 * alpha_d2))
            if d3 != False: print ("weighted chi2_3: %f" % (chisq3 * alpha_d3))
            if d1 != False: print ("Max Frame chi2_1: %f" % chisq1_max)
            if d2 != False: print ("Max Frame chi2_2: %f" % chisq2_max)
            if d3 != False: print ("Max Frame chi2_3: %f" % chisq3_max)

            if final == True:
                if d1 != False: print ("All chisq1:",chisq1_List)
                if d2 != False: print ("All chisq2:",chisq2_List)
                if d3 != False: print ("All chisq3:",chisq3_List)

            # Now deal with the a flow, if necessary
            if R_flow['alpha'] != 0.0:
                flow_tv =  squared_gradient_flow(Flow)
                s_dynamic += flow_tv*R_flow['alpha_flow_tv']
                print ("Weighted Flow TV: %f" % (flow_tv*R_flow['alpha_flow_tv']))
                s_dynamic += Rflow(Frames, Flow, **R_flow)*R_flow['alpha']
                print ("Weighted R_Flow: %f" % (Rflow(Frames, Flow, **R_flow)*R_flow['alpha']))

            if s1 != 0.0: print ("weighted s1: %f" % (s1))
            if s2 != 0.0: print ("weighted s2: %f" % (s2))
            if s_dF != 0.0: print ("weighted s_dF: %f" % (s_dF))
            if s_dS != 0.0: print ("weighted s_dS: %f" % (s_dS))
            print ("weighted s_dynamic: %f" % (s_dynamic))
            if alpha_centroid > 0.0: print ("weighted COM: %f" % cm)

            if alpha_flux > 0.0:
                print ("weighted flux constraint: %f" % (alpha_flux * movie_flux_constraint(Frames, flux_List)))

            if stochastic_optics == True:
                chisq_epsilon = sum(EpsilonList*EpsilonList)/((N*N-1.0)/2.0)
                regterm_scattering = alpha_phi * (chisq_epsilon - 1.0)
                print("Epsilon chi^2 : %0.2f " % (chisq_epsilon))
                print("Weighted Epsilon chi^2 : %0.2f " % (regterm_scattering))
                print("Max |Epsilon| : %0.2f " % (max(abs(EpsilonList))))

            if nit%refresh_interval == 0:
                print ("Plotting Functionality Temporarily Disabled...")

    loginit = np.hstack(loginit_List).flatten()
    if R_flow['alpha'] == 0.0:
        x0 = loginit
    else:
        Flow_Init_embed = np.transpose([Flow_Init.ravel()[::2][embed_mask_List[0]],Flow_Init.ravel()[1::2][embed_mask_List[0]]]).ravel()
        x0 = np.concatenate( (loginit, Flow_Init_embed) )

    if stochastic_optics == True:
        x0 = np.concatenate((x0,np.zeros(N**2-1)))


    print ("Total Pixel #: ",(N_pixel*N_pixel*N_frame))
    print ("Clipped Pixel #: ",(len(loginit)))

    print ("Initial Values:")
    plotcur(x0)

    # Minimize
    optdict = {'maxiter':maxit, 'ftol':stop, 'maxcor':NHIST, 'gtol': 1e-10} # minimizer params
    tstart = time.time()
    res = opt.minimize(objfunc, x0, method=minimizer_method, jac=objgrad, options=optdict, callback=plotcur)
    tstop = time.time()

    Frames = np.zeros((N_frame, N_pixel, N_pixel))
    log_Frames = np.zeros((N_frame, N_pixel, N_pixel))

    init_i = 0
    for i in range(N_frame):
        cur_len = np.sum(embed_mask_List[i])
        log_Frames[i] = embed(res.x[init_i:(init_i+cur_len)], embed_mask_List[i]).reshape((N_pixel, N_pixel))
        #Impose the prior mask in linear space for the output
        Frames[i] = np.exp(log_Frames[i])*(embed_mask_List[i].reshape((N_pixel, N_pixel)))
        init_i += cur_len

    Flow = EpsilonList = False

    if R_flow['alpha'] != 0.0:
        print ("Collecting Flow...")
        cur_len = np.sum(embed_mask_List[0])
        Flow_x = embed(res.x[init_i:(init_i+2*cur_len-1):2],   embed_mask_List[i]).reshape((N_pixel, N_pixel))
        Flow_y = embed(res.x[(init_i+1):(init_i+2*cur_len):2], embed_mask_List[i]).reshape((N_pixel, N_pixel))
        Flow = np.transpose([Flow_x.ravel(),Flow_y.ravel()]).reshape((N_pixel, N_pixel,2))
        init_i += 2*cur_len

    if stochastic_optics == True:
        EpsilonList = res.x[init_i:(init_i + N**2-1)]
        init_i += len(EpsilonList)


    plotcur(res.x, final=True)

    # Print stats
    print ("time: %f s" % (tstop - tstart))
    print ("J: %f" % res.fun)
    print (res.message)

    #Note: the global variables are *not* released to avoid recalculation
    if processes != -1:
        pool.close()

    #Return Frames

    outim = [image.Image(Frames[i].reshape(Prior.ydim, Prior.xdim), Prior.psize,
                         Prior.ra, Prior.dec, rf=Obsdata_List[i].rf, source=Prior.source,
                         mjd=Prior.mjd, pulse=Prior.pulse) for i in range(N_frame)]

    if R_flow['alpha'] == 0.0 and stochastic_optics == False:
        return outim
    else:
        return {'Frames':outim, 'Flow':Flow, 'EpsilonList':EpsilonList }



def multifreq_dynamical_imaging(Obsdata_Multifreq_List, InitIm_Multifreq_List, Prior, flux_Multifreq_List = [],
d1='vis', d2=False, d3=False,
alpha_d1=10, alpha_d2=10, alpha_d3=10,
systematic_noise1=0.0, systematic_noise2=0.0, systematic_noise3=0.0,
entropy1="tv2", entropy2="l1",
alpha_s1=1.0, alpha_s2=1.0, norm_reg=True, alpha_A=1.0,
R_dI  ={'alpha':0.0, 'metric':'SymKL', 'p':2.0},
R_dt  ={'alpha':0.0, 'metric':'SymKL', 'sigma_dt':0.0, 'p':2.0},
R_dt_multifreq ={'alpha':0.0, 'metric':'SymKL', 'sigma_dt':0.0, 'p':2.0},
alpha_centroid=0.0, alpha_flux=0.0, alpha_dF=0.0, alpha_dS1=0.0, alpha_dS2=0.0, #other regularizers
Target_Dynamic_Range = 10000.0,
maxit=200, J_factor = 0.001, stop=1.0e-10, ipynb=False, refresh_interval = 1000, minimizer_method = 'L-BFGS-B', NHIST = 25,  update_interval = 1, clipfloor=0., recalculate_chisqdata = True,  ttype = 'nfft', fft_pad_factor=2):
    """Run dynamic imager
       Uses I = exp(I') change of variables.
       Obsdata_List is a list of Obsdata objects, InitIm_List is a list of Image objects, and Prior is an Image object.
       Returns list of Image objects, one per frame (unless a flow or stochastic optics is used)
       ttype = 'direct' or 'fast' or 'nfft'
    """

    global A1_List, A2_List, A3_List, data1_List, data2_List, data3_List, sigma1_List, sigma2_List, sigma3_List

    N_freq  = len(Obsdata_Multifreq_List)
    N_frame = len(Obsdata_Multifreq_List[0])
    N_pixel = Prior.xdim #pixel dimension

    # Flatten the input lists
    flux_List    = [x for y in flux_Multifreq_List    for x in y]
    InitIm_List  = [x for y in InitIm_Multifreq_List  for x in y]
    Obsdata_List = [x for y in Obsdata_Multifreq_List for x in y]

    # Determine the appropriate final resolution
    all_res = [[] for j in range(N_freq)]
    for j in range(len(Obsdata_Multifreq_List)):
        for obs in Obsdata_Multifreq_List[j]:
            if len(obs.data) > 0:
                all_res[j].append(obs.res())

    # Determine the beam size for each frequency
    beam_size = [np.min(all_res[j]) for j in range(N_freq)]
    print("Maximal Resolutions:",beam_size)

    if alpha_flux > 0.0 and len(flux_Multifreq_List[0]) != N_frame:
        raise Exception("Number of elements in the list of total flux densities does not match the number of frames!")

    # Make the blurring kernel for R_dt
    # Note: There are odd problems when sigma_dt is too small. I can't figure out why it causes the convolution to crash.
    # However, having sigma_dt -> 0 is not a problem in theory. So we'll just set the kernel to be zero in that case and then ignore it later in the convolution.

    if R_dt['sigma_dt'] > 0.0:
        B_dt = np.abs(np.array([[np.exp(-1.0*(float(i)**2+float(j)**2)/(2.*R_dt['sigma_dt']**2))
                              for i in np.linspace((Prior.xdim-1)/2., -(Prior.xdim-1)/2., num=Prior.xdim)]
                              for j in np.linspace((Prior.ydim-1)/2., -(Prior.ydim-1)/2., num=Prior.ydim)]))
        if np.max(B_dt) == 0.0 or np.sum(B_dt) == 0.0:
            raise Exception("Error with the blurring kernel!")
        B_dt = B_dt / np.sum(B_dt) # normalize to be flux preserving
    else:
        B_dt = np.zeros((Prior.ydim,Prior.xdim))

    if R_dt_multifreq['sigma_dt'] > 0.0:
        B_dt_multifreq = np.abs(np.array([[np.exp(-1.0*(float(i)**2+float(j)**2)/(2.*R_dt_multifreq['sigma_dt']**2))
                              for i in np.linspace((Prior.xdim-1)/2., -(Prior.xdim-1)/2., num=Prior.xdim)]
                              for j in np.linspace((Prior.ydim-1)/2., -(Prior.ydim-1)/2., num=Prior.ydim)]))
        if np.max(B_dt_multifreq) == 0.0 or np.sum(B_dt_multifreq) == 0.0:
            raise Exception("Error with the blurring kernel!")
        B_dt_multifreq = B_dt_multifreq / np.sum(B_dt_multifreq) # normalize to be flux preserving
    else:
        B_dt_multifreq = np.zeros((Prior.ydim,Prior.xdim))

    embed_mask_List = [Prior.imvec > clipfloor for j in range(N_freq*N_frame)]
    embed_mask_All = np.array(embed_mask_List).flatten()

    embed_totals = [np.sum(embed_mask) for embed_mask in embed_mask_List]

    logprior_List = [None,] * (N_freq * N_frame)
    loginit_List = [None,] * (N_freq * N_frame)

    nprior_embed_List = [None,] * (N_freq * N_frame)
    nprior_List = [None,] * (N_freq * N_frame)

    ninit_embed_List = [InitIm_List[i].imvec for i in range(N_freq * N_frame)]
    ninit_List = [ninit_embed_List[i][embed_mask_List[i]] for i in range(N_freq * N_frame)]

    if (recalculate_chisqdata == True and ttype == 'direct') or ttype != 'direct':
        print ("Calculating lists/matrices for chi-squared terms...")
        A1_List = [None,] * N_freq * N_frame
        A2_List = [None,] * N_freq * N_frame
        A3_List = [None,] * N_freq * N_frame
        data1_List = [[],] * N_freq * N_frame
        data2_List = [[],] * N_freq * N_frame
        data3_List = [[],] * N_freq * N_frame
        sigma1_List = [None,] * N_freq * N_frame
        sigma2_List = [None,] * N_freq * N_frame
        sigma3_List = [None,] * N_freq * N_frame

    # Get data and Fourier matrices for the data terms
    for i in range(N_frame*N_freq):
        pixel_max = np.max(InitIm_List[i].imvec)
        prior_flux_rescale = 1.0
        if len(flux_List) > 0:
            prior_flux_rescale = flux_List[i]/Prior.total_flux()

        nprior_embed_List[i] = Prior.imvec * prior_flux_rescale

        nprior_List[i] = nprior_embed_List[i][embed_mask_List[i]]
        logprior_List[i] = np.log(nprior_List[i])
        loginit_List[i] = np.log(ninit_List[i] + pixel_max/Target_Dynamic_Range/1.e6)  #add the dynamic range floor here

        if len(Obsdata_List[i].data) == 0:  #This allows the algorithm to create frames for periods with no data
            continue

        if (recalculate_chisqdata == True and ttype == 'direct') or ttype != 'direct':
            (data1_List[i], sigma1_List[i], A1_List[i]) = chisqdata(Obsdata_List[i], Prior, embed_mask_List[i], d1, ttype=ttype, fft_pad_factor=fft_pad_factor, systematic_noise=systematic_noise1)
            (data2_List[i], sigma2_List[i], A2_List[i]) = chisqdata(Obsdata_List[i], Prior, embed_mask_List[i], d2, ttype=ttype, fft_pad_factor=fft_pad_factor, systematic_noise=systematic_noise2)
            (data3_List[i], sigma3_List[i], A3_List[i]) = chisqdata(Obsdata_List[i], Prior, embed_mask_List[i], d3, ttype=ttype, fft_pad_factor=fft_pad_factor, systematic_noise=systematic_noise3)

    # Coordinate matrix for COM constraint
    coord = np.array([[[x,y] for x in np.linspace(Prior.xdim/2,-Prior.xdim/2,Prior.xdim)]
                                           for y in np.linspace(Prior.ydim/2,-Prior.ydim/2,Prior.ydim)])
    coord = coord.reshape(Prior.ydim*Prior.xdim, 2)

    # Define the objective function and gradient
    def objfunc(x):
        Frames = np.zeros((N_freq*N_frame, N_pixel, N_pixel))
        log_Frames = np.zeros((N_freq*N_frame, N_pixel, N_pixel))

        init_i = 0
        for i in range(N_freq*N_frame):
            cur_len = np.sum(embed_mask_List[i])
            log_Frames[i] = embed(x[init_i:(init_i+cur_len)], embed_mask_List[i]).reshape((N_pixel, N_pixel))
            Frames[i] = np.exp(log_Frames[i])*(embed_mask_List[i].reshape((N_pixel, N_pixel)))
            init_i += cur_len

        s1 = s2 = s_multifreq = s_dynamic = cm = flux = s_dS = s_dF = 0.0

        # Multifrequency part
        if R_dt_multifreq['alpha'] != 0.0: 
            for j in range(N_frame):
                s_multifreq += Rdt(Frames[j::N_frame], B_dt_multifreq, **R_dt_multifreq)*R_dt_multifreq['alpha']

        # Individual frequencies
        for j in range(N_freq):
            i1 = j*N_frame
            i2 = (j+1)*N_frame

            if alpha_s1 != 0.0:
                s1 += static_regularizer(Frames[i1:i2], nprior_embed_List[i1:i2], embed_mask_List[i1:i2], Prior.total_flux(), Prior.psize, entropy1, norm_reg=norm_reg, beam_size=beam_size[j], alpha_A=alpha_A)*alpha_s1
            if alpha_s2 != 0.0:
                s2 += static_regularizer(Frames[i1:i2], nprior_embed_List[i1:i2], embed_mask_List[i1:i2], Prior.total_flux(), Prior.psize, entropy2, norm_reg=norm_reg, beam_size=beam_size[j], alpha_A=alpha_A)*alpha_s2

            if R_dI['alpha'] != 0.0: s_dynamic += RdI(Frames[i1:i2], **R_dI)*R_dI['alpha']
            if R_dt['alpha'] != 0.0: s_dynamic += Rdt(Frames[i1:i2], B_dt, **R_dt)*R_dt['alpha']

            if alpha_dS1 != 0.0: s_dS += RdS(Frames[i1:i2], nprior_embed_List[i1:i2], embed_mask_List[i1:i2], entropy1, norm_reg, beam_size=beam_size[j], alpha_A=alpha_A)*alpha_dS1
            if alpha_dS2 != 0.0: s_dS += RdS(Frames[i1:i2], nprior_embed_List[i1:i2], embed_mask_List[i1:i2], entropy2, norm_reg, beam_size=beam_size[j], alpha_A=alpha_A)*alpha_dS2

            if alpha_dF != 0.0: s_dF += RdF_clip(Frames[i1:i2], embed_mask_List[i1:i2])*alpha_dF

        if alpha_centroid != 0.0: cm = centroid(Frames, coord) * alpha_centroid

        if alpha_flux > 0.0:
            flux = alpha_flux * movie_flux_constraint(Frames, flux_List)

        chisq = np.array([get_chisq(j, Frames[j].ravel()[embed_mask_List[j]], d1, d2, d3, ttype, embed_mask_List[j]) for j in range(N_frame*N_freq)])

        chisq = ((np.sum(chisq[:,0])/(N_freq*N_frame) - 1.0)*alpha_d1 +
                 (np.sum(chisq[:,1])/(N_freq*N_frame) - 1.0)*alpha_d2 +
                 (np.sum(chisq[:,2])/(N_freq*N_frame) - 1.0)*alpha_d3)

        return (s1 + s2 + s_dF + s_dS + s_multifreq + s_dynamic + chisq + cm + flux)*J_factor

    def objgrad(x):
        Frames = np.zeros((N_freq*N_frame, N_pixel, N_pixel))
        log_Frames = np.zeros((N_freq*N_frame, N_pixel, N_pixel))

        init_i = 0
        for i in range(N_freq*N_frame):
            cur_len = np.sum(embed_mask_List[i])
            log_Frames[i] = embed(x[init_i:(init_i+cur_len)], embed_mask_List[i]).reshape((N_pixel, N_pixel))
            Frames[i] = np.exp(log_Frames[i])*(embed_mask_List[i].reshape((N_pixel, N_pixel)))
            init_i += cur_len

        s1 = s2 = s_dS = s_dF = np.zeros((N_freq*N_frame*cur_len))
        s_dynamic_grad = cm_grad = flux_grad = np.zeros((N_freq*N_frame*N_pixel*N_pixel))
        s_multifreq = 0.0

        # Multifrequency part
        if R_dt_multifreq['alpha'] != 0.0:
            s_multifreq = np.zeros((N_freq*N_frame, N_pixel*N_pixel))
            for j in range(N_frame):
                s_multifreq[j::N_frame] += Rdt_gradient(Frames[j::N_frame], B_dt_multifreq, **R_dt_multifreq).reshape((N_freq,N_pixel*N_pixel))*R_dt_multifreq['alpha']
            s_multifreq = s_multifreq.reshape(N_freq*N_frame*N_pixel*N_pixel)

        # Individual frequencies
        for j in range(N_freq):
            i1 = j*N_frame
            i2 = (j+1)*N_frame
            f1 = j*N_frame*N_pixel*N_pixel 
            f2 = (j+1)*N_frame*N_pixel*N_pixel
            mf1 = j*N_frame*cur_len # Note: This assumes that all priors have the same number of masked pixels!
            mf2 = (j+1)*N_frame*cur_len


            if alpha_s1 != 0.0:
                s1[mf1:mf2] = static_regularizer_gradient(Frames[i1:i2], nprior_embed_List[i1:i2], embed_mask_List[i1:i2], Prior.total_flux(), Prior.psize, entropy1, norm_reg=norm_reg, beam_size=beam_size[j], alpha_A=alpha_A)*alpha_s1
            if alpha_s2 != 0.0:
                s2[mf1:mf2] = static_regularizer_gradient(Frames[i1:i2], nprior_embed_List[i1:i2], embed_mask_List[i1:i2], Prior.total_flux(), Prior.psize, entropy2, norm_reg=norm_reg, beam_size=beam_size[j], alpha_A=alpha_A)*alpha_s2

            if R_dI['alpha'] != 0.0: s_dynamic_grad[f1:f2] += RdI_gradient(Frames[i1:i2],**R_dI)*R_dI['alpha']
            if R_dt['alpha'] != 0.0: s_dynamic_grad[f1:f2] += Rdt_gradient(Frames[i1:i2], B_dt, **R_dt)*R_dt['alpha']

            if alpha_dS1 != 0.0: s_dS[mf1:mf2] += RdS_gradient(Frames[i1:i2], nprior_embed_List[i1:i2], embed_mask_List[i1:i2], entropy1, norm_reg, beam_size=beam_size[j], alpha_A=alpha_A)*alpha_dS1
            if alpha_dS2 != 0.0: s_dS[mf1:mf2] += RdS_gradient(Frames[i1:i2], nprior_embed_List[i1:i2], embed_mask_List[i1:i2], entropy2, norm_reg, beam_size=beam_size[j], alpha_A=alpha_A)*alpha_dS2

            if alpha_dF != 0.0: s_dF[mf1:mf2] += RdF_gradient_clip(Frames[i1:i2], embed_mask_List[i1:i2])*alpha_dF

        if alpha_centroid != 0.0: cm_grad = centroid_gradient(Frames, coord) * alpha_centroid

        if alpha_flux > 0.0:
            flux_grad = alpha_flux * movie_flux_constraint_grad(Frames, flux_List)

        chisq_grad = np.array([get_chisqgrad(j, Frames[j].ravel()[embed_mask_List[j]], d1, d2, d3, ttype, embed_mask_List[j]) for j in range(N_freq*N_frame)])

        # Now add the Jacobian factor and concatenate
        for j in range(N_freq*N_frame):
            chisq_grad[j,0] = chisq_grad[j,0]*Frames[j].ravel()[embed_mask_List[j]]
            chisq_grad[j,1] = chisq_grad[j,1]*Frames[j].ravel()[embed_mask_List[j]]
            chisq_grad[j,2] = chisq_grad[j,2]*Frames[j].ravel()[embed_mask_List[j]]

        chisq_grad = (np.concatenate([embed(chisq_grad[i,0], embed_mask_List[i]) for i in range(N_freq*N_frame)])/(N_freq*N_frame)*alpha_d1
                    + np.concatenate([embed(chisq_grad[i,1], embed_mask_List[i]) for i in range(N_freq*N_frame)])/(N_freq*N_frame)*alpha_d2
                    + np.concatenate([embed(chisq_grad[i,2], embed_mask_List[i]) for i in range(N_freq*N_frame)])/(N_freq*N_frame)*alpha_d3)

#        print((s1.shape, s2.shape, s_dF.shape, s_dS.shape))
#        print((s_multifreq.shape, s_dynamic_grad.shape, chisq_grad.shape, cm_grad.shape, flux_grad.shape))
#        print((s_multifreq[embed_mask_All].shape))

        return ((s1 + s2 + s_dF + s_dS + (s_multifreq + s_dynamic_grad + chisq_grad + cm_grad + flux_grad)[embed_mask_All])*J_factor)

    # Plotting function for each iteration
    global nit
    nit = 0
    def plotcur(x, final=False):
        global nit
        nit += 1

        if nit%update_interval == 0 or final == True:
            print ("iteration %d" % nit)

            Frames = np.zeros((N_freq*N_frame, N_pixel, N_pixel))
            log_Frames = np.zeros((N_freq*N_frame, N_pixel, N_pixel))

            init_i = 0
            for i in range(N_freq*N_frame):
                cur_len = np.sum(embed_mask_List[i])
                log_Frames[i] = embed(x[init_i:(init_i+cur_len)], embed_mask_List[i]).reshape((N_pixel, N_pixel))
                Frames[i] = np.exp(log_Frames[i])*(embed_mask_List[i].reshape((N_pixel, N_pixel)))
                init_i += cur_len

            s1 = s2 = s_multifreq = s_dynamic = cm = s_dS = s_dF = 0.0

            # Multifrequency part
            if R_dt_multifreq['alpha'] != 0.0: 
                for j in range(N_frame):
                    s_multifreq += Rdt(Frames[j::N_frame], B_dt_multifreq, **R_dt_multifreq)*R_dt_multifreq['alpha']

            # Individual frequencies
            for j in range(N_freq):
                i1 = j*N_frame
                i2 = (j+1)*N_frame

                if alpha_s1 != 0.0:
                    s1 += static_regularizer(Frames[i1:i2], nprior_embed_List[i1:i2], embed_mask_List[i1:i2], Prior.total_flux(), Prior.psize, entropy1, norm_reg=norm_reg, beam_size=beam_size[j], alpha_A=alpha_A)*alpha_s1
                if alpha_s2 != 0.0:
                    s2 += static_regularizer(Frames[i1:i2], nprior_embed_List[i1:i2], embed_mask_List[i1:i2], Prior.total_flux(), Prior.psize, entropy2, norm_reg=norm_reg, beam_size=beam_size[j], alpha_A=alpha_A)*alpha_s2

                if R_dI['alpha'] != 0.0: s_dynamic += RdI(Frames[i1:i2], **R_dI)*R_dI['alpha']
                if R_dt['alpha'] != 0.0: s_dynamic += Rdt(Frames[i1:i2], B_dt, **R_dt)*R_dt['alpha']

                if alpha_dS1 != 0.0: s_dS += RdS(Frames[i1:i2], nprior_embed_List[i1:i2], embed_mask_List[i1:i2], entropy1, norm_reg, beam_size=beam_size[j], alpha_A=alpha_A)*alpha_dS1
                if alpha_dS2 != 0.0: s_dS += RdS(Frames[i1:i2], nprior_embed_List[i1:i2], embed_mask_List[i1:i2], entropy2, norm_reg, beam_size=beam_size[j], alpha_A=alpha_A)*alpha_dS2

                if alpha_dF != 0.0: s_dF += RdF_clip(Frames[i1:i2], embed_mask_List[i1:i2])*alpha_dF

            if alpha_centroid != 0.0: cm = centroid(Frames, coord) * alpha_centroid

            chisq = np.array([get_chisq(j, Frames[j].ravel()[embed_mask_List[j]],
                                  d1, d2, d3, ttype, embed_mask_List[j]) for j in range(N_freq*N_frame)])

            chisq1_List = chisq[:,0]
            chisq2_List = chisq[:,1]
            chisq3_List = chisq[:,2]
            chisq1 = np.sum(chisq1_List)/len(chisq1_List)
            chisq2 = np.sum(chisq2_List)/len(chisq1_List)
            chisq3 = np.sum(chisq3_List)/len(chisq1_List)
            chisq1_max = np.max(chisq1_List)
            chisq2_max = np.max(chisq2_List)
            chisq3_max = np.max(chisq3_List)
            if d1 != False: print ("chi2_1: %f" % chisq1)
            if d2 != False: print ("chi2_2: %f" % chisq2)
            if d3 != False: print ("chi2_3: %f" % chisq3)
            if d1 != False: print ("weighted chi2_1: %f" % (chisq1 * alpha_d1))
            if d2 != False: print ("weighted chi2_2: %f" % (chisq2 * alpha_d2))
            if d3 != False: print ("weighted chi2_3: %f" % (chisq3 * alpha_d3))
            if d1 != False: print ("Max Frame chi2_1: %f" % chisq1_max)
            if d2 != False: print ("Max Frame chi2_2: %f" % chisq2_max)
            if d3 != False: print ("Max Frame chi2_3: %f" % chisq3_max)

            if final == True:
                if d1 != False: print ("All chisq1:",chisq1_List)
                if d2 != False: print ("All chisq2:",chisq2_List)
                if d3 != False: print ("All chisq3:",chisq3_List)

            if s1 != 0.0: print ("weighted s1: %f" % (s1))
            if s2 != 0.0: print ("weighted s2: %f" % (s2))
            if s_dF != 0.0: print ("weighted s_dF: %f" % (s_dF))
            if s_dS != 0.0: print ("weighted s_dS: %f" % (s_dS))
            print ("weighted s_dynamic: %f" % (s_dynamic))
            print ("weighted s_multifreq: %f" % (s_multifreq))
            if alpha_centroid > 0.0: print ("weighted COM: %f" % cm)

            if alpha_flux > 0.0:
                print ("weighted flux constraint: %f" % (alpha_flux * movie_flux_constraint(Frames, flux_List)))

            if nit%refresh_interval == 0:
                print ("Plotting Functionality Temporarily Disabled...")

    loginit = np.hstack(loginit_List).flatten()

    x0 = loginit

    print ("Total Pixel #: ",(N_pixel*N_pixel*N_frame*N_freq))
    print ("Clipped Pixel #: ",(len(loginit)))

    print ("Initial Values:")
    plotcur(x0)

    # Minimize
    optdict = {'maxiter':maxit, 'ftol':stop, 'maxcor':NHIST, 'gtol': 1e-10} # minimizer params
    tstart = time.time()
    res = opt.minimize(objfunc, x0, method=minimizer_method, jac=objgrad, options=optdict, callback=plotcur)
    tstop = time.time()

    Frames = np.zeros((N_freq*N_frame, N_pixel, N_pixel))
    log_Frames = np.zeros((N_freq*N_frame, N_pixel, N_pixel))

    init_i = 0
    for i in range(N_freq*N_frame):
        cur_len = np.sum(embed_mask_List[i])
        log_Frames[i] = embed(res.x[init_i:(init_i+cur_len)], embed_mask_List[i]).reshape((N_pixel, N_pixel))
        #Impose the prior mask in linear space for the output
        Frames[i] = np.exp(log_Frames[i])*(embed_mask_List[i].reshape((N_pixel, N_pixel)))
        init_i += cur_len

    plotcur(res.x, final=True)

    # Print stats
    print ("time: %f s" % (tstop - tstart))
    print ("J: %f" % res.fun)
    print (res.message)

    #Return Frames
    outim = [[image.Image(Frames[i + j*N_frame].reshape(Prior.ydim, Prior.xdim), Prior.psize,
                         Prior.ra, Prior.dec, rf=Obsdata_Multifreq_List[j][i].rf, source=Prior.source,
                         mjd=Prior.mjd, pulse=Prior.pulse) for i in range(N_frame)] for j in range(N_freq)]

    return outim






##################################################################################################
# Plotting Functions
##################################################################################################

def plot_im_List_Set(im_List_List, plot_log_amplitude=False, ipynb=False):
    plt.ion()
    plt.clf()

    Prior = im_List_List[0][0]

    xnum = len(im_List_List[0])
    ynum = len(im_List_List)

    for i in range(xnum*ynum):
        plt.subplot(ynum, xnum, i+1)
        im = im_List_List[(i-i%xnum)//xnum][i%xnum]
        if plot_log_amplitude == False:
            plt.imshow(im.imvec.reshape(im.ydim,im.xdim), cmap=plt.get_cmap('afmhot'), interpolation='gaussian')
        else:
            plt.imshow(np.log(im.imvec.reshape(im.ydim,im.xdim)), cmap=plt.get_cmap('afmhot'), interpolation='gaussian')
        xticks = ticks(im.xdim, im.psize/RADPERAS/1e-6)
        yticks = ticks(im.ydim, im.psize/RADPERAS/1e-6)
        plt.xticks(xticks[0], xticks[1])
        plt.yticks(yticks[0], yticks[1])
        if i == 0:
            plt.xlabel('Relative RA ($\mu$as)')
            plt.ylabel('Relative Dec ($\mu$as)')
        else:
            plt.xlabel('')
            plt.ylabel('')
            plt.title('')

    plt.draw()

    if ipynb:
        display.clear_output()
        display.display(plt.gcf())

def plot_im_List(im_List, plot_log_amplitude=False, ipynb=False):

    plt.ion()
    plt.clf()

    Prior = im_List[0]

    for i in range(len(im_List)):
        plt.subplot(1, len(im_List), i+1)
        if plot_log_amplitude == False:
            plt.imshow(im_List[i].imvec.reshape(Prior.ydim,Prior.xdim), cmap=plt.get_cmap('afmhot'), interpolation='gaussian')
        else:
            plt.imshow(np.log(im_List[i].imvec.reshape(Prior.ydim,Prior.xdim)), cmap=plt.get_cmap('afmhot'), interpolation='gaussian')
        xticks = ticks(Prior.xdim, Prior.psize/RADPERAS/1e-6)
        yticks = ticks(Prior.ydim, Prior.psize/RADPERAS/1e-6)
        plt.xticks(xticks[0], xticks[1])
        plt.yticks(yticks[0], yticks[1])
        if i == 0:
            plt.xlabel('Relative RA ($\mu$as)')
            plt.ylabel('Relative Dec ($\mu$as)')
        else:
            plt.xlabel('')
            plt.ylabel('')
            plt.title('')


    plt.draw()

    if ipynb:
        display.clear_output()
        display.display(plt.gcf())

def plot_i_dynamic(im_List, Prior, nit, chi2, s, s_dynamic, ipynb=False):

    plt.ion()
    plt.clf()

    for i in range(len(im_List)):
        plt.subplot(1, len(im_List), i+1)
        plt.imshow(im_List[i].reshape(Prior.ydim,Prior.xdim), cmap=plt.get_cmap('afmhot'), interpolation='gaussian')
        xticks = ticks(Prior.xdim, Prior.psize/RADPERAS/1e-6)
        yticks = ticks(Prior.ydim, Prior.psize/RADPERAS/1e-6)
        plt.xticks(xticks[0], xticks[1])
        plt.yticks(yticks[0], yticks[1])
    if i == 0:
        plt.xlabel('Relative RA ($\mu$as)')
        plt.ylabel('Relative Dec ($\mu$as)')
        plt.title("step: %i  $\chi^2$: %f  $s$: %f  $s_{t}$: %f" % (nit, chi2, s, s_dynamic), fontsize=20)
    else:
        plt.xlabel('')
        plt.ylabel('')
        plt.title('')


    plt.draw()

    if ipynb:
        display.clear_output()
        display.display(plt.gcf())

##################################################################################################
#BU blazar CLEAN file loading functions
##################################################################################################

class MOJAVEHTMLParser(HTMLParser):
    #standard overriding of the python HTMLParser to suit the format of the BU blazar library
    dates = []
    mod_files = []
    uv_files = []

    #checks for CLEAN files linked on the page by looking for file ending
    def handle_starttag(self, tag, attrs):
        for attr in attrs:
            if len(attr) == 2:
                attr1s = attr[1].strip().split('.')
                if (len(attr1s) >1):
                    fileType = attr1s[1]
                    if fileType == 'icn':
                        self.mod_files.append(str(attr[1]).strip())
                    if fileType == 'uvf':
                        self.uv_files.append(str(attr[1]).strip())

    #extracts date information and standardizes format to DD + month string + YYYY = DDMONYYYY
    def handle_data(self,data):
        ds = data.split(' ')
        lds = len(ds)
        if ds[lds-1].isdigit():
            if lds == 3 or lds == 4:
                day = str(ds[lds-3])
                month = str(ds[lds-2])
                year = str(ds[lds-1])
                if len(day) == 1:
                    day = '0' + day
                month = month[0:3]
                monthNum = str(list(calendar.month_abbr).index(month))
                if len(monthNum) == 1:
                    monthNum = '0' + monthNum
                newDate = year +monthNum + day
                self.dates.append(str(newDate))

class BlazarHTMLParser(HTMLParser):
    #standard overriding of the python HTMLParser to suit the format of the BU blazar library
    dates = []
    mod_files = []
    uv_files = []

    #checks for CLEAN files linked on the page by looking for file ending
    def handle_starttag(self, tag, attrs):
        for attr in attrs:
            if len(attr) == 2:
                attr1s = attr[1].strip().split('.')
                if (len(attr1s) >1):
                    fileType = attr1s[1]
                    if fileType == 'MOD' or fileType=='mod':
                        self.mod_files.append(str(attr[1]).strip())
                    if fileType == 'UVP' or fileType=='uvp':
                        self.uv_files.append(str(attr[1]).strip())

    #extracts date information and standardizes format to DD + month string + YYYY = DDMONYYYY
    def handle_data(self,data):
        ds = data.split(' ')
        lds = len(ds)
        if ds[lds-1].isdigit():
            if lds == 3 or lds == 4:
                day = str(ds[lds-3])
                month = str(ds[lds-2])
                year = str(ds[lds-1])
                if len(day) == 1:
                    day = '0' + day
                month = month[0:3]
                monthNum = str(list(calendar.month_abbr).index(month))
                if len(monthNum) == 1:
                    monthNum = '0' + monthNum
                newDate = year +monthNum + day
                self.dates.append(str(newDate))

def generateMOJAVEdates(url, sourceName, path = './'):
    #Creates a list of observation dates and .mod filenames for a  particular source in the MOJAVE library
    #Returns the filename of the output file
    #url is the URL of the MOJAVE library page of files for a particular source
    r = requests.get(url)
    parser = MOJAVEHTMLParser()
    parser.feed(r.text)
    outputFileName = sourceName+'_dates_and_CLEAN_filenames.txt'
    outputFile = open(outputFileName, 'w')
    outputFile.write("#Observation dates, UV files, and CLEAN models obtained from " + url + '\n')
    for i in range(len(parser.dates)):
        outputFile.write(parser.dates[i]+','+parser.uv_files[i]+','+parser.mod_files[i]+'\n')
    outputFile.close()
    return outputFileName

def generateCLEANdates(url, sourceName, path = './'):
    #Creates a list of observation dates and .mod filenames for a  particular source in the BU blazar library
    #Returns the filename of the output file
    #url is the URL of the BU blazar library page of files for a particular source
    r = requests.get(url)
    parser = BlazarHTMLParser()
    parser.feed(r.text)
    outputFileName = sourceName+'_dates_and_CLEAN_filenames.txt'
    outputFile = open(outputFileName, 'w')
    outputFile.write("#Observation dates, UV files, and CLEAN models obtained from " + url + '\n')
    for i in range(len(parser.dates)):
        outputFile.write(parser.dates[i]+','+parser.uv_files[i]+','+parser.mod_files[i]+'\n')
    outputFile.close()
    return outputFileName

def sourceNameFromURL(url):
    #returns a string containing the BU url designation for a particular source, i.e. "3c454" for 3c454.3
    urls = url.split('/')
    lurls = len(urls)
    sourcehtmls = urls[lurls-1].split('.')
    sn = sourcehtmls[0]
    return sn

def downloadMOJAVEfiles(url, path = './'):
    #Downloads data and image from MOJAVE library
    #url is the URL of the MOJAVE page of files for a particular source

    sn =  sourceNameFromURL(url)
    #Find the base directory in the BU library from the url. This is a really hacky way to do this.
    #source name in BU library
    lsn = len(sn)
    baseurl = url[:-(5+lsn)]

    #Make a new directory for the downloaded files
    CLEANpath = path + '/' + sn + '_CLEAN_images'
    if not os.path.exists(CLEANpath):
        os.mkdir(CLEANpath)

    print ("Downloading CLEAN images to " + CLEANpath)

    UVpath = path + '/' + sn+"_uvf_files"
    if not os.path.exists(UVpath):
        os.mkdir(UVpath)

    print ("Downloading uvfits files to " + UVpath)

    #Generate the bookkeeping file and iterate on it, downloading files and saving them in a better format
    #Files are saved in a new directory named after the source i.e. "3c454_CLEAN_files"
    guideFileName = generateCLEANdates(url, sn, path=path)
    observations = np.loadtxt(guideFileName, dtype = str, delimiter = ',', skiprows = 1)
    for obs in observations:
        date = obs[0]
        UVbuFileName = obs[1]
        CLEANbuFileName = obs[2]
        UVurl = baseurl+UVbuFileName
        CLEANurl = baseurl + CLEANbuFileName
        # If it doesn't already exist, download the UV file
        if os.path.isfile(UVpath+'/'+date+'_'+ sn+".uvf") == False:
            print ("Downloading " + (UVpath+'/'+date+'_'+ sn+".uvf"))
            response = requests.get(UVurl, stream=True)
            with open(UVpath+'/'+date+'_'+ sn+".uvf",'wb') as handle:
                handle.write(response.raw.read())
        else:
            print ("Already Downloaded " + (UVpath+'/'+date+'_'+ sn+".uvf"))

        # If it doesn't already exist, download the CLEAN file
        if os.path.isfile(CLEANpath+'/'+date+'_'+sn+".icn.fits.gz") == False:
            print ("Downloading " + (CLEANpath+'/'+date+'_'+sn+".icn.fits.gz"))
            response = requests.get(CLEANurl, stream=True)
            with open(CLEANpath+'/'+date+'_'+sn+".icn.fits.gz",'wb') as handle:
                for chunk in response.iter_content(chunk_size = 128):
                    handle.write(chunk)
        else:
            print ("Already Downloaded " + (CLEANpath+'/'+date+'_'+sn+".icn.fits.gz"))

def downloadCLEANfiles(url, path = './'):
    #Downloads all CLEAN files from a single source in the BU blazar library
    #url is the URL of the BU blazar library page of files for a particular source

    sn =  sourceNameFromURL(url)
    #Find the base directory in the BU library from the url. This is a really hacky way to do this.
    #source name in BU library
    lsn = len(sn)
    baseurl = url[:-(5+lsn)]

    #Make a new directory for the downloaded files
    CLEANpath = path + '/' + sn + '_CLEAN_files'
    if not os.path.exists(CLEANpath):
        os.mkdir(CLEANpath)

    print ("Downloading CLEAN files to " + CLEANpath)

    UVpath = path + '/' + sn+"_UVP.gz_files"
    if not os.path.exists(UVpath):
        os.mkdir(UVpath)

    print ("Downloading uvfits files to " + UVpath)

    #Generate the bookkeeping file and iterate on it, downloading files and saving them in a better format
    #Files are saved in a new directory named after the source i.e. "3c454_CLEAN_files"
    guideFileName = generateCLEANdates(url, sn, path=path)
    observations = np.loadtxt(guideFileName, dtype = str, delimiter = ',', skiprows = 1)
    for obs in observations:
        date = obs[0]
        UVbuFileName = obs[1]
        CLEANbuFileName = obs[2]
        UVurl = baseurl+UVbuFileName
        CLEANurl = baseurl + CLEANbuFileName
        # If it doesn't already exist, download the UV file
        if os.path.isfile(UVpath+'/'+date+'_'+ sn+"_UV.UVP.gz") == False:
            print ("Downloading " + (UVpath+'/'+date+'_'+ sn+"_UV.UVP.gz"))
            response = requests.get(UVurl, stream=True)
            with open(UVpath+'/'+date+'_'+ sn+"_UV.UVP.gz",'wb') as handle:
                handle.write(response.raw.read())
        else:
            print ("Already Downloaded " + (UVpath+'/'+date+'_'+ sn+"_UV.UVP.gz"))

        # If it doesn't already exist, download the CLEAN file
        if os.path.isfile(CLEANpath+'/'+date+'_'+sn+"_CLEAN.mod") == False:
            print ("Downloading " + (CLEANpath+'/'+date+'_'+sn+"_CLEAN.mod"))
            response = requests.get(CLEANurl, stream=True)
            with open(CLEANpath+'/'+date+'_'+sn+"_CLEAN.mod",'wb') as handle:
                for chunk in response.iter_content(chunk_size = 128):
                    handle.write(chunk)
        else:
            print ("Already Downloaded " + (CLEANpath+'/'+date+'_'+sn+"_CLEAN.mod"))

def minDeltaMJD(inputMJD, im_List):
    #returns the image whose MJD most closely matches the inputMJD
    index = 0
    minDelta = 10000
    for i in range(len(im_List)):
        oldDelta = minDelta
        minDelta = min(minDelta, abs(inputMJD - im_List[i].mjd))
        if not(minDelta == oldDelta):
            index = i
    #return im_List[index].copy()
    return index

def ReadCLEAN(nameF, reference_obs, npix, fov=0, beamPar=(0,0,0.)):
#This should be able to load CLEAN Model data
#such as given here https://www.bu.edu/blazars/VLBA_GLAST/3c454.html
#nameF - name of the CLEAN Model file to load (3columns: Flux in Jy, r in mas, theta in deg)
#npix - number of pixels in one dimension
#fov - field of view (radians)
#beamPar - parameters of Gaussian beam, same as beamparams in image.blur_gauss

    #read data
    #first remove multiple models in a single file
    linesF = open(nameF).readlines()
    DelTrig = 0
    TrigString = '!'
    linesMax = 0
    for cou in range(len(linesF)):
        if (linesF[cou].find(TrigString) != -1)*(cou>8):
            open(nameF, 'w').writelines(linesF[:cou])
            break

    #skip headline
    TableMOD = np.genfromtxt(nameF, skip_header=4)
    ScaleR = 1.
    FluxConst = 1.
    Flux = FluxConst*TableMOD[:,0]
    xPS = ScaleR*TableMOD[:,1]*np.cos(np.pi/2.-(np.pi/180.)*TableMOD[:,2])*(1.e3)*RADPERUAS #to radians
    yPS = ScaleR*TableMOD[:,1]*np.sin(np.pi/2.-(np.pi/180.)*TableMOD[:,2])*(1.e3)*RADPERUAS #to radians
    NumbPoints = np.shape(yPS)[0]

    #set image parameters
    if fov==0:
        MaxR = np.amax(TableMOD[:,1]) #in mas
        fov = 1.*MaxR*(1.e3)*RADPERUAS

    image0 = np.zeros((int(npix),int(npix)))
    im = image.Image(image0, fov/npix, 0., 0., rf=86e9)

    beamMaj = beamPar[0]
    if beamMaj==0:
        beamMaj = 4.*fov/npix

    beamMin = beamPar[1]
    if beamMin==0:
        beamMin = 4.*fov/npix

    beamTh = beamPar[2]

    sigma_maj = beamMaj / (2. * np.sqrt(2. * np.log(2.)))
    sigma_min = beamMin / (2. * np.sqrt(2. * np.log(2.)))
    cth = np.cos(beamTh)
    sth = np.sin(beamTh)
    xfov = im.xdim * im.psize
    yfov = im.ydim * im.psize
    xlist = np.arange(0,-im.xdim,-1)*im.psize + (im.psize*im.xdim)/2.0 - im.psize/2.0
    ylist = np.arange(0,-im.ydim,-1)*im.psize + (im.psize*im.ydim)/2.0 - im.psize/2.0

    gauss = image0
    for couP in range(NumbPoints):
        x = xPS[couP]
        y = yPS[couP]
        xM, yM = np.meshgrid(xlist, ylist)
        gaussNew = np.exp(-((yM-y)*cth + (xM-x)*sth)**2/(2*sigma_maj**2) - ((xM-x)*cth - (yM-y)*sth)**2/(2.*sigma_min**2))
        gauss = gauss + gaussNew*Flux[couP]

    gauss /= (2.0*np.pi*sigma_maj*sigma_min)/(fov/npix)**2 #Normalize the Gaussian
    gauss = (gauss > 0.)*gauss + 1.e-10

    imageCLEAN = image.Image(gauss, fov/npix, reference_obs.ra, reference_obs.dec, rf=reference_obs.rf)
    imageCLEAN.mjd = reference_obs.mjd
    return imageCLEAN

def Cont(imG):
#This is meant to create plots similar to the ones from
#https://www.bu.edu/blazars/VLBA_GLAST/3c454.html
#for the visual comparison

    import matplotlib.pyplot as plt
    plt.figure()
    Z = np.reshape(imG.imvec,(imG.xdim,imG.ydim))
    pov = imG.xdim*imG.psize
    pov_mas = pov/(RADPERUAS*1.e3)
    Zmax = np.amax(Z)
    print(Zmax)

    levels = np.array((-0.00125*Zmax,0.00125*Zmax,0.0025*Zmax, 0.005*Zmax, 0.01*Zmax,
                        0.02*Zmax, 0.04*Zmax, 0.08*Zmax, 0.16*Zmax, 0.32*Zmax, 0.64*Zmax))
    CS = plt.contour(Z, levels,
                     origin='lower',
                     linewidths=2,
                     extent=(-pov_mas/2., pov_mas/2., -pov_mas/2., pov_mas/2.))
    plt.show()



def ReadSeriesImages(pathCLEAN, Obs, npix,fov,beamPar, obsNumbList):

    listCLEAN = os.listdir(pathCLEAN)
    listCLEAN = sorted(listCLEAN)
    listCLEAN = list( listCLEAN[i] for i in obsNumbList )

    im_List = [None]*len(listCLEAN)
    for cou in range(len(listCLEAN)):
        nameF = pathCLEAN+listCLEAN[cou]
        print(nameF)
        im_List[cou] = ReadCLEAN(nameF,Obs[cou], npix, fov, beamPar)

    return im_List

def SaveSeriesImages(pathCLEAN, im_List, sourceName, outputDirectory='default'):
    #saves a list of images returned by ReadSeriesImages according to the naming convention in pathCLEAN and the source name
    if outputDirectory == 'default':
        outputDirectory = sourceName+'_READ_CLEAN_files'
    if not os.path.exists(outputDirectory):
        os.mkdir(outputDirectory)
    outputSuffix = '_'+sourceName+'_READ_CLEAN.txt'

    listCLEAN = os.listdir(pathCLEAN)
    listCLEAN = sorted(listCLEAN)
    datesCLEAN = [filename.split('_')[0] for filename in listCLEAN]
    for i in range(len(datesCLEAN)):
        outputName = outputDirectory+'/'+datesCLEAN[i] + outputSuffix
        im_List[i].save_txt(outputName)

def LoadSeriesImages(sourceName):
    #loads a list of images saved according to the BU Blazar library convention
    dirName = sourceName+'_READ_CLEAN_files'
    fileList = os.listdir(dirName)
    filenameList = [dirName+'/'+filename for filename in fileList]
    im_List = [image.load_txt(filename) for filename in filenameList]
    return im_List
