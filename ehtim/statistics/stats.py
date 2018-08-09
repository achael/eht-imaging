# stats.py
# variety of statistical functions useful for 
#
#    Copyright (C) 2018 Maciek Wielgus
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

from __future__ import division
from __future__ import print_function
from builtins import str
from builtins import map
from builtins import range

import numpy as np
import numpy.random as npr
import sys

from ehtim.const_def import *

def circular_mean(theta, unit='deg'):
    '''circular mean for averaging angular quantities
    Args:
        theta: list/vector of angles to average
        unit: degrees ('deg') or radians (any other string)

    Returns:
        circular mean
    '''
    theta = np.asarray(theta, dtype=np.float32)
    theta= theta.flatten()
    theta = theta[theta==theta]
    if unit=='deg':
        theta *= np.pi/180.
    if len(theta)==0:
        return None
    else:
        C = np.mean(np.cos(theta))
        S = np.mean(np.sin(theta))
        circ_mean = np.arctan2(S,C)
        if unit=='deg':
            circ_mean *= 180./np.pi
            return np.mod(circ_mean+180.,360.)-180.
        else:
            return np.mod(circ_mean+np.pi,2.*np.pi)-np.pi

def circular_std(theta,unit='deg'):
    '''standard deviation of a circular distribution
    Args:
        theta: list/vector of angles
        unit: degrees ('deg') or radians (any other string)

    Returns:
        circular standard deviation
    '''
    theta = np.asarray(theta, dtype=np.float32)
    theta= theta.flatten()
    theta = theta[theta==theta]
    if unit=='deg':
        theta *= np.pi/180.
    if len(theta)<2:
        return None
    else:
        C = np.mean(np.cos(theta))
        S = np.mean(np.sin(theta))
        circ_std = np.sqrt(-2.*np.log(np.sqrt(C**2+S**2)))
        if unit=='deg':
            circ_std *= 180./np.pi
        return circ_std

def circular_std_of_mean(theta,unit='deg'):
    '''standard deviation of mean for a circular distribution
    Args:
        theta: list/vector of angles
        unit: degrees ('deg') or radians (any other string)

    Returns:
        circular standard deviation of mean
    '''
    theta = np.asarray(theta, dtype=np.float32)
    theta= theta.flatten()
    theta = theta[theta==theta]
    return circular_std(theta,unit)/np.sqrt(len(theta))

def mean_incoh_amp(amp,sigma,debias=True,err_type='predicted',num_samples=int(1e3)):
    """amplitude from ensemble of Rice-distributed measurements with debiasing
    Args:
        amp: vector of (biased) amplitudes
        sigma: vector of errors
        debias: whether debiasing is applied
    Returns:
        amp0: estimator of unbiased amplitude
    """
    if (not hasattr(amp, "__len__")):
        amp = [amp]
    amp = np.asarray(amp, dtype=np.float32) 
    N = len(amp)
    if (not hasattr(sigma, "__len__")):
        sigma = sigma*np.ones(N)
    elif len(sigma)==1:
        sigma = sigma*np.ones(N)
    sigma = np.asarray(sigma, dtype=np.float32)
    if len(sigma)!=len(amp):
        print('Inconsistent length of amp and sigma')
        return None
    else:
        amp_clean=amp[(amp==amp)&(sigma==sigma)&(sigma>0)&(amp>0)]
        sigma_clean=sigma[(amp==amp)&(sigma==sigma)&(sigma>0)&(amp>0)]
        #eq. 9.86 from Thompson et al.
        if debias==True:
            amp0sq = ( np.mean(amp_clean**2 - (2. - 1./N)*sigma_clean**2) )
        else: amp0sq = np.mean(amp_clean**2)
        amp0sq = np.maximum(amp0sq,0.)
        amp0 = np.sqrt(amp0sq)
        
        #getting errors
        if err_type=='predicted':
            sigma0 = np.sqrt(np.sum(sigma_clean**2)/len(sigma_clean)**2)
        elif err_type=='measured':
            ampfoo, ci = bootstrap(amp_clean, np.mean, num_samples=num_samples,wrapping_variable=False)
            sigma0 = 0.5*(ci[1]-ci[0])
        return amp0,sigma0

def mean_incoh_amp_from_vis(vis,sigma,debias=True,err_type='predicted',num_samples=int(1e3)):
    """Amplitude from ensemble of visibility measurements with debiasing
        Args:
            amp: vector of (biased) amplitudes
            sigma: vector of errors
            debias: whether debiasing is applied
        Returns:
            amp0: estimator of unbiased amplitude
    """
    if (not hasattr(vis, "__len__")):
        vis = [vis]
    vis= np.asarray(vis) 
    vis= vis[vis==vis]
    amp=np.abs(vis)

    N = len(amp)
    if (not hasattr(sigma, "__len__")):
        sigma = sigma*np.ones(N)
    elif len(sigma)==1:
        sigma = sigma*np.ones(N)
    sigma = np.asarray(sigma, dtype=np.float32)
    if len(sigma)!=len(amp):
        print('Inconsistent length of amp and sigma')
        return None, None
    else:
        amp_clean=amp[(amp==amp)&(sigma==sigma)&(sigma>=0)&(amp>=0)]
        sigma_clean=sigma[(amp==amp)&(sigma==sigma)&(sigma>=0)&(amp>=0)]
        Nc=len(amp_clean)
        if Nc<1:
            return None, None
        else:
            #eq. 9.86 from Thompson et al.
            if debias==True:
                amp0sq = ( np.mean(amp_clean**2 - (2. - 1./Nc)*sigma_clean**2) )
            else: amp0sq = np.mean(amp_clean**2)
            if (amp0sq!=amp0sq): amp0sq=0.
            amp0sq = np.maximum(amp0sq,0.)
            amp0 = np.sqrt(amp0sq)
            #getting errors
            if err_type=='predicted':
                #sigma0 = np.sqrt(np.sum(sigma_clean**2)/Nc**2)
                #Esigma = np.median(sigma_clean)
                #snr0 = amp0/Esigma
                #snrA = 1./(np.sqrt(1. + 2./np.sqrt(Nc)*(1./snr0)*np.sqrt(1.+1./snr0**2)) - 1.)
                #sigma0=amp0/snrA
                sigma0 = np.sqrt(np.sum(sigma_clean**2)/Nc**2)

            elif err_type=='measured':
                ampfoo, ci = bootstrap(amp_clean, np.mean, num_samples=num_samples,wrapping_variable=False,alpha='1sig')
                sigma0 = 0.5*(ci[1]-ci[0])
            return amp0,sigma0

def bootstrap(data, statistic, num_samples=int(1e3), alpha='1sig',wrapping_variable=False):
    """bootstrap estimate of 100.0*(1-alpha) confidence interval for a given statistic
        Args:
            data: vector of data to estimate bootstrap statistic on
            statistic: function representing the statistic to be evaluated
            num_samples: number of bootstrap (re)samples
            alpha: parameter of the confidence interval, '1s' gets an analog of 1 sigma confidence for a normal variable
            wrapping_variable: True for circular variables, attempts to avoid problem related to estimating variablity of wrapping variable

        Returns:
            bootstrap_value: bootstrap-estimated value of the statistic
            bootstrap_CI: bootstrap-estimated confidence interval
    """
    if alpha=='1sig':
        alpha=0.3173
    elif alpha=='2sig':
        alpha=0.0455
    elif alpha=='3sig':
        alpha=0.0027
    stat = np.zeros(num_samples)
    data = np.asarray(data)
    if wrapping_variable==True:
        m=statistic(data)
    else:
        m=0
    data = data-m
    n = len(data)
    idx = npr.randint(0, n, (num_samples, n))
    samples = data[idx]
    for cou in range(num_samples):
        stat[cou] = statistic(samples[cou,:])
    stat = np.sort(stat)
    bootstrap_value = np.median(stat)+m
    bootstrap_CI = [stat[int((alpha/2.0)*num_samples)]+m, stat[int((1-alpha/2.0)*num_samples)]+m]
    return bootstrap_value, bootstrap_CI

def mean_incoh_avg(x,debias=True):
    amp = np.abs(np.asarray([y[0] for y in x]))
    sig = np.asarray([y[1] for y in x])
    ampN = amp[(amp==amp)&(amp>=0)&(sig==sig)&(sig>=0)]
    sigN = sig[(amp==amp)&(amp>=0)&(sig==sig)&(sig>=0)]
    amp = ampN
    sig = sigN
    Nc = len(sig)
    if Nc==0:
        amp0 = -1
        sig0 = -1
    elif Nc==1:
        amp0 = amp[0]
        sig0 = sig[0]
    else:
        if debias==True:
            amp0 = deb_amp(amp,sig)
        else: 
            amp0= np.sqrt(np.maximum(np.mean(amp**2),0.))
        sig0 = inc_sig(amp,sig)
        #sig0 = coh_sig(amp,sig)
    return amp0,sig0

def deb_amp(amp,sig):
    #eq. 9.86 from Thompson et al.
    amp = np.abs(np.asarray(amp))
    sig = np.asarray(sig)
    Nc = len(amp)
    amp0sq = ( np.mean(amp**2 - (2. - 1./Nc)*sig**2) )
    amp0sq = np.maximum(amp0sq,0.)
    amp0 = np.sqrt(amp0sq)
    return amp0

def inc_sig(amp,sig):
    amp = np.abs(np.asarray(amp))
    sig = np.asarray(sig)
    Nc = len(amp)
    amp0 = deb_amp(amp,sig)
    Esigma = np.median(sig)
    snr0 = amp0/Esigma
    snrA = 1./(np.sqrt(1. + 2./np.sqrt(Nc)*(1./snr0)*np.sqrt(1.+1./snr0**2)) - 1.)
    if snrA>0:
        sigma0=amp0/snrA
    else: sigma0=coh_sig(amp,sig)
    return sigma0

def coh_sig(amp,sig):
    amp = np.abs(np.asarray(amp))
    sig = np.asarray(sig)
    Nc = len(amp)
    sigma0 = np.sqrt(np.sum(sig**2)/Nc**2)
    return sigma0
