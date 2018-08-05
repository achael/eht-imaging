# pulses.py
# image restoring pulse functions
#
#    Copyright (C) 2018 Katie Bouman
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

# If dom="I", we are in real space, if dom="F" we are in Fourier (uv) space
# Coordinates in real space are in radian, coordinates in Fourier space are in lambda

from __future__ import division
import math
import numpy as np
import scipy.special as spec

# Delta Function  Pulse
def deltaPulse2D(x, y, pdim, dom='F'):
    if dom=='I':
        if x==y==0.0: return 1.0
        else: return 0.0
    elif dom=='F':
        return 1.0

# Square  Wave Pulse
def rectPulse2D(x, y, pdim, dom='F'):
    if dom=='I':
        return rectPulse_I(x, pdim) * rectPulse_I(y,pdim)
    elif dom=='F':
        return rectPulse_F(x, pdim) * rectPulse_F(y,pdim)

def rectPulse_I(x, pdim):
    if abs(x) >= pdim/2.0:
        return 0.0
    else:
        return 1.0/pdim

def rectPulse_F(omega, pdim):
    if (omega == 0):
        return 1.0
    else:
        return (2.0/(pdim*omega)) * math.sin((pdim*omega)/2.0)

# Triangle  Wave Pulse
def trianglePulse2D(x, y, pdim, dom='F'):
    if dom=='I':
        return trianglePulse_I(x,pdim) * trianglePulse_I(y,pdim)
    elif dom=='F':
        return trianglePulse_F(x, pdim)*trianglePulse_F(y, pdim)

def trianglePulse_I(x, pdim):
    if abs(x) > pdim: return 0.0
    else: return -(1.0/(pdim**2))*abs(x) + 1.0/pdim

def trianglePulse_F(omega, pdim):
    if (omega == 0):
        return 1.0
    else:
        return (4.0/(pdim**2 * omega**2)) * ( math.sin (( pdim * omega )/2.0) )**2

# Gaussian Pulse
def GaussPulse2D(x, y, pdim, dom='F'):
    sigma = pdim/3.  #Gaussian SD (sigma) vs pixelwidth (pdim)
    a = 1./2./sigma/sigma
    if dom=='I':
        return (a/np.pi)*np.exp(-a*(x**2 + y**2))

    elif dom=='F':
        return np.exp(-(x**2 + y**2)/4./a)

# Cubic Pulse
def cubicPulse2D(x, y, pdim, dom='F'):
    if dom=='I':
        return cubicPulse_I(x,pdim) * cubicPulse_I(y,pdim)

    elif dom=='F':
        return cubicPulse_F(x, pdim)*cubicPulse_F(y, pdim)

def cubicPulse_I(x, pdim):
    if abs(x) < pdim: return (1.5*(abs(x)/pdim)**3 - 2.5*(x/pdim)**2 +1.)/pdim
    elif  abs(abs(x)-1.5*pdim) <= 0.5*pdim: return (-0.5*(abs(x)/pdim)**3 + 2.5*(x/pdim)**2 - 4.*(abs(x)/pdim) +2.)/pdim
    else: return 0.

def cubicPulse_F(omega, pdim):
    if (omega == 0):
        return 1.0
    else:
        return 2.*((3./omega/pdim)*math.sin(omega*pdim/2.)-math.cos(omega*pdim/2.))*((2./omega/pdim)*math.sin(omega*pdim/2.))**3

# Sinc Pulse
def sincPulse2D(x, y, pdim, dom='F'):
    if dom=='I':
        return sincPulse_I(x,pdim) * sincPulse_I(y,pdim)

    elif dom=='F':
        return sincPulse_F(x, pdim) * sincPulse_F(y, pdim)

def sincPulse_I(x, pdim):
    if (x == 0):
        return 1./pdim
    else: return (1./pdim)*math.sin(np.pi*x/pdim)/(np.pi*x/pdim)

def sincPulse_F(omega, pdim):
    if (abs(omega) < np.pi/pdim):
        return 1.0
    else:
        return 0.
        
# Circular Disk Pulse
#def circPulse2D(x, y, pdim, dom='F'):
#        rm = 0.5*pdim #max radius of the disk
#        if dom=='I':
#            if x**2 + y**2 <= rm**2:
#                return 1./np.pi/rm**2
#            else: return 0.
#        elif dom=='F':
#            return 2.*spec.j1(rm*np.sqrt(x**2 + y**2))/np.sqrt(x**2 + y**2)/rm**2


# Cubic Spline Pulse
# def cubicsplinePulse2D_F(omegaX, omegaY, pdim):
#       return cubicsplinePulse(omegaX, pdim)*cubicsplinePulse(omegaY,pdim)
#
# def cubicsplinePulse_F(omega, delta):
#       if (omega == 0):
#               coeff = delta
#       else:
#               omega_delta = omega*delta
#
#               coeff = delta * ( (4.0/omega_delta**3)*math.sin(omega_delta)*(2.0*math.cos(omega_delta) + 1.0) +
#                   (24.0/omega_delta**4)*math.cos(omega_delta)*(math.cos(omega_delta) - 1.0) )
#
#       return coeff / pdim # TODO : CHECK IF YOU DIVIDE BY PDIM FOR CLUBIC SPLINE PULSE
