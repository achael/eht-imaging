#pulses.py
#07/10/16
#All of the real & fourier space representations of the restoring pulses we are using
#If dom="I", we are in real space, if dom="F" we are in Fourier space
#pdim is in radian, Coordinates in real space are in radian, coordinates in Fourier space are in ANGULAR. spatial freq. 

import math

def deltaPulse2D(x, y, pdim, dom='F'):
	if dom=='I':
		if x==y==0.0: return 1.0
		else: return 0.0
	elif dom=='F':
		return 1.0
		
def rectPulse2D(x, y, pdim, dom='F'):
	if dom=='I':
		return rectPulse_I(x, pdim) * rectPulse_I(y,pdim)
	elif dom=='F':
		return rectPulse_F(x, pdim) * rectPulse_F(y,pdim)

def rectPulse_I(x, pdim):
	if abs(x) > pdim/2.:
		return 0.0
	else:
		return 1./pdim
		
def rectPulse_F(omega, pdim):
	if (omega == 0):
		return 1.0 
	else: 
		return (2.0/(pdim*omega)) * math.sin((pdim * omega)/2.0)
		
def trianglePulse2D(x, y, pdim, dom='F'):
	if dom=='I':
		return trianglePulse_I(x,pdim) * trianglePulse_I(y,pdim)
	
	elif dom=='F': 
		return trianglePulse_F(x, pdim)*trianglePulse_F(y, pdim)
	
def trianglePulse_I(x, pdim):
	if abs(x) > pdim: return 0.0
	else: return -(1./pdim**2)*abs(x) + 1./pdim
	
def trianglePulse_F(omega, pdim): 
	if (omega == 0):
		return 1.0
	else: 
		return (4.0/(pdim**2 * omega**2)) * ( math.sin (( pdim * omega )/2.0) )**2 
		
			

# def cubicsplinePulse2D_F(omegaX, omegaY, pdim): 
# 	return cubicsplinePulse(omegaX, pdim)*cubicsplinePulse(omegaY,pdim)
# 
# def cubicsplinePulse_F(omega, delta):
# 	if (omega == 0):
# 		coeff = delta
# 	else: 
# 		omega_delta = omega*delta
# 
# 		coeff = delta * ( (4.0/omega_delta**3)*math.sin(omega_delta)*(2.0*math.cos(omega_delta) + 1.0) + 
# 		    (24.0/omega_delta**4)*math.cos(omega_delta)*(math.cos(omega_delta) - 1.0) )
# 		    
# 	return coeff / pdim # TODO : CHECK IF YOU DIVIDE BY PDIM FOR CLUBIC SPLINE PULSE