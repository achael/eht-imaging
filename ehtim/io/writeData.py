# writeData.py
# functionto save observation to OIFITS
# author: Katie Bouman

from __future__ import division
from __future__ import print_function
from builtins import range

import numpy as np
import numpy as np
from numpy import pi, cos, sin
import ehtim.io.oifits
from ehtim.const_def import *

def writeOIFITS(filename, RA, DEC, frequency, bandWidth, intTime,
                visamp, visamperr, visphi, visphierr, u, v, ant1, ant2, timeobs,
                t3amp, t3amperr, t3phi, t3phierr, uClosure, vClosure, antOrder, timeClosure,
                antennaNames, antennaDiam, antennaX, antennaY, antennaZ):


	speedoflight = C;
	flagVis = False; # do not flag any data

	# open a new oifits file
	data = ehtim.io.oifits.oifits();

	# put in the target information - RA and DEC should be in degrees
	name = 'TARGET_NAME';
	data.target = np.append(data.target, ehtim.io.oifits.OI_TARGET(name, RA, DEC, veltyp='LSR') )

	#calulate wavelength and bandpass
	wavelength = speedoflight/frequency
	bandlow    = speedoflight/(frequency+(0.5*bandWidth))
	bandhigh   = speedoflight/(frequency-(0.5*bandWidth))
	bandpass   = bandhigh-bandlow

    # put in the wavelength information - only using a single frequency
	data.wavelength['WAVELENGTH_NAME'] = ehtim.io.oifits.OI_WAVELENGTH(wavelength, eff_band=bandpass)

	# put in information about the telescope stations in the array
	stations = [];
	for i in range(0, len(antennaNames)):
		stations.append( (antennaNames[i], antennaNames[i], i+1, antennaDiam[i], [antennaX[i], antennaY[i], antennaZ[i]]) )
	data.array['ARRAY_NAME'] = ehtim.io.oifits.OI_ARRAY('GEOCENTRIC', [0, 0, 0], stations);

	print('Warning: set cflux and cfluxerr = False because otherwise problems were being generated...are they the total flux density?')
	print('Warning: are there any true flags?')

	# put in the visibility information - note this does not include phase errors!
	for i in range(0, len(u)):
		station_curr = (data.array['ARRAY_NAME'].station[ int(ant1[i] - 1) ] , data.array['ARRAY_NAME'].station[ int(ant2[i] - 1) ]);
		currVis = ehtim.io.oifits.OI_VIS(timeobs[i], intTime, visamp[i], visamperr[i], visphi[i], visphierr[i], flagVis, u[i]*wavelength, v[i]*wavelength, data.wavelength['WAVELENGTH_NAME'], data.target[0], array=data.array['ARRAY_NAME'], station=station_curr, cflux=False, cfluxerr=False);
		data.vis = np.append( data.vis, currVis );

	# put in bispectrum information
	for j in range(0, len(uClosure)):
		station_curr = (data.array['ARRAY_NAME'].station[ int(antOrder[j][0] - 1) ] , data.array['ARRAY_NAME'].station[ int(antOrder[j][1] - 1) ], data.array['ARRAY_NAME'].station[ int(antOrder[j][2] - 1) ]);
		currT3 = ehtim.io.oifits.OI_T3(timeClosure[j], intTime, t3amp[j], t3amperr[j], t3phi[j], t3phierr[j], flagVis, uClosure[j][0]*wavelength, vClosure[j][0]*wavelength, uClosure[j][1]*wavelength, vClosure[j][1]*wavelength, data.wavelength['WAVELENGTH_NAME'], data.target[0], array=data.array['ARRAY_NAME'], station=station_curr);
		data.t3 = np.append(data.t3, currT3 );

	# put in visibility squared information
	for k in range(0, len(u)):
		station_curr = (data.array['ARRAY_NAME'].station[ int(ant1[k] - 1) ] , data.array['ARRAY_NAME'].station[ int(ant2[k] - 1) ]);
		currVis2 = ehtim.io.oifits.OI_VIS2(timeobs[k], intTime, visamp[k]**2, 2.0*visamp[k]*visamperr[k], flagVis, u[k]*wavelength, v[k]*wavelength, data.wavelength['WAVELENGTH_NAME'], data.target[0], array=data.array['ARRAY_NAME'], station=station_curr);
		data.vis2 = np.append(data.vis2, currVis2 );

    #save oifits file
	data.save(filename)

def arrayUnion(array, union):
    for item in array:
        if not (item in list(union.keys())):
            union[item] = len(union)+1
    return union

def convertStrings(array, union):
    returnarray = np.zeros(array.shape)
    for i in range(len(array)):
        returnarray[i] = union[array[i]]
    return returnarray
