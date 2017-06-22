import string
import numpy as np
import numpy.lib.recfunctions as rec
import matplotlib.pyplot as plt
import scipy.optimize as opt
import itertools as it

import ehtim.image
import ehtim.observing.obs_simulate 
import ehtim.io.save 
import ehtim.io.load

from ehtim.const_def import *
from ehtim.observing.obs_helpers import *

##################################################################################################
# Obsdata object
##################################################################################################

class Obsdata(object):
    """A polarimetric VLBI observation of visibility amplitudes and phases (in Jy).
    
       Attributes:
           source (str): The source name
           ra (float): The source Right Ascension in fractional hours
           dec (float): The source declination in fractional degrees
           mjd (int): The integer MJD of the observation 
           tstart (float): The start time of the observation in hours
           tstop (float): The end time of the observation in hours
           rf (float): The observation frequency in Hz
           bw (float): The observation bandwidth in Hz
           ampcal (bool): True if amplitudes calibrated
           phasecal (bool): True if phases calibrated 
           opacitycal (bool): True if time-dependent opacities correctly accounted for in sigmas
           frcal (bool): True if feed rotation calibrated out of visibilities
           dcal (bool): True if D terms calibrated out of visibilities
           timetype (str): How to interpret tstart and tstop; either 'GMST' or 'UTC' 

           tarr (numpy.recarray): The array of telescope data with datatype DTARR 
           tkey (dict): A dictionary of rows in the tarr for each site name 
           data (numpy.recarray): the basic data with datatype DTPOL
    """
    
    def __init__(self, ra, dec, rf, bw, datatable, tarr, source=SOURCE_DEFAULT, mjd=MJD_DEFAULT, ampcal=True, phasecal=True, opacitycal=True, dcal=True, frcal=True, timetype='UTC'):
        
        if len(datatable) == 0:
            raise Exception("No data in input table!")
        if (datatable.dtype != DTPOL):
            raise Exception("Data table should be a recarray with datatable.dtype = %s" % DTPOL)
        
        # Set the various parameters
        self.source = str(source)
        self.ra = float(ra)
        self.dec = float(dec)
        self.rf = float(rf)
        self.bw = float(bw)
        self.ampcal = bool(ampcal)
        self.phasecal = bool(phasecal)
        self.opacitycal = bool(opacitycal)
        self.dcal = bool(dcal)
        self.frcal = bool(frcal)
        self.timetype = timetype
        self.tarr = tarr
        
        # Dictionary of array indices for site names
        self.tkey = {self.tarr[i]['site']: i for i in range(len(self.tarr))}
        
        # Time partition the datatable
        datalist = []
        for key, group in it.groupby(datatable, lambda x: x['time']):
            datalist.append(np.array([obs for obs in group]))
        
        # Remove conjugate baselines
        obsdata = []
        for tlist in datalist:
            blpairs = []
            for dat in tlist:
                if not (set((dat['t1'], dat['t2']))) in blpairs:

                     # Reverse the baseline in the right order for uvfits:
                     if(self.tkey[dat['t1']] < self.tkey[dat['t2']]):                        
                        (dat['t1'], dat['t2']) = (dat['t2'], dat['t1'])
                        dat['u'] = -dat['u']
                        dat['v'] = -dat['v']
                        dat['vis'] = np.conj(dat['vis'])
                        dat['uvis'] = np.conj(dat['uvis'])
                        dat['qvis'] = np.conj(dat['qvis'])
                        dat['vvis'] = np.conj(dat['vvis'])

                     # Append the data point
                     blpairs.append(set((dat['t1'],dat['t2'])))    
                     obsdata.append(dat) 

        obsdata = np.array(obsdata, dtype=DTPOL)
        
        # Sort the data by time
        obsdata = obsdata[np.argsort(obsdata, order=['time','t1'])]
        
        # Save the data             
        self.data = obsdata
            
        # Get tstart, mjd and tstop
        times = self.unpack(['time'])['time']
        self.tstart = times[0]
        self.mjd = int(mjd)
        #self.mjd = fracmjd(mjd, self.tstart)
        self.tstop = times[-1]
        if self.tstop < self.tstart: 
            self.tstop += 24.0
  
    def copy(self):
        """Copy the observation object.
        """
        newobs = Obsdata(self.ra, self.dec, self.rf, self.bw, self.data, self.tarr, source=self.source, mjd=self.mjd, 
                         ampcal=self.ampcal, phasecal=self.phasecal, opacitycal=self.opacitycal, dcal=self.dcal, frcal=self.frcal)
        return newobs
        
    def data_conj(self):
        """Return a data array of same format as self.data but including all conjugate baselines.
        """
        
        data = np.empty(2*len(self.data), dtype=DTPOL)        
        
        # Add the conjugate baseline data
        for f in DTPOL:
            f = f[0]
            if f in ["t1", "t2", "tau1", "tau2"]:
                if f[-1]=='1': f2 = f[:-1]+'2'
                else: f2 = f[:-1]+'1'
                data[f] = np.hstack((self.data[f], self.data[f2]))
            elif f in ["u","v"]:
                data[f] = np.hstack((self.data[f], -self.data[f]))
            elif f in ["vis","qvis","uvis","vvis"]:
                data[f] = np.hstack((self.data[f], np.conj(self.data[f])))
            else:
                data[f] = np.hstack((self.data[f], self.data[f]))
        
        # Sort the data by time
        #!AC TODO should we apply some sorting within equal times? 
        data = data[np.argsort(data['time'])]
        return data

    def tlist(self, conj=False):
        """Return the data in a list of equal time observation datatables.
        """
        
        if conj: 
            data = self.data_conj()
        else: 
            data = self.data
        
        # Use itertools groupby function to partition the data
        datalist = []
        for key, group in it.groupby(data, lambda x: x['time']):
            datalist.append(np.array([obs for obs in group]))
        
        return np.array(datalist)
  
    def split_obs(self):
        """Split single observation into multiple observation files, one per scan.
        """

        print "Splitting Observation File into " + str(len(self.tlist())) + " scans"

        #Note that the tarr of the output includes all sites, even those that don't participate in the scan
        splitlist = [Obsdata(self.ra, self.dec, self.rf, self.bw, tdata, self.tarr, source=self.source,
                             mjd=self.mjd, ampcal=self.ampcal, phasecal=self.phasecal) 
                     for tdata in self.tlist() 
                    ]

        return splitlist   
           
    def unpack_bl(self, site1, site2, in_fields, ang_unit='deg', debias=False):
        """Unpack the data for the given fields in_fields over time on the selected baseline site1-site2.
        """

        # If we only specify one field
        fields=['time']
        if type(in_fields) == str: fields.append(in_fields)
        else: 
            for i in range(len(in_fields)): fields.append(in_fields[i])
            
        # Get field data on selected baseline   
        allout = []    
        
        # Get the data from data table on the selected baseline
        tlist = self.tlist(conj=True)
        for scan in tlist:
            for obs in scan:
                if (obs['t1'], obs['t2']) == (site1, site2):
                    obs = np.array([obs])
                    out = self.unpack_dat(obs, fields, ang_unit=ang_unit, debias=debias)             
                    allout.append(out)
        return np.array(allout)            
    
    def unpack(self, fields, conj=False, ang_unit='deg', mode='all', debias=False):
        """Return a recarray of all the data for the given fields extracted from the dtable.
           If conj=True, will also return conjugate baselines.
           ang_unit can be 'deg' or 'radian'
           debias=True will debias visibility amplitudes
        """
        
        if not mode in ('time', 'all'):
            raise Exception("possible options for mode are 'time' and 'all'")
                    
        # If we only specify one field
        if type(fields) == str: fields = [fields]
        
        if mode=='all':    
            if conj:
                data = self.data_conj()     
            else:
                data = self.data
            allout=self.unpack_dat(data, fields, ang_unit=ang_unit, debias=debias)

        elif mode=='time':
            allout=[]
            tlist = self.tlist(conj=True)
            for scan in tlist:
                out=self.unpack_dat(scan, fields, ang_unit=ang_unit, debias=debias)
                allout.append(out)
        
        return np.array(allout)
    
    def unpack_dat(self, data, fields, conj=False, ang_unit='deg', debias=False):
        """Return a recarray of data for the given fields extracted from the datatable 'data'.
           If conj=True, will also return conjugate baselines.
           ang_unit can be 'deg' or 'radian'
           debias=True will debias visibility amplitudes
        """
       
        if ang_unit=='deg': angle=DEGREE
        else: angle = 1.0
        
        # Get field data    
        allout = []    
        for field in fields:
            if field in ["u","v","tint","time","tau1","tau2"]: 
                out = data[field]
                ty = 'f8'
            elif field in ["uvdist"]: 
                out = np.abs(data['u'] + 1j * data['v'])
                ty = 'f8'
            elif field in ["t1","el1","par_ang1","hr_ang1"]: 
                sites = data["t1"]
                keys = [self.tkey[site] for site in sites]
                tdata = self.tarr[keys]
                out = sites
                ty = 'a32'
            elif field in ["t2","el2","par_ang2","hr_ang2"]: 
                sites = data["t2"]
                keys = [self.tkey[site] for site in sites]
                tdata = self.tarr[keys]
                out = sites
                ty = 'a32'
            elif field in ["vis","amp","phase","snr","sigma","sigma_phase"]: 
                out = data['vis']
                sig = data['sigma']
                ty = 'c16'
            elif field in ["qvis","qamp","qphase","qsnr","qsigma","qsigma_phase"]: 
                out = data['qvis']
                sig = data['qsigma']
                ty = 'c16'
            elif field in ["uvis","uamp","uphase","usnr","usigma","usigma_phase"]: 
                out = data['uvis']
                sig = data['usigma']
                ty = 'c16'
            elif field in ["vvis","vamp","vphase","vsnr","vsigma","vsigma_phase"]: 
                out = data['vvis']
                sig = data['vsigma']
                ty = 'c16'               
            elif field in ["pvis","pamp","pphase","psnr","psigma","psigma_phase"]: 
                out = data['qvis'] + 1j * data['uvis']
                sig = np.sqrt(data['qsigma']**2 + data['usigma']**2)
                ty = 'c16'
            elif field in ["m","mamp","mphase","msnr","msigma","msigma_phase"]: 
                out = (data['qvis'] + 1j * data['uvis']) / data['vis']
                sig = merr(data['sigma'], data['qsigma'], data['usigma'], data['vis'], out)
                ty = 'c16'
            
            else: raise Exception("%s is not valid field \n" % field + 
                                  "valid field values are " + string.join(FIELDS)) 

            # Elevation and Parallactic Angles
            if field in ["el1","el2","hr_ang1","hr_ang2","par_ang1","par_ang2"]:
                if self.timetype=='GMST':
                    times_sid = data['time']
                else:
                    times_sid = utc_to_gmst(data['time'], self.mjd)

                thetas = np.mod((times_sid - self.ra)*HOUR, 2*np.pi)
                coords = tdata[['x','y','z']].view(('f8', 3))
                el_angle = elev(earthrot(coords, thetas), self.sourcevec())
                latlon = xyz_2_latlong(coords)
                hr_angles = hr_angle(times_sid*HOUR, latlon[:,1], self.ra*HOUR)

                if field in ["el1","el2"]:
                    out=el_angle/angle
                    ty = 'f8'
                if field in ["hr_ang1","hr_ang2"]:
                    out = hr_angles/angle
                    ty = 'f8'
                if field in ["par_ang1","par_ang2"]:
                    par_ang = par_angle(hr_angles, latlon[:,0], self.dec*DEGREE)
                    out = par_ang/angle
                    ty = 'f8'
                                
            # Get arg/amps/snr
            if field in ["amp", "qamp", "uamp","vamp","pamp","mamp"]: 
                out = np.abs(out)

                # !AC debias here? 
                if debias:
                    print "Debiasing amplitudes in unpack!"
                    out = amp_debias(out, sig)

                ty = 'f8'
            elif field in ["phase", "qphase", "uphase", "vphase","pphase", "mphase"]: 
                out = np.angle(out)/angle
                ty = 'f8'
            elif field in ["sigma","qsigma","usigma","vsigma","psigma","msigma"]:
                out = np.abs(sig)
                ty = 'f8'
            elif field in ["sigma_phase","qsigma_phase","usigma_phase","vsigma_phase","psigma_phase","msigma_phase"]:
                out = np.abs(sig)/np.abs(out)/angle
                ty = 'f8'                                                
            elif field in ["snr", "qsnr", "usnr", "vsnr", "psnr", "msnr"]:
                out = np.abs(out)/np.abs(sig)
                ty = 'f8'
                                            
            # Reshape and stack with other fields
            out = np.array(out, dtype=[(field, ty)])

            if len(allout) > 0: #N.B.: This throws an error sometimes 
                allout = rec.merge_arrays((allout, out), asrecarray=True, flatten=True)
            else:
                allout = out
            
        return allout
    
    def sourcevec(self):
        """Return the source position vector in geocentric coordinates at 0h GMST.
        """
        return np.array([np.cos(self.dec*DEGREE), 0, np.sin(self.dec*DEGREE)])
        
    def res(self):
        """Return the nominal resolution (1/longest baseline) of the observation in radians.
        """
        return 1.0/np.max(self.unpack('uvdist')['uvdist'])
        
    def bispectra(self, vtype='vis', mode='time', count='min'):
        """Return a recarray of the equal time bispectra.
           
           Args: 
               vtype (str): The visibilty type ('vis','qvis','uvis','vvis','pvis') from which to assemble bispectra
               mode (str): If 'time', return phases in a list of equal time arrays, if 'all', return all phases in a single array
               count (str): If 'min', return minimal set of phases, if 'max' return all closure phases up to reordering

           Returns:
               numpy.recarry: A recarray of the bispectra values with datatype DTBIS
           
        """

        if not mode in ('time', 'all'):
            raise Exception("possible options for mode are 'time' and 'all'")
        if not count in ('min', 'max'):
            raise Exception("possible options for count are 'min' and 'max'")
        if not vtype in ('vis','qvis','uvis','vvis','pvis'):
            raise Exception("possible options for vtype are 'vis','qvis','uvis','vvis','pvis'")
            
        # Generate the time-sorted data with conjugate baselines
        tlist = self.tlist(conj=True)    
        outlist = []
        bis = []
        for tdata in tlist:
            time = tdata[0]['time']
            sites = list(set(np.hstack((tdata['t1'],tdata['t2']))))
                                        
            # Create a dictionary of baselines at the current time incl. conjugates;
            l_dict = {}
            for dat in tdata:
                l_dict[(dat['t1'], dat['t2'])] = dat
            
            # Determine the triangles in the time step

            # Minimal Set
            if count == 'min':
                # If we want a minimal set, choose triangles with the minimum sefd reference
                # Unless there is no sefd data, in which case choose the northernmost
                # !AC TODO This should probably be an sefdr + sefdl average
                if len(set(self.tarr['sefdr'])) > 1:
                    ref = sites[np.argmin([self.tarr[self.tkey[site]]['sefdr'] for site in sites])]
                else:
                    ref = sites[np.argmax([self.tarr[self.tkey[site]]['z'] for site in sites])]
                sites.remove(ref)
                
                # Find all triangles that contain the ref                    
                tris = list(it.combinations(sites,2))
                tris = [(ref, t[0], t[1]) for t in tris]

            # Maximal  Set - find all triangles
            elif count == 'max':
                tris = list(it.combinations(sites,3))
            
            # Generate bispectra for each triangle
            for tri in tris:
                # The ordering is north-south
                a1 = np.argmax([self.tarr[self.tkey[site]]['z'] for site in tri])
                a3 = np.argmin([self.tarr[self.tkey[site]]['z'] for site in tri])
                a2 = 3 - a1 - a3
                tri = (tri[a1], tri[a2], tri[a3])
                    
                # Select triangle entries in the data dictionary
                try:
                    l1 = l_dict[(tri[0], tri[1])]
                    l2 = l_dict[(tri[1],tri[2])]
                    l3 = l_dict[(tri[2], tri[0])]
                except KeyError:
                    continue
                    
                (bi, bisig) = make_bispectrum(l1,l2,l3,vtype)

                # Append to the equal-time list
                bis.append(np.array((time, tri[0], tri[1], tri[2], 
                                     l1['u'], l1['v'], l2['u'], l2['v'], l3['u'], l3['v'],
                                     bi, bisig), dtype=DTBIS))                 
            
            # Append to outlist    
            if mode=='time' and len(bis) > 0:
                outlist.append(np.array(bis))
                bis = []    

            elif mode=='all':
                outlist = np.array(bis)
        
        return np.array(outlist)
   
    def unique_c_phases(self):
        """Return an array of all unique closure phase triangles.
        """

        biarr = self.bispectra(mode="all", count="min")
        catsites = np.vstack((np.vstack((biarr['t1'],biarr['t2'])), biarr['t3'] ))
        uniqueclosure = np.vstack({tuple(row) for row in catsites.T})

        return uniqueclosure

    def c_phases(self, vtype='vis', mode='time', count='min', ang_unit='deg'):
        """Return a recarray of the equal time closure phases.
           
           Args: 
               vtype (str): The visibilty type ('vis','qvis','uvis','vvis','pvis') from which to assemble closure phases
               mode (str): If 'time', return phases in a list of equal time arrays, if 'all', return all phases in a single array
               count (str): If 'min', return minimal set of phases, if 'max' return all closure phases up to reordering
               ang_unit (str): If 'deg', return closure phases in degrees, else return in radians

           Returns:
               numpy.recarry: A recarray of the closure phases with datatype DTPHASE
        """
     
        if not mode in ('time', 'all'):
            raise Exception("possible options for mode are 'time' and 'all'")  
        if not count in ('max', 'min'):
            raise Exception("possible options for count are 'max' and 'min'")  
        if not vtype in ('vis','qvis','uvis','vvis','pvis'):
            raise Exception("possible options for vtype are 'vis','qvis','uvis','vvis','pvis'")
                
        if ang_unit=='deg': angle=DEGREE
        else: angle = 1.0
                    
        # Get the bispectra data
        bispecs = self.bispectra(vtype=vtype, mode='time', count=count)
        
        # Reformat into a closure phase list/array
        outlist = []
        cps = []
        for bis in bispecs:
            for bi in bis:
                if len(bi) == 0: continue
                bi.dtype.names = ('time','t1','t2','t3','u1','v1','u2','v2','u3','v3','cphase','sigmacp')
                bi['sigmacp'] = np.real(bi['sigmacp']/np.abs(bi['cphase'])/angle)
                bi['cphase'] = np.real((np.angle(bi['cphase'])/angle))
                cps.append(bi.astype(np.dtype(DTCPHASE)))
            if mode == 'time' and len(cps) > 0:
                outlist.append(np.array(cps))
                cps = []
                
        if mode == 'all':
            outlist = np.array(cps)

        return np.array(outlist)    
         
    def c_amplitudes(self, vtype='vis', mode='time', count='min', ctype='camp', debias=True):
        """Return a recarray of the equal time closure amplitudes.
           
           Args: 
               vtype (str): The visibilty type ('vis','qvis','uvis','vvis','pvis') from which to assemble closure amplitudes
               ctype (str): The closure amplitude type ('camp' or 'logcamp')
               mode (str): If 'time', return amplitudes in a list of equal time arrays, if 'all', return all amplitudes in a single array
               count (str): If 'min', return minimal set of amplitudes, if 'max' return all closure amplitudes up to inverses
               debias (bool): If True, debias the closure amplitude - the individual visibility amplitudes are always debiased.

           Returns:
               numpy.recarry: A recarray of the closure amplitudes with datatype DTCAMP
           
        """
     
        if not mode in ('time','all'):
            raise Exception("possible options for mode are 'time' and 'all'")
        if not count in ('max', 'min'):
            raise Exception("possible options for count are 'max' and 'min'")  
        if not vtype in ('vis','qvis','uvis','vvis','pvis'):
            raise Exception("possible options for vtype are 'vis','qvis','uvis','vvis','pvis'")
        if not (ctype in ['camp', 'logcamp']):
            raise Exception("closure amplitude type must be 'camp' or 'logcamp'!")

        # Get data sorted by time
        tlist = self.tlist(conj=True) 
        outlist = []
        cas = []
        for tdata in tlist:
            time = tdata[0]['time']
            sites = np.array(list(set(np.hstack((tdata['t1'],tdata['t2'])))))
            if len(sites) < 4:
                continue
                                            
            # Create a dictionary of baseline data at the current time including conjugates;
            l_dict = {}
            for dat in tdata:
                l_dict[(dat['t1'], dat['t2'])] = dat
            
            # Minimal set
            if count == 'min':
                # If we want a minimal set, choose the minimum sefd reference
                # !AC TODO this should probably be an sefdr + sefdl average
                sites = sites[np.argsort([self.tarr[self.tkey[site]]['sefdr'] for site in sites])]
                ref = sites[0]
                
                # Loop over other sites >=3 and form minimal closure amplitude set
                for i in xrange(3, len(sites)):
                    if (ref, sites[i]) not in l_dict: 
                        continue

                    blue1 = l_dict[ref, sites[i]] # MJ: This causes a KeyError in some cases, probably with flagged data or something
                    for j in xrange(1, i):
                        if j == i-1: k = 1
                        else: k = j+1
                        
                        if (sites[i], sites[j]) not in l_dict: # MJ: I tried joining these into a single if statement usng or without success... no idea why...
                            continue

                        if (ref, sites[k]) not in l_dict:
                            continue

                        if (sites[j], sites[k]) not in l_dict:
                            continue

                        red1 = l_dict[sites[i], sites[j]]
                        red2 = l_dict[ref, sites[k]]
                        blue2 = l_dict[sites[j], sites[k]] 
                        # Compute the closure amplitude and the error
                        (camp, camperr) = make_closure_amplitude(red1, red2, blue1, blue2, vtype, ctype=ctype)
   
                        # Add the closure amplitudes to the equal-time list  
                        # Our site convention is (12)(34)/(14)(23)       
                        cas.append(np.array((time, 
                                             ref, sites[i], sites[j], sites[k],
                                             blue1['u'], blue1['v'], blue2['u'], blue2['v'], 
                                             red1['u'], red1['v'], red2['u'], red2['v'],
                                             camp, camperr),
                                             dtype=DTCAMP)) 

            # Maximal Set
            elif count == 'max':
                # Find all quadrangles
                quadsets = list(it.combinations(sites,4))
                for q in quadsets:
                    # Loop over 3 closure amplitudes
                    # Our site convention is (12)(34)/(14)(23)
                    for quad in (q, [q[0],q[2],q[1],q[3]], [q[0],q[1],q[3],q[2]]): 
                        
                        # Blue is numerator, red is denominator
                        blue1 = l_dict[quad[0], quad[1]] #MJ: Need to add checks here
                        blue2 = l_dict[quad[2], quad[3]]
                        red1 = l_dict[quad[0], quad[3]]
                        red2 = l_dict[quad[1], quad[2]]
                        
                        # Compute the closure amplitude and the error
                        (camp, camperr) = make_closure_amplitude(red1, red2, blue1, blue2, vtype, ctype=ctype)
                                                                              
                        # Add the closure amplitudes to the equal-time list 
                        # Our site convention is (12)(34)/(14)(23)        
                        cas.append(np.array((time, 
                                             quad[0], quad[1], quad[2], quad[3],
                                             blue1['u'], blue1['v'], blue2['u'], blue2['v'], 
                                             red1['u'], red1['v'], red2['u'], red2['v'],
                                             camp, camperr),
                                             dtype=DTCAMP)) 

            # Append all equal time closure amps to outlist    
            if mode=='time':
                outlist.append(np.array(cas))
                cas = []    

            elif mode=='all':
                outlist = np.array(cas)
        
        return np.array(outlist)
    
    def dirtybeam(self, npix, fov, pulse=PULSE_DEFAULT):
        """Make an image observation dirty beam.
           
           Args:
               npix (int): The pixel size of the square output image. 
               fov (float): The field of view of the square output image in radians.
               pulse (function): The function convolved with the pixel values for continuous image. 

           Returns:
               Image: an Image object with the dirty beam.
        """


        # !AC TODO add different types of beam weighting
        pdim = fov/npix
        u = self.unpack('u')['u']
        v = self.unpack('v')['v']
        
        xlist = np.arange(0,-npix,-1)*pdim + (pdim*npix)/2.0 - pdim/2.0
        
        im = np.array([[np.mean(np.cos(2*np.pi*(i*u + j*v)))
                  for i in xlist] 
                  for j in xlist])    
        
        im = im[0:npix, 0:npix]
        
        # Normalize to a total beam power of 1
        im = im/np.sum(im)
        
        src = self.source + "_DB"
        return ehtim.image.Image(im, pdim, self.ra, self.dec, rf=self.rf, source=src, mjd=self.mjd, pulse=pulse)
        
    def cleanbeam(self, npix, fov, pulse=PULSE_DEFAULT):
        """Make an image of observation clean beam.
           
           Args:
               npix (int): The pixel size of the square output image. 
               fov (float): The field of view of the square output image in radians.
               pulse (function): The function convolved with the pixel values for continuous image. 

           Returns:
               Image: an Image object with the clean beam.
        """

        # !AC TODO include other beam weightings
        im = ehtim.image.make_square(self, npix, fov, pulse=pulse)
        beamparams = self.fit_beam()
        im = im.add_gauss(1.0, beamparams)
        return im
        
    def fit_beam(self):
        """Fit a gaussian to the dirty beam and return the parameters (fwhm_maj, fwhm_min, theta).

           Returns:
               tuple: a tuple (fwhm_maj, fwhm_min, theta) of the dirty beam parameters in radians.
        """

        # !AC TODO include other beam weightings
        # Define the sum of squares function that compares the quadratic expansion of the dirty image
        # with the quadratic expansion of an elliptical gaussian
        def fit_chisq(beamparams, db_coeff):
            
            (fwhm_maj2, fwhm_min2, theta) = beamparams
            a = 4 * np.log(2) * (np.cos(theta)**2/fwhm_min2 + np.sin(theta)**2/fwhm_maj2)
            b = 4 * np.log(2) * (np.cos(theta)**2/fwhm_maj2 + np.sin(theta)**2/fwhm_min2)
            c = 8 * np.log(2) * np.cos(theta) * np.sin(theta) * (1/fwhm_maj2 - 1/fwhm_min2)
            gauss_coeff = np.array((a,b,c))
            
            chisq = np.sum((np.array(db_coeff) - gauss_coeff)**2)
            
            return chisq
        
        # These are the coefficients (a,b,c) of a quadratic expansion of the dirty beam
        # For a point (x,y) in the image plane, the dirty beam expansion is 1-ax^2-by^2-cxy
        u = self.unpack('u')['u']
        v = self.unpack('v')['v']
        n = float(len(u))
        abc = (2.*np.pi**2/n) * np.array([np.sum(u**2), np.sum(v**2), 2*np.sum(u*v)])                
        abc = 1e-20 * abc # Decrease size of coefficients
        
        # Fit the beam 
        guess = [(50)**2, (50)**2, 0.0]
        params = opt.minimize(fit_chisq, guess, args=(abc,), method='Powell')
        
        # Return parameters, adjusting fwhm_maj and fwhm_min if necessary
        if params.x[0] > params.x[1]:
            fwhm_maj = 1e-10*np.sqrt(params.x[0])
            fwhm_min = 1e-10*np.sqrt(params.x[1])
            theta = np.mod(params.x[2], np.pi)
        else:
            fwhm_maj = 1e-10*np.sqrt(params.x[1])
            fwhm_min = 1e-10*np.sqrt(params.x[0])
            theta = np.mod(params.x[2] + np.pi/2, np.pi)

        return np.array((fwhm_maj, fwhm_min, theta))

    def dirtyimage(self, npix, fov, pulse=PULSE_DEFAULT):
        """Make the observation dirty image (direct Fourier transform).
           
           Args:
               npix (int): The pixel size of the square output image. 
               fov (float): The field of view of the square output image in radians.
               pulse (function): The function convolved with the pixel values for continuous image. 

           Returns:
               Image: an Image object with dirty image.
        """

        # !AC TODO add different types of beam weighting  
        pdim = fov/npix
        u = self.unpack('u')['u']
        v = self.unpack('v')['v']
        vis = self.unpack('vis')['vis']
        qvis = self.unpack('qvis')['qvis']
        uvis = self.unpack('uvis')['uvis']
        vvis = self.unpack('vvis')['vvis']
        
        xlist = np.arange(0,-npix,-1)*pdim + (pdim*npix)/2.0 - pdim/2.0

        # Take the DTFTS
        # Shouldn't need to real about conjugate baselines b/c unpack does not return them
        im  = np.array([[np.mean(np.real(vis)*np.cos(2*np.pi*(i*u + j*v)) - 
                                 np.imag(vis)*np.sin(2*np.pi*(i*u + j*v)))
                  for i in xlist] 
                  for j in xlist])    
        qim = np.array([[np.mean(np.real(qvis)*np.cos(2*np.pi*(i*u + j*v)) - 
                                 np.imag(qvis)*np.sin(2*np.pi*(i*u + j*v)))
                  for i in xlist] 
                  for j in xlist])     
        uim = np.array([[np.mean(np.real(uvis)*np.cos(2*np.pi*(i*u + j*v)) - 
                                 np.imag(uvis)*np.sin(2*np.pi*(i*u + j*v)))
                  for i in xlist] 
                  for j in xlist])    
        vim = np.array([[np.mean(np.real(vvis)*np.cos(2*np.pi*(i*u + j*v)) - 
                                 np.imag(vvis)*np.sin(2*np.pi*(i*u + j*v)))
                  for i in xlist] 
                  for j in xlist])    
                                                             
        dim = np.array([[np.mean(np.cos(2*np.pi*(i*u + j*v)))
                  for i in xlist] 
                  for j in xlist])   
        
        # Normalization   
        im = im/np.sum(dim)
        qim = qim/np.sum(dim)
        uim = uim/np.sum(dim)
        vim = vim/np.sum(dim)
        
        im = im[0:npix, 0:npix]
        qim = qim[0:npix, 0:npix]
        uim = uim[0:npix, 0:npix]   
        vim = vim[0:npix, 0:npix]
        
        out = ehtim.image.Image(im, pdim, self.ra, self.dec, rf=self.rf, source=self.source, mjd=self.mjd, pulse=pulse)
        out.add_qu(qim, uim)
        out.add_v(vim)
        
        return out

    def deblur(self):
        """Deblur the observation obs by dividing by the Sgr A* scattering kernel.

           Returns:
               Obsdata: a new deblurred observation object.
        """
        
        # make a copy of observation data
        datatable = (self.copy()).data

        vis = datatable['vis']
        qvis = datatable['qvis']
        uvis = datatable['uvis']
        vvis = datatable['vvis']
        sigma = datatable['sigma']
        qsigma = datatable['qsigma']
        usigma = datatable['usigma']
        vsigma = datatable['vsigma']            
        u = datatable['u']
        v = datatable['v']
        
        # divide visibilities by the scattering kernel
        for i in range(len(vis)):
            ker = sgra_kernel_uv(self.rf, u[i], v[i])
            vis[i] = vis[i] / ker
            qvis[i] = qvis[i] / ker
            uvis[i] = uvis[i] / ker
            vvis[i] = vvis[i] / ker
            sigma[i] = sigma[i] / ker
            qsigma[i] = qsigma[i] / ker
            usigma[i] = usigma[i] / ker
            vsigma[i] = vsigma[i] / ker
                                
        datatable['vis'] = vis
        datatable['qvis'] = qvis
        datatable['uvis'] = uvis
        datatable['vvis'] = vvis
        datatable['sigma'] = sigma
        datatable['qsigma'] = qsigma
        datatable['usigma'] = usigma
        datatable['vsigma'] = vsigma    
        
        obsdeblur = Obsdata(self.ra, self.dec, self.rf, self.bw, datatable, self.tarr, source=self.source, mjd=self.mjd, 
                            ampcal=self.ampcal, phasecal=self.phasecal, opacitycal=self.opacitycal, dcal=self.dcal, frcal=self.frcal)
        return obsdeblur

    def fit_gauss(self, flux=1.0, fittype='amp', paramguess=(100*RADPERUAS, 100*RADPERUAS, 0.)):
        """Fit a gaussian to either Stokes I complex visibilities or Stokes I visibility amplitudes.
        """
        vis = self.data['vis']
        u = self.data['u']
        v = self.data['v']
        sig = self.data['sigma']
        
        # error function
        if fittype=='amp':
            def errfunc(p):
            	vismodel = gauss_uv(u,v, flux, p, x=0., y=0.)
            	err = np.sum((np.abs(vis)-np.abs(vismodel))**2/sig**2)
            	return err
        else:
            def errfunc(p):
            	vismodel = gauss_uv(u,v, flux, p, x=0., y=0.)
            	err = np.sum(np.abs(vis-vismodel)**2/sig**2)
            	return err
        
        optdict = {'maxiter':5000} # minimizer params
        res = opt.minimize(errfunc, paramguess, method='Powell',options=optdict)
        return res.x
    
    def plotall(self, field1, field2, ebar=True, rangex=False, rangey=False, conj=False, show=True, axis=False, color='b', ang_unit='deg', debias=True):
        """Make a scatter plot of 2 real observation fields with errors.
           If conj==True, display conjugate baselines.
        """
        
        # Determine if fields are valid
        if (field1 not in FIELDS) and (field2 not in FIELDS):
            raise Exception("valid fields are " + string.join(FIELDS))
                              
        # Unpack x and y axis data
        data = self.unpack([field1, field2], conj=conj, ang_unit=ang_unit, debias=debias)
        
        # X error bars
        if sigtype(field1):
            sigx = self.unpack(sigtype(field2), conj=conj, ang_unit=ang_unit)[sigtype(field1)]
        else:
            sigx = None
            
        # Y error bars
        if sigtype(field2):
            sigy = self.unpack(sigtype(field2), conj=conj, ang_unit=ang_unit)[sigtype(field2)]
        else:
            sigy = None
        
        # Debias amplitudes if appropriate:
        # !AC TODO - now debiasing done in unpack -- ok?
        #if field1 in ['amp', 'qamp', 'uamp', 'vamp', 'pamp', 'mamp']:
        #    print "De-biasing amplitudes for plot x values!"
        #    data[field1] = amp_debias(data[field1], sigx)
        
        #if field2 in ['amp', 'qamp', 'uamp', 'vamp', 'pamp', 'mamp']:
        #    print "De-biasing amplitudes for plot y values!"
        #    data[field2] = amp_debias(data[field2], sigy)
           
        # Data ranges
        if not rangex:
            rangex = [np.min(data[field1]) - 0.2 * np.abs(np.min(data[field1])), 
                      np.max(data[field1]) + 0.2 * np.abs(np.max(data[field1]))] 
        if not rangey:
            rangey = [np.min(data[field2]) - 0.2 * np.abs(np.min(data[field2])), 
                      np.max(data[field2]) + 0.2 * np.abs(np.max(data[field2]))] 
        
        # Plot the data
        if axis:
            x = axis
        else:
            fig=plt.figure()
            x = fig.add_subplot(1,1,1)
         
        if ebar and (np.any(sigy) or np.any(sigx)):
            x.errorbar(data[field1], data[field2], xerr=sigx, yerr=sigy, fmt='.', color=color)
        else:
            x.plot(data[field1], data[field2], '.', color=color)
            
        x.set_xlim(rangex)
        x.set_ylim(rangey)
        x.set_xlabel(field1)
        x.set_ylabel(field2)

        if show:
            plt.show(block=False)
        return x
                        
    def plot_bl(self, site1, site2, field, ebar=True, rangex=False, rangey=False, show=True, axis=False, color='b', ang_unit='deg', debias=True):
        """Plot a field over time on a baseline site1-site2. 
        """
                
        if ang_unit=='deg': angle=DEGREE
        else: angle = 1.0
        
        # Determine if fields are valid
        if field not in FIELDS:
            raise Exception("valid fields are " + string.join(FIELDS))
        
        plotdata = self.unpack_bl(site1, site2, field, ang_unit=ang_unit, debias=debias)
        if not rangex: 
            rangex = [self.tstart,self.tstop]
        if not rangey:
            rangey = [np.min(plotdata[field]) - 0.2 * np.abs(np.min(plotdata[field])), 
                      np.max(plotdata[field]) + 0.2 * np.abs(np.max(plotdata[field]))] 
        # Plot the data                        
        if axis:
            x = axis
        else:
            fig = plt.figure()
            x = fig.add_subplot(1,1,1)       

        if ebar and sigtype(field)!=False:
            errdata = self.unpack_bl(site1, site2, sigtype(field), ang_unit=ang_unit, debias=debias)
            x.errorbar(plotdata['time'][:,0], plotdata[field][:,0], yerr=errdata[sigtype(field)][:,0], fmt='b.', color=color)
        else:
            x.plot(plotdata['time'][:,0], plotdata[field][:,0],'b.', color=color)
            
        x.set_xlim(rangex)
        x.set_ylim(rangey)
        x.set_xlabel('hr')
        x.set_ylabel(field)
        x.set_title('%s - %s'%(site1,site2))
        
        if show:
            plt.show(block=False)    
        return x
                
                
    def plot_cphase(self, site1, site2, site3, vtype='vis', ebar=True, rangex=False, rangey=False, show=True, axis=False, color='b', ang_unit='deg'):
        """Plot closure phase over time on a triangle (1-2-3). 
        """
                
        if ang_unit=='deg': angle=DEGREE 
        else: angle = 1.0
        
        # Get closure phases (maximal set)
        cphases = self.c_phases(mode='time', count='max', vtype=vtype)
        
        # Get requested closure phases over time
        tri = (site1, site2, site3)
        plotdata = []
        for entry in cphases:
            for obs in entry:
                obstri = (obs['t1'],obs['t2'],obs['t3'])
                if set(obstri) == set(tri):
                    # Flip the sign of the closure phase if necessary
                    parity = paritycompare(tri, obstri) 
                    plotdata.append([obs['time'], parity*obs['cphase'], obs['sigmacp']])
                    continue
        
        plotdata = np.array(plotdata)
        
        if len(plotdata) == 0: 
            print "No closure phases on this triangle!"
            return
        
        # Data ranges
        if not rangex: 
            rangex = [self.tstart,self.tstop]
        if not rangey:
            rangey = [np.min(plotdata[:,1]) - 0.2 * np.abs(np.min(plotdata[:,1])), 
                      np.max(plotdata[:,1]) + 0.2 * np.abs(np.max(plotdata[:,1]))] 
        
        # Plot the data                        
        if axis:
            x = axis
        else:
            fig=plt.figure()
            x = fig.add_subplot(1,1,1)       

        if ebar and np.any(plotdata[:,2]):
            x.errorbar(plotdata[:,0], plotdata[:,1], yerr=plotdata[:,2], fmt='b.', color=color)
        else:
            x.plot(plotdata[:,0], plotdata[:,1],'b.', color=color)
            
        x.set_xlim(rangex)
        x.set_ylim(rangey)
        x.set_xlabel('GMT (h)')
        x.set_ylabel('Closure Phase (deg)')
        x.set_title('%s - %s - %s' % (site1,site2,site3))
        if show:
            plt.show(block=False)    
        return x
        
    def plot_camp(self, site1, site2, site3, site4, vtype='vis',ctype='camp', debias=True, 
                        ebar=True, rangex=False, rangey=False, show=True, axis=False, color='b'):
        """Plot closure amplitude over time on a quadrange (1-2)(3-4)/(1-4)(2-3).
        """
        quad = (site1, site2, site3, site4)
        b1 = set((site1, site2))
        r1 = set((site1, site4))
              
        # Get the closure amplitudes
        camps = self.c_amplitudes(mode='time', count='max', vtype='vis', ctype=ctype, debias=debias)
        plotdata = []
        for entry in camps:
            for obs in entry:
                obsquad = (obs['t1'],obs['t2'],obs['t3'],obs['t4'])
                if set(quad) == set(obsquad):
                    num = [set((obs['t1'], obs['t2'])), set((obs['t3'], obs['t4']))] 
                    denom = [set((obs['t1'], obs['t4'])), set((obs['t2'], obs['t3']))]
                    
                    if (b1 in num) and (r1 in denom):
                        plotdata.append([obs['time'], obs['camp'], obs['sigmaca']])
                    elif (r1 in num) and (b1 in denom):
                        plotdata.append([obs['time'], 1./obs['camp'], obs['sigmaca']/(obs['camp']**2)])
                    continue
                
                    
        plotdata = np.array(plotdata)
        if len(plotdata) == 0: 
            print "No closure amplitudes on this quadrangle!"
            return

        # Data ranges
        if not rangex: 
            rangex = [self.tstart,self.tstop]
        if not rangey:
            rangey = [np.min(plotdata[:,1]) - 0.2 * np.abs(np.min(plotdata[:,1])), 
                      np.max(plotdata[:,1]) + 0.2 * np.abs(np.max(plotdata[:,1]))] 
        
        # Plot the data                        
        if axis:
            x = axis
        else:
            fig=plt.figure()
            x = fig.add_subplot(1,1,1)       
            
        if ebar and np.any(plotdata[:,2]):
            x.errorbar(plotdata[:,0], plotdata[:,1], yerr=plotdata[:,2], fmt='b.', color=color)
        else:
            x.plot(plotdata[:,0], plotdata[:,1],'b.', color=color)
            
        x.set_xlim(rangex)
        x.set_ylim(rangey)
        x.set_xlabel('GMT (h)')
        if ctype=='camp':
            x.set_ylabel('Closure Amplitude')
        elif ctype=='logcamp':
            x.set_ylabel('Log Closure Amplitude')
        x.set_title('(%s - %s)(%s - %s)/(%s - %s)(%s - %s)'%(site1,site2,site3,site4,
                                                           site1,site4,site2,site3))
        if show:
            plt.show(block=False)    
            return
        else:
            return x
                	        
    def save_txt(self, fname):
        """Save visibility data to a text file."""

        ehtim.io.save.save_obs_txt(self,fname)
        return
    
    #!AC TODO how can we save dterm and field rotation arrays to uvfits?
    def save_uvfits(self, fname):
        """Save visibility data to uvfits. 
        """
        ehtim.io.save.save_obs_uvfits(self,fname)
        return
    
    def save_oifits(self, fname, flux=1.0):
        """ Save visibility data to oifits. Polarization data is NOT saved.
        """
        #Antenna diameter currently incorrect and the exact times are not correct in the datetime object
        #Please contact Katie Bouman (klbouman@mit.edu) for any questions on this function 
        
        ehtim.io.save.save_obs_oifits(self, fname, flux=flux)
        return

##################################################################################################
# Observation creation functions
##################################################################################################

def merge_obs(obs_List):
    """Merge a list of observations into a single observation file.
    """

    if (len(set([obs.ra for obs in obs_List])) > 1 or 
        len(set([obs.dec for obs in obs_List])) > 1 or 
        len(set([obs.rf for obs in obs_List])) > 1 or 
        len(set([obs.bw for obs in obs_List])) > 1 or 
        len(set([obs.source for obs in obs_List])) > 1 or 
        len(set([np.floor(obs.mjd) for obs in obs_List])) > 1):

        raise Exception("All observations must have the same parameters!")
        return 

    #The important things to merge are the mjd, the data, and the list of telescopes
    data_merge = np.hstack([obs.data for obs in obs_List]) 

    mergeobs = Obsdata(obs_List[0].ra, obs_List[0].dec, obs_List[0].rf, obs_List[0].bw, data_merge, np.unique(np.concatenate([obs.tarr for obs in obs_List])), 
                       source=obs_List[0].source, mjd=obs_List[0].mjd, ampcal=obs_List[0].ampcal, phasecal=obs_List[0].phasecal) 

    return mergeobs

def load_txt(fname):
    """Read an observation from a text file and return an Obsdata object, with the same format as output from Obsdata.savedata().
    """
    return ehtim.io.load.load_obs_txt(fname)

def load_uvfits(fname, flipbl=False):
    """Load observation data from a uvfits file.
    """
    return ehtim.io.load.load_obs_uvfits(fname, flipbl=flipbl)

def load_oifits(fname, flux=1.0):
    """Load data from an oifits file. Does NOT currently support polarization.
    """
    return ehtim.io.load.load_obs_oifits(fname, flux=flux)

def load_maps(arrfile, obsspec, ifile, qfile=0, ufile=0, vfile=0, src=SOURCE_DEFAULT, mjd=MJD_DEFAULT, ampcal=False, phasecal=False):
    """Read an observation from a maps text file and return an Obsdata object.
    """
    return ehtim.io.load.load_obs_maps(arrfile, obsspec, ifile, qfile=qfile, ufile=ufile, vfile=vfile, src=src, mjd=mjd, ampcal=ampcal, phasecal=phasecal)

