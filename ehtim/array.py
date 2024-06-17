# array.py
# a interferometric telescope array class
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

from __future__ import division
from __future__ import print_function

from builtins import str
from builtins import range
from builtins import object

import numpy as np
import copy
from astropy.time import Time
import matplotlib.pyplot as plt
import matplotlib
import ehtim.observing.obs_simulate as simobs
import ehtim.observing.obs_helpers as obsh
import ehtim.io.save
import ehtim.io.load
import ehtim.const_def as ehc
from ehtim.caltable import plot_tarr_dterms


###################################################################################################
# Array object
###################################################################################################


class Array(object):

    """A VLBI array of telescopes with site locations, SEFDs, and other data.

       Attributes:
           tarr (numpy.recarray): The array of telescope data with datatype DTARR
           tkey (dict): A dictionary of rows in the tarr for each site name
           ephem (dict): A dictionary of TLEs for each space antenna,
                         Space antennas have x=y=z=0 in the tarr
    """

    def __init__(self, tarr, ephem={}):
        self.tarr = tarr
        self.ephem = ephem

        # check to see if ephemeris is correct
        for line in self.tarr:
            if np.any(np.isnan([line['x'], line['y'], line['z']])):
                sitename = str(line['site'])
                try:
                    elen = len(ephem[sitename])
                except NameError:
                    raise Exception('no ephemeris for site %s !' % sitename)
                if elen != 3:
                    raise Exception('wrong ephemeris format for site %s !' % sitename)

        # Dictionary of array indices for site names
        self.tkey = {self.tarr[i]['site']: i for i in range(len(self.tarr))}

    @property 
    def tarr(self):
        return self._tarr
        
    @tarr.setter 
    def tarr(self, tarr):
        self._tarr = tarr
        self.tkey = {tarr[i]['site']: i for i in range(len(tarr))}

    def copy(self):
        """Copy the array object.

           Args:

           Returns:
               (Array): a copy of the Array object.
        """

        newarr = copy.deepcopy(self)
        return newarr

       
    def listbls(self):
        """List all baselines.

           Args:
           Returns:
                numpy.array : array of baselines
        """

        bls = []
        for i1 in sorted(self.tarr['site']):
            for i2 in sorted(self.tarr['site']):
                if not ([i1, i2] in bls) and not ([i2, i1] in bls) and i1 != i2:
                    bls.append([i1, i2])
        bls = np.array(bls)

        return bls

    def obsdata(self, ra, dec, rf, bw, tint, tadv, tstart, tstop,
                mjd=ehc.MJD_DEFAULT, timetype='UTC', polrep='stokes',
                elevmin=ehc.ELEV_LOW, elevmax=ehc.ELEV_HIGH,
                no_elevcut_space=False,
                tau=ehc.TAUDEF, fix_theta_GMST=False):
        """Generate u,v points and baseline uncertainties.

           Args:
               ra (float): the source right ascension in fractional hours
               dec (float): the source declination in fractional degrees
               tint (float): the scan integration time in seconds
               tadv (float): the uniform cadence between scans in seconds
               tstart (float): the start time of the observation in hours
               tstop (float): the end time of the observation in hours

               mjd (int): the mjd of the observation
               timetype (str): how to interpret tstart and tstop; either 'GMST' or 'UTC'
               polrep (str): polarization representation, either 'stokes' or 'circ'
               elevmin (float): station minimum elevation in degrees
               elevmax (float): station maximum elevation in degrees
               no_elevcut_space (bool): if True, do not apply elevation cut to orbiters
               tau (float): the base opacity at all sites, or a dict giving one opacity per site
               fix_theta_GMST (bool): if True, stops earth rotation to sample fixed u,v points
               
           Returns:
               Obsdata: an observation object with no data

        """

        obsarr = simobs.make_uvpoints(self, ra, dec, rf, bw,
                                      tint, tadv, tstart, tstop,
                                      mjd=mjd, polrep=polrep, tau=tau,
                                      elevmin=elevmin, elevmax=elevmax, 
                                      no_elevcut_space=no_elevcut_space,
                                      timetype=timetype, fix_theta_GMST=fix_theta_GMST)

        uniquetimes = np.sort(np.unique(obsarr['time']))
        scans = np.array([[time - 0.5 * tadv, time + 0.5 * tadv] for time in uniquetimes])
        source = str(ra) + ":" + str(dec)
        obs = ehtim.obsdata.Obsdata(ra, dec, rf, bw, obsarr, self.tarr,
                                    source=source, mjd=mjd, timetype=timetype, polrep=polrep,
                                    ampcal=True, phasecal=True, opacitycal=True,
                                    dcal=True, frcal=True,
                                    scantable=scans)
        return obs

    def make_subarray(self, sites):
        """Make a subarray from the Array object array that only includes the sites listed.

           Args:
               sites (list) : list of sites in the subarray
           Returns:
               Array: an Array object with specified sites and metadata
        """
        all_sites = [t[0] for t in self.tarr]
        mask = np.array([t in sites for t in all_sites])
        subarr = Array(self.tarr[mask], ephem=self.ephem)
        return subarr

    def save_txt(self, fname):
        """Save the array data in a text file.

           Args:
               fname (str) : path to output array file
        """
        ehtim.io.save.save_array_txt(self, fname)
        return

    def plot_dterms(self, sites='all', label=None, legend=True, clist=ehc.SCOLORS,
                    rangex=False, rangey=False, markersize=2 * ehc.MARKERSIZE,
                    show=True, grid=True, export_pdf=""):
        """Make a plot of the D-terms.

           Args:
               sites (list) : list of sites to plot
               label (str) : title for plot
               legend (bool) : add telescope legend or not
               clist (list) : list of colors for different stations
               rangex (list) : lower and upper x-axis limits  
               rangey (list) : lower and upper y-axis limits 
               markersize (float) : marker size
               show (bool) : display the plot or not
               grid (bool) : add a grid to the plot or not
               export_pdf (str) : save a pdf file to this path 
               
           Returns:
               matplotlib.axes
        """
        # sites
        if (isinstance(sites,str) and sites.lower() == 'all'):
            sites = list(self.tkey.keys())

        if not isinstance(sites, list):
            sites = [sites]

        if len(sites)==0:
            sites = list(self.tkey.keys())
                    
        keys = [self.tkey[site] for site in sites]

        axes = plot_tarr_dterms(self.tarr, keys=keys, label=label, legend=legend, clist=clist,
                                rangex=rangex, rangey=rangey, markersize=markersize,
                                show=show, grid=grid, export_pdf=export_pdf)

        return axes

    def add_site(self, site, coords, sefd=10000, 
                 fr_par=0, fr_elev=0, fr_off=0, 
                 dr=0.+0.j, dl=0.+0.j):
    
        """Add a ground station to the array
             
        """
        tarr_old = self.tarr.copy()
        ephem_old = self.ephem.copy()
        
        
        tarr_newline = np.array((str(site), float(coords[0]), float(coords[1]), float(coords[2]), 
                                 float(sefd), float(sefd), 
                                 dr, dl, 
                                 float(fr_par), float(fr_elev), float(fr_off)), dtype=ehc.DTARR)
        tarr_new = np.append(tarr_old, tarr_newline)
        
        arr_out = Array(tarr_new, ephem_old)
        return arr_out
            
    def remove_site(self, site):
        """Remove a site from the array
           
        """
        tarr_old = self.tarr.copy()
        ephem_old = self.ephem.copy()
        ephem_new = ephem_old.copy()
        
        try:
            tarr_new = np.delete(tarr_old.copy(), self.tkey[site])
            if site in ephem_old.keys():
                ephem_new.pop(site) 
        except:
            raise Exception("could not find site %s to delete from Array!"%site)
        
        arr_out = Array(tarr_new, ephem_new)
        return arr_out

    def add_satellite_tle(self, tlelist, sefd=10000):
    
        """Add an earth-orbiting satellite to the array from a TLE
        
           Args: 
             tlearr (str) : 3 element list with [name, tle line 1, tle line 2] as strings
             sefd (float) : assumed sefd for the array file (assumes sefdl = sefdr)     
        """
        satname = tlearr[0]
        tarr_new = self.tarr.copy()
        ephem_new = self.ephem.copy()
        
        tarr_newline = np.array((str(satname), 0., 0., 0.,
                                 float(sefd), float(sefd), 
                                 0., 0., 0., 0., 0.), dtype=ehc.DTARR)
        tarr_new = np.append(tarr_new, tarr_newline)
        ephem_new[satname] = tlearr
        arr_out = Array(tarr_new, ephem_new)
        
        return arr_out

    def add_satellite_elements(self, satname, 
                               perigee_mjd=Time.now().mjd, 
                               period_days=1., eccentricity=0.,
                               inclination=0., arg_perigee=0., long_ascending=0., 
                               sefd=10000):
        """Add an earth-orbiting satellite to the array from simple keplerian elements
           perfect keplerian orbit is assumed, no derivatives
                   
           Args: 
               perigee time given in mjd
               period given in days
               inclination, arg_perigee, long_ascending given in degrees
        """

        tarr_new = self.tarr.copy()
        ephem_new = self.ephem.copy()
        
        tarr_newline = np.array((str(satname), 0., 0., 0.,
                                 float(sefd), float(sefd), 
                                 0., 0., 0., 0., 0.), dtype=ehc.DTARR)
        tarr_new = np.append(tarr_new, tarr_newline)
        
        ephem_new[satname] = [perigee_mjd, period_days, eccentricity, inclination, arg_perigee, long_ascending]
        arr_out = Array(tarr_new, ephem_new)
                
        return arr_out
        
    def plot_satellite_orbits(self, tstart_mjd=Time.now().mjd, tstop_mjd=Time.now().mjd+1, npoints=1000):
        earth_radius_polar = 6357. #km
        earth_radius_eq = 6378.
    
        fig = plt.figure(figsize=(18,6))
        gs = matplotlib.gridspec.GridSpec(1,3,width_ratios=[1,1,1])
        
        satellites = self.ephem.keys()
        for i,satellite in enumerate(satellites):
        
            if i==0: color='k'
            else: color=ehc.SCOLORS[i-1]
            
            # get skyfield satelllite object
            if len(self.ephem[satellite])==3: # TLE
                line1 = self.ephem[satellite][1]
                line2 = self.ephem[satellite][2]            
                sat = obsh.sat_skyfield_from_tle(satellite, line1, line2)
            elif len(self.ephem[satellite])==6: #keplerian elements
                elements = self.ephem[satellite]
                sat = obsh.sat_skyfield_from_elements(satellite, tstart_mjd,
                                                      elements[0],elements[1],elements[2],elements[3],elements[4],elements[5])
            else:
                raise Exception("ephemeris format not recognized for %s"%satellite)    
        
            # get GCRS positions
            fracmjds = np.linspace(tstart_mjd, tstop_mjd, npoints)
            positions = obsh.orbit_skyfield(sat, fracmjds, whichout='gcrs')
            positions *= 1.e-3 # convert to km
            distances = np.sqrt(positions[0]**2 + positions[1]**2 + positions[2]**2)
            maxdist = np.max(distances)
            
            ax1 = fig.add_subplot(gs[0])
            ax1.set_aspect(1)
            plt.plot(positions[0], positions[1], color=color, marker='.',ls='None')
            circle1 = matplotlib.patches.Circle((0, 0), earth_radius_eq, color='b')
            plt.gca().add_patch(circle1)
            plt.xlabel('x (km)')
            plt.ylabel('y (km)')
            plt.xlim(-1.1*maxdist, 1.1*maxdist)
            plt.ylim(-1.1*maxdist, 1.1*maxdist)
            plt.grid()

            ax2 = fig.add_subplot(gs[1])
            ax2.set_aspect(1)
            plt.plot(positions[1], positions[2], color=color, marker='.',ls='None')
            circle1 = matplotlib.patches.Ellipse((0, 0), 2*earth_radius_eq, 2*earth_radius_polar, color='b')
            plt.gca().add_patch(circle1)
            plt.xlabel('y (km)')
            plt.ylabel('z (km)')
            plt.xlim(-1.1*maxdist, 1.1*maxdist)
            plt.ylim(-1.1*maxdist, 1.1*maxdist)
            plt.grid()  
            
            ax3 = fig.add_subplot(gs[2])
            ax3.set_aspect(1)
            plt.plot(positions[0], positions[2], color=color, marker='.',ls='None', label=satellite)
            circle1 = matplotlib.patches.Ellipse((0, 0), 2*earth_radius_eq, 2*earth_radius_polar, color='b')
            plt.gca().add_patch(circle1)
            plt.xlabel('x (km)')
            plt.ylabel('z (km)')
            plt.xlim(-1.1*maxdist, 1.1*maxdist)
            plt.ylim(-1.1*maxdist, 1.1*maxdist)
            plt.legend(frameon=False,loc='center left', bbox_to_anchor=(1, 0.5))
            plt.grid()

        plt.subplots_adjust(wspace=1)
        ehc.show_noblock()            
        return
                                 
##########################################################################
# Array creation functions
##########################################################################


def load_txt(fname, ephemdir='ephemeris'):
    """Read an array from a text file.
       Sites with x=y=z=0 are spacecraft, TLE ephemerides read from ephemdir.

       Args:
           fname (str) : path to input array file
           ephemdir (str) : path to directory with TLE ephemerides for spacecraft
       Returns:
           Array: an Array object loaded from file
    """

    return ehtim.io.load.load_array_txt(fname, ephemdir=ephemdir)
