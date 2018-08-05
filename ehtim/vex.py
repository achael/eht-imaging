# vex.py
# a interferometric array vex schedule class
#
#    Copyright (C) 2018 Hotaka Shiokawa
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
from builtins import range
from builtins import object

import numpy as np
import re

#import jdcal
from astropy.time import Time
import os
import ehtim.array
from ehtim.const_def import *
from ehtim.observing.obs_helpers import *

###########################################################################################################################################
# Vex object
###########################################################################################################################################
class Vex(object):
    """Read in observing schedule data from .vex files.
       Assumes there is only 1 MODE in vex file

       Attributes:
           filename (str): The .vex filename.
           source (str): The source name.
           metalist (list): The observation information.
           sched (list): The schedule information.
           array (Array): an Array object of sites.
    """

    def __init__(self, filename):

        f = open(filename)
        raw = f.readlines()
        f.close()

        self.filename = filename

        # Divide 'raw' data into sectors of '$' marks
        # ASSUMING '$' is the very first character in a line (no space in front)
        metalist = [] # meaning list of metadata

        for i in range(len(raw)):
            if raw[i][0]=='$':
                temp = [raw[i]]
                break

        for j in range(i+1,len(raw)):
            if raw[j][0]!='$':
                temp.append(raw[j])
            elif raw[j][0]=='$':
                metalist.append(temp)
                temp = [raw[j]]
            else:
                print('Something is wrong.')
        metalist.append(temp) # don't forget to add the final one
        self.metalist = metalist


        # Extract desired information
        # SOURCE ========================================================
        SOURCE = self.get_sector('SOURCE')
        source = []
        indef = False

        for i in range(len(SOURCE)):

            line = SOURCE[i]
            if line[0:3]=="def":
                indef=True

            if indef:
                ret = self.get_variable("source_name",line)
                if len(ret)>0: source_name = ret
                ret = self.get_variable("ra",line)
                if len(ret)>0: ra = ret
                ret = self.get_variable("dec",line)
                if len(ret)>0: dec = ret
                ret = self.get_variable("ref_coord_frame",line)
                if len(ret)>0: ref_coord_frame = ret

                if line[0:6]=="enddef":
                    source.append({'source':source_name,'ra':ra,'dec':dec,'ref_coord_frame':ref_coord_frame})
                    indef=False

        self.source = source

        # FREQ ==========================================================
        FREQ = self.get_sector('FREQ')
        indef = False
        nfreq = 0
        for i in range(len(FREQ)):

            line = FREQ[i]
            if line[0:3]=="def":
                if nfreq>0: print("Not implemented yet.")
                nfreq += 1
                indef=True

            if indef:
                idx = line.find('chan_def')
                if idx>=0 and line[0]!='*':
                    chan_def = re.findall("[-+]?\d+[\.]?\d*",line)
                    self.freq = float(chan_def[0])*1.e6
                    self.bw_hz = float(chan_def[1])*1.e6

                if line[0:6]=="enddef": indef=False


        # SITE ==========================================================
        SITE = self.get_sector('SITE')
        sites = []
        site_ID_dict = {}
        indef = False

        for i in range(len(SITE)):

            line = SITE[i]
            if line[0:3]=="def": indef=True

            if indef:
                # get site_name and SEFD
                ret = self.get_variable("site_name",line)
                if len(ret)>0:
                    site_name = ret
                    SEFD = self.get_SEFD(site_name)

                # making dictionary of site_ID:site_name
                ret = self.get_variable("site_ID",line)
                if len(ret)>0:
                    site_ID_dict[ret] = site_name

                # get site_position
                ret = self.get_variable("site_position",line)
                if len(ret)>0:
                    site_position = re.findall("[-+]?\d+[\.]?\d*",line)

                # same format as Andrew's array tables
                if line[0:6]=="enddef":
                    sites.append([site_name,site_position[0],site_position[1],site_position[2],SEFD])
                    indef=False


        # Construct Array() object of Andrew's format
        # mimic the function "load_array(filename)"
        # TODO this does not store d-term and pol cal. information!
        tdataout = [np.array((x[0],float(x[1]),float(x[2]),float(x[3]),float(x[4]),float(x[4]),0.0, 0.0, 0.0, 0.0, 0.0),
                               dtype=DTARR) for x in sites]
        tdataout = np.array(tdataout)
        self.array = ehtim.array.Array(tdataout)

        # SCHED  =========================================================
        SCHED = self.get_sector('SCHED')
        sched = []
        inscan = False

        for i in range(len(SCHED)):

            line = SCHED[i]
            if line[0:4]=="scan":
                inscan=True
                temp={}
                temp['scan']={}
                cnt = 0

            if inscan:
                ret = self.get_variable("start",line)
                if len(ret)>0:
                    mjd,hr = vexdate_to_MJD_hr(ret) # convert vex time format to mjd and hour
                    temp['mjd_floor'] = mjd
                    temp['start_hr'] = hr

                ret = self.get_variable("mode",line)
                if len(ret)>0: temp['mode'] = ret

                ret = self.get_variable("source",line)
                if len(ret)>0: temp['source'] = ret

                ret = self.get_variable("station",line)
                if len(ret)>0:
                    site_ID = ret
                    site_name = site_ID_dict[site_ID] # convert to more familier site name
                    sdur = re.findall("[-+]?\d+[\.]?\d*",line)
                    s_st = float(sdur[0]) # start time in sec
                    s_en = float(sdur[1]) # end time in sec
                    d_size = float(sdur[2]) # data size(?) in GB
                    temp['scan'][cnt] = {'site':site_name,'scan_sec_start':s_st,'scan_sec':s_en,'data_size':d_size}
                    cnt +=1

                if line[0:7]=="endscan":
                    sched.append(temp)
                    inscan=False

        self.sched = sched


    # Function to obtain a desired sector from 'metalist'
    def get_sector(self, sname):
        """Obtain a desired sector from 'metalist'. 
        """

        for i in range(len(self.metalist)):
            if sname in self.metalist[i][0]:
                return self.metalist[i]
        print('No sector named %s'%sname)
        return False

    # Function to get a value of 'vname' in a line which has format of
    # 'vname' = value ;(or :)
    def get_variable(self, vname, line):
        """Function to get a value of 'vname' in a line.
        """

        idx = self.find_variable(vname,line)
        name = ''
        if idx>=0:
            start = False
            for i in range(idx+len(vname),len(line)):
                if start==True:
                    if line[i]==';' or line[i]==':': break
                    elif line[i]!=' ': name += line[i]
                if start==False and line[i]!=' ' and line[i]!='=': break
                if line[i]=='=': start = True
        return name

    # check if a variable 'vname' exists by itself in a line.
    # returns index of vname[0] in a line, or -1
    def find_variable(self, vname, line):
        """Function to find a variable 'vname' in a line.
        """
        idx = line.find(vname)
        if ((idx>0 and line[idx-1]==' ') or idx==0) and line[0]!='*':
            if idx+len(vname)==len(line): return idx
            if line[idx+len(vname)]=='=' or line[idx+len(vname)]==' ' or line[idx+len(vname)]==':' or line[idx+len(vname)]==';': return idx
        return -1

    # Find SEFD for a given station name.
    # For now look for it in Andrew's tables
    # Vex files could have SEFD sector.
    def get_SEFD(self, station):
        """Find SEFD for a given station.
        """
        f = open(os.path.dirname(os.path.abspath(__file__)) + "/../arrays/SITES.txt")
        sites = f.readlines()
        f.close()
        for i in range(len(sites)):
            if sites[i].split()[0]==station:
                return float(re.findall("[-+]?\d+[\.]?\d*",sites[i])[3])
        print('No station named %s'%station)
        return 10000. # some arbitrary value

    # Find the time that any station starts observing the source in MJD.
    # Find the time that the last station stops observing the source in MHD.
    def get_obs_timerange(self, source):
        """Find the time that any station starts observing the source in MJD,
           and the time that the last station stops observing the source.
        """

        sched = self.sched
        first = True
        for i_scan in range(len(sched)):
            if sched[i_scan]['source']==source and first==True:
                Tstart_hr = sched[i_scan]['start_hr']
                mjd_s = sched[i_scan]['mjd_floor'] + Tstart_hr/24.
                first = False
            if sched[i_scan]['source']==source and first==False:
                Tstop_hr = sched[i_scan]['start_hr'] + sched[i_scan]['scan'][0]['scan_sec']/3600.
                mjd_e = sched[i_scan]['mjd_floor'] + Tstop_hr/24.

        return mjd_s, mjd_e


#=================================================================
#=================================================================

# Function to find MJD (int!) and hour in UT from vex format,
# e.g, 2016y099d05h00m00s
def vexdate_to_MJD_hr(vexdate):
    """Find the integer MJD and UT hour from vex format date. 
    """

    time = re.findall("[-+]?\d+[\.]?\d*",vexdate)
    year = int(time[0])
    date = int(time[1])
    yeardatetime = ("%04i"%year)+':'+("%03i"%date)+":00:00:00.000"
    t = Time(yeardatetime, format='yday')
    mjd = t.mjd
    #mjd = jdcal.gcal2jd(year,1,1)[1]+date-1
    #print (mjd,mjd2,mjd-mjd2)
    hour = int(time[2]) + float(time[3])/60. + float(time[4])/60./60.
    return mjd,hour
