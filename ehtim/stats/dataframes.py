# DataFrames.py
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
try:
    import pandas as pd
except ImportError:
    print("Warning: pandas not installed!")
    print("Please install pandas to use statistics package!")

import datetime as datetime
from astropy.time import Time
from ehtim.stats.statistics import *

def make_df(obs,polarization='unknown',band='unknown',round_s=0.1):

    """converts visibilities from obs.data to DataFrame format

    Args:
        obs: ObsData object
        round_s: accuracy of datetime object in seconds

    Returns:
        df: observation visibility data in DataFrame format
    """
    sour=obs.source
    df = pd.DataFrame(data=obs.data)
    df['fmjd'] = df['time']/24.
    df['mjd'] = obs.mjd + df['fmjd']
    telescopes = list(zip(df['t1'],df['t2']))
    telescopes = [(x[0].decode('unicode_escape'),x[1].decode('unicode_escape')) for x in telescopes]
    df['baseline'] = [x[0]+'-'+x[1] for x in telescopes]
    df['amp'] = list(map(np.abs,df['vis']))
    df['phase'] = list(map(lambda x: (180./np.pi)*np.angle(x),df['vis']))
    df['datetime'] = Time(df['mjd'], format='mjd').datetime
    df['datetime'] =list(map(lambda x: round_time(x,round_s=round_s),df['datetime']))
    df['jd'] = Time(df['mjd'], format='mjd').jd
    df['polarization'] = polarization
    df['band'] = band
    df['snr'] = df['amp']/df['sigma']
    df['source'] = sour
    df['baselength'] = np.sqrt(np.asarray(df.u)**2+np.asarray(df.v)**2)
    return df

def make_cphase_df(obs,band='unknown',polarization='unknown',mode='all',count='max',round_s=0.1):

    """generate DataFrame of closure phases

    Args: 
        obs: ObsData object
        round_s: accuracy of datetime object in seconds

    Returns:
        df: closure phase data in DataFrame format
    """

    data=obs.c_phases(mode=mode,count=count)
    sour=obs.source
    df = pd.DataFrame(data=data).copy()
    df['fmjd'] = df['time']/24.
    df['mjd'] = obs.mjd + df['fmjd']
    df['triangle'] = list(map(lambda x: x[0].decode('unicode_escape')+'-'+x[1].decode('unicode_escape')+'-'+x[2].decode('unicode_escape'),zip(df['t1'],df['t2'],df['t3'])))
    df['datetime'] = Time(df['mjd'], format='mjd').datetime
    df['datetime'] =list(map(lambda x: round_time(x,round_s=round_s),df['datetime']))
    df['jd'] = Time(df['mjd'], format='mjd').jd
    df['polarization'] = polarization
    df['band'] = band
    df['source'] = sour
    return df

def make_camp_df(obs,ctype='logcamp',debias=False,band='unknown',polarization='unknown',mode='all',count='max',debias_type='ExactLog',round_s=0.1):

    """generate DataFrame of closure amplitudes

    Args: 
        obs: ObsData object
        round_s: accuracy of datetime object in seconds

    Returns:
        df: closure amplitude data in DataFrame format
    """

    data = obs.c_amplitudes(mode=mode,count=count,debias=debias,ctype=ctype,debias_type=debias_type)
    sour=obs.source
    df = pd.DataFrame(data=data).copy()
    df['fmjd'] = df['time']/24.
    df['mjd'] = obs.mjd + df['fmjd']
    df['quadrangle'] = list(map(lambda x: x[0].decode('unicode_escape')+'-'+x[1].decode('unicode_escape')+'-'+x[2].decode('unicode_escape')+'-'+x[3].decode('unicode_escape'),zip(df['t1'],df['t2'],df['t3'],df['t4'])))
    df['datetime'] = Time(df['mjd'], format='mjd').datetime
    df['datetime'] =list(map(lambda x: round_time(x,round_s=round_s),df['datetime']))
    df['jd'] = Time(df['mjd'], format='mjd').jd
    df['polarization'] = polarization
    df['band'] = band
    df['source'] = sour
    df['catype'] = ctype
    return df

def make_bsp_df(obs,band='unknown',polarization='unknown',mode='all',count='min',round_s=0.1):

    """generate DataFrame of bispectra

    Args: 
        obs: ObsData object
        round_s: accuracy of datetime object in seconds

    Returns:
        df: bispectra data in DataFrame format
    """

    data = obs.bispectra(mode=mode,count=count)
    sour=obs.source
    df = pd.DataFrame(data=data).copy()
    df['fmjd'] = df['time']/24.
    df['mjd'] = obs.mjd + df['fmjd']
    df['triangle'] = list(map(lambda x: x[0].decode('unicode_escape')+'-'+x[1].decode('unicode_escape')+'-'+x[2].decode('unicode_escape'),zip(df['t1'],df['t2'],df['t3'])))
    df['datetime'] = Time(df['mjd'], format='mjd').datetime
    df['datetime'] =list(map(lambda x: round_time(x,round_s=round_s),df['datetime']))
    df['jd'] = Time(df['mjd'], format='mjd').jd
    df['polarization'] = polarization
    df['band'] = band
    df['source'] = sour
    return df

def average_cphases(cdf,dt,return_type='rec',err_type='predicted',num_samples=1000):

    """averages DataFrame of cphases

    Args:
        cdf: data frame of closure phases
        dt: integration time in seconds
        return_type: 'rec' for numpy record array (as used by ehtim), 'df' for data frame
        err_type: 'predicted' for modeled error, 'measured' for bootstrap empirical variability estimator

    Returns:
        cdf2: averaged closure phases
    """

    cdf2 = cdf.copy()
    t0 = datetime.datetime(1960,1,1)
    cdf2['round_time'] = list(map(lambda x: np.round((x- t0).total_seconds()/float(dt)),cdf2.datetime))  
    grouping=['polarization','band','triangle','t1','t2','t3','round_time']
    #column just for counting the elements
    cdf2['number'] = 1
    aggregated = {'datetime': np.min, 'time': np.mean,
    'number': lambda x: len(x), 'u1':np.mean, 'u2': np.mean, 'u3':np.mean,'v1':np.mean, 'v2': np.mean, 'v3':np.mean}

    #AVERAGING-------------------------------    
    if err_type=='measured':
        cdf2['dummy'] = cdf2['cphase']
        aggregated['dummy'] = lambda x: bootstrap(x, circular_mean, num_samples=num_samples,wrapping_variable=True)
    elif err_type=='predicted':
        aggregated['cphase'] = circular_mean
        aggregated['sigmacp'] = lambda x: np.sqrt(np.sum(x**2)/len(x)**2)
    else:
        print("Error type can only be 'predicted' or 'measured'! Assuming 'predicted'.")
        aggregated['cphase'] = circular_mean
        aggregated['sigmacp'] = lambda x: np.sqrt(np.sum(x**2)/len(x)**2)

    #ACTUAL AVERAGING
    cdf2 = cdf2.groupby(grouping).agg(aggregated).reset_index()

    if err_type=='measured':
        cdf2['cphase'] = [x[0] for x in list(cdf2['dummy'])]
        cdf2['sigmacp'] = [0.5*(x[1][1]-x[1][0]) for x in list(cdf2['dummy'])]

    #round datetime
    cdf2['datetime'] =  list(map(lambda x: t0 + datetime.timedelta(seconds= int(dt*x)), cdf2['round_time']))
    
    #ANDREW TODO-- this can lead to big problems!!
    #drop values averaged from less than 3 datapoints
    #cdf2.drop(cdf2[cdf2.number < 3.].index, inplace=True)
    if return_type=='rec':
        return df_to_rec(cdf2,'cphase')
    elif return_type=='df':
        return cdf2


def average_bispectra(cdf,dt,return_type='rec',num_samples=int(1e3)):

    """averages DataFrame of bispectra

    Args:
        cdf: data frame of bispectra
        dt: integration time in seconds
        return_type: 'rec' for numpy record array (as used by ehtim), 'df' for data frame

    Returns:
        cdf2: averaged bispectra
    """

    cdf2 = cdf.copy()
    t0 = datetime.datetime(1960,1,1)
    cdf2['round_time'] = list(map(lambda x: np.round((x- t0).total_seconds()/float(dt)),cdf2.datetime))  
    grouping=['polarization','band','triangle','t1','t2','t3','round_time']
    #column just for counting the elements
    cdf2['number'] = 1
    aggregated = {'datetime': np.min, 'time': np.mean,
    'number': lambda x: len(x), 'u1':np.mean, 'u2': np.mean, 'u3':np.mean,'v1':np.mean, 'v2': np.mean, 'v3':np.mean}

    #AVERAGING-------------------------------    
    aggregated['bispec'] = np.mean
    aggregated['sigmab'] = lambda x: np.sqrt(np.sum(x**2)/len(x)**2)

    #ACTUAL AVERAGING
    cdf2 = cdf2.groupby(grouping).agg(aggregated).reset_index()

    #round datetime
    cdf2['datetime'] =  list(map(lambda x: t0 + datetime.timedelta(seconds= int(dt*x)), cdf2['round_time']))
    
    #ANDREW TODO -- this can lead to big problems!!
    #drop values averaged from less than 3 datapoints
    #cdf2.drop(cdf2[cdf2.number < 3.].index, inplace=True)
    if return_type=='rec':
        return df_to_rec(cdf2,'bispec')
    elif return_type=='df':
        return cdf2


def average_camp(cdf,dt,return_type='rec',err_type='predicted',num_samples=int(1e3)):

    """averages DataFrame of closure amplitudes

    Args:
        cdf: data frame of closure amplitudes
        dt: integration time in seconds
        return_type: 'rec' for numpy record array (as used by ehtim), 'df' for data frame
        err_type: 'predicted' for modeled error, 'measured' for bootstrap empirical variability estimator

    Returns:
        cdf2: averaged closure amplitudes
    """

    cdf2 = cdf.copy()
    t0 = datetime.datetime(1960,1,1)
    cdf2['round_time'] = list(map(lambda x: np.round((x- t0).total_seconds()/float(dt)),cdf2.datetime))  
    grouping=['polarization','band','quadrangle','t1','t2','t3','t4','round_time']
    #column just for counting the elements
    cdf2['number'] = 1
    aggregated = {'datetime': np.min, 'time': np.mean,
    'number': lambda x: len(x), 'u1':np.mean, 'u2': np.mean, 'u3':np.mean, 'u4': np.mean, 'v1':np.mean, 'v2': np.mean, 'v3':np.mean,'v4':np.mean}

    #AVERAGING-------------------------------    
    if err_type=='measured':
        cdf2['dummy'] = cdf2['camp']
        aggregated['dummy'] = lambda x: bootstrap(x, np.mean, num_samples=num_samples,wrapping_variable=False)
    elif err_type=='predicted':
        aggregated['camp'] = np.mean
        aggregated['sigmaca'] = lambda x: np.sqrt(np.sum(x**2)/len(x)**2)
    else:
        print("Error type can only be 'predicted' or 'measured'! Assuming 'predicted'.")
        aggregated['camp'] = np.mean
        aggregated['sigmaca'] = lambda x: np.sqrt(np.sum(x**2)/len(x)**2)

    #ACTUAL AVERAGING
    cdf2 = cdf2.groupby(grouping).agg(aggregated).reset_index()

    if err_type=='measured':
        cdf2['camp'] = [x[0] for x in list(cdf2['dummy'])]
        cdf2['sigmaca'] = [0.5*(x[1][1]-x[1][0]) for x in list(cdf2['dummy'])]

    #round datetime
    cdf2['datetime'] =  list(map(lambda x: t0 + datetime.timedelta(seconds= int(dt*x)), cdf2['round_time']))
    
    #ANDREW TODO -- this can lead to big problems!!
    #drop values averaged from less than 3 datapoints
    #cdf2.drop(cdf2[cdf2.number < 3.].index, inplace=True)
    if return_type=='rec':
        return df_to_rec(cdf2,'camp')
    elif return_type=='df':
        return cdf2

def df_to_rec(df,product_type):

    """converts DataFrame to numpy recarray used by ehtim

    Args:
        df: DataFrame to convert
        product_type: vis, cphase, camp
    """
    if product_type=='cphase':
         out= df[['time','t1','t2','t3','u1','v1','u2','v2','u3','v3','cphase','sigmacp']].to_records(index=False)
         return np.array(out,dtype=DTCPHASE)
    elif product_type=='camp':
         out=  df[['time','t1','t2','t3','t4','u1','v1','u2','v2','u3','v3','u4','v4','camp','sigmaca']].to_records(index=False)
         return np.array(out,dtype=DTCAMP)
    elif product_type=='vis':
         out=  df[['time','tint','t1','t2','tau1','tau2','u','v','vis','qvis','uvis','vvis','sigma','qsigma','usigma','vsigma']].to_records(index=False)
         return np.array(out,dtype=DTPOL)
    elif product_type=='bispec':
         out=  df[['time','t1','t2','t3','u1','v1','u2','v2','u3','v3','bispec','sigmab']].to_records(index=False)
         return np.array(out,dtype=DTBIS)

def round_time(t,round_s=0.1):

    """rounding time to given accuracy

    Args:
        t: time
        round_s: delta time to round to in seconds

    Returns:
        round_t: rounded time
    """
    t0 = datetime.datetime(t.year,1,1)
    foo = t - t0
    foo_s = foo.days*24*3600 + foo.seconds + foo.microseconds*(1e-6)
    foo_s = np.round(foo_s/round_s)*round_s
    days = np.floor(foo_s/24/3600)
    seconds = np.floor(foo_s - 24*3600*days)
    microseconds = int(1e6*(foo_s - days*3600*24 - seconds))
    round_t = t0+datetime.timedelta(days,seconds,microseconds)
    return round_t
