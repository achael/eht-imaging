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
from ehtim.statistics.stats import *

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
    telescopes = [(x[0],x[1]) for x in telescopes]
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


def make_amp(obs,debias=True,polarization='unknown',band='unknown',round_s=0.1):

    """converts visibilities from obs.data to amplitudes inDataFrame format

    Args:
        obs: ObsData object
        debias (str): whether to debias the amplitudes
        round_s: accuracy of datetime object in seconds

    Returns:
        df: observation visibility data in DataFrame format
    """
    sour=obs.source
    df = pd.DataFrame(data=obs.data)
    df['fmjd'] = df['time']/24.
    df['mjd'] = obs.mjd + df['fmjd']
    telescopes = list(zip(df['t1'],df['t2']))
    telescopes = [(x[0],x[1]) for x in telescopes]
    df['baseline'] = [x[0]+'-'+x[1] for x in telescopes]
    df['amp'] = list(map(np.abs,df['vis']))
    if debias==True:
        amp2 = np.maximum(np.asarray(df['amp'])**2-np.asarray(df['sigma'])**2,np.asarray(df['sigma'])**2)
        df['amp'] = np.sqrt(amp2)
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

def coh_avg_vis(obs,dt=0,scan_avg=False,return_type='rec',err_type='predicted',num_samples=int(1e3)):
    """coherently averages visibilities
    Args:
        obs: ObsData object
        dt (float): integration time in seconds
        return_type (str): 'rec' for numpy record array (as used by ehtim), 'df' for data frame
        err_type (str): 'predicted' for modeled error, 'measured' for bootstrap empirical variability estimator
        num_samples: 'bootstrap' resample set size for measured error
        scan_avg (bool): should scan-long averaging be performed. If True, overrides dt
    Returns:
        vis_avg: coherently averaged visibilities
    """
    if (dt<=0)&(scan_avg==False):
        return obs.data
    else:
        vis = make_df(obs)
        if scan_avg==False:
            #TODO
            #we don't have to work on datetime products at all
            #change it to only use 'time' in mjd
            t0 = datetime.datetime(1960,1,1) 
            vis['round_time'] = list(map(lambda x: np.floor((x- t0).total_seconds()/float(dt)),vis.datetime))  
            grouping=['tau1','tau2','polarization','band','baseline','t1','t2','round_time']
        else:
            bins, labs = get_bins_labels(obs.scans)
            vis['scan'] = list(pd.cut(vis.time, bins,labels=labs))
            grouping=['tau1','tau2','polarization','band','baseline','t1','t2','scan']
        #column just for counting the elements
        vis['number'] = 1
        aggregated = {'datetime': np.min, 'time': np.min,
        'number': lambda x: len(x), 'u':np.mean, 'v':np.mean,'tint': np.sum}

        if err_type not in ['measured', 'predicted']:
            print("Error type can only be 'predicted' or 'measured'! Assuming 'predicted'.")
            err_type='predicted'

        #AVERAGING-------------------------------    
        if err_type=='measured':
            vis['dummy'] = vis['vis']
            vis['udummy'] = vis['uvis']
            vis['vdummy'] = vis['vvis']
            vis['qdummy'] = vis['qvis']
            aggregated['vis'] = np.mean
            aggregated['uvis'] = np.mean
            aggregated['qvis'] = np.mean
            aggregated['vvis'] = np.mean
            aggregated['dummy'] = lambda x: bootstrap(np.abs(x), np.mean, num_samples=num_samples,wrapping_variable=False)
            aggregated['udummy'] = lambda x: bootstrap(np.abs(x), np.mean, num_samples=num_samples,wrapping_variable=False)
            aggregated['vdummy'] = lambda x: bootstrap(np.abs(x), np.mean, num_samples=num_samples,wrapping_variable=False)
            aggregated['qdummy'] = lambda x: bootstrap(np.abs(x), np.mean, num_samples=num_samples,wrapping_variable=False)
   
        elif err_type=='predicted':
            aggregated['vis'] = np.mean
            aggregated['uvis'] = np.mean
            aggregated['qvis'] = np.mean
            aggregated['vvis'] = np.mean
            aggregated['sigma'] = lambda x: np.sqrt(np.sum(x**2)/len(x)**2)
            aggregated['usigma'] = lambda x: np.sqrt(np.sum(x**2)/len(x)**2)
            aggregated['qsigma'] = lambda x: np.sqrt(np.sum(x**2)/len(x)**2)
            aggregated['vsigma'] = lambda x: np.sqrt(np.sum(x**2)/len(x)**2)

        #ACTUAL AVERAGING
        vis_avg = vis.groupby(grouping).agg(aggregated).reset_index()
        
        if err_type=='measured':
            vis_avg['sigma'] = [0.5*(x[1][1]-x[1][0]) for x in list(vis_avg['dummy'])]
            vis_avg['usigma'] = [0.5*(x[1][1]-x[1][0]) for x in list(vis_avg['udummy'])]
            vis_avg['qsigma'] = [0.5*(x[1][1]-x[1][0]) for x in list(vis_avg['qdummy'])]
            vis_avg['vsigma'] = [0.5*(x[1][1]-x[1][0]) for x in list(vis_avg['vdummy'])]

        vis_avg['amp'] = list(map(np.abs,vis_avg['vis']))
        vis_avg['phase'] = list(map(lambda x: (180./np.pi)*np.angle(x),vis_avg['vis']))
        vis_avg['snr'] = vis_avg['amp']/vis_avg['sigma']
        if scan_avg==False:
            #round datetime and time to the begining of the bucket and add half of a bucket time
            half_bucket = dt/2.
            vis_avg['datetime'] =  list(map(lambda x: t0 + datetime.timedelta(seconds= int(dt*x) + half_bucket), vis_avg['round_time']))
            vis_avg['time']  = list(map(lambda x: (Time(x).mjd-obs.mjd)*24., vis_avg['datetime']))
        else:
            #drop values that couldn't be matched to any scan
            vis_avg.drop(list(vis_avg[vis_avg.scan<0].index.values),inplace=True)
        if err_type=='measured':
            vis_avg.drop(labels=['udummy','vdummy','qdummy','dummy'],axis='columns',inplace=True)      
        if return_type=='rec':
            return df_to_rec(vis_avg,'vis')
        elif return_type=='df':
            return vis_avg

def incoh_avg_vis(obs,dt=0,debias=True,scan_avg=False,return_type='rec',rec_type='vis',err_type='predicted',num_samples=int(1e3)):
    """incoherently averages visibilities
    Args:
        obs: ObsData object
        dt (float): integration time in seconds
        return_type (str): 'rec' for numpy record array (as used by ehtim), 'df' for data frame
        rec_type (str): 'vis' for DTPOL and 'amp' for DTAMP
        err_type (str): 'predicted' for modeled error, 'measured' for bootstrap empirical variability estimator
        num_samples: 'bootstrap' resample set size for measured error
        scan_avg (bool): should scan-long averaging be performed. If True, overrides dt
    Returns:
        vis_avg: coherently averaged visibilities
    """
    if (dt<=0)&(scan_avg==False):
        print('Either averaging time must be positive, or scan_avg option should be selected!')
        return obs.data
    else:
        vis = make_df(obs)
        if scan_avg==False:
            #TODO
            #we don't have to work on datetime products at all
            #change it to only use 'time' in mjd
            t0 = datetime.datetime(1960,1,1) 
            vis['round_time'] = list(map(lambda x: np.floor((x- t0).total_seconds()/float(dt)),vis.datetime))  
            grouping=['tau1','tau2','polarization','band','baseline','t1','t2','round_time']
        else:
            bins, labs = get_bins_labels(obs.scans)
            vis['scan'] = list(pd.cut(vis.time, bins,labels=labs))
            grouping=['tau1','tau2','polarization','band','baseline','t1','t2','scan']
        #column just for counting the elements
        vis['number'] = 1
        aggregated = {'datetime': np.min, 'time': np.min,
        'number': lambda x: len(x), 'u':np.mean, 'v':np.mean,'tint': np.sum}

        if err_type not in ['measured', 'predicted']:
            print("Error type can only be 'predicted' or 'measured'! Assuming 'predicted'.")
            err_type='predicted'

        #AVERAGING------------------------------- 
        vis['dummy'] = list(zip(np.abs(vis['vis']),vis['sigma']))
        vis['udummy'] = list(zip(np.abs(vis['uvis']),vis['usigma']))
        vis['vdummy'] = list(zip(np.abs(vis['vvis']),vis['vsigma']))
        vis['qdummy'] = list(zip(np.abs(vis['qvis']),vis['qsigma']))

        if err_type=='predicted':
            aggregated['dummy'] = lambda x: mean_incoh_avg(x,debias=debias)
            aggregated['udummy'] = lambda x: mean_incoh_avg(x,debias=debias)
            aggregated['vdummy'] = lambda x: mean_incoh_avg(x,debias=debias)
            aggregated['qdummy'] = lambda x: mean_incoh_avg(x,debias=debias)

        elif err_type=='measured':
            aggregated['dummy'] = lambda x: bootstrap(np.abs(np.asarray([y[0] for y in x])), np.mean, num_samples=num_samples,wrapping_variable=False)
            aggregated['udummy'] = lambda x: bootstrap(np.abs(np.asarray([y[0] for y in x])), np.mean, num_samples=num_samples,wrapping_variable=False)
            aggregated['vdummy'] = lambda x: bootstrap(np.abs(np.asarray([y[0] for y in x])), np.mean, num_samples=num_samples,wrapping_variable=False)
            aggregated['qdummy'] = lambda x: bootstrap(np.abs(np.asarray([y[0] for y in x])), np.mean, num_samples=num_samples,wrapping_variable=False)

        #ACTUAL AVERAGING
        vis_avg = vis.groupby(grouping).agg(aggregated).reset_index()

        if err_type=='predicted':
            vis_avg['vis'] = [x[0] for x in list(vis_avg['dummy'])]
            vis_avg['uvis'] = [x[0] for x in list(vis_avg['udummy'])]
            vis_avg['qvis'] = [x[0] for x in list(vis_avg['qdummy'])]
            vis_avg['vvis'] = [x[0] for x in list(vis_avg['vdummy'])]
            vis_avg['sigma'] = [x[1] for x in list(vis_avg['dummy'])]
            vis_avg['usigma'] = [x[1] for x in list(vis_avg['udummy'])]
            vis_avg['qsigma'] = [x[1] for x in list(vis_avg['qdummy'])]
            vis_avg['vsigma'] = [x[1] for x in list(vis_avg['vdummy'])]
        
        elif err_type=='measured':
            vis_avg['vis'] = [x[0] for x in list(vis_avg['dummy'])]
            vis_avg['uvis'] = [x[0] for x in list(vis_avg['udummy'])]
            vis_avg['qvis'] = [x[0] for x in list(vis_avg['qdummy'])]
            vis_avg['vvis'] = [x[0] for x in list(vis_avg['vdummy'])]
            vis_avg['sigma'] = [0.5*(x[1][1]-x[1][0]) for x in list(vis_avg['dummy'])]
            vis_avg['usigma'] = [0.5*(x[1][1]-x[1][0]) for x in list(vis_avg['udummy'])]
            vis_avg['qsigma'] = [0.5*(x[1][1]-x[1][0]) for x in list(vis_avg['qdummy'])]
            vis_avg['vsigma'] = [0.5*(x[1][1]-x[1][0]) for x in list(vis_avg['vdummy'])]

        vis_avg['amp'] = list(map(np.abs,vis_avg['vis']))
        vis_avg['phase'] = 0
        vis_avg['snr'] = vis_avg['amp']/vis_avg['sigma']
        if scan_avg==False:
            #round datetime and time to the begining of the bucket and add half of a bucket time
            half_bucket = dt/2.
            vis_avg['datetime'] =  list(map(lambda x: t0 + datetime.timedelta(seconds= int(dt*x) + half_bucket), vis_avg['round_time']))
            vis_avg['time']  = list(map(lambda x: (Time(x).mjd-obs.mjd)*24., vis_avg['datetime']))
        else:
            #drop values that couldn't be matched to any scan
            vis_avg.drop(list(vis_avg[vis_avg.scan<0].index.values),inplace=True)
            
        vis_avg.drop(labels=['udummy','vdummy','qdummy','dummy'],axis='columns',inplace=True)    
        if return_type=='rec':
            return df_to_rec(vis_avg,rec_type)
        elif return_type=='df':
            return vis_avg


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
    df['triangle'] = list(map(lambda x: x[0]+'-'+x[1]+'-'+x[2],zip(df['t1'],df['t2'],df['t3'])))
    df['datetime'] = Time(df['mjd'], format='mjd').datetime
    df['datetime'] =list(map(lambda x: round_time(x,round_s=round_s),df['datetime']))
    df['jd'] = Time(df['mjd'], format='mjd').jd
    df['polarization'] = polarization
    df['band'] = band
    df['source'] = sour
    return df

def make_camp_df(obs,ctype='logcamp',debias=False,band='unknown',polarization='unknown',mode='all',count='max',round_s=0.1):

    """generate DataFrame of closure amplitudes

    Args: 
        obs: ObsData object
        round_s: accuracy of datetime object in seconds

    Returns:
        df: closure amplitude data in DataFrame format
    """

    data = obs.c_amplitudes(mode=mode,count=count,debias=debias,ctype=ctype)
    sour=obs.source
    df = pd.DataFrame(data=data).copy()
    df['fmjd'] = df['time']/24.
    df['mjd'] = obs.mjd + df['fmjd']
    df['quadrangle'] = list(map(lambda x: x[0]+'-'+x[1]+'-'+x[2]+'-'+x[3],zip(df['t1'],df['t2'],df['t3'],df['t4'])))
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
    df['triangle'] = list(map(lambda x: x[0]+'-'+x[1]+'-'+x[2],zip(df['t1'],df['t2'],df['t3'])))
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
    elif product_type=='amp':
         out=  df[['time','tint','t1','t2','u','v','vis','amp','sigma']].to_records(index=False)
         return np.array(out,dtype=DTAMP)
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

def get_bins_labels(intervals,dt=0.00001):
    '''gets bins and labels necessary to perform averaging by scan
    Args:
        intervals:
        dt (float): time margin to add to the scan limits
    ''' 

    def fix_midnight_overlap(x):
        if x[1] < x[0]:
            x[1]+= 24.
        return x

    def is_overlapping(interval0,interval1):
        if ((interval1[0]<=interval0[0])&(interval1[1]>=interval0[0]))|((interval1[0]<=interval0[1])&(interval1[1]>=interval0[1])):
            return True
        else: return False
    
    def merge_overlapping_intervals(intervals):
        return (np.min([x[0] for x in intervals]),np.max([x[1] for x in intervals]))

    def replace_overlapping_intervals(intervals,element_ind):
        indic_not_overlap=[not is_overlapping(x,intervals[element_ind]) for x in intervals]
        indic_overlap=[is_overlapping(x,intervals[element_ind]) for x in intervals]
        fooarr=np.asarray(intervals)
        return sorted([tuple(x) for x in fooarr[indic_not_overlap]]+[merge_overlapping_intervals(list(fooarr[indic_overlap]))])
    
    intervals = sorted(list(set(zip(intervals[:,0],intervals[:,1]))))
    intervals = [fix_midnight_overlap(x) for x in intervals]
    cou=0
    while cou < len(intervals): 
        intervals = replace_overlapping_intervals(intervals,cou)
        cou+=1
        
    binsT=[None]*(2*np.shape(intervals)[0])
    binsT[::2] = [x[0]-dt for x in intervals]
    binsT[1::2] = [x[1]+dt for x in intervals]
    labels=[None]*(2*np.shape(intervals)[0]-1)
    labels[::2] = [cou for cou in range(1,len(intervals)+1)]
    labels[1::2] = [-cou for cou in range(1,len(intervals))]
    
    return binsT, labels
