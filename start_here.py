# Import required packages
import os
import sys
import cdms
import numpy as np
import matplotlib.pyplot as plt
from rawio.IO import *
#from scdmsPyTools.BatTools.IO import *
import pandas as pd
import qetpy
from qetpy import Noise
from qetpy.utils import lowpassfilter
import uproot
from qetpy import autocuts
from scipy import interpolate
import scipy.optimize as spo
from scipy.linalg import eigh
from scipy import ndimage
from scipy import fft
from scipy import signal
import pickle
import time
import inspect
sys.path.append("/gpfs/slac/staas/fs1/g/supercdms/users/jnels1/my_functions")
import cute_utils_v2 as cute_utils

def gf(trace):
    return ndimage.gaussian_filter1d(trace, sigma=2.5, order=0, mode='wrap')

def pull_traces( series, run, det, Ntrace, trigger='Random', amp_range=[-np.inf,np.inf] ):
    
    '''
    Docstring goes here
    '''

    trace_list=list()
    delay_list=list()

    rq_data = cute_utils.fetch_data( series, run )

    det_list = list(rq_data['Events']['Trigger'])
    det_list.remove('Random')   
 
    if trigger not in ['Random', 'Det']:
        print( 'trigger must be one of: ', ['Random', 'Det'] )
        return np.nan
    
    elif det not in det_list:
        print( 'det must be one of: ', det_list)
        return np.nan
    
    # Assume in run randoms for now
    if trigger=='Random':
        trig = rq_data['Events']['Trigger']['Random']['InRun']
    else:
        trig = rq_data['Events']['Trigger'][det]

    in_range = (rq_data['Det'][det]['OFamp']*1e6>amp_range[0])&(rq_data['Det'][det]['OFamp']*1e6<amp_range[1])
    
    events_oi = rq_data['Events']['EventNum'][trig&in_range]
    delay_oi = rq_data['Det'][det]['OFdelay'][trig&in_range]

    traces_oi = cute_utils.fetch_traces( series, run, event_numbers=events_oi, max_N=Ntrace, sub_base=False  )[det]

    for i, trace in enumerate(traces_oi):
        # 1st check for pileup
        bs = np.mean(trace[:15000])
        # might need to tweek height value
        peak_index, peak_prop = signal.find_peaks( trace, height=120*1e-9, distance=200 )
        if trigger=='Det':
            if len(peak_index)==1:
                trace_list.append(trace)
        elif trigger=='Random':
            if len(peak_index)==0:
                trace_list.append(trace)

        trace_list.append(trace)
        delay_list.append(delay_oi.iloc[i])

    save_dict={
        'series':series,
        'run':run,
        'det':det,
        'Ntrace0':Ntrace,
        'Ntrace':len(trace_list),
        'trigger':trigger,
        'amp_range':amp_range,
        'EventNum':np.hstack(events_oi),
        'OFdelay':np.hstack(delay_oi),
        'traces':np.vstack(trace_list)
    }
    return save_dict

def trace_shaper( trace_dict, do_filter=False, downsample=1, time_window=[0, 2**15] ):
    fs = 625000

    traces0 = trace_dict['traces']
    
    # subtract baseline
    bs_array = np.mean(traces0[:,:15000], axis=1)
    traces1 = traces0 - bs_array[:,np.newaxis]

    # apply time shift (and filter if needed)
    traces2_temp=list()
    for i, trace in enumerate(traces1):
        if do_filter: 
            traces2_temp.append( np.roll(gf(trace), -int(trace_dict['OFdelay'][i]*fs)) )
        else:
            traces2_temp.append( np.roll(trace, -int(trace_dict['OFdelay'][i]*fs)) )
    traces2 = np.vstack(traces2_temp)

    # Do downsampling
    if downsample==1:
        traces3 = traces2
    else:
        traces3 = bulk_ds(traces2, downsample)

    # Trim traces outside the time window
    win_start = int(time_window[0]/downsample)
    win_stop = int(time_window[1]/downsample)
    traces4 = traces3[:,win_start:win_stop]

    return traces4;

def bulk_ds( trace, Nd ):

    ds_grid = Nd*np.arange((len(trace[0])/Nd+1), dtype=int)
    avg_array = list()
    for i in range(len(ds_grid)-1):
        avg_array.append( np.mean(trace[:, ds_grid[i]:ds_grid[i+1] ], axis=1) )
    res_array = np.transpose(np.vstack( avg_array ))

    return res_array;

def make_basis( shaped_traces, neig ):
    m = len(shaped_traces[0])

    C = np.cov(shaped_traces, rowvar=False)
    w, v = eigh(C, eigvals=(m-neig, m-1))

    ret_dict={
        'Vec':np.fliplr(v),
        'Val':np.flip(w)
    }
    return ret_dict

def main():


    det='PD2'
    series = '23201125_214136'
    run = 18

    test_data = pull_traces( series, run, det, 100, trigger='Det' )
    test_shape = trace_shaper( test_data, do_filter=True, time_window=[15500,15500+4096] )
    test_basis = make_basis(test_shape, 10)

    

    return

if __name__=="__main__":
    main()
