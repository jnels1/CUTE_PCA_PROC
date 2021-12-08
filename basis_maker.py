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
import trace_shaper

def make_psd( series, run, det ):

    # kind of a special case of pull traces

    psd_traces = pull_traces(series, run, det, 1000, trigger='Random')
    # remove the OFdelay
    psd_traces['OFdelay'] = np.zeros( len(psd_traces['OFdelay']) )

    return

def pull_traces( series, run, det, Ntrace, trigger='Random', amp_range=[-np.inf,np.inf] ):
    
    '''
    Docstring goes here
    '''

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


    # If looking at triggered data, do simple pileup check
    if trigger=='Det':
        trace_list=list()
        for i, trace in enumerate(traces_oi):
            bs = np.mean(trace[:15000])
            # might need to tweek height value
            peak_index, peak_prop = signal.find_peaks( trace-bs, height=100*1e-9, distance=200 )
            if len(peak_index)==1:
                trace_list.append(trace)
        traces = np.vstack(trace_list)

    # If randoms, perform autocuts
    else:
        acut = autocuts(np.vstack(traces_oi))
        traces = np.vstack(traces_oi)[acut]

#    for i, trace in enumerate(traces_oi):
        # 1st check for pileup
#        bs = np.mean(trace[:15000])
        # might need to tweek height value
#        peak_index, peak_prop = signal.find_peaks( trace-bs, height=120*1e-9, distance=200 )
#        if trigger=='Det':
#            if len(peak_index)==1:
#                trace_list.append(trace)
#        elif trigger=='Random':
#            if len(peak_index)==0:
#                trace_list.append(trace)

#        trace_list.append(trace)
#        delay_list.append(delay_oi.iloc[i])

    save_dict={
        'series':series,
        'run':run,
        'det':det,
        'Ntrace0':Ntrace,
        'Ntrace':len(traces),
        'trigger':trigger,
        'amp_range':amp_range,
        'EventNum':np.hstack(events_oi),
        'OFdelay':np.hstack(delay_oi),
        'traces':traces
    }
    return save_dict



def eval_basis( shaped_traces, neig ):
    traces = shaped_traces['traces']
    m = len(traces[0])

    C = np.cov(traces, rowvar=False)
    w, v = eigh(C, eigvals=(m-neig, m-1))

    ret_dict={
        'Vec':np.fliplr(v),
        'Val':np.flip(w)
    }
    return ret_dict

def make_basis(config_dict):

    
    #det='PD2'
    #series = '23201125_214136'
    #run = 18

    #det = sys.argv[1]
    #series = sys.argv[2]
    #run = int(sys.argv[3])

    series = config_dict['series']
    run = config_dict['run']
    det = config_dict['det']
    Ntrain = config_dict['train#']
    trig_cond = config_dict['train_trigger']
    arange = config_dict['train_range']
    filter_bool = config_dict['shape_filter']
    shape_window = config_dict['shape_window']
    shape_ds = config_dict['shape_ds']
    Nbase = config_dict['N_basis']

    basis_traces = pull_traces( series, run, det, Ntrain, trigger=trig_cond, amp_range=arange )
    if trig_cond=='Random':
        psd_traces = basis_traces
    else:
        psd_traces = pull_traces( series, run, det, Ntrain, trigger='Random' )
    psd_traces['OFdelay'] = np.zeros( len(psd_traces['OFdelay']) )

    basis_shaped = trace_shaper.trace_shaper( basis_traces, do_filter=filter_bool, downsample=shape_ds, time_window=shape_window )
    psd_shaped = trace_shaper.trace_shaper( psd_traces, do_filter=filter_bool, downsample=shape_ds, time_window=shape_window )
   
    temp_psd=list()
    for trace in psd_shaped['traces']:
        frq, pxx = signal.periodogram(trace, fs=psd_shaped['fs'], return_onesided=True)
        pxx[0] = np.inf
        temp_psd.append(pxx)
    psd = np.mean( np.vstack(temp_psd) ,axis=0)
 
    mean_basis_trace = np.mean(np.vstack(basis_shaped['traces']), axis=0)

    derived_basis = eval_basis(basis_shaped, Nbase)

    ret_dict = dict()
    ret_dict['N_basis']=Nbase
    ret_dict['Basis']=derived_basis
    ret_dict['PSD']=psd
    ret_dict['BasisMean']=mean_basis_trace

    return ret_dict
