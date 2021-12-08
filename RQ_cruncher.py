import os
import inspect
import sys
import cdms
import numpy as np
import matplotlib.pyplot as plt
from rawio.IO import *
#from scdmsPyTools.BatTools.IO import *
from scdmsPyTools.Traces.Stats import slope
from pandas import DataFrame,MultiIndex,Series
import qetpy
from qetpy import Noise
from qetpy.utils import lowpassfilter
import uproot
from qetpy import autocuts
from scipy import interpolate
import scipy.optimize as spo
from scipy.optimize import curve_fit
from scipy import signal
from scipy import ndimage
from scipy.linalg import eigh, inv
from scipy import fft
import pickle
import time
sys.path.append("/gpfs/slac/staas/fs1/g/supercdms/users/jnels1/my_functions")
import cute_utils_v2 as cute_utils
import trace_shaper

def gf(trace, fs):
    k = 2.5/625000
    return ndimage.gaussian_filter1d(trace, sigma=k*fs, order=0, mode='wrap')

def get_rise_index(trace, fs, thresh):

    # 1st filter and normalize trace
    filt_trace = gf(trace, fs)
    norm_trace = (1./np.amax(filt_trace))*filt_trace

    max_ind=np.argmax(trace)
    pre_trace=norm_trace[:max_ind]
    try:
        rise_index=np.argmax(-np.abs(pre_trace-thresh))
    except:
        rise_index=np.nan
    return rise_index;

def get_fall_index(trace, fs, thresh):
    # 1st filter and normalize trace
    filt_trace = gf(trace, fs)
    norm_trace = (1./np.amax(filt_trace))*filt_trace
    
    max_ind=np.argmax(trace)
    post_trace=norm_trace[max_ind:]
    try:
        fall_index=np.argmax(-np.abs(post_trace-thresh))+max_ind
    except:
        fall_index=np.nan
    return fall_index;

def calc_chi2( trace, temp, my_psd ):

    dt = 52*1e-6 # seconds
    N = len(trace)
    norm = dt/N
    #template0=amp0*np.roll(template['PD2'][1:-1], int(fs*dt0))

    compound_ft = fft.rfft( trace-temp )
    comp_chi_array = (norm/8.)*(compound_ft*np.conj(compound_ft))/(my_psd)

    return np.sum(comp_chi_array, axis=-1)/N;

def pca_fit( trace, pcs, Npc ):
    #print(pcs)
    if Npc>np.shape(pcs)[1]:
        return
    elif Npc==1:
        proj_mat = pcs[:,-1][:,np.newaxis] @ np.transpose(pcs[:,-1])[np.newaxis,:]
    else:
        proj_mat = pcs[:,-Npc:] @ np.transpose(pcs[:,-Npc:])

    return trace @ proj_mat

def get_res(trace, fit, order):
    return np.sum(np.power(np.power(trace-fit,2),order/2), axis=-1)


def pull_data(series, run, det, event_numbers):

    trace_list=list()
    delay_list=list()

    rq_data = cute_utils.fetch_data( series, run )

    det_list = list(rq_data['Events']['Trigger'])
    det_list.remove('Random')

    if det not in det_list:
        print( 'det must be one of: ', det_list)
        return np.nan

    good_events = np.in1d( rq_data['Events']['EventNum'], event_numbers )
    delay_oi = rq_data['Det'][det]['OFdelay'][good_events]

    traces_oi = cute_utils.fetch_traces( series, run, event_numbers=event_numbers, max_N=len(event_numbers), sub_base=False  )[det]

    ret_dict={
        'series':series,
        'run':run,
        'det':det,
        'EventNum':np.hstack(event_numbers),
        'OFdelay':np.hstack(delay_oi),
        'traces':np.vstack(traces_oi)
    }
    
    ret_dict['EventTime_ref']=rq_data['Events']['EventTime'][good_events]
    ret_dict['EventTime_trig']=rq_data['Events']['TriggerTime'][good_events]
    ret_dict['OFamp']=np.array(rq_data['Det'][det]['OFamp'][good_events])
    ret_dict['OFdelay']=np.array(rq_data['Det'][det]['OFdelay'][good_events])
    ret_dict['chi2LF']=np.array(rq_data['Det'][det]['chi2LF'][good_events])

    return ret_dict

def proc_block(data_block, basis):

    ret_dict=dict()

    # store all the information we already have (except raw traces)
    for key in data_block:
        if key=='traces':
            continue
        ret_dict[key] = data_block[key]
    for key in basis:
        ret_dict[key] = basis[key]

    v = basis['Basis']['Vec']
    traces = data_block['traces']

    # calculate rise/fall times
    ret_dict['GFFT2']=np.zeros(len(traces))
    ret_dict['GFFT5']=np.zeros(len(traces))
    ret_dict['GFFT8']=np.zeros(len(traces))
    ret_dict['GFRT2']=np.zeros(len(traces))
    ret_dict['GFRT5']=np.zeros(len(traces))
    ret_dict['GFRT8']=np.zeros(len(traces))

    for i, trace in enumerate(traces):
        ret_dict['GFFT2'][i]=(get_fall_index(trace,ret_dict['fs'], 0.2))
        ret_dict['GFFT5'][i]=(get_fall_index(trace,ret_dict['fs'], 0.5))
        ret_dict['GFFT8'][i]=(get_fall_index(trace,ret_dict['fs'], 0.8))
        ret_dict['GFRT2'][i]=(get_rise_index(trace,ret_dict['fs'], 0.2))
        ret_dict['GFRT5'][i]=(get_rise_index(trace,ret_dict['fs'], 0.5))
        ret_dict['GFRT8'][i]=(get_rise_index(trace,ret_dict['fs'], 0.8))

    # calculate PCA values
    projected_traces = pca_fit(traces, v, basis['N_basis'])
    ret_dict['chi2'] = calc_chi2( traces, projected_traces, basis['PSD'] )
    ret_dict['res2'] = get_res( traces, projected_traces, 2 )
    ret_dict['pc_amp'] = np.amax( projected_traces, axis=1 )
    ret_dict['pc_comp'] = traces @ v
    ret_dict['INT'] = np.sum( traces, axis=1 )/ret_dict['fs'] 

    return ret_dict

def series_looper( config_dict, basis ):

    # break series into 1000 event blocks
    # pull/shape traces

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


    rq_data = cute_utils.fetch_data(series, run)
    det_trig = rq_data['Events']['Trigger'][det]

    temp_out=dict()
    
    N_split = int(np.count_nonzero(det_trig)/1000)+1
    block_events = np.array_split(rq_data['Events']['EventNum'][det_trig], N_split)
    for block in block_events:
        block_data = pull_data(series, run, det, block)
        block_shaped = trace_shaper.trace_shaper( block_data, do_filter=filter_bool, downsample=shape_ds, time_window=shape_window )
        block_res = proc_block(block_shaped, basis)
        for key in block_res:
            if key not in temp_out:
                temp_out[key] = list()
            temp_out[key].append(block_res[key])
    
    ret_dict=dict()
    for key in temp_out:
        #print(temp_out[key])
        #print(key, np.shape(temp_out[key]))
        try:
            ret_dict[key] = np.concatenate(temp_out[key], axis=-1)
        except:
            ret_dict[key]=temp_out[key]
    return ret_dict;

if __name__=="__main__":
    main()
