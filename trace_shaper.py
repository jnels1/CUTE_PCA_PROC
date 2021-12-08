import numpy as np
import pandas as pd
from scipy import ndimage

def gf(trace, fs):
    k = 2.5/625000
    return ndimage.gaussian_filter1d(trace, sigma=fs*k, order=0, mode='wrap')

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
            traces2_temp.append( np.roll(gf(trace, fs), -int(trace_dict['OFdelay'][i]*fs)) )
        else:
            traces2_temp.append( np.roll(trace, -int(trace_dict['OFdelay'][i]*fs)) )
    traces2 = np.vstack(traces2_temp)

    # Do downsampling
    if downsample==1:
        traces3 = traces2
    else:
        traces3 = bulk_ds(traces2, downsample)
    fs = fs/downsample

    # Trim traces outside the time window
    win_start = int(time_window[0]/downsample)
    win_stop = int(time_window[1]/downsample)
    traces4 = traces3[:,win_start:win_stop]

    ret_dict = {
        'fs':fs,
        'bs':bs_array,
        'OFdelay':trace_dict['OFdelay'],
        'traces':np.vstack(traces4)
    }
    for key in trace_dict:
        if key=='traces':
            continue
        else:
            ret_dict[key] = trace_dict[key]

    return ret_dict;

def bulk_ds( trace, Nd ):

    ds_grid = Nd*np.arange((len(trace[0])/Nd+1), dtype=int)
    avg_array = list()
    for i in range(len(ds_grid)-1):
        avg_array.append( np.mean(trace[:, ds_grid[i]:ds_grid[i+1] ], axis=1) )
    res_array = np.transpose(np.vstack( avg_array ))

    return res_array;

