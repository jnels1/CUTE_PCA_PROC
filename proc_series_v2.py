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

import pickle
import time
sys.path.append("/gpfs/slac/staas/fs1/g/supercdms/users/jnels1/my_functions")
import cute_utils_v2 as cute_utils

def gf(trace):
    return ndimage.gaussian_filter1d(trace, sigma=2.5, order=0, mode='wrap')

def get_rise_index(trace, thresh):
    max_ind=np.argmax(trace)
    pre_trace=trace[:max_ind]
   
    try: 
        rise_guess=np.argmax(-np.abs(pre_trace-thresh))
    except:
        rise_index = np.nan
        return rise_index

    try:
        f_0d = interpolate.interp1d( np.arange(len(pre_trace)), pre_trace-thresh )
        sol = spo.root_scalar( f_0d, bracket=[rise_guess-1, rise_guess+1], method='brentq' )
        rise_index = sol.root
    except:
        rise_index=rise_guess

    return rise_index;

def get_fall_index(trace, thresh):
    max_ind=np.argmax(trace)
    post_trace=trace[max_ind:]
    try:
        fall_guess=np.argmax(-np.abs(post_trace-thresh))
    except:
        fall_index = np.nan
        return fall_index
    try:
        f_0d = interpolate.interp1d( np.arange(len(post_trace)), post_trace-thresh )
        sol = spo.root_scalar( f_0d, bracket=[fall_guess-1, fall_guess+1], method='brentq' )
        fall_index = sol.root + max_ind
    except:
        fall_index=fall_guess+max_ind
    return fall_index;

def est_t0_point(trace):
    zeroth_d = ndimage.gaussian_filter1d(trace[10000:20000], sigma=2.5, order=0, mode='wrap')
    first_d = ndimage.gaussian_filter1d(trace[10000:20000], sigma=2.5, order=1, mode='wrap')
    second_d = ndimage.gaussian_filter1d(trace[10000:20000], sigma=2.5, order=2, mode='wrap')
    
    # find the point of maximal slope, allowing for interpolation between data points
    f_0d = interpolate.interp1d( np.arange(10000), zeroth_d )    
    f_1d = interpolate.interp1d( np.arange(10000), first_d )    
    f_2d = interpolate.interp1d( np.arange(10000), second_d )
    
    max_guess = np.argmax(first_d)
    #print(max_guess)
    try:
        sol = spo.root_scalar( f_2d, bracket=[max_guess-1, max_guess+1], method='brentq' )
    
        #tmax = np.argmax(first_d) # time when slope is maximal
        tmax = sol.root
    except:
        tmax = max_guess
    tslope = f_0d(tmax)/f_1d(tmax) # characteristic time at maximal slop
    return tmax-tslope+10000

def bulk_ds( trace ):
    
    Nd = 8 # downsampling factor
    
    #bs = np.mean(trace[:,:15000], axis=1)
    #trace0=trace-bs[:,np.newaxis]

    ds_grid = Nd*np.arange((len(trace[0])/Nd+1), dtype=int)
    #avg_grid = Nd*np.arange((len(trace), len(trace[0])/Nd+1), dtype=int)
    
    #averaged_trace = np.vstack( [np.mean( trace0[ avg_grid[i]:avg_grid[i+1] ] ) for i in range(len(ds_sgrid))] )
    avg_array = list()
    for i in range(len(ds_grid)-1):
        avg_array.append( np.mean(trace[:, ds_grid[i]:ds_grid[i+1] ], axis=1) )
    res_array = np.transpose(np.vstack( avg_array ))
    
    return res_array;

def check_coh(traces):
    # Look at a collection of traces we want to use for deriving the PCA components
    norm_array = np.zeros(len(traces))
    for i, trace in enumerate(traces):
        norm_array[i] = np.sqrt( np.sum( np.power( trace, 2 ) ) )
    norm_traces = np.power(norm_array, -1)[:,np.newaxis]*traces    
    coh_mat = norm_traces @ np.transpose(norm_traces)
    mean_coh = np.mean( coh_mat, axis=1 )    
    return mean_coh;

def frac_roll( trace, dt ):
    # *effectively* shift a trace by a non-integer number of samples
    # get the integer and decmil part of our shift
    dt_frac = dt % 1
    
    lower_bound = np.roll( trace, -int(np.floor(dt)) )
    upper_bound = np.roll( trace, -int(np.ceil(dt)) )
    interp_trace = dt_frac*upper_bound + (1-dt_frac)*lower_bound
    return interp_trace

def main():

    # Let's try to process an entire series using PCA
    #series = '23201125_214136'

    series = sys.argv[1]
    run = int(sys.argv[2])
    det = sys.argv[3]
    save_dir = '/sdf/home/j/jnels1/my_data/cute_proc/PCA_proc/'
    #series = '23201126_175642'
    #run = 18
    #det = 'PD2'

    print('processing run ', run, 'series ', series)
    time_start = time.time()
    rq_data = cute_utils.fetch_data(series, run)

    det_trig = rq_data['Events']['Trigger'][det]
    noise_trig = rq_data['Events']['Trigger']['Random']['BeginRun']
    of_range = (rq_data['Det'][det]['OFamp']*1e6>0.25)&(rq_data['Det'][det]['OFamp']*1e6<1.5)

    #print(np.count_nonzero(det_trig), np.count_nonzero(det_trig&of_range))
    get_train = True
    
    if get_train:
        # Get training traces
        train_events = rq_data['Events']['EventNum'][det_trig&of_range]
        train_delay = rq_data['Det'][det]['OFdelay'][det_trig&of_range]
        train_traces = cute_utils.fetch_traces( series, run, event_numbers=train_events, max_N=500 )[det]

        # Align training traces
        tshift_traces = list()
        for i, trace in enumerate(train_traces):
            bs = np.mean(trace[:15000])
            trace0 = trace-bs
            t0 = est_t0_point(trace)
            tshift = t0 - 16040
            tshift_traces.append( frac_roll(trace0, tshift) )

        # subtract baseline and downsample each trace
        train_array0 = bulk_ds( np.vstack(tshift_traces)  )

        # check coherence of training set, make a very rough cut
        # should really make this cut energy dependant
        train_coh = check_coh(train_array0)
        good_coh = train_coh>0.8
        train_array = train_array0[good_coh]

        with open(save_dir+'PCA_training_'+series+'.p', 'wb') as f:
            pickle.dump(train_array, f)


        # Get noise traces, don't try time shift!
        noise_events = rq_data['Events']['EventNum'][noise_trig]
        noise_traces = cute_utils.fetch_traces( series, run, event_numbers=noise_events, max_N=500 )[det]
        
        noise_cut = autocuts(np.vstack(noise_traces))
        noise_bs = np.mean( noise_traces[:15000], axis=1 )
        noise_array = bulk_ds((noise_traces-noise_bs[:,np.newaxis])[noise_cut])

        with open(save_dir+'PCA_noise_'+series+'.p', 'wb') as f:
            pickle.dump(noise_array, f)
        

    else:
        with open(save_dir+'PCA_training_'+series+'.p', 'rb') as f:
            train_array = pickle.load(f)
        with open(save_dir+'PCA_noise_'+series+'.p', 'rb') as f:
            noise_array = pickle.load(f)

    # Once we have the training traces, calculate the covariance matrix and diagonalize it
    
    Neig = 10
    C = np.cov( train_array, rowvar=False )
    m = 2**12
    w, v = eigh( C, eigvals=(m-Neig, m-1))


    C_noise = np.cov(noise_array, rowvar=False)
    w_noise, v_noise = eigh( C_noise, eigvals=(m-Neig, m-1))

    # Now it's time to get down to buisness
    # need to pull every triggered trace and determine its PCA decomposition

    # Define container for our results
    # Also record relavant precalculated RQs from OF alg.
    ret_dict=dict()
    ret_dict['series']=series
    ret_dict['events']=rq_data['Events']['EventNum'][det_trig]
    ret_dict['EventTime_ref']=rq_data['Events']['EventTime'][det_trig]
    ret_dict['EventTime_trig']=rq_data['Events']['TriggerTime'][det_trig]
    ret_dict['OFamp']=np.array(rq_data['Det'][det]['OFamp'][det_trig])
    ret_dict['OFdelay']=np.array(rq_data['Det'][det]['OFdelay'][det_trig])
    ret_dict['chi2LF']=np.array(rq_data['Det'][det]['chi2LF'][det_trig])
    ret_dict['PC_vecs']=np.fliplr(v)
    ret_dict['PC_vals']=np.flip(w)
    ret_dict['PC_noise_vecs']=np.fliplr(v_noise)
    ret_dict['PC_noise_vals']=np.flip(w_noise)

    pc_amp=list()
    pc_comp=list()
    gf_amp_list=list()
    bs_list=list()
    delta_bs_list=list()
    t0_list=list()

    int1_list=list()
    int2_list=list()
    int3_list=list()

    gfint1_list=list()
    gfint2_list=list()
    gfint3_list=list()
    
    gfTmax_list=list()
    gfRT2_list=list()
    gfRT5_list=list()
    gfRT8_list=list()
    gfFT2_list=list()
    gfFT5_list=list()
    gfFT8_list=list()

    proj1_res=list()
    proj3_res=list()
    noise1_res=list()
    noise3_res=list()

    # Now we can fit each trace
    # Only load 1000 at a time, so that we don't overload memory

    # Split the above events into groups of 1000
    N_split = int(np.count_nonzero(det_trig)/1000) +1

    block_events = np.array_split(ret_dict['events'], N_split)

    for block in block_events:
        in_block = np.in1d(rq_data['Events']['EventNum'], block)
        block_delays = rq_data['Det'][det]['OFdelay'][in_block]
        block_amps = rq_data['Det'][det]['OFamp'][in_block]

        traces_oi = cute_utils.fetch_traces(series, run, event_numbers=block, max_N=1000, sub_base=False)[det]
        # Perform time shift and calculate non-PCA RQs
        tshift_traces = list()
        for i, trace in enumerate(traces_oi):
            # calculate baseline quantities
            bs = np.mean(trace[:15000])
            bs_list.append(bs)
            delta_bs_list.append( np.mean(trace[-5000:]) - np.mean(trace[:5000]) )

            # subtrace mean baseline from raw trace
            trace0 = trace - bs
            
            # estimate start time
            t0 = est_t0_point(trace0)
            t0_list.append(t0/625000)
            # Calculate integral quantities
            int1_list.append( np.sum(trace0[int(t0)-100:int(t0)+900])/625000 ) 
            int2_list.append( np.sum(trace0[int(t0)-100:int(t0)+2400])/625000 ) 
            int3_list.append( np.sum(trace0[int(t0)-100:int(t0)+9900])/625000 ) 
            
            #gfint1_list.append( np.sum(gf(trace0[int(t0)-100:int(t0)+900]))/625000 ) 
            #gfint2_list.append( np.sum(gf(trace0[int(t0)-100:int(t0)+2400]))/625000 ) 
            #gfint3_list.append( np.sum(gf(trace0[int(t0)-100:int(t0)+9900]))/625000 ) 
            
            gfint1_list.append( np.sum(gf(trace0)[int(t0)-100:int(t0)+900])/625000 ) 
            gfint2_list.append( np.sum(gf(trace0)[int(t0)-100:int(t0)+2400])/625000 ) 
            gfint3_list.append( np.sum(gf(trace0)[int(t0)-100:int(t0)+9900])/625000 ) 

            # Get gf amplitude
            gf_amp = np.amax(gf(trace0))         
            gf_amp_list.append(gf_amp)

            # Normalize trace and get rise/fall times
            norm_gf_trace = (1./gf_amp)*gf(trace0)
            gfTmax_list.append(np.argmax(norm_gf_trace)/625000)
            gfRT2_list.append(get_rise_index( norm_gf_trace, 0.2 )/625000)
            gfRT5_list.append(get_rise_index( norm_gf_trace, 0.5 )/625000)
            gfRT8_list.append(get_rise_index( norm_gf_trace, 0.8 )/625000)
            
            gfFT2_list.append(get_fall_index( norm_gf_trace, 0.2 )/625000)
            gfFT5_list.append(get_fall_index( norm_gf_trace, 0.5 )/625000)
            gfFT8_list.append(get_fall_index( norm_gf_trace, 0.8 )/625000)

            tshift = t0 - 16040
            shifted_trace = frac_roll(trace0, tshift)
            tshift_traces.append(shifted_trace)

        down_traces = bulk_ds(np.vstack(tshift_traces))
        projected_traces_full = down_traces @ v @ np.transpose(v)
        pc_amp.append(np.amax( projected_traces_full, axis=1 ))
        pc_comp.append( down_traces @ v )

        #print(np.shape(down_traces), np.shape(v))
        #print(np.shape(v[:,-1]), np.shape(v[:,-1][:,np.newaxis] @ np.transpose(v[:,-1])[np.newaxis:,]))

        projected_trace1 = down_traces @ (v[:,-1][:,np.newaxis] @ np.transpose(v[:,-1])[np.newaxis,:])
        projected_trace3 = down_traces @ (v[:,-3:] @ np.transpose(v[:,-3:]))
        
        projected_noise1 = down_traces @ (v_noise[:,-1][:,np.newaxis] @ np.transpose(v_noise[:,-1])[np.newaxis,:])
        projected_noise3 = down_traces @ (v_noise[:,-3:] @ np.transpose(v_noise[:,-3:]))

        proj1_res.append( np.sum(np.power(projected_trace1-down_traces, 2), axis=1) )
        proj3_res.append( np.sum(np.power(projected_trace3-down_traces, 2), axis=1) )
        
        noise1_res.append( np.sum(np.power(projected_noise1-down_traces, 2), axis=1) )
        noise3_res.append( np.sum(np.power(projected_noise3-down_traces, 2), axis=1) )
 
    ret_dict['gf_amp']=np.hstack(gf_amp_list)
    ret_dict['pc_amp']=np.hstack(pc_amp)
    ret_dict['bs']=np.hstack(bs_list)
    ret_dict['delta_bs']=np.hstack(delta_bs_list)
    ret_dict['t0']=np.hstack(t0_list)

    ret_dict['INT1']=np.hstack(int1_list)
    ret_dict['INT2']=np.hstack(int2_list)
    ret_dict['INT3']=np.hstack(int3_list)
    ret_dict['gfINT1']=np.hstack(gfint1_list)
    ret_dict['gfINT2']=np.hstack(gfint2_list)
    ret_dict['gfINT3']=np.hstack(gfint3_list)
    
    ret_dict['gfTmax']=np.hstack(gfTmax_list)
    ret_dict['gfRT2']=np.hstack(gfRT2_list)
    ret_dict['gfRT5']=np.hstack(gfRT5_list)
    ret_dict['gfRT8']=np.hstack(gfRT8_list)
    ret_dict['gfFT2']=np.hstack(gfFT2_list)
    ret_dict['gfFT5']=np.hstack(gfFT5_list)
    ret_dict['gfFT8']=np.hstack(gfFT8_list)

    ret_dict['proj1_res']=np.hstack(proj1_res)
    ret_dict['proj3_res']=np.hstack(proj3_res)
    ret_dict['noise1_res']=np.hstack(noise1_res)
    ret_dict['noise3_res']=np.hstack(noise3_res)

    comp_stack = np.vstack(pc_comp)
    for i in range(Neig):
        ret_dict['PC'+str(Neig-i)] = comp_stack[:,i]
    
    print('processed ', np.count_nonzero(det_trig), 'events') 
    print('that took ', (time.time()-time_start)/60, ' minutes')

    with open(save_dir+det+'_res_'+series+'.p', 'wb') as f:
        pickle.dump(ret_dict, f)

    return

if __name__=="__main__":
    main()
