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
import basis_maker
import RQ_cruncher

def pull_data(series, run, det, event_numbers):

    trace_list=list()
    delay_list=list()

    rq_data = cute_utils.fetch_data( series, run )

    det_list = list(rq_data['Events']['Trigger'])
    det_list.remove('Random')

    if det not in det_list:
        print( 'det must be one of: ', det_list)
        return np.nan

    good_events = np.in1d( event_numbers, rq_data['Events']['EventNum'] )
    delay_oi = rq_data['Det'][det]['OFdelay'][good_events]

    traces_oi = cute_utils.fetch_traces( series, run, event_numbers=events_oi, max_N=len(events_oi), sub_base=False  )[det]

    ret_dict={
        'series':series,
        'run':run,
        'det':det,
        'EventNum':np.hstack(event_numbers),
        'OFdelay':np.hstack(delay_oi),
        'traces':np.vstack(trace_list)
    }

    ret_dict['EventTime_ref']=rq_data['Events']['EventTime'][good_events]
    ret_dict['EventTime_trig']=rq_data['Events']['TriggerTime'][good_events]
    ret_dict['OFamp']=np.array(rq_data['Det'][det]['OFamp'][good_events])
    ret_dict['OFdelay']=np.array(rq_data['Det'][det]['OFdelay'][good_events])
    ret_dict['chi2LF']=np.array(rq_data['Det'][det]['chi2LF'][good_events])

    return ret_dict


def main():

    # Let's try to process an entire series using PCA
    #series = '23201125_214136'

    data_dir = '/sdf/home/j/jnels1/my_data/cute_proc/PCA_proc/CUTE_PCA_PROC/'
   
    series = sys.argv[1]
    run = int(sys.argv[2])
    det = sys.argv[3]
 
    filter_str = sys.argv[4]
    ds_str = sys.argv[5]
    train_str = sys.argv[6]

    config_dict = {
    'series':'23201126_175642',
    'run':18,
    'det':'PD2',
    'train#':1000,
    'train_range':np.array([0., 1.5]),
    'N_basis':15
    }

    config_dict['series']=series
    config_dict['run']=run
    config_dict['det']=det
    config_dict['train_trigger']=train_str

    if filter_str=='true':
        config_dict['shape_filter']=True
    else:
        config_dict['shape_filter']=False

    if ds_str=='true':
        config_dict['shape_ds']=8
        config_dict['shape_window']=[0,2**15]
    else:
        config_dict['shape_ds']=1
        config_dict['shape_window']=[15500,15500+4096]

    basis_savename = det+str(run)+'_'+series+'basis_filter_'+filter_str+'_ds_'+ds_str+'_'+train_str+'.p'
    RQ_savename = det+str(run)+'_'+series+'RQ_filter_'+filter_str+'_ds_'+ds_str+'_'+train_str+'.p'

    time_start = time.time()
    do_basis=True
    if do_basis:
        my_basis = basis_maker.make_basis(config_dict)
        with open(data_dir+'basis'+basis_savename, 'wb') as f:
            pickle.dump(my_basis, f)
    else:
        with open(data_dir+'basis'+basis_savename, 'rb') as f:
            my_basis = pickle.load(f)

    basis_result = RQ_cruncher.series_looper(config_dict, my_basis)
    time_stop = time.time()-time_start
    print('That took', round(time_stop/60), 'minutes')

    with open(data_dir+'RQs'+RQ_savename, 'wb') as f:
        pickle.dump(basis_result, f)


    return

if __name__=="__main__":
    main()
