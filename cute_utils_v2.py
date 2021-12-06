# Define some convenience functions for grabbing cute data

import numpy as np
import pandas as pd
import uproot
import qetpy
from rawio.IO import *

#from scdmsPyTools.BatTools.IO import *


chan_map = {

        'R11':{'zip1':{'PES1':'PD2'},
               'zip2':{'PAS2':'T5Z2_A2', 'PBS2':'T5Z2_B2', 'PDS2':'T5Z2_C2', 'PFS2':'T5Z2_D2', 'PT':'T5Z2_T'}},
        
        'R14':{'zip1':{'PES1':'PD2'},
               'zip2':{'PAS2':'T5Z2_A2', 'PBS2':'T5Z2_B2', 'PDS2':'T5Z2_C2', 'PFS2':'T5Z2_D2', 'PT':'T5Z2_T'}},
        
        'R15':{'zip1':{'PES1':'PD2'},
               'zip2':{'PAS2':'T5Z2_A2', 'PBS2':'T5Z2_B2', 'PDS2':'T5Z2_C2', 'PFS2':'T5Z2_D2', 'PT':'T5Z2_T'}},
        
        'R18':{'zip1':{'PBS1':'CPD', 'PES1':'PD2'},
               'zip2':{'PAS2':'T5Z2_A2', 'PBS2':'T5Z2_B2', 'PDS2':'T5Z2_C2', 'PFS2':'T5Z2_D2', 'PT':'T5Z2_T'}},
        
        'R19':{'zip1':{'PAS1':'T5Z2_B1', 'PBS1':'T5Z2_A2', 'PDS1':'T5Z2_B2', 'PES1':'T5Z2_C2', 'PFS1':'T5Z2_D2', 'PT':'T5Z2_T'},
               'zip2':{'PBS2':'CPD', 'PFS2':'PD2'}},
        
        'R20':{'zip1':{'PAS1':'T5Z2_B1', 'PBS1':'T5Z2_A2', 'PDS1':'T5Z2_B2', 'PES1':'T5Z2_C2', 'PFS1':'T5Z2_D2', 'PT':'T5Z2_T'},
               'zip2':{'PBS2':'CPD', 'PFS2':'PD2'}}


        }

det_map = {'R11':{'PD2':1, 'T5Z2':2},
           'R14':{'PD2':1, 'T5Z2':2},
           'R15':{'PD2':1, 'T5Z2':2},
           'R18':{'PD2':1, 'CPD':1, 'T5Z2':2},
           'R19':{'CPD':2, 'T5Z2':1},
           'R20':{'CPD':2, 'T5Z2':1}}


noise_base='/gpfs/slac/staas/fs1/g/supercdms/data/CDMS/CUTE/'

raw_base='/gpfs/slac/staas/fs1/g/supercdms/data/CDMS/CUTE/'
raw_end='/Raw/'

proc_base='/gpfs/slac/staas/fs1/g/supercdms/data/CDMS/CUTE/'

use_Eabs=False

if use_Eabs:
    proc_map = {
            'R11':'R11/Processed/Tests/ProdFeb242021/Submerged/ProdFeb242021_',
            'R14':'R14/Processed/Tests/ProdFeb242021/Submerged/ProdFeb242021_',
            'R15':'R15/Processed/Tests/ProdFeb242021/Submerged/ProdFeb242021_',
            'R18':'R18/Processed/Tests/ProdFeb242021/Submerged/ProdFeb242021_',
            'R19':'R19/Processed/Tests/ProdFeb242021/Submerged/ProdFeb242021_',
            'R20':'R20/Processed/Tests/ProdCPDb/Submerged/ProdCPDb_'
            }

    noise_map = {
            'R11':'R11/Processed/Tests/ProdFeb242021/Noise/ProdFeb242021_Filter_',
            'R14':'R14/Processed/Tests/ProdFeb242021/Noise/ProdFeb242021_Filter_',
            'R15':'R15/Processed/Tests/ProdFeb242021/Noise/ProdFeb242021_Filter_',
            'R18':'R18/Processed/Tests/ProdFeb242021/Noise/ProdFeb242021_Filter_',
            'R19':'R19/Processed/Tests/ProdFeb242021/Noise/ProdFeb242021_Filter_',
            'R20':'R20/Processed/Tests/ProdCPDb/Noise/ProdCPDb_Filter_'
            }
else:    
    proc_map = {
            'R11':'R11/Processed/Tests/Prod2T/Submerged/Prod2T_',
            'R14':'R14/Processed/Tests/Prod2T/Submerged/Prod2T_',
            'R15':'R15/Processed/Tests/Prod2T/Submerged/Prod2T_',
            'R18':'R18/Processed/Tests/ProdPD2F/Submerged/ProdPD2F_',
            #'R18':'R18/Processed/Tests/Prod2T/Submerged/Prod2T_',
            'R19':'R19/Processed/Tests/ProdPD2/Submerged/ProdPD2_',
            'R20':'R20/Processed/Tests/ProdCPDb/Submerged/ProdCPDb_'
            }

    noise_map = {
            'R11':'R11/Processed/Tests/Prod2T/Noise/Prod2T_Filter_',
            'R14':'R14/Processed/Tests/Prod2T/Noise/Prod2T_Filter_',
            'R15':'R15/Processed/Tests/Prod2T/Noise/Prod2T_Filter_',
            'R18':'R18/Processed/Tests/ProdPD2F/Noise/ProdPD2F_Filter_',
            #'R18':'R18/Processed/Tests/Prod2T/Noise/Prod2T_Filter_',
            'R19':'R19/Processed/Tests/ProdPD2/Noise/ProdPD2_Filter_',
            'R20':'R20/Processed/Tests/ProdCPDb/Noise/ProdCPDb_Filter_'
            }

def fetch_data( series_num, run_num  ):
    
    map_key='R'+str(run_num)
    
    filepath_raw = '/gpfs/slac/staas/fs1/g/supercdms/data/CDMS/CUTE/R'+str(run_num)+'/Raw/'
    
    Rshunt = 5.0e-3 # shunt resistance
    Rfb = 5000.0 # feedback resistance
    ADCperVolt = 65536.0/8.0 # the number of ADC bins per V for the readout
    LoopRatio = 2.4 # SQUID turn ratio 
 
#    dsettings = getDetectorSettings(filepath_raw,series_num)
    #print(list(dsettings))

    rq_map={'OFamps':'OFamp',
            'OFamps0':'OF0amp',
            'OFdelay':'OFdelay',
            'OFchisq':'chi2',
            'OFchisqLF':'chi2LF',
            'INTall':'INTall',
            'INTtrunc':'INTtrunc',
            'OF1X2Pamps':'TTOFamp_slow',
            'OF1X2Ramps':'TTOFamp_fast',
            'OF1X2chisq':'TTOFchi2',
            'OF1X2delay':'TTOFdelay',
            'Eabs':'Eabs',
            'std':'std'}
  
    series_path = proc_base+proc_map[map_key]+series_num+'.root'
    #print(series_path)

    file_obj = uproot.open(series_path)
    try:
        file_obj.allkeys()
        zip1 = file_obj.get('rqDir/zip1').pandas.df("*")
        zip2 = file_obj.get('rqDir/zip2').pandas.df("*")
        eventTree = file_obj.get('rqDir/eventTree').pandas.df("*")


    # Sometimes the above block doesn't work, but this one does
    # Some version conflict? Really should figure this out
    # update: I think as long as we're using >V4 of the cdms env on CVMFS we should be fine

    except:
        zip1 = file_obj.get('rqDir/zip1').arrays(library="pd")
        zip2 = file_obj.get('rqDir/zip2').arrays(library="pd")
        eventTree = file_obj.get('rqDir/eventTree').arrays(library="pd")

    not_empty = eventTree['NumTriggers']>0

    EventNumber = np.array(eventTree['EventNumber'], dtype=int)[not_empty]
    EventTime = np.array(eventTree['EventTime'])[not_empty]
    TriggerTime = np.array(eventTree['TriggerTime'])[not_empty]
    TriggerType = np.array(eventTree['TriggerType'])[not_empty]
    TrigDet = np.array(eventTree['TriggerDetectorNum'])[not_empty]
    TriggerMask = np.array(eventTree['TriggerMask'])[not_empty]


    det_dict=dict()
    for ch in chan_map[map_key]['zip1']:
        english_channel = chan_map[map_key]['zip1'][ch]
        det_dict[english_channel]=dict()
#        if ch=='PT':
#            DriverGain=np.inf
#        else:
#            DriverGain = dsettings['Z1'][ch]['driverGain']*4.0
#        conv2Amps = 1/(Rfb*DriverGain*LoopRatio*ADCperVolt)
        for rq in rq_map:
            try:
#                if rq=='bs':
#                    det_dict[english_channel][rq_map[rq]]=zip1[ch+rq][not_empty]*conv2Amps
#                else:
                 det_dict[english_channel][rq_map[rq]]=zip1[ch+rq][not_empty]
            except:
                continue    

    for ch in chan_map[map_key]['zip2']:
        english_channel = chan_map[map_key]['zip2'][ch]
        det_dict[english_channel]=dict()
#        if ch=='PT':
#            DriverGain=np.inf
#        else:
#            DriverGain = dsettings['Z2'][ch]['driverGain']*4.0
#        conv2Amps = 1/(Rfb*DriverGain*LoopRatio*ADCperVolt)
        for rq in rq_map:
            try: 
#                if rq=='bs':
#                    det_dict[english_channel][rq_map[rq]]=zip2[ch+rq][not_empty]*conv2Amps
#                else:
                det_dict[english_channel][rq_map[rq]]=zip2[ch+rq][not_empty] 
            except:
                continue

    triggers = {'Random':{'BeginRun':(TriggerType==2),
                          'InRun':(TriggerType==3),
                          'EndRun':(TriggerType==4) }} 
    
    for det in det_map[map_key]:
        det_trig=(TrigDet==det_map[map_key][det])
        is_det2_empty=np.count_nonzero((TriggerType==1)&det_trig&(TriggerMask==2))==0
        if det=='CPD' and not is_det2_empty:
            right_mask=(TriggerMask==2)
        else:
            right_mask=(TriggerMask==1)
        triggers[det]=(TriggerType==1)&det_trig&right_mask

    series_data = { 'Events':{'EventNum':EventNumber, 'EventTime':EventTime, 'TriggerTime':TriggerTime, 'Trigger':triggers},
                    'Det':det_dict }

    return series_data;

def fetch_pure_rqs(series_num, run_num):
    map_key='R'+str(run_num)

    series_path = proc_base+proc_map[map_key]+series_num+'.root'

    file_obj = uproot.open(series_path)
    file_obj.allkeys()
    zip1 = file_obj.get('rqDir/zip1').pandas.df("*")
    zip2 = file_obj.get('rqDir/zip2').pandas.df("*")
    
    eventTree = file_obj.get('rqDir/eventTree').pandas.df("*")
    rq_dict = {
               'zip1':zip1,
               'zip2':zip2,
               'eventTree':eventTree
              } 

    return rq_dict;

def fetch_template(series_number, run_number):

    map_key='R'+str(run_number)
    chan_config = chan_map[map_key]
    
    noise_path = proc_base+noise_map[map_key]+series_number+'.root'
    noise_obj = uproot.open(noise_path)
    template_dict=dict()

    for zip_num in chan_config:
        for det in chan_config[zip_num]:
            #print(list(noise_file[zip_num]))
            try:
                template_dict[chan_config[zip_num][det]]=np.array(noise_obj[zip_num+'/'+det+'TemplateTime'])
                #template_dict[chan_config[zip_num][det]]=noise_obj.get(zip_num+'/'+det+'TemplateTime').pandas.df("*")
            except:
                template_dict[chan_config[zip_num][det]]=np.nan
    #template = { 'PD2':np.array(noise_file['zip1/PES1TemplateTime']) }
    
    #for ch in channels:
    #    template['T5Z2_'+ch]=np.array(noise_file['zip2/'+channels[ch]+'TemplateTime'])

    return template_dict;

def fetch_psd(series_number, run_number):

    map_key='R'+str(run_number)
    chan_config = chan_map[map_key]
    
    noise_path = proc_base+noise_map[map_key]+series_number+'.root'
    noise_file = uproot.open(noise_path)

    psd_dict=dict()

    for zip_num in chan_config:
        for det in chan_config[zip_num]:
            #print(list(noise_file[zip_num]))
            try:
                psd_dict[chan_config[zip_num][det]]=np.array(noise_file[zip_num+'/'+det+'NoisePSD'])
            except:
                psd_dict[chan_config[zip_num][det]]=np.nan

    return psd_dict;

def fetch_traces(series_number, run_number, event_numbers, max_N=1000, sub_base=True):

    map_key='R'+str(run_number)
    filepath_raw = '/gpfs/slac/staas/fs1/g/supercdms/data/CDMS/CUTE/R'+str(run_number)+'/Raw/'
    
    Rshunt = 5.0e-3 # shunt resistance
    Rfb = 5000.0 # feedback resistance
    ADCperVolt = 65536.0/8.0 # the number of ADC bins per V for the readout
    LoopRatio = 2.4 # SQUID turn ratio 
 
    dsettings = getDetectorSettings(filepath_raw,series_number)

    #traces_full = getRawEvents(filepath_raw, series_number, outputFormat=1, eventNumbers=list(event_numbers), maxNumEvents=max_N)
    #traces_full = getRawEvents(filepath_raw, series_number, outputFormat=1, eventNumbers=list(event_numbers), maxNumEvents=max_N, skipEmptyEvents=False)
    traces_full = getRawEvents(filepath_raw, series_number, outputFormat=1, eventNumbers=list(event_numbers), maxNumEvents=max_N, skipEmptyEvents=False)
    #print(list(traces_full[('Z1','PES1')]))
    trace_dict=dict()

    #print(len(traces_full[0]['Z1']['PES1']))
    #print(len(list(event_numbers)), len(np.vstack(traces_full[('Z1', 'PES1')])))

    for zip_num in chan_map[map_key]:
        if zip_num=='zip1':
            z_key='Z1'
        elif zip_num=='zip2':
            z_key='Z2'
        
        for det in chan_map[map_key][zip_num]:

            if det=='PT':
                continue
             
            DriverGain = dsettings[z_key][det]['driverGain']*4.0
            conv2Amps = 1/(Rfb*DriverGain*LoopRatio*ADCperVolt)
            #print(traces_full[(z_key, det)])

            #try:
            #    trace_array=np.vstack(traces_full[(z_key, det)])
            #except:
            # some of the events are empty...
            trace_list=list()
                # first, get the trace length
            for trace in traces_full[(z_key, det)]:
                if np.count_nonzero(np.isnan(trace))==0:
                    trace_len=len(trace)
                    break
                
            # now go through and put in placeholder with appropriate length
            for trace in traces_full[(z_key, det)]:
                #print(trace, np.count_nonzero(np.isnan(trace)))
                if np.count_nonzero(np.isnan(trace))>0:
                    trace_list.append(np.ones((1,trace_len))*-99999)
                else:
                    trace_list.append(trace)
            trace_array=np.vstack(trace_list)
            if sub_base:        
                trace_dict[chan_map[map_key][zip_num][det]] = conv2Amps*( trace_array - np.mean( trace_array, axis=1)[:,np.newaxis] )
            else:
                trace_dict[chan_map[map_key][zip_num][det]] = conv2Amps*( trace_array )
    return trace_dict;
