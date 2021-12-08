import numpy as np
import pandas as pd
import pickle

import RQ_cruncher
import cute_utils_v2 as cute_utils

def main():

    series = '23210326_012655'
    run = 20
    det = 'CPD'

    rq_data = cute_utils.fetch_data(series, run)
    det_trig = rq_data['Events']['Trigger'][det]
    ran_trig = rq_data['Events']['Trigger']['Random']['InRun']
    in_range = (rq_data['Det'][det]['OFamp']*1e6>1)&(rq_data['Det'][det]['OFamp']*1e6<2)
    det_events = rq_data['Events']['EventNum'][det_trig&in_range]
    noise_events = rq_data['Events']['EventNum'][ran_trig]

    test_data_det = RQ_cruncher.pull_data( series, run, det, det_events[:100] )
    test_data_ran = RQ_cruncher.pull_data( series, run, det, noise_events[:100] )
    
    with open('test_data_det.p', 'wb') as f:
        pickle.dump(test_data_det, f)
    with open('test_data_ran.p', 'wb') as f:
        pickle.dump(test_data_ran, f)

    return

if __name__=="__main__":
    main()
