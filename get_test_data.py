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
    in_range = (rq_data['Det'][det]['OFamp']*1e6>1)&(rq_data['Det'][det]['OFamp']*1e6<2)
    events_oi = rq_data['Events']['EventNum'][det_trig&in_range]

    test_data = RQ_cruncher.pull_data( series, run, det, events_oi[:100] )
    with open('test_data.p', 'wb') as f:
        pickle.dump(test_data, f)

    return

if __name__=="__main__":
    main()
