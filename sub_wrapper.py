import numpy as np
import pandas as pd
import sys
import subprocess

#import fit_2pulse_v2


def job_sub(series, run, det, filter_str, ds_str, train_str):
    
    log_dir='/sdf/home/j/jnels1/my_data/cute_proc/PCA_proc/CUTE_PCA_PROC/logs/'
    log_name = log_dir+det+str(run)+'_'+series+'_filter_'+filter_str+'_ds_'+ds_str+'_'+train_str+'.log'

    subprocess.check_call("sbatch -o %s proc_sub.sh %s %s %s %s %s %s" % (log_name, series, run, det, filter_str, ds_str, train_str), shell=True)
    #subprocess.check_call("sbatch -e %s proc_sub.sh %s %s %s" % (log_name, series, run, det), shell=True)

    return;

def main():

    # submit all series with specified run and data type
    #run = int(sys.argv[1])
    #data_type = sys.argv[2]
    
    #run=20
    #series_list = read_data_log(run, data_type)
    #series_list = ['23210325_211520', '23210326_012655', '23210326_130758', '23210327_010110',  '23210327_190020', '23210328_051029', '23210328_142945', '23210329_021134', '23210329_114716']

    job_dict={
        '1':{'series':'23210326_012655','run':20,'det':'CPD','filter':'true','ds':'true','train':'Det'},
        '2':{'series':'23210326_012655','run':20,'det':'CPD','filter':'true','ds':'true','train':'Random'},
        '3':{'series':'23210326_012655','run':20,'det':'CPD','filter':'false','ds':'true','train':'Det'},
        '4':{'series':'23210326_012655','run':20,'det':'CPD','filter':'false','ds':'true','train':'Random'},
        '5':{'series':'23210326_012655','run':20,'det':'CPD','filter':'true','ds':'false','train':'Det'},
        '6':{'series':'23210326_012655','run':20,'det':'CPD','filter':'true','ds':'false','train':'Random'},
        '7':{'series':'23210326_012655','run':20,'det':'CPD','filter':'false','ds':'false','train':'Det'},
        '8':{'series':'23210326_012655','run':20,'det':'CPD','filter':'false','ds':'false','train':'Random'}
    } 


 
    for key in job_dict:
        job_sub(job_dict[key]['series'], job_dict[key]['run'], job_dict[key]['det'], job_dict[key]['filter'], job_dict[key]['ds'], job_dict[key]['train'])

    return;

if __name__=="__main__":
    main()
