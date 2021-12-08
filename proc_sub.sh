#!/bin/bash
# JOB HEADERS HERE
#SBATCH --mem=32gb
#SBATCH --partition=supercdms

. /cvmfs/cdms.opensciencegrid.org/setup_cdms.sh V04-00
#python proc_series.py $1 $2
python proc_series.py $1 $2 $3 $4 $5 $6
