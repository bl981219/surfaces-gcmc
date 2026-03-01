#!/bin/bash 
#SBATCH -J superLSF  #sensible name for the job
#SBATCH -n 64 #Request tasks (40 cores/nodes)
#SBATCH -N 2 #Request node
#SBATCH -t 3:00:00 #Request runtime
#SBATCH -C centos7 #Request only Centos7 nodes
#SBATCH -p sched_mit_nse #Run on sched_engaging_default partition

module load python/3.9.4
python calculation.py