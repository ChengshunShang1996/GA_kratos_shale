#!/bin/bash
#SBATCH --job-name=BTS-Q-Ep6.2e10-T1e3-f0.1
#SBATCH --output=BTS-Q-Ep6.2e10-T1e3-f0.1%j.out
#SBATCH --error=BTS-Q-Ep6.2e10-T1e3-f0.1%j.err
#SBATCH --partition=HighParallelization
#SBATCH --ntasks-per-node=1

##Optional - Required memory in MB per node, or per core. Defaults are 1GB per core.
##SBATCH --mem=3096
#SBATCH --mem-per-cpu=3096
##SBATCH --exclusive

##Optional - Estimated execution time
##Acceptable time formats include  "minutes",   "minutes:seconds",
##"hours:minutes:seconds",   "days-hours",   "days-hours:minutes" ,"days-hours:minutes:seconds".
#SBATCH --time=10-0

########### Further details -> man sbatch ##########
#export OMP_NUM_THREADS=1

python3 MainKratos.py