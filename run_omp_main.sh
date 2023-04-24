#!/bin/bash
#SBATCH --job-name=GA_XGBoost_Controller_Hola_Barcelona
#SBATCH --output=GA_XGBoost_Controller%j.out
#SBATCH --error=GA_XGBoost_Controller%j.err
#SBATCH --partition=HM
#SBATCH --ntasks-per-node=4

##Optional - Required memory in MB per node, or per core. Defaults are 1GB per core.
##SBATCH --mem=3096
#SBATCH --mem-per-cpu=3096
##SBATCH --exclusive

##Optional - Estimated execution time
##Acceptable time formats include  "minutes",   "minutes:seconds",
##"hours:minutes:seconds",   "days-hours",   "days-hours:minutes" ,"days-hours:minutes:seconds".
#SBATCH --time=10-0

########### Further details -> man sbatch ##########
#export OMP_NUM_THREADS=4

python3 GA_ML_Kratos_shale.py