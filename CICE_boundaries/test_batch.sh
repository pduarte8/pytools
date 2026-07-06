#! /usr/bin/bash

## test script for make_CICE_bc_from_NORESM_ver2.0.py
#SBATCH --account=nn9300k
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=128
#SBATCH --partition=normal
#SBATCH --time=24:00:00
#SBATCH --job-name=slurm

module load Python/3.13.5-GCCcore-14.3.0
module load CDO/2.5.3-gompi-2025b
source /cluster/home/pduarte/python_env/bin/activate 

#import numpy as np
#import netCDF4, datetime, glob, os, sys
#from numpy import dtype
# Only if SciPy isn't already installed
#python -m pip install scipy

srun --ntasks=1 python make_CICE_bc_from_NORESM_ver2.0.py

