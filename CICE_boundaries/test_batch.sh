#! /usr/bin/env python

## test script for make_CICE_bc_from_NORESM_ver2.0.py
#SBATCH --account=nn9300k
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=128
#SBATCH --partition=normal
#SBATCH --time=24:00:00
#SBATCH --job-name=slurm

import numpy as np
import netCDF4, datetime, glob, os, sys
from numpy import dtype

os.system("srun python make_CICE_bc_from_NORESM_ver2.0.py")

