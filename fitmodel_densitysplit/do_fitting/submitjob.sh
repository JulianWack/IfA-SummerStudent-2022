#!/bin/bash
#SBATCH --job-name=fitmodel
#SBATCH --time=20:00:00
#SBATCH --ntasks=1
#SBATCH --mem=1G

module unload anaconda/3 # needed when module has been loaded previously 
module load anaconda/3.7
python3 modelfit.py >> Log.$SLURM_JOBID