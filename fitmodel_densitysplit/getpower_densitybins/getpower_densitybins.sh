#!/bin/tcsh
#SBATCH --job-name=getpower_densitybins
#SBATCH --time=20:00:00
#SBATCH --ntasks=1
#SBATCH --mem=100G

set LogFile=Log-Serial.$SLURM_JOBID
python3 getpower_densitybins.py >> $LogFile