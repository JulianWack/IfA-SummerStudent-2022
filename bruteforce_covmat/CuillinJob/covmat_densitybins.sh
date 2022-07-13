#!/bin/tcsh
#SBATCH --job-name=covmat_densitybins
#SBATCH --time=00:10:00
#SBATCH --ntasks=1
#SBATCH --mem=1G

set LogFile=Log-Serial.$SLURM_JOBID
python3 power_covmat_densitybins.py >> $LogFile