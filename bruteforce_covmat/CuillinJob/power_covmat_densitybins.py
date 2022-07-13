import numpy as np
import matplotlib.pyplot as plt
import time
from datetime import timedelta
from os import mkdir, listdir

import cat_power_algos as catpk
import classylss
import fitsio
from nbodykit.lab import *

wdir = '/home/jwack/main/bruteforce_covmat/' # main working directory
LOS = [0,0,1]
redshift = 0
BoxSize = 500
cosmo_paras = classylss.load_ini(wdir+'Planck18_LCDM.ini')
cosmo = cosmology.cosmology.Cosmology.from_dict(cosmo_paras)
Plin = cosmology.LinearPower(cosmo, redshift, transfer='EisensteinHu') 

kmin, kmax, dk = 0, 2, 0.01
Nmesh = 128

ells = [0, 2]
percent_edges = [0,10,20,30,40,50,60,70,80,90,100]
n_bins = len(percent_edges)-1 # number bins = number edges - 1

# base file path for data and density 
data_basepath = '/disk01/DESI/cosmosim/FirstGenMocks/AbacusSummit/CubicBox/BGS_v2/z0.200/small/'
density_basepath = '/disk01/DESI/cosmosim/FirstGenMocks/AbacusSummit/CubicBox/BGS_v2/z0.200/small_vtf/'
# the data for individual boxes is stored in these base paths according to:
# - Data: AbacusSummit_small_c000_phX/BGS_box_phX.fits with X between 3000 and 5000 inclusive.
# - Density: BGS_box_phX.vtf_mean.fits.gz with the same X as above
# note: some values of X have been skipped and not all for which data is present had density computed 


### Compute and store power spectrum for each density bins for all boxes ###
start_time = time.time()
count = 0
for density_file in listdir(density_basepath):
    # check if X in [3000,5000] and if file has aleady been computed
    idx = int(density_file[10:-17])
    if not (3000 <= idx <= 5000): continue
    if 'BGS_box_ph%d'%idx in listdir(wdir+'density_splits'): continue 
    mkdir(wdir+"density_splits/BGS_box_ph%d"%idx)
    
    data_path = data_basepath + 'AbacusSummit_small_c000_ph%d/BGS_box_ph%d.fits'%(idx,idx)
    density_path = density_basepath + density_file
    
    t1 = time.time()
    # make catalog and find value of denisty at percentile edges
    cat = FITSCatalog(data_path)
    cat = catpk.prep_fitscat(cat, cosmo=cosmo, LOS=LOS, z=redshift)
    fits = fitsio.FITS(density_path)
    ptile_split = np.percentile(fits[1]['rho'][:], percent_edges)
    np.savetxt(wdir+'density_splits/BGS_box_ph%d/percentile_edges.txt'%idx, ptile_split, 
               header='Value of density bin edges corresponding to percentage edges' + str(percent_edges))
    
    # compute power for each density bin
    for i in range(n_bins):
        insel = fits[1].where('rho>%f && rho<=%f'%(ptile_split[i],ptile_split[i+1]))
        mesh = cat[insel].to_mesh(position='RSDPosition', resampler='tsc', BoxSize=BoxSize, Nmesh=Nmesh, compensated=True)
        r = FFTPower(mesh, mode='2d', Nmu=5, los=LOS, poles=ells, kmin=kmin, kmax=kmax, dk=dk)
        r.save(wdir+'density_splits/BGS_box_ph%d/power_ptile_%d.json'%(idx,i))
    count += 1
        
end_time = time.time()
print('Total time for %d files: %s'%(count, str(timedelta(seconds=end_time-start_time))))


### Compute and store covariance matrix for each density bin ###
n_boxes = len(listdir(wdir+'density_splits'))

for i in range(n_bins):
    for s,box_folder in enumerate(listdir(wdir+'density_splits')): 
        if box_folder == '.ipynb_checkpoints': continue
        r = FFTPower.load(wdir+'density_splits/'+box_folder+'/power_ptile_%d.json'%i)
        poles = r.poles 
        
        Pk_ells = np.empty((len(ells), len(poles['k']))) # later flattened 
        for j,ell in enumerate(ells):
            Pk_ell = poles['power_%d' %ell].real
            if ell == 0: 
                Pk_ell = Pk_ell - poles.attrs['shotnoise']

            Pk_ells[j] = Pk_ell
        Pk_ells = Pk_ells.flatten()
        # add multipole from current box to data vector for current percentile
        if s == 0:
            prev = Pk_ells
            continue
        prev = np.vstack((prev, Pk_ells))
        
    cov_mat = np.cov(prev.T)
    np.savetxt(wdir+'covariance_matricies/cov_ptile_%d.txt'%i, cov_mat, 
               header='Elements corresponding to k: %s and l: %s'%(str(poles['k']), str(ells)))
    
print("Stored covariance matrix for all density bins")