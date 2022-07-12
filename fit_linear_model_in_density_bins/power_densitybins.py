import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta
import time

import cat_power_algos as catpk
import classylss
import fitsio
from nbodykit.lab import *


LOS = [0,0,1]
redshift = 0
BoxSize = 2000
cosmo_paras = classylss.load_ini('Planck18_LCDM.ini')
cosmo = cosmology.cosmology.Cosmology.from_dict(cosmo_paras)
Plin = cosmology.LinearPower(cosmo, redshift, transfer='EisensteinHu') # matter power spectrum 

kmin, kmax, dk = 0, 0.05, 0.01
Nmesh = 128

# load and split data
start_time = time.time()
print("Loading catalog")
filepath = '/disk11/salam/FirstGenMocks/AbacusSummit/CubicBox/BGS_v2/z0.200/AbacusSummit_base_c000_ph006/BGS_box_ph006.fits'
cat = FITSCatalog(filepath)
cat = catpk.prep_fitscat(cat, cosmo=cosmo, LOS=LOS, z=redshift)

print("Loading density data")
filepath_density = '/disk11/salam/FirstGenMocks/AbacusSummit/CubicBox/BGS_v2/z0.200/AbacusSummit_base_c000_ph006/BGS_box_ph006.vtf_mean.fits.gz'
fits = fitsio.FITS(filepath_density)

print("Splitting density data")
percent_edges = [0,10,20,30,40,50,60,70,80,90,100]
ptile_split = np.percentile(fits[1]['rho'][:], percent_edges) # gets edges of density bins
np.savetxt("density_bins/percentile_edges.txt", ptile_split, 
           header='Value of density bin edges corresponding to percentage edges' + str(percent_edges))
end_time = time.time()
print('Total time for loading and splitting data: %s'%str(timedelta(seconds=end_time-start_time)))

# save power spectrum for each density bin
ells = [0,2]
n_ptile = len(ptile_split)-1 # number of bins = number of edges - 1

for i in range(n_ptile):
    t1 = time.time()
    # get indcies of i-th percentile
    insel = fits[1].where('rho>%f && rho<=%f'%(ptile_split[i],ptile_split[i+1]))
        
    mesh = cat[insel].to_mesh(position='RSDPosition', resampler='tsc', BoxSize=BoxSize, Nmesh=Nmesh, compensated=True)
    # large number of mu allows to integrate over mu. Needed for covariance matrix using data P(k,mu) as input. 
    # See ABC fitting notebook
    r = FFTPower(mesh, mode='2d', Nmu=51, los=LOS, poles=ells, kmin=kmin, kmax=kmax, dk=dk)
    r.save('density_bins/ptile_%d.json'%i)
    print('Computed power of percentile %d'%i)