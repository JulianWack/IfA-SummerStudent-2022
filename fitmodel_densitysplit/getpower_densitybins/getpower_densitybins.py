# computes and stores the 2D power spectrum for each density bin and the whole box
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta
import time

from astropy.cosmology import Planck18 as Planck18_astropy 
import fitsio
from nbodykit.lab import *



### Helper functions ###
def vec_projection(vector, direction):
    '''Projects vector on direction vector (can be non-normalised).'''
    direction = numpy.asarray(direction, dtype='f8')
    direction = direction / (direction ** 2).sum() ** 0.5
    projection = (vector * direction).sum(axis=-1)
    projection = projection[:, None] * direction[None, :]

    return projection


def prep_fitscat(cat, cosmo, LOS=[0,0,0], z=0):
    '''Adds Position, Velocity, and RSDPosition to FITSCatalog.'''
    cat['Position'] = np.stack([cat['x'].compute(), cat['y'].compute(), cat['z'].compute()], axis=1)
    cat['Velocity'] = np.stack([cat['vx'].compute(), cat['vy'].compute(), cat['vz'].compute()], axis=1)
    
    if LOS != [0,0,0]:
        # RSD position = position + velocity offset along LOS
        rsd_factor = (1+z) / (100 * cosmo.efunc(z))
        cat['RSDPosition'] = cat['Position'] + rsd_factor * vec_projection(cat['Velocity'], LOS)
        
    return cat
#######################


### Find power spectrum for each density bin ###
LOS = [0,0,1]
redshift = 0.2
BoxSize = 2000

cosmo = cosmology.Cosmology.from_astropy(Planck18_astropy)

kmin, kmax, dk = 0, 1.4, 0.01
Nmesh = 128
ells = [0,2]

# load and split data
start_time = time.time()
filepath = '/disk11/salam/FirstGenMocks/AbacusSummit/CubicBox/BGS_v2/z0.200/AbacusSummit_base_c000_ph006/BGS_box_ph006.fits'
cat = FITSCatalog(filepath)
cat = prep_fitscat(cat, cosmo=cosmo , LOS=LOS, z=redshift)

filepath_density = '/disk11/salam/FirstGenMocks/AbacusSummit/CubicBox/BGS_v2/z0.200/AbacusSummit_base_c000_ph006/BGS_box_ph006.vtf_mean.fits.gz'
fits = fitsio.FITS(filepath_density)

percent_edges = [0,10,20,30,40,50,60,70,80,90,100]
ptile_split = np.percentile(fits[1]['rho'][:], percent_edges) # gets edges of density bins
np.savetxt("../power_densitybins/percentile_edges.txt", ptile_split, 
           header='Value of density bin edges corresponding to percentage edges' + str(percent_edges))
end_time = time.time()
print('Total time for loading and splitting catalog: %s'%str(timedelta(seconds=end_time-start_time)))

# save power spectrum for whole box
# large number of mu bins allows to integrate over mu in later analysis.
mesh = cat.to_mesh(position='RSDPosition', resampler='tsc', BoxSize=BoxSize, Nmesh=128, compensated=True)
r = FFTPower(mesh, mode='2d', Nmu=51, los=LOS, poles=ells, kmin=kmin, kmax=kmax, dk=dk)
r.save('../power_densitybins/all_bins.json')

# save power spectrum for each density bin
n_ptile = len(ptile_split)-1 # number of bins = number of edges - 1

for i in range(n_ptile):
    t1 = time.time()
    # get indcies of i-th percentile
    insel = fits[1].where('rho>%f && rho<=%f'%(ptile_split[i],ptile_split[i+1]))
    np.save('../power_densitybins/selection_ptile_%d.npy'%i, insel)
    mesh = cat[insel].to_mesh(position='RSDPosition', resampler='tsc', BoxSize=BoxSize, Nmesh=Nmesh, compensated=True)
    r = FFTPower(mesh, mode='2d', Nmu=51, los=LOS, poles=ells, kmin=kmin, kmax=kmax, dk=dk)
    r.save('../power_densitybins/ptile_%d.json'%i)
    t2 = time.time()
    print('Computed power of percentile %d in %s'%(i, str(timedelta(seconds=t2-t1))))
################################################