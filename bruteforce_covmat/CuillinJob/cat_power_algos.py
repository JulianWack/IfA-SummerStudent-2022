from nbodykit.lab import *
from nbodykit import setup_logging, style
import numpy as np


def vec_projection(vector, direction):
    '''Projects vector on direction vector (can be non-normalised).'''
    direction = numpy.asarray(direction, dtype='f8')
    direction = direction / (direction ** 2).sum() ** 0.5
    projection = (vector * direction).sum(axis=-1)
    projection = projection[:, None] * direction[None, :]

    return projection


def make_cat(file_path, cosmo=cosmology.Planck15, LOS=[0,0,0], z=0):
    '''Creates ArrayCatalog from input file. Final catalog contains Position, Velocity, and RSDPosition as columns. 
    RSD added according to Getting Started->Discrete data catalogs->Common data operations->Adding RSD. 
    To aviod RSD computation, leave LOS as default
    Input types: 
        - file_path: must lead to binary file with cartesian position and velocity as first column
        - LOS: list of 3 floats such as [1,0,0]
        - redshift: float  
    '''
    print(f"Loading {file_path}", flush=True)
    inp = np.loadtxt(file_path) # 2D np array, each row: ['x','y','z','vx','vy','vz','sigV','Mhalo','flag', 'haloindex']
    pos = inp[:,0:3]
    vel = inp[:,3:6]
    
    data = numpy.empty(inp.shape[0], dtype=[('Position', ('f8', 3)), ('Velocity', ('f8', 3))])
    data['Position'] = pos
    data['Velocity'] = vel

    cat = ArrayCatalog(data)
    
    if LOS != [0,0,0]:
        # RSD position = position + velocity offset along LOS
        rsd_factor = (1+z) / (100 * cosmo.efunc(z))
        cat['RSDPosition'] = cat['Position'] + rsd_factor * vec_projection(cat['Velocity'], LOS) # identical to rsd_factor*cat['Velocity']*LOS
    
    return cat


def prep_fitscat(cat, cosmo, LOS=[0,0,0], z=0):
    '''Adds Position, Velocity, and RSDPosition to FITSCatalog.'''
    cat['Position'] = np.stack([cat['x'].compute(), cat['y'].compute(), cat['z'].compute()], axis=1)
    cat['Velocity'] = np.stack([cat['vx'].compute(), cat['vy'].compute(), cat['vz'].compute()], axis=1)
    
    if LOS != [0,0,0]:
        # RSD position = position + velocity offset along LOS
        rsd_factor = (1+z) / (100 * cosmo.efunc(z))
        cat['RSDPosition'] = cat['Position'] + rsd_factor * vec_projection(cat['Velocity'], LOS)
        
    return cat


def get_binned_Pk(mesh_in, kbin=[0.05,2,20,'lin'], outfile=''):
    '''Computes the binned 1D power spectrum. 
    Returns two 1D np arrays containing k, Pk respectively.'''
    kbin_edges, kbin_mid = bin_k(kbin)
    r = FFTPower(mesh_in, mode='1d', dk=0.005, kmin=kbin[0])
    Pk = r.power
    pkbin = bin_pk(Pk, kbin_edges)
    
    if(outfile!=''):
        out = np.column_stack((kbin_mid,pkbin))
        np.savetxt(outfile,out,header='k, pk')
        print('written: ',outfile)
    
    return kbin_mid, pkbin


def get_binned_Pkmu(mesh_in, Nmu, LOS, kbin=[0.05,2,20,'lin'], outfile=''):
    '''Computes the binned 2D power spectrum.
    Returns 1D np array containing binned k and 2D np array containing Pk for every value of mu in its columns.
    Also returns values of mu considered.'''
    kbin_edges, kbin_mid = bin_k(kbin)
    r = FFTPower(mesh_in, mode='2d', Nmu=Nmu, los=LOS, dk=0.005, kmin=kbin[0])
    Pkmu = r.power # dims (k, mu)
    
    pksbin = np.empty((len(kbin_mid), Nmu))
    for i in range(Nmu):
        Pk = Pkmu[:,i]
        pksbin[:,i] = bin_pk(Pk, kbin_edges)
    
    if(outfile!=''):
        out = np.column_stack((kbin_mid, pksbin))
        np.savetxt(outfile,out,header='k, pkmu for mu:' + str(Pkmu.coords['mu']))
        print('written: ',outfile)
        
    return kbin_mid, pksbin, Pkmu.coords['mu']

    
def bin_k(kbin):
    '''Perform binning of k. Return edges of bins and their midpoint.'''
    if(kbin[3]=='lin'):
        kbin_ed = np.linspace(kbin[0],kbin[1],kbin[2])
    elif(kbin[3]=='log'):
        kbin_ed = np.power(10,np.linspace(np.log10(kbin[0]),np.log10(kbin[1]),kbin[2]))
        
    kbin_mid = 0.5*(kbin_ed[1:]+kbin_ed[:-1])
    
    return kbin_ed, kbin_mid
    
    
def bin_pk(Pk, kbin_ed):
    '''See binning_explantion notebook for aiding understanding.'''
    pkin = np.column_stack([Pk['k'], Pk['power'].real - Pk.attrs['shotnoise']])
    
    mode,hh = np.histogram(pkin[:,0],bins=kbin_ed)
    pkbin,hh = np.histogram(pkin[:,0],bins=kbin_ed,weights=pkin[:,1])

    # perform pkbin=pkbin/mode but aviod divide by 0 error
    pkbin = np.divide(pkbin, mode, out=np.zeros(pkbin.shape), where=mode!=0)
        
    return pkbin