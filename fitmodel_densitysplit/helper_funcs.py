# Helper functions to load power spectrum data, compute power spectrum multipoles numerically, and compute (and slice) the inverse covariance matrix either analytically or via brute force result.
import numpy as np
from scipy.special import legendre
from scipy.integrate import simpson
from scipy.linalg import LinAlgError, inv, pinv
import scipy.sparse as ss

from nbodykit.lab import FFTPower



### Finding data P(k,mu) and multipoles ###
def load_power_data(base_path, ells, get_data_Pkmus=True):
    '''Loads in pre-computed power spectra for all density bins to find multipoles for each bin.
    The base path specifies the location of the folder containing the density bin edges and the power spectra for each bin.
    Optionally returns data based P(k,mu) and associated mu bins according to
    eq 24 of Grieb et al. 2016: https://arxiv.org/pdf/1509.04293.pdf.
    
    We only need to subtract the shotnoise from the monopole due to the orthogonality of the Legendre polynomials.
    As the shotnoise is constant wrt to $\mu$, it is proportional to the 0th Legendre polynomial s.t. only the monopole is affected.
    '''
    ptile_split = np.loadtxt(base_path+'power_densitybins/percentile_edges.txt')
    n_ptile = len(ptile_split)-1 # number of bins = number of edges - 1
    
    # need number of k and mu bins to initilize arrays.
    aux = FFTPower.load(base_path+'power_densitybins/ptile_0.json')
    k = aux.poles['k']
    mus = aux.power.coords['mu']
    shotnoise = aux.attrs['shotnoise']
    
    Pk_ells = np.empty((n_ptile, len(ells), len(k))) # for each denisty bin store all multipoles
    Pkmus = np.empty((n_ptile, len(k), len(mus))) # for each density bin store 2D power spectrum 

    for i in range(n_ptile):
        r = FFTPower.load(base_path+'power_densitybins/ptile_%d.json'%i)
        poles = r.poles 

        Pkmu_nl = np.zeros((len(k), len(mus)))
        for j,ell in enumerate(ells):
            Pk_ell = poles['power_%d' %ell].real
            if ell == 0: 
                Pk_ell = Pk_ell - poles.attrs['shotnoise']

            Pk_ells[i][j] = Pk_ell
            Pkmu_nl += np.outer(Pk_ell, legendre(ell)(mus))

        Pkmus[i] = Pkmu_nl
    
    if get_data_Pkmus:
        return k, shotnoise, n_ptile, Pk_ells, mus, Pkmus 
    else:
        return k, shotnoise, n_ptile, Pk_ells

    

### Finding model P(k,mu) and multipoles ###
def kaiser_pkmu(k, mu, b1, beta, Plin):
    """Returns power spectrum in redshift space, following Kaiser's linear results."""
    return (1 + beta*mu**2)**2 *b1**2 * Plin(k)


def fog_kaiser_pkmu(k, mu, b1, beta, sigma, Plin):
    '''Introduces phenomenological Finger of God term to Kaiser model.
    Common forms of this correction term are Gaussian or Lorentzian, both bring one new free parameter sigma to the model.
    See eq 7.2 - 7.4 in Hamilton https://arxiv.org/pdf/astro-ph/9708102.pdf.'''
    return np.exp(-0.5*(sigma*k*mu)**2)*kaiser_pkmu(k, mu, b1, beta, Plin)


def make_Pkmu(k, b1, beta, sigma, Plin):
    '''Make 2D array containing the model P(k,mu) with rows iterating k bins and columns iterating mu bins.
    Later seek to integrate up each row i.e. integrate over mu. Nmu defines the discretiation.
    Plin is function to compute the linear matter power spectrum such as nbodykit.cosmology.power.linear.LinearPower.'''
    Nmu = 51
    mus = np.linspace(-1,1,Nmu)
    Pkmu = np.empty((len(k), Nmu))
    
    for i,mu in enumerate(mus):
        Pkmu[:,i] = fog_kaiser_pkmu(k,mu,b1,beta,sigma,Plin)
        
    return Pkmu, mus


def model_multipole(k, ell, b1, beta, sigma, Plin):
    '''Computes ell-th multipole of damped Kaiser model P(k,mu) by projecting on Legendre polynominal.'''
    Pkmu, mus = make_Pkmu(k, b1, beta, sigma, Plin)
    L_ell = legendre(ell)(mus)
    integrand = L_ell*Pkmu # each column gets multiplied by value of Legendre poly at assocaiated mu

    return (2*ell+1)/2 * simpson(integrand, mus) # integrate up all rows separtely

    

### Finding inverse of analytic covariance matrix ###
def per_mode_cov(k, l1, l2, BoxSize, shotnoise, dk, Pkmu, mus):
    '''Construct per mode covariance. See eq 15, 16 (for factor f) of Grieb et al. 2016: https://arxiv.org/pdf/1509.04293.pdf
    Pkmu is a 2D array of shape (# k bins, # mu bins). The above paper presents two possible choices in eq 23, 24 with the former 
    being based on the model and later based on the data.'''
    V = BoxSize**3
    V_k = 4/3*np.pi*((k+dk/2)**3 - (k-dk/2)**3)
    f = 2*(2*np.pi)**4 / V_k**2 * k**2 * dk
    
    L_l1, L_l2 = legendre(l1)(mus), legendre(l2)(mus)
    integrand = (Pkmu + shotnoise)**2 * L_l1*L_l2
    
    return f*(2*l1+1)**2 * (2*l2+1)**2 / V * simpson(integrand, mus) # 1D array containing per mode cov for each k bin


def gaussian_cov_mat_inv(k, ells, BoxSize, shotnoise, dk, Pkmu, mus):
    '''See Grieb et al. 2016: https://arxiv.org/pdf/1509.04293.pdf or Fitting_b1_different_kmax.ipynb for explanation of structure of 
    covariance matrix. Uses sparse matricies for fast inversion.
    scipy.sparse.bmat allows to combine matricies by passing structure of larger matrix in terms of submatricies.'''
    # initialize array accepting matricies as elements and fill with diagonal C_l1,l2 matricies
    C = np.empty((len(ells), len(ells)), dtype='object')
    for i,l1 in enumerate(ells):
        for j,l2 in list(enumerate(ells))[i:]:
            C[i][j] = ss.diags(per_mode_cov(k,l1,l2,BoxSize,shotnoise,dk,Pkmu,mus))
            if j!=i:
                C[j][i] = C[i][j]
                
    cov_mat = ss.bmat(C).tocsc() # convert to efficient scipy matrix format
    
    # deal with inverting signular matrix
    try: 
        inverse = ss.linalg.inv(cov_mat).toarray()
    except LinAlgError or RuntimeError:
        inverse = pinv(cov_mat.toarray())
        
    return inverse



### Finding inverse of mock covariance matrix ###
def slice_covmat(cov_mat, k_full, kmax):
    '''Slices down full covariance matrix to k<=kmax<2 under the assumption that ells = [0,2].
    The first k bin will be also be sliced away as the quadrupole vanishes in that bin, leading to a divide by 0 error 
    when computing the correlation matrix.'''    
    k_slice = (k_full <= kmax)
    k_slice[0] = False # remove problematic first bin for fitting analysis
    new_size = int(2*np.sum(k_slice)) # slices matrix will be of shape (new_size, new_size) 
    mask = np.concatenate((k_slice, k_slice)) # when ells = [0,2,4] concatenate 3 times
    mat_mask = np.outer(mask, mask) 
    
    return np.reshape(cov_mat[mat_mask], (new_size, new_size))


def mock_cov_mat_inv(cov_mat, k, kmax):
    '''Returns inverse of covariance matrix determined by brute force from many N body realizations. 
    Covariance matricies for every density bin are stored on disk and computed for k<2. For the analysis at hand
    we are often interested in a smaller kmax such that we need the slice the covariance matrix down first.
    Note that the number of k bins from the mock boxes used for the covariance matrix must be the same as a the number of k bins
    in the measurment box .i.e need to pass the same k_min, k_max, dk to FFTPower. The full set of the so obtained k values is k_full.
    kmax is the maximum k until we seek to fit.
    
    Further note that the boxes used to get the covariance matrix have a BoxSize which is 4 times smaller than the BoxSize 
    of the measurment simulation (500 vs 2000 Mpc/h). To account for this, divide covariance matrix by 4^3.'''
    
    sliced_covmat = slice_covmat(cov_mat, k, kmax) / 4**3
    
    # deal with inverting signular matrix
    try: 
        inverse = inv(sliced_covmat)
    except LinAlgError or RuntimeError:
        inverse = pinv(sliced_covmat)
        
    return inverse