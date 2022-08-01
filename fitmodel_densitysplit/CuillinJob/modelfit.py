# Fit Kaiser with FoG term as explained in modelfit.ipynb
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import legendre, erf
from datetime import timedelta
import time
from os import mkdir, listdir

import inv_cov_funcs as icf
import cat_power_algos as catpk
import classylss
import fitsio
import zeus 
from nbodykit.lab import *


### MCMC functions###
def logprior(theta, i, kmax):
    ''' The natural logarithm of the prior probability. Assume parameters independent such that log priors add.
    Note that normalization is irrelevant for MCMC.'''
    lp = 0.
    b1, beta, sigma = theta
    
    sigma_min, sigma_max = 1, 5
    sigma_max = 5 if kmax < 0.075 else 60
    lp_sigma = 0. if sigma_min < sigma < sigma_max else -np.inf
        
    b1_min, b1_max = 0, 3
    if i == 0:
        beta_min, beta_max = -3, 3
    else:
        beta_min, beta_max = 0, 3
        
    lp_b1 = 0. if b1_min < b1 < b1_max else -np.inf
    lp_beta = 0. if beta_min < beta < beta_max else -np.inf
    
    return lp_b1 + lp_beta + lp_sigma


def loglike(theta, data_multipoles, k, C_inv):
    '''Return logarithm of likelihood i.e. -0.5*chi2.
    data_multipoles must be an array of shape (len(ells), len(k)). theta is parameter vector: [b1, beta, sigma].'''
    ells = [0,2]
    model_multipoles = np.empty((len(ells), len(k)))

    b1, beta, sigma = theta
    model_multipoles[0] = ( 1/(2*(k*sigma)**5) * (np.sqrt(2*np.pi)*erf(k*sigma/np.sqrt(2))*(3*beta**2+(k*sigma)**4+2*beta*(k*sigma)**2) + 
                                                np.exp(-0.5*(k*sigma)**2)*(-2*beta*(beta+2)*(k*sigma)**3-6*beta**2*k*sigma) ) ) * b1**2 * Plin(k)
    model_multipoles[1] = ( -5/(4*(k*sigma)**7) * (np.sqrt(2*np.pi)*erf(k*sigma/np.sqrt(2))*(-45*beta**2+(k*sigma)**6+(2*beta-3)*(k*sigma)**4+3*(beta-6)*beta*(k*sigma)**2) + 
                                                np.exp(-0.5*(k*sigma)**2)*((4*beta*(beta+2)+6)*(k*sigma)**5+12*beta*(2*beta+3)*(k*sigma)**3+90*beta**2*k*sigma) ) ) * b1**2 * Plin(k)

    D_M = (data_multipoles - model_multipoles).flatten()
    
    return -0.5*D_M@(C_inv @ D_M)


def logpost(theta, i, data_multipoles, k, C_inv):
    '''Returns the logarithm of the posterior. By Bayes' theorem, this is just the sum of the log prior and log likelihood (up 
    to a irrelavant constant).
    Uses values for theta from pre-analysis step to inform prior
    ''' 
    return logprior(theta, i, k[-1]) + loglike(theta, data_multipoles, k, C_inv)
#####################


### Set up MCMC ###
LOS = [0,0,1]
redshift = 0
BoxSize = 2000
cosmo_paras = classylss.load_ini('/home/jwack/main/Planck18_LCDM.ini')
cosmo = cosmology.cosmology.Cosmology.from_dict(cosmo_paras)
Plin = cosmology.LinearPower(cosmo, redshift, transfer='CLASS')
sigma8_lin = Plin.sigma8
sigma8_true = 0.8111 
f0_true = cosmo.scale_independent_growth_rate(redshift)

dk = 0.01
ells = [0,2]

# load computed power spectra to deduce multipoles in each bin and P(k,mu) from data
k_full, shotnoise, n_ptile, Pk_ells_full = icf.load_power_data('/home/jwack/main/fitmodel_densitysplit/', ells, get_data_Pkmus=False)
# for given BoxSize, k is NaN above 0.034
possible_kmax = k_full[k_full<=0.343][1:] # ignore first k bin

kmax_range = possible_kmax #[:21]
Nkmax = len(kmax_range)

b1_fits, beta_fits, sigma_fits, delta_fs8 = np.full((n_ptile, Nkmax), np.nan), np.full((n_ptile, Nkmax), np.nan), np.full((n_ptile, Nkmax), np.nan), np.full((n_ptile, Nkmax), np.nan) 
b1_stds, beta_stds, sigma_stds, delta_fs8_stds = np.full((n_ptile, Nkmax), np.nan), np.full((n_ptile, Nkmax), np.nan), np.full((n_ptile, Nkmax), np.nan), np.full((n_ptile, Nkmax), np.nan)
reduced_chi2 = np.full((n_ptile, Nkmax), np.nan)

nsteps = 2500
ndim = 3
nwalkers = 8 
start_b1 = 0.5 + 1*np.random.random(nwalkers)
start_beta = 0.5 + 1*np.random.random(nwalkers)
start_sigma = 1 + 4*np.random.random(nwalkers) 
start = np.column_stack([start_b1, start_beta, start_sigma])
###################

########################################

### Load results from terminated job ###
root_path = '/home/jwack/main/fitmodel_densitysplit/fit_results/CuillinJob/'
for i in range(n_ptile):
    if 'chains_ptile%d'%i not in listdir(root_path):
        break
    store_path = root_path+'chains_ptile%d/'%i
    
    for j,kmax in enumerate(kmax_range):
        if 'k%d.npy'%j not in listdir(store_path):
            break
        chain = np.load(store_path+'k%d.npy'%j)

        b1_fits[i][j], b1_stds[i][j] = np.mean(chain[:,0]), np.std(chain[:,0]) 
        # parameter space is sym about b1=0 for Kaiser model. To get non negative fs8 assure that b1 and beta have the same sign
        if i == 0:
            b1_fits[i][j] *= -1
        beta_fits[i][j], beta_stds[i][j] = np.mean(chain[:,1]), np.std(chain[:,1])
        delta_fs8[i][j] = 1 - sigma8_lin*(beta_fits[i][j]*b1_fits[i][j])/(f0_true*sigma8_true)
        delta_fs8_stds[i][j] = np.abs(sigma8_lin/(f0_true*sigma8_true)*(beta_stds[i][j]*b1_fits[i][j]+beta_fits[i][j]*b1_stds[i][j]))
        sigma_fits[i][j], sigma_stds[i][j] = np.mean(chain[:,2]), np.std(chain[:,2])
        
        mask = np.full(len(k_full), False)
        mask = k_full <= kmax
        mask[0] = False 
        k_sliced = k_full[mask]
        Pk_ells_i = Pk_ells_full[:,:,mask][i]
        cov_mat = np.loadtxt('/home/jwack/main/bruteforce_covmat/covariance_matricies/cov_ptile_%d.txt'%i)
        C_inv = icf.mock_cov_mat_inv(cov_mat, k_full, kmax)
        reduced_chi2[i][j] = -2*loglike([b1_fits[i][j], beta_fits[i][j], sigma_fits[i][j]], Pk_ells_i, k_sliced, C_inv) / (len(ells)*len(k_sliced)-ndim)
        
    print('loaded in fits of ptile %d'%i)
########################################

########################################


### Run MCMC ###
root_path = '/home/jwack/main/fitmodel_densitysplit/fit_results/CuillinJob/'
print("Fitting up to kmax=%.3f"%kmax_range[-1])

for i in range(n_ptile):
    store_path = root_path+'chains_ptile%d/'%i
    if 'chains_ptile%d'%i not in listdir(root_path):
        mkdir(store_path)
    cov_mat = np.loadtxt('/home/jwack/main/bruteforce_covmat/covariance_matricies/cov_ptile_%d.txt'%i)
    t1 = time.time()
    for j,kmax in enumerate(kmax_range):
        if 'k%d.npy'%j in listdir(store_path):
            continue
        # slice up to increasingly large kmax and find delta_fs8 for each bin
        mask = np.full(len(k_full), False)
        mask = k_full <= kmax
        mask[0] = False 
        k_sliced = k_full[mask]
        Pk_ells_i = Pk_ells_full[:,:,mask][i]
        C_inv = icf.mock_cov_mat_inv(cov_mat, k_full, kmax)
        
        sampler = zeus.EnsembleSampler(nwalkers, ndim, logpost, maxiter=1e5, verbose=False, args=[i, Pk_ells_i, k_sliced, C_inv]) 
        sampler.run_mcmc(start, nsteps)
        
        chain = sampler.get_chain(flat=True, discard=nsteps//2)
        np.save(store_path+'k%d'%j, chain)
        
        b1_fits[i][j], b1_stds[i][j] = np.mean(chain[:,0]), np.std(chain[:,0]) 
        # parameter space is sym about b1=0 for Kaiser model. To get non negative fs8 assure that b1 and beta have the same sign
        if i == 0:
            b1_fits[i][j] *= -1
        beta_fits[i][j], beta_stds[i][j] = np.mean(chain[:,1]), np.std(chain[:,1])
        delta_fs8[i][j] = 1 - sigma8_lin*(beta_fits[i][j]*b1_fits[i][j])/(f0_true*sigma8_true)
        delta_fs8_stds[i][j] = np.abs(sigma8_lin/(f0_true*sigma8_true)*(beta_stds[i][j]*b1_fits[i][j]+beta_fits[i][j]*b1_stds[i][j]))
        sigma_fits[i][j], sigma_stds[i][j] = np.mean(chain[:,2]), np.std(chain[:,2])
        reduced_chi2[i][j] = -2*loglike([b1_fits[i][j], beta_fits[i][j], sigma_fits[i][j]], Pk_ells_i, k_sliced, C_inv) / (len(ells)*len(k_sliced)-ndim)

            
    t2 = time.time()
    print('Fitted %d-th percentile in %s'%(i,str(timedelta(seconds=t2-t1))))
################


### Store fit result ###
np.savetxt(root_path+'b1_fits.txt', b1_fits)
np.savetxt(root_path+'b1_stds.txt', b1_stds)

np.savetxt(root_path+'beta_fits.txt', beta_fits)
np.savetxt(root_path+'beta_stds.txt', beta_stds)

np.savetxt(root_path+'delta_fs8.txt', delta_fs8)
np.savetxt(root_path+'delta_fs8_stds.txt', delta_fs8_stds)

np.savetxt(root_path+'sigma_fits.txt', sigma_fits)
np.savetxt(root_path+'sigma_stds.txt', sigma_stds)

np.savetxt(root_path+'reduced_chi2.txt', reduced_chi2)
########################


### Make fs8 plot ###
fig = plt.figure(figsize=(20,8))

for i in range(n_ptile):
    plt.plot(kmax_range, delta_fs8[i], label='%d-th'%i)
    plt.fill_between(kmax_range, delta_fs8[i]-delta_fs8_stds[i,:], delta_fs8[i]+delta_fs8_stds[i,:], alpha=0.1)
    
# add 10% diviation lines
plt.hlines([-0.1, 0.1], kmax_range[0], kmax_range[-1], linestyle='dotted', color='k', linewidth=2)

plt.ylim([-0.3, 1.2])

plt.title(r'relative difference to true $f_0*\sigma_8$')
plt.xlabel(r'$k_{max}$ [$h \ \mathrm{Mpc}^{-1}$]')
plt.ylabel(r'$1 - \sigma_8^{lin}*\beta*b1/(f_0\sigma_8)^{true}$')

handles, labels = plt.gca().get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=n_ptile)

fig.savefig("/home/jwack/main/fitmodel_densitysplit/plots/Kaiserdamped_fullk_largesigma_dfs8_vs_kmax.pdf")
#####################


### Make fit plot ###
fig = plt.figure(figsize=(26,18))

ax_b1 = plt.subplot(2,3,1)
ax_beta = plt.subplot(2,3,2)
ax_sigma = plt.subplot(2,3,3)
ax_chi2 = plt.subplot(2,3,(4,6))

for i in range(n_ptile):
    ax_b1.plot(kmax_range, b1_fits[i], label='%d-th'%i)
    ax_b1.fill_between(kmax_range, b1_fits[i]-b1_stds[i], b1_fits[i]+b1_stds[i], alpha=0.1)
    
    ax_beta.plot(kmax_range, beta_fits[i], label='%d-th'%i)
    ax_beta.fill_between(kmax_range, beta_fits[i]-beta_stds[i], beta_fits[i]+beta_stds[i], alpha=0.1)
    
    ax_sigma.plot(kmax_range, sigma_fits[i], label='%d-th'%i)
    ax_sigma.fill_between(kmax_range, sigma_fits[i]-sigma_stds[i], sigma_fits[i]+sigma_stds[i], alpha=0.1)
    
    ax_chi2.plot(kmax_range[1:], reduced_chi2[i][1:], label='%d-th'%i) # first element negative, s.t. not shown on log scale
  
ax_b1.set_title('b1 fit')
ax_b1.set_xlabel(r'$k_{max}$ [$h \ \mathrm{Mpc}^{-1}$]')
ax_b1.set_ylabel(r'$b1$')

ax_beta.set_title(r'$\beta$ fit')
ax_beta.set_xlabel(r'$k_{max}$ [$h \ \mathrm{Mpc}^{-1}$]')
ax_beta.set_ylabel(r'$\beta$')

ax_sigma.set_title(r'$\sigma$ fit')
ax_sigma.set_xlabel(r'$k_{max}$ [$h \ \mathrm{Mpc}^{-1}$]')
ax_sigma.set_ylabel(r'$\sigma$')

ax_chi2.set_title(r'reduced $\chi^2$')
ax_chi2.set_yscale('log')
ax_chi2.set_xlabel(r'$k_{max}$ [$h \ \mathrm{Mpc}^{-1}$]')
ax_chi2.set_ylabel(r'$\chi^2 / dof$')

handles, labels = plt.gca().get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, +0.05), ncol=n_ptile)

fig.savefig("/home/jwack/main/fitmodel_densitysplit/plots/Kaiserdamped_fullk_largesigma_fits.pdf")
#####################