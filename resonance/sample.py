# Monte Carlo inversion of b(omega)-TR-TE encoded MRI data into
# nonparameteric D(omega)-R1-R2 distributions
#
# Code to be shared with the paper
# Massively multidimensional relaxation-diffusion correlation MRI
# O Narvaez, L Svenningsson, M Yon, A Sierra, and D Topgaard
# Front Phys special issue https://www.frontiersin.org/research-topics/21291/capturing-biological-complexity-and-heterogeneity-using-multidimensional-mri
#
# Adapted from https://github.com/daniel-topgaard/md-dmri
# Tested for Python 3.7.3

import os
import time
import math
import random
from pathlib import Path
import copy
import multiprocessing as mp
from pprint import pprint

import torch
import numpy as np #https://numpy.org
import scipy.io as sio #https://www.scipy.org
from scipy.optimize import nnls
import nibabel as nib #https://nipy.org
import matplotlib.pyplot as plt

from config import max_components

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Inversion options
do_multicpu = 0
range_bs = range(0,100)
# range_bs = np.flipud(range_bs)

method_name = 'dtor1r2d'
dist_varnams = ('w','dpar','dperp','theta','phi','d0','rpar','rperp','r1','r2')

opt = {
    'method': method_name,
    'dmin': 5e-12,
    'dmax': 5e-9,
    'rmin': 0.1,
    'rmax': 1e5,
    'r1min': 0.2,
    'r1max': 20,
    'r2min': 2,
    'r2max': 200,
    'n_in': 200,
    'n_out': 10,
    'n_prol': 20,
    'n_darwin': 20,
    'dfuzz': .1,
    'ofuzz': .1*2*np.pi,
    'rfuzz': .1,
    'r1fuzz': .1,
    'r2fuzz': .1} # Saved later for book keeping

# Define paths
script_path = os.path.dirname(os.path.realpath(__file__))
data_path = os.path.join(script_path, 'indata')
data_name = 'data'

# data_fn = os.path.join(data_path, data_name+'.nii.gz')
# mask_fn = os.path.join(data_path, data_name+'_mask_small.nii.gz')
# mask_fn = os.path.join(data_path, data_name+'_mask.nii.gz')

# # Experimental parameters are loaded from _xps.mat
xps_fn = os.path.join(data_path, data_name+'_xps.mat')


# Extract relevant experimental parameters
xps_dict = sio.loadmat(xps_fn)
xps_struct = xps_dict['xps']
btomega = xps_struct[0,0]['btomega']
domega = xps_struct[0,0]['domega']
domegav = np.tile(domega[0], (1,btomega.shape[1]//6))
domegav[0,0] = 0
omega = torch.Tensor(np.cumsum(domegav,axis=1))

xps = {
    'btomega': btomega,
    'omega'  : omega,
    'tr'     : xps_struct[0,0]['tr'],
    'te'     : xps_struct[0,0]['te']}

sqrt2 = np.sqrt(2)

# Define functions
# Functions to generate and manipulate DTD components
def notfin2zero(x):
    x[np.isinf(x)] = 0
    x[np.isneginf(x)] = 0
    x[np.isnan(x)] = 0

    return x


def dist_rand(n_components):
    # Distribution with random parameters within the min/max boundaries 
    dist = {
    'w'    : np.zeros((n_components,1)),
    'dpar' : opt['dmin']*np.power(opt['dmax']/opt['dmin'],np.random.rand(n_components,1)),
    'dperp': opt['dmin']*np.power(opt['dmax']/opt['dmin'],np.random.rand(n_components,1)),
    'theta': np.arccos(2*np.random.rand(n_components,1)-1),
    'phi'  : 2*np.pi*np.random.rand(n_components,1),
    'd0'   : opt['dmin']*np.power(opt['dmax']/opt['dmin'],np.random.rand(n_components,1)),
    'rpar' : opt['rmin']*np.power(opt['rmax']/opt['rmin'],np.random.rand(n_components,1)),
    'rperp': opt['rmin']*np.power(opt['rmax']/opt['rmin'],np.random.rand(n_components,1)),
    'r1'   : opt['r1min']*np.power(opt['r1max']/opt['r1min'],np.random.rand(n_components,1)),
    'r2'   : opt['r2min']*np.power(opt['r2max']/opt['r2min'],np.random.rand(n_components,1))}
 
    dpar = copy.deepcopy(dist['dpar'])
    dperp = copy.deepcopy(dist['dperp'])
    d0 = copy.deepcopy(dist['d0'])
    
    ind = d0<dpar
    d0[ind] = dpar[ind]
    ind = d0<dperp
    d0[ind] = dperp[ind]    

    for varnam in ('dpar','dperp','d0'):
        dist.update( {varnam:eval(varnam)} )
           
    return dist

    
# Functions for fitting
def xpsdist2dtomega(xps,dist):
    """

    Vectorized computation of the signal dtomega for a population of 200 solution candidates

    """
    
    # Convert experimental and distribution parameters to D(omega) array 
    Nomega = xps['omega'].shape[1] # this is an integer of 50
    Ncomp = dist['dpar'].shape[0]

    omega_tile = torch.tile(omega, (Ncomp,1))
    dist_tile = {}
    for varnam in dist_varnams: # 'w', 'dpar', 'dperp', 'theta', 'phi', 'd0', 'rpar', 'rperp', 'r1', 'r2'
        val = copy.deepcopy(dist[varnam])       
        dist_tile.update( {varnam:torch.tile(val.reshape(Ncomp,1), (1,Nomega))} )    
   
    dparo  = dist_tile['d0'] - (dist_tile['d0'] - dist_tile['dpar' ])/(1 + torch.pow(omega_tile,2)/torch.pow(dist_tile['rpar' ],2))
    dperpo = dist_tile['d0'] - (dist_tile['d0'] - dist_tile['dperp'])/(1 + torch.pow(omega_tile,2)/torch.pow(dist_tile['rperp'],2))
    dparo  = notfin2zero(dparo)
    dperpo = notfin2zero(dperpo)
    dperpo[dperpo==0] = opt['dmin']
    dparo[dparo==0]   = opt['dmin']
    
    diso = (dparo + 2*dperpo)/3
    ddelta = notfin2zero((dparo-dperpo)/(3*diso))

    xcos = torch.cos(dist_tile['phi'])*torch.sin(dist_tile['theta'])
    ycos = torch.sin(dist_tile['phi'])*torch.sin(dist_tile['theta'])
    zcos = torch.cos(dist_tile['theta'])

    dxx = diso*(1 + ddelta*(3*xcos*xcos - 1))
    dxy = diso*(0 + ddelta*(3*xcos*ycos - 0))
    dxz = diso*(0 + ddelta*(3*xcos*zcos - 0))
    dyy = diso*(1 + ddelta*(3*ycos*ycos - 1))
    dyz = diso*(0 + ddelta*(3*ycos*zcos - 0))
    dzz = diso*(1 + ddelta*(3*zcos*zcos - 1))

    dtomega = torch.cat((dxx,dyy,dzz,sqrt2*dxy,sqrt2*dxz,sqrt2*dyz),axis=1)

    return dtomega
    
def kernel(xps,dist):
    """

    Parameters to be drawn
    ---------- 
    dist:
        'w', 'dpar', 'dperp', 'theta', 'phi', 'd0', 'rpar', 'rperp', 'r1', 'r2'
    
    dist['w']: np.array (n_components,1). w is proportional to fitness, allocated at nnlsq
    

    Fix experimental parameters
    -------
    xps['btomega'] : np.array (809,300) 
    xps['omega']: np.array (1,50) # Is not justed. only shape of array is used in xpsdist2omega
    xps['tr']: np.array (809,1)
    xps['te']: np.array (809,1)
    
    
    Draw data
    ---------
    Draw parameters as done in dist_rand()
    Draw w: max 10 components. Draw number of components as uniform(0,10) or Dirichlet dist.
    
    Alternatives to draw batch data:
        1. Rewrite xpsdist2omega with additional vectorization
        2. Draw row vector of max_components*batch_size and reshape
        3. Call xpsdist2omega batch_size number of times
    
    Output
    ------
    K: np.array (809, max_components), signal of individual components
    signal_mixture: np.array (809), linear combination of components K w.r.t weights w
    """
    
    
    
    
    # The Voxels [X,Y,Z,M] is multiplied by the binary mask matrix and ordered as a row of length N
    # The signal input for each voxel is ordered by [N, M] where N is number of voxeles and M is number of sample sets.
    # The Voxel vector N can probably be used with mask to rectreate the coxel image.
    # The final output is saved in mfs.npz files, though a mat file option would be useful here.
    # Generate inversion kernel from experimental and distribution parameters
    # xps is the measurment input. similar to "t" in exp(-k*t). These are called by name and have length N
    # dist is the MC sampled variables that we want to obtain
    dtomega = xpsdist2dtomega(xps,dist)
    Kd = torch.exp(-torch.matmul(xps['btomega'], dtomega.T))
    Kr1 = 1-torch.exp(-torch.matmul(xps['tr'], dist['r1'].T))
    Kr2 = torch.exp(torch.matmul(xps['te'], dist['r2'].T))
    K = Kd*Kr1*Kr2
    

    # signal_mixture = K @ w
    
    return K


def draw(device, batch_size, weighting_scheme):

    dist = dist_rand(n_components=max_components*batch_size)

    if weighting_scheme == 'uniform':
        components = torch.randint(low=1, high=max_components, size=(batch_size,))
        w = torch.rand((batch_size, max_components))

        for i, c in enumerate(components):
            w[i, c:] = 0
            w = (w/w.sum(axis=1).unsqueeze(1)).to(device)

    elif weighting_scheme == 'dirichlet':
        scale_factors = random.rand(batch_size)*1 + 1/100
        _w = np.array([random.dirichlet(np.ones(max_components)*s, size = 1) for s in scale_factors])
        w = torch.Tensor(_w).squeeze().to(device)
    w = w.unsqueeze(1)
    dist['w'] = w
    
    # Cast to torch tensors
    for k in xps.keys():
        xps[k] = torch.Tensor(xps[k])
    for k in dist.keys():
        dist[k] = torch.Tensor(dist[k])

    K = kernel(xps, dist)

    # Format into batches
    K = torch.reshape(K, (batch_size, K.shape[0], max_components))

    for k in dist.keys():
        dist[k] = torch.reshape(dist[k], (batch_size, 1, max_components))

    # Take signal mixture as linear cominbation w.r.t weights w
    signal_mixture = (K * dist['w']).sum(axis=2)
    signal_mixture = signal_mixture.to(device)

    return signal_mixture, dist

def linear_combination(dist, w):
    pass


if __name__ == '__main__':
    draw('cpu', 128, 'uniform')