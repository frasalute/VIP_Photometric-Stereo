# -*- coding: utf-8 -*-
"""
Project: Vision and Image Processing

File: ps_utils.py

Description: Collections of routines for Photomoetric Stereo 
Processing.

Main content is:

* Two methods of integration of a normal field to a depth function
  by solving a Poisson equation. They are both written for normal field integration-

  - The first, unbiased_integrate(), unbiased, implements a Poisson solver on an 
    irregular domain. Rather standard approach, with efficient implementation ported
    from Yvain's MatLab one.
  - The second, simchony_integrate() implements the Simchony et al. method for 
    integration of a normal field.
  
* A specialised implementation of Fishler and Bolles' Random Sampling 
   Consensus -- RANSAC -- for estimating a 3D vector from numerical measurements
   and a light matrix

* A numerical diffusion for normal fields as a simple exercises in Riemannian
    geometry and diffusion:-)

* A 3D rendering display function of a surface patch via Mayavi.


copyright 2015-2020 François Lauze, University of Copenhagen

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

import sys
import warnings 
from mpl_toolkits.mplot3d import Axes3D


surf_backend = ""
try:
    from mayavi import mlab
    surf_backend = "mayavi"
except:
    surf_backend = "mpl3d"

import matplotlib.pyplot  as plt
from matplotlib.colors import LightSource
import numpy as np
from scipy import fftpack as fft
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
import random




def ransac_3dvector(data, threshold, max_data_tries=100, max_iters=1000, 
                    p=0.9, det_threshold=1e-1, verbose=2):
    """
    A RANSAC implementation for fitting a vector in a linear model I = s.m
    with I a scalar, s a 3D vector and m the vector to be fitted. 
    For us, s should represent a directional light source, I, observed 
    intensity and m = rho*n = albedo x normal
    Parameters:
    -----------
    data: tuple-like
        pair (I, S) with S the (K, 3) matrix of 3D vectors and I the 
        K-vector of observations.
    threshold: float
        |I-s.m|^2 <= threshold: inlier, else outlier.
    max_data_tries: int (optional, default=100)
        maximum tries in sampling 3 vectors from S. If repeatably fail 
        to get a 3x3 submatrix with determinant large enough, then report
        an impossible match here - None instead of a (3D vector, inliers, best_fit)
    max_iters: int (optional, default=1000)
        maximum number of iterations, i.e. explore up to max_iters potential models.
    p: float (optional, default=0.9)
        desired propability to al least pick an essentially error free sample
        (z in Fishler & Bolles paper)
    det_threshold: float
        threshold to decide wether a determinant (in absolute value) is large enough.
    verbose: int
        0 = silent, 1 = some comments, >= 2: a lot of things

    Returns:
    --------
    m, inliers, best_fit: tuple (ndarray, int, float) or None
        if successful, m is the estimated 3D vector, inliers a list
        of indices in [0..K-1] of the data points considered as inliers 
        and best_fit the best average fit for the selected model.
        if unsuccessful, returns None

    Reference:
    ----------
    Martin A. Fischler & Robert C. Bolles "Random Sample Consensus: A Paradigm for 
    Model Fitting with Applications to Image Analysis and Automated Cartography". 
    Comm. ACM. 24 (6): 381–395. 1981.
    
    """


    # minimum number of  points to select a model
    global s
    n_model_points = 3
    
    # initialisation of model to None,
    best_m = None
    
    # a count of attempts at selecting a good data set
    trial_count = 0
    
    # score of currently best selected model
    best_score = 0
    
    # best average fit for a model
    best_fit = float("Inf")
    
    # number of trials needed to pick a correct a correct dataset 
    # with probability p. Will be updated during the run (same name 
    # as in Fishler-Bolles.
    k = 1 
    
    I, S = data
    #S = S.T
    S = S.copy().astype(float)
    I = I.copy().astype(float)
    ndata = len(I)
    
    while k > trial_count:
        if verbose >= 2:   
            print("ransac_3dvector(): at trial ",trial_count)

        i = 0
        while i < max_data_tries:
            # select 3 pairs s, I randomly and check whether
            # they allow to compute a proper model: |det(s_i1, s_i2, s_i3| >> 0.
            idx = random.sample(range(ndata), n_model_points)
            if verbose >= 2:
                print("ransac_3dvector(): selected indices = ", idx)
            s = S[idx]
            if abs(np.linalg.det(s))>= det_threshold:
                Is = I[idx]
                break
            i += 1
        if i == max_data_tries:
            if verbose >= 1:
                print("ransac_3dvector(): no dataset found, degenerate model?")
            return None
        
        # here, we can evaluate a candidate model
        m = np.linalg.inv(s) @ Is
        """
        if np.any(np.isnan(m)) or np.any(np.isinf(m)) or (np.linalg.norm(m) < 1e-10):
            print('Estimated vector:', m)
            print('Data points used:', Is)
            print('Light sub-matrix determinant', np.linalg.det(s))
            raise ValueError('NaN of Infinite or null vector encountered.')
        """

        if verbose >= 2:
            print("ransac_3dvector(): estimated model", m)
        # then its inliers. For that we fist compute fitting values
        fit = np.abs(I - S @ m)
        inliers = np.where(fit <= threshold)[0]
        n_inliers = len(inliers)
        if verbose >= 2:
            print("ransac_3dvector(): number of inliers for this model", n_inliers)

        
        if n_inliers > best_score:
            best_score = n_inliers
            best_inliers = inliers
            # we reevaluate m on the inliers' subset
            s = S[inliers]
            Is = I[inliers]
            best_m = np.linalg.pinv(s) @ Is
            # This should match Yvain's version?
            # best_m = m.copy()
            best_fit = np.mean(np.abs(Is - s@best_m))
            if verbose >= 2:
                print("ransac_3dvector(), updating best model to", best_m)
            
            frac_inliers = n_inliers / ndata
            # p_outliers is the 1 - b of Fishler-Bolles
            # the number of needed points to select a model is 3
            p_outliers = 1 - frac_inliers**n_model_points
            # preveny NaN/Inf  in estimation of k
            eps = np.spacing(p_outliers)
            p_outliers = min(1-eps, max(eps, p_outliers))
            k = np.log(1-p)/np.log(p_outliers)
            if verbose >= 2:
                print("ransac_3dvector(): estimate of runs to select"
                      " enough inliers with probability {0}: {1}".format(p, k))

        trial_count += 1
        if trial_count > max_iters:
            if verbose:
                print("ransac_3dvector(): reached maximum number of trials.")
            break

    if best_m is None:
        if verbose: 
            print("ransac_3dvector(): unable to find a good enough solution.")
        return None
    else:
        if verbose >= 2:
            print("ransac_3dvector(): returning after {0} iterations.".format(trial_count))
        return best_m, best_inliers, best_fit


def cdx(f):
    """
    central differences for f in x direction
    """
    m = f.shape[0]
    west = [0] + list(range(m-1))
    east = list(range(1,m)) + [m-1]
    return 0.5*(f[east,:] - f[west,:])
    
def cdy(f):
    """
    central differences for f in y direction
    """
    n = f.shape[1]
    south = [0] + list(range(n-1))
    north = list(range(1,n)) + [n-1]
    return 0.5*(f[:,north] - f[:,south])
    

def tolist(A):
    """
    Linearize array to a 1D list
    """
    return list(np.reshape(A, A.size))
    


def make_bc_data(mask):
    """
    Create the data structure used to enforce some  null Neumann BC condition on
    some PDEs used in my Photometric Stereo Experiments.  
    Argument:
    ---------
    mask: numpy array
        a binary mask of size (m,n). 
    Returns:
    --------
        west, north, east, south, inside, n_pixels with
        west[i]  index of point at the "west"  of mask[inside[0][i],inside[1][i]]
        north[i] index of point at the "north" of mask[inside[0][i],inside[1][i]]
        east[i]  index of point at the "east"  of mask[inside[0][i],inside[1][i]]
        south[i] index of point at the "south" of mask[inside[0][i],inside[1][i]]
        inside: linear indices of points inside the mask
        n_pixels: number of inside / in domain pixels
    """
    m,n = mask.shape
    inside = np.where(mask)
    x, y = inside
    n_pixels = len(x)
    m2i = -np.ones(mask.shape)
    # m2i[i,j] = -1 if (i,j) not in domain, index of (i,j) else.
    m2i[(x,y)] = range(n_pixels)
    west  = np.zeros(n_pixels, dtype=int)
    north = np.zeros(n_pixels, dtype=int)
    east  = np.zeros(n_pixels, dtype=int)
    south = np.zeros(n_pixels, dtype=int)

   
    for i in range(n_pixels):
        xi = x[i]
        yi = y[i]
        wi = x[i] - 1
        ni = y[i] + 1
        ei = x[i] + 1
        si = y[i] - 1

        west[i]  = m2i[wi,yi] if (wi > 0) and (mask[wi, yi] > 0) else i
        north[i] = m2i[xi,ni] if (ni < n) and (mask[xi, ni] > 0) else i
        east[i]  = m2i[ei,yi] if (ei < m) and (mask[ei, yi] > 0) else i
        south[i] = m2i[xi,si] if (si > 0) and (mask[xi, si] > 0) else i

    return west, north, east, south, inside, n_pixels


def project_orthogonal(p, n):
    """
    Project p orthogonally on the orthogonal of n, i.e.,
    compute p - (p.n)n
    Arguments:
    ----------
    p : numpy array
        vector field to be projected, the vectorial dimension is
        the last one.

    n : numpy array 
        normal to projection direction. n is assumed to be a normal 
        vector field. Assumes p.shape = n.shape
    Returns:
    --------
        projection of p on orthogonal(n), same shape as p
    """    
    sshape = p.shape[:-1] + (1,)
    h = (p*n).sum(axis=-1)
    return p - n*h.reshape(sshape)

def sphere_Exp_map(v, n, eps=1e-7):
    """
    Riemannian exponential map(s) Exp_n(v) on sphere S^k
    Arguments:
    ----------
    n : numpy array
        (k+1) Normal vector field - i.e., each vector has norm 1, belongs
        to S^k. The vectorial dimension is the last one
    v : numpy array
        vector field on S^k. Same dimensions as n. v and n are assumed to
        be point-wise orthogonal.
    eps: float
        to filter out essentially 0-length vectors v.
    Returns:
        A new normal vector field.
    """
    sshape = v.shape[:-1] + (1,)
    nv = np.linalg.norm(v, axis=-1)
    cv = np.cos(nv).reshape(sshape)
    sv = np.sin(nv).reshape(sshape)
    
    # to avoid division by 0, when |nv| is < eps, replace by 1
    # do not change the evolution!
    np.place(nv, nv < eps, 1.0)
    vhat = v/nv.reshape(sshape)
    return n*cv + vhat*sv
    

def smooth_normal_field(n1, n2, n3, mask, bc_list=None, iters=100, tau=0.05, verbose=False):
    """
    Runs a few iterations of a minimization of 2-norm squared
    of a field with domain given by mask and value on the 2-sphere
    Runs dn/dt = -||Grad n||^2, n^Tn = 1

    Arguments:
    ----------
    n1, n2, n3: numpy arrays:
        the x, y and z components of the normal field.
    mask: numpy array
        mask[x,y] == 1 if (x,y) in domain, 0 else.
    bc_list: list
        data needed for Null Neumann boundary conditions. If None,
        created from mask.
    iters: int
        number of iterations in descent:
        n^(i+1) = Exp_(S^2,n^(i))(-tau Laplace_Beltrami(n^(i)))
    tau: float
        descent step size
    verbose: bool
        if True, display iteration numbers

    Returns:
    --------
    Smoothed version of field (n1, n2, n3)

    
    """
    if bc_list is None:
        bc_list = make_bc_data(mask)
    west, north, east, south, inside, n_pixels = bc_list
    N = np.zeros((n_pixels, 3))
    N[:,0] = n1[inside]
    N[:,1] = n2[inside]
    N[:,2] = n3[inside]

    for i in range(iters):
        if verbose:
            sys.stdout.write(f'\rsmoothing iteration {i} out of {iters}\t')
        # Tension (a.k.a vector-valued Laplace Beltrami on proper bundle)
        v3 = N[west] + N[north] + N[east] + N[south] - 4.0*N

        grad = project_orthogonal(v3, N)
        # Riemannian Exponential map for evolution
        N = sphere_Exp_map(tau*grad, N)

    if verbose:
        print('\n')

    N = N.T
    N1 = np.zeros(mask.shape)
    N2 = np.zeros(mask.shape)
    N3 = np.ones(mask.shape)

    N1[inside] = N[0]
    N2[inside] = N[1]
    N3[inside] = N[2]
    return N1, N2, N3


def tichonov_regularisation_normal_field(n1, n2, n3, mu, mask, bc_list = None, 
                                        iters=10, tau=0.05, verbose=False, eps=1e-7):
    """
    A Sphere - Tichonov regularisation of normal field n0=(n1,n2,n3). 
    Square distance between two fiels is
    \int_x d(n(x),n0(x))^2 dx where d is th arccos distance on S2. 

    Arguments:
    ----------
    n1, n2, n3: numpy arrays
        the x, y and z components of the normal field n=(n1,n2,n3). 
        Should satisfy n1**2 + n2**2 + n3**2 = 1
    mu: float
        weight of the data term
    mask: numpy array
        field domain mask 
    bc_list: list or None
        data needed for Null Neumann boundary conditions. If None,
        created from mask.
    iters: int
        number of iterations in descent
        dn/dt = LaplaceBeltrami(n) - mu*gradient_n d^2(n, n0) 
    tau: float
        descent time step
    eps: float
        default 1e-7: to avoid indetermined form in arcos(x)/sqrt(1-x^2)
        when x too closed to 1. The case x -> -1 is excluded, it would 
        mean a maximum change in local value of vector field !

    Returns:
    --------
        n1, n2, n3: regularised field


    """
    # Some of the distance^2 gradients calculations could result in 
    # division by zero warning, ignore them.
    with warnings.catch_warnings():
            warnings.filterwarnings('ignore', r'divide by zero encountered in true_divide')

    if bc_list is None:
        bc_list = make_bc_data(mask)
    west, north, east, south, inside, n_pixels = bc_list

    N = np.zeros((n_pixels, 3)) 
    N[:,0] = n1[inside]
    N[:,1] = n2[inside]
    N[:,2] = n3[inside]
    N0 = N.copy()
    dterm= np.zeros(n_pixels)

    for i in range(iters):
        if verbose:
            sys.stdout.write('\rTichonov iteration {0} out of {1}\t'.format(i, iters))
        
        # Tension (a.k.a vector-valued Laplace Beltrami on proper bundle)
        v3 = N[west] + N[north] + N[east] + N[south] -4.0*N

        # distance derived term
        NN0 = ((N*N0).sum(axis=1)) # (N.N0)
        NN02 = NN0**2
        np.place(NN02, NN02 > 1.0, 1.0)
        unstable = np.where(NN02 >= 1-eps)
        #stable = np.where(NN02 <= 1-eps)
        
        dterm = np.arccos(NN0)/np.sqrt(1-NN02) 
        dterm[unstable] = 0.0
        d3 = np.reshape(dterm, (-1,1))*N0

        grad = project_orthogonal(mu*d3 + v3, N)
        N = sphere_Exp_map(tau*grad, N)
        
    if verbose:
        print('\n')

    N = N.T
    N1 = np.zeros(mask.shape)
    N2 = np.zeros(mask.shape)
    N3 = np.ones(mask.shape)

    N1[inside] = N[0]
    N2[inside] = N[1]
    N3[inside] = N[2]
    return N1, N2, N3



def simchony_integrate(n1, n2, n3, mask):
    """
    Integration of the normal field recovered from observations onto 
    a depth map via Simchony et al. hybrid DCT / finite difference
    methods.
    
    Done by solving via DCT a finite difference equation discretizing
    the equation:
        Laplacian(z) - Divergence((n1/n3, n2/n3)) = 0
    under proper boundary conditions ("natural" boundary conditions on 
    a rectangular domain)
    
    Arguments:
    ----------
    n1, n2, n3: nympy float arrays 
        the 3 components of the normal. They must be 2D arrays
        of size (m,n). The gradient field is computed as (p=-n2/n3, q=-n1/n3)
    mask: characteristic function of domain.
       
    Returns:
    --------
        z : depth map obtained by integration of the field (p, q)

    Reference:
    ----------
    Tal Simchony, Rama Chellappa, and Min Shao. "Direct analytical methods for 
    solving Poisson equations in computer vision problems." IEEE transactions 
    on Pattern Analysis and Machine Intelligence 12.5 (1990): 435-446.
    """
    
    m,n = n1.shape
    p = -n2/n3
    q = -n1/n3

    outside = np.where(mask == 0)
    p[outside] = 0
    q[outside] = 0

    # divergence of (p,q)
    px = cdx(p)
    qy = cdy(q)
    
    f = px + qy      

    # 4 edges
    f[0,1:-1]  = 0.5*(p[0,1:-1] + p[1,1:-1])    
    f[-1,1:-1] = 0.5*(-p[-1,1:-1] - p[-2,1:-1])
    f[1:-1,0]  = 0.5*(q[1:-1,0] + q[1:-1,1])
    f[1:-1,-1] = 0.5*(-q[1:-1,-1] - q[1:-1,-2])

    # 4 corners
    f[ 0, 0] = 0.5*(p[0,0] + p[1,0] + q[0,0] + q[0,1])
    f[-1, 0] = 0.5*(-p[-1,0] - p[-2,0] + q[-1,0] + q[-1,1])
    f[ 0,-1] = 0.5*(p[0,-1] + p[1,-1] - q[0,-1] - q[1,-1])
    f[-1,-1] = 0.5*(-p[-1,-1] - p[-2,-1] -q[-1,-1] -q[-1,-2])

    # cosine transform f (reflective conditions, a la matlab, 
    # might need some check)
    fs = fft.dctn(f, norm='ortho')
    
    x, y = np.mgrid[0:m,0:n]
    denum = (2*np.cos(np.pi*x/m) - 2) + (2*np.cos(np.pi*y/n) -2)
    Z = fs/denum
    Z[0,0] = 0.0 
    # or what Yvain proposed, it does not really matters
    # Z[0,0] = Z[1,0] + Z[0,1]
    z = fft.idctn(Z, norm='ortho')
    # fill outside with Nan, i.e., undefined.
    z[np.where(mask == 0)] = np.nan
    return z
# simchony()





def unbiased_integrate(n1, n2, n3, mask, order=2):
    """
    Constructs the finite difference matrix, domain and other information
    for solving the Poisson system and solve it. Port of Yvain's implementation.
    Finally even modified some of the original comments from MatLab
    
    It creates a matrix A which is a finite difference approximation of 
    the neg-Laplacian operator for the domain encoded by the mask, and a
    b matrix which encodes the neg divergence of (n2/n3, n1/n3).
    (Not -n1/n3, -n2/n3, because first dimension in Python / MatLab arrays
    is vertical, second is horizontal, and there are a few weird things with the data)
    
    The depth is obtained by solving the discretized Poisson system
    Az = b, 
    z needs to be reformated/reshaped after that.
    
    Arguments:
    ----------
    n1, n2, n3: numpy float arrays 
        the 3 components of the normal. They must be 2D arrays
        of size (m,n). The gradient field is computed as (p=-n2/n3, q=-n1/n3)
    mask: numpy array 
        characteristic function of domain (0 out, 1 in)
    order: int
        for type of numerical schemed used, 1, or 2. Frankly, use 
        the default value (2). 
    Returns:
    --------
        z : depth map obtained by integration of the field (p, q)

    
    """
    
    p = -n2/n3
    q = -n1/n3        
    
    # Calculate some usefuk masks
    m,n = mask.shape
    Omega = np.zeros((m,n,4))
    Omega_padded = np.pad(mask, (1,1), mode='constant', constant_values=0)
    Omega[:, :, 0] = Omega_padded[2:, 1:-1]*mask   # value 1 iff bottom neighbor also in mask
    Omega[:, :, 1] = Omega_padded[:-2, 1:-1]*mask  # value 1 iff top neighbor also in mask
    Omega[:, :, 2] = Omega_padded[1:-1, 2:]*mask   # value 1 iff right neighbor also in mask
    Omega[:, :, 3] = Omega_padded[1:-1, :-2]*mask  # value 1 iff left neighbor also in mask
    del Omega_padded
    
    # Mapping between 2D indices and an linear indices of
    # pixels inside the mask
    indices_mask = np.where(mask > 0)
    lidx = len(indices_mask[0])
    mapping_matrix = np.zeros(p.shape, dtype=int)
    mapping_matrix[indices_mask] = list(range(lidx))
    
    if order == 1:
        pbar = p.copy()
        qbar = q.copy()
    elif order == 2:
        pbar = 0.5*(p + p[list(range(1,m)) + [m-1], :])  # p <- (p + south(p))/2
        qbar = 0.5*(q + q[:, list(range(1,n)) + [n-1]])  # q <- (q + east(q))/2
        
    # System
    I = []
    J = []
    K = []
    b = np.zeros(lidx)

    # In mask, right neighbor also in mask
    rset = Omega[:,:,2]
    X, Y = np.where(rset > 0)
    I_center = mapping_matrix[(X,Y)].astype(int)
    I_neighbors = mapping_matrix[(X,Y+1)]
    lic = len(X)
    A_center = np.ones(lic)
    A_neighbors = -A_center
    K += tolist(A_center) + tolist(A_neighbors)
    I += tolist(I_center) + tolist(I_center)
    J += tolist(I_center) + tolist(I_neighbors)
    b[I_center] -= qbar[(X,Y)]

    # In mask, left neighbor in mask
    lset = Omega[:,:,3]
    X, Y = np.where(lset > 0)
    I_center = mapping_matrix[(X,Y)].astype(int)
    I_neighbors = mapping_matrix[(X,Y-1)]
    lic = len(X)
    A_center = np.ones(lic)
    A_neighbors = -A_center
    K += tolist(A_center) + tolist(A_neighbors)
    I += tolist(I_center) + tolist(I_center)
    J += tolist(I_center) + tolist(I_neighbors)  
    b[I_center] += qbar[(X,Y-1)]

    # In mask, top neighbor in mask
    tset = Omega[:,:,1]
    X, Y = np.where(tset > 0)
    I_center = mapping_matrix[(X,Y)].astype(int)
    I_neighbors = mapping_matrix[(X-1,Y)]
    lic = len(X)
    A_center = np.ones(lic)
    A_neighbors = -A_center
    K += tolist(A_center) + tolist(A_neighbors)
    I += tolist(I_center) + tolist(I_center)
    J += tolist(I_center) + tolist(I_neighbors)
    b[I_center] += pbar[(X-1,Y)]

    #	In mask, bottom neighbor in mask
    bset = Omega[:,:,0]
    X, Y = np.where(bset > 0)
    I_center = mapping_matrix[(X,Y)].astype(int)
    I_neighbors = mapping_matrix[(X+1,Y)]
    lic = len(X)
    A_center = np.ones(lic)
    A_neighbors = -A_center
    K += tolist(A_center) + tolist(A_neighbors)
    I += tolist(I_center) + tolist(I_center)
    J += tolist(I_center) + tolist(I_neighbors)
    b[I_center] -= pbar[(X,Y)]
    
    # Construction de A (compressed sparse column matrix)
    A = sp.csc_matrix((K, (I, J)))
    A = A + sp.eye(A.shape[0])*1e-9
    z = np.nan*np.ones(mask.shape)
    z[indices_mask] = spsolve(A, b)
    return z
    



def display_surface(z, albedo=None):
    """
    frontend routine for surface display

    Parameters
    ----------
    z : ndarray float
        (m,n) depth image to be rendered
    albedo : ndarray float or None
        Albedo of the shape to be rendered.

    Returns
    -------
    None.
    """
    if surf_backend == "mayavi":
        display_surface_mayavi(z, albedo=albedo)
    else:
        display_surface_matplotlib(z, albedo=albedo)


def display_surface_mayavi(z, albedo=None):
    """
    Display the computed depth function as a surface using 
    mayavi mlab.

    Arguments:
    ----------
    z : numpy array
        2D depth map to be displayed
    albedo: numpy array
        Same size as z if present. Used to render / texture 
        the surface 
    """

    m, n = z.shape
    x, y = np.mgrid[0:m, 0:n]
    x2 = m/2
    y2 = n/2
    if albedo is None:
        scalars = z
    else:
        scalars = albedo.max() - albedo
    mlab.mesh(x, y, z, scalars=scalars, colormap="Greys")
    mlab.view(azimuth=-60, elevation=25.0, focalpoint=(x2, y2,-1.0), distance=2.5*max(m, n))
    mlab.show()

     
def display_surface_matplotlib(z, albedo=None):
    """
    Same as above but using matplotlib instead. It is passed but unused, just for
    keeping the same signature as the mayavi based function...
    """
    if albedo is not None:
        print("albedo mapping is not implemented yet...")
    m, n = z.shape
    x, y = np.mgrid[0:m, 0:n]
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ls = LightSource(azdeg=-60, altdeg=25.0)
    greyvals = ls.shade(z, plt.cm.Greys)
    ax.plot_surface(x, y, z, rstride=1, cstride=1, linewidth=0, antialiased=False, facecolors=greyvals)
    # not implemented yet!
    # plt.axis('equal')
    plt.show()
    
    
def read_data_file(filename):
    """
    Read a matlab PS data file and returns
    - the images as a 3D array of size (m,n,nb_images)
    - the mask as a 2D array of size (m,n) with 
      mask > 0 meaning inside the mask
    - the light matrix S as a (nb_images, 3) matrix
    """
    from scipy.io import loadmat
    
    data = loadmat(filename)
    I = data['I']
    mask = data['mask']
    S = data['S']
    return I, mask, S
