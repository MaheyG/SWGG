import sys
import logging
import pdb
import itertools
import numpy as np
import torch
from tqdm.autonotebook import tqdm
from joblib import Parallel, delayed
import geomloss
import ot
import matplotlib.pylab as pl

from .sqrtm import sqrtm, sqrtm_newton_schulz
from .utils import process_device_arg

logger = logging.getLogger(__name__)


def bures_distance(Σ1, Σ2, sqrtΣ1, commute=False, squared=True):
    """ Bures distance between PDF matrices. Simple, non-batch version.
        Potentially deprecated.
    """
    if not commute:
        sqrtΣ1 = sqrtΣ1 if sqrtΣ1 is not None else sqrtm(Σ1)
        bures = torch.trace(
            Σ1 + Σ2 - 2 * sqrtm(torch.mm(torch.mm(sqrtΣ1, Σ2), sqrtΣ1)))
    else:
        bures = ((sqrtm(Σ1) - sqrtm(Σ2))**2).sum()
    if not squared:
        bures = torch.sqrt(bures)
    return torch.relu(bures)  # i.e., max(bures,0)


def bbures_distance(Σ1, Σ2, sqrtΣ1=None, inv_sqrtΣ1=None,
                    diagonal_cov=False, commute=False, squared=True, sqrt_method='spectral',
                    sqrt_niters=20):
    """ Bures distance between PDF. Batched version. """
    if sqrtΣ1 is None and not diagonal_cov:
        sqrtΣ1 = sqrtm(Σ1) if sqrt_method == 'spectral' else sqrtm_newton_schulz(Σ1, sqrt_niters)  # , return_inverse=True)

    if diagonal_cov:
        bures = ((torch.sqrt(Σ1) - torch.sqrt(Σ2))**2).sum(-1)
    elif commute:
        sqrtΣ2 = sqrtm(Σ2) if sqrt_method == 'spectral' else sqrtm_newton_schulz(Σ2, sqrt_niters)
        bures = ((sqrtm(Σ1) - sqrtm(Σ2))**2).sum((-2, -1))
    else:
        if sqrt_method == 'spectral':
            cross = sqrtm(torch.matmul(torch.matmul(sqrtΣ1, Σ2), sqrtΣ1))
        else:
            cross = sqrtm_newton_schulz(torch.matmul(torch.matmul(
                sqrtΣ1, Σ2), sqrtΣ1), sqrt_niters)
        ## pytorch doesn't have batched trace yet!
        bures = (Σ1 + Σ2 - 2 * cross).diagonal(dim1=-2, dim2=-1).sum(-1)
    if not squared:
        bures = torch.sqrt(bures)
    return torch.relu(bures)


def wasserstein_gauss_distance(μ_1, μ_2, Σ1, Σ2, sqrtΣ1=None, cost_function='euclidean',
                               squared=False,**kwargs):
    """
    Returns 2-Wasserstein Distance between Gaussians:

         W(α, β)^2 = || μ_α - μ_β ||^2 + Bures(Σ_α, Σ_β)^2


    Arguments:
        μ_1 (tensor): mean of first Gaussian
        kwargs (dict): additional arguments for bbures_distance.

    Returns:
        d (tensor): the Wasserstein distance

    """
    if cost_function == 'euclidean':
        mean_diff = ((μ_1 - μ_2)**2).sum(axis=-1)  # I think this is faster than torch.norm(μ_1-μ_2)**2
    else:
        mean_diff = cost_function(μ_1,μ_2)
        pdb.set_trace(header='TODO: what happens to bures distance for embedded cost function?')

    cova_diff = bbures_distance(Σ1, Σ2, sqrtΣ1=sqrtΣ1, squared=True, **kwargs)
    d = torch.relu(mean_diff + cova_diff)
    if not squared:
        d = torch.sqrt(d)
    return d


def pwdist_gauss(M1, S1, M2, S2, symmetric=False, return_dmeans=False, nworkers=1,
                 commute=False):
    """ POTENTIALLY DEPRECATED.
        Computes Wasserstein Distance between collections of Gaussians,
        represented in terms of their means (M1,M2) and Covariances (S1,S2).

        Arguments:
            parallel (bool): Whether to use multiprocessing via joblib


     """
    n1, n2 = len(M1), len(M2)  # Number of clusters in each side
    if symmetric:
        ## If tasks are symmetric (same data on both sides) only need combinations
        pairs = list(itertools.combinations(range(n1), 2))
    else:
        ## If tasks are assymetric, need n1 x n2 comparisons
        pairs = list(itertools.product(range(n1), range(n2)))

    D = torch.zeros((n1, n2)).to(device)

    if nworkers > 1:
        results = Parallel(n_jobs=nworkers, verbose=1, backend="threading")(
            delayed(wasserstein_gauss_distance)(M1[i], M2[j], S1[i], S2[j], squared=True) for i, j in pairs)
        for (i, j), d in zip(pairs, results):
            D[i, j] = d
            if symmetric:
                D[j, i] = D[i, j]
    else:
        for i, j in tqdm(pairs, leave=False):
            D[i, j] = wasserstein_gauss_distance(
                M1[i], M2[j], S1[i], S2[j], squared=True, commute=commute)
            if symmetric:
                D[j, i] = D[i, j]

    if return_dmeans:
        D_means = torch.cdist(M1, M2)  # For viz purposes only
        return D, D_means
    else:
        return D


def efficient_pwdist_gauss(M1, S1, M2=None, S2=None, sqrtS1=None, sqrtS2=None,
                        symmetric=False, diagonal_cov=False, commute=False,
                        sqrt_method='spectral',sqrt_niters=20,sqrt_pref=0,
                        device='cpu',nworkers=1,
                        cost_function='euclidean',
                        return_dmeans=False, return_sqrts=False):
    """ [Formerly known as efficient_pwassdist] Efficient computation of pairwise
    label-to-label Wasserstein distances between various distributions. Saves
    computation by precomputing and storing covariance square roots."""
    if M2 is None:
        symmetric = True
        M2, S2 = M1, S1

    n1, n2 = len(M1), len(M2)  # Number of clusters in each side
    if symmetric:
        ## If tasks are symmetric (same data on both sides) only need combinations
        pairs = list(itertools.combinations(range(n1), 2))
    else:
        ## If tasks are assymetric, need n1 x n2 comparisons
        pairs = list(itertools.product(range(n1), range(n2)))

    D = torch.zeros((n1, n2), device = device, dtype=M1.dtype)

    sqrtS = []
    ## Note that we need inverses of only one of two datasets.
    ## If sqrtS of S1 provided, use those. If S2 provided, flip roles of covs in Bures
    both_sqrt = (sqrtS1 is not None) and (sqrtS2 is not None)
    if (both_sqrt and sqrt_pref==0) or (sqrtS1 is not None):
        ## Either both were provided and S1 (idx=0) is prefered, or only S1 provided
        flip = False
        sqrtS = sqrtS1
    elif sqrtS2 is not None:
        ## S1 wasn't provided
        if sqrt_pref == 0: logger.warning('sqrt_pref=0 but S1 not provided!')
        flip = True
        sqrtS = sqrtS2  # S2 playes role of S1
    elif len(S1) <= len(S2):  # No precomputed squareroots provided. Compute, but choose smaller of the two!
        flip = False
        S = S1
    else:
        flip = True
        S = S2  # S2 playes role of S1

    if not sqrtS:
        logger.info('Precomputing covariance matrix square roots...')
        for i, Σ in tqdm(enumerate(S), leave=False):
            if diagonal_cov:
                assert Σ.ndim == 1
                sqrtS.append(torch.sqrt(Σ)) # This is actually not needed.
            else:
                sqrtS.append(sqrtm(Σ) if sqrt_method ==
                         'spectral' else sqrtm_newton_schulz(Σ, sqrt_niters))

    logger.info('Computing gaussian-to-gaussian wasserstein distances...')
    pbar = tqdm(pairs, leave=False)
    pbar.set_description('Computing label-to-label distances')
    for i, j in pbar:
        if not flip:
            D[i, j] = wasserstein_gauss_distance(M1[i], M2[j], S1[i], S2[j], sqrtS[i],
                                                 diagonal_cov=diagonal_cov,
                                                 commute=commute, squared=True,
                                                 cost_function=cost_function,
                                                 sqrt_method=sqrt_method,
                                                 sqrt_niters=sqrt_niters)
        else:
            D[i, j] = wasserstein_gauss_distance(M2[j], M1[i], S2[j], S1[i], sqrtS[j],
                                                 diagonal_cov=diagonal_cov,
                                                 commute=commute, squared=True,
                                                 cost_function=cost_function,
                                                 sqrt_method=sqrt_method,
                                                 sqrt_niters=sqrt_niters)
        if symmetric:
            D[j, i] = D[i, j]

    if return_dmeans:
        D_means = torch.cdist(M1, M2)  # For viz purposes only
        if return_sqrts:
            return D, D_means, sqrtS
        else:
            return D, D_means
    elif return_sqrts:
        return D, sqrtS
    else:
        return D

def pwdist_means_only(M1, M2=None, symmetric=False, device=None):
    if M2 is None or symmetric:
        symmetric = True
        M2 = M1
    D = torch.cdist(M1, M2)
    if device:
        D = D.to(device)
    return D

def pwdist_upperbound(M1, S1, M2=None, S2=None,symmetric=False, means_only=False,
                          diagonal_cov=False, commute=False, device=None,
                          return_dmeans=False):
    """ Computes upper bound of the Wasserstein distance between distributions
    with given mean and covariance.
    """

    if M2 is None:
        symmetric = True
        M2, S2 = M1, S1

    n1, n2 = len(M1), len(M2)  # Number of clusters in each side
    if symmetric:
        ## If tasks are symmetric (same data on both sides) only need combinations
        pairs = list(itertools.combinations(range(n1), 2))
    else:
        ## If tasks are assymetric, need n1 x n2 comparisons
        pairs = list(itertools.product(range(n1), range(n2)))

    D = torch.zeros((n1, n2), device = device, dtype=M1.dtype)

    logger.info('Computing gaussian-to-gaussian wasserstein distances...')
    pbar = tqdm(pairs, leave=False)
    pbar.set_description('Computing label-to-label distances')

    if means_only or return_dmeans:
        D_means = torch.cdist(M1, M2)

    if not means_only:
        for i, j in pbar:
            if means_only:
                D[i,j] = ((M1[i]-  M2[j])**2).sum(axis=-1)
            else:
                D[i,j] = ((M1[i]-  M2[j])**2).sum(axis=-1) + (S1[i] + S2[j]).diagonal(dim1=-2, dim2=-1).sum(-1)
            if symmetric:
                D[j, i] = D[i, j]
    else:
        D = D_means

    if return_dmeans:
        D_means = torch.cdist(M1, M2)  # For viz purposes only
        return D, D_means
    else:
        return D

def pwdist_exact(X1, Y1, X2=None, Y2=None, symmetric=False, loss='sinkhorn',
                 cost_function='euclidean', p=2, debias=True, entreg=1e-1, device='cpu'):

    """ Efficient computation of pairwise label-to-label Wasserstein distances
    between multiple distributions, without using Gaussian assumption.

    Args:
        X1,X2 (tensor): n x d matrix with features
        Y1,Y2 (tensor): labels corresponding to samples
        symmetric (bool): whether X1/Y1 and X2/Y2 are to be treated as the same dataset
        cost_function (callable/string): the 'ground metric' between features to
            be used in optimal transport problem. If callable, should take follow
            the convection of the cost argument in geomloss.SamplesLoss
        p (int): power of the cost (i.e. order of p-Wasserstein distance). Ignored
            if cost_function is a callable.
        debias (bool): Only relevant for Sinkhorn. If true, uses debiased sinkhorn
            divergence.


    """
    device = process_device_arg(device)
    if X2 is None:
        symmetric = True
        X2, Y2 = X1, Y1

    c1 = torch.unique(Y1)
    c2 = torch.unique(Y2)
    n1, n2 = len(c1), len(c2)

    ## We account for the possibility that labels are shifted (c1[0]!=0), see below

    if symmetric:
        ## If tasks are symmetric (same data on both sides) only need combinations
        pairs = list(itertools.combinations(range(n1), 2))
    else:
        ## If tasks are assymetric, need n1 x n2 comparisons
        pairs = list(itertools.product(range(n1), range(n2)))


    if cost_function == 'euclidean':
        if p == 1:
            cost_function = lambda x, y: geomloss.utils.distances(x, y)
        elif p == 2:
            cost_function = lambda x, y: geomloss.utils.squared_distances(x, y)
        else:
            raise ValueError()
   
    if loss == 'sinkhorn':
        distance = geomloss.SamplesLoss(
            loss=loss, p=p,
            cost=cost_function,
            debias=debias,
            blur=entreg**(1 / p),
        )
    elif loss == 'wasserstein':
        print(X1.shape)
        def distance(Xa, Xb):
            C = cost_function(Xa, Xb).cpu()
            a=torch.ones(Xa.shape[0])/Xa.shape[0]
            b=torch.ones(Xb.shape[0])/Xb.shape[0]
            #print(ot.emd2(a,b,C))
            return ot.emd2(a,b,C,numItermax=1000000).clone().detach()#, verbose=True)
    else:
        raise ValueError('Wrong loss')


    logger.info('Computing label-to-label (exact) wasserstein distances...')
    pbar = tqdm(pairs, leave=False)
    pbar.set_description('Computing label-to-label distances')
    D = torch.zeros((n1, n2), device = device, dtype=X1.dtype)
    for i, j in pbar:
        try:
            D[i, j] = distance(X1[Y1==c1[i]].to(device), X2[Y2==c2[j]].to(device)).item()
        except:
            print("This is awkward. Distance computation failed. Geomloss is hard to debug" \
                  "But here's a few things that might be happening: "\
                  " 1. Too many samples with this label, causing memory issues" \
                  " 2. Datatype errors, e.g., if the two datasets have different type")
            sys.exit('Distance computation failed. Aborting.')
        if symmetric:
            D[j, i] = D[i, j]
    return D
    
def pwdist_SWGG(X1, Y1, X2=None, Y2=None, symmetric=False, loss='sinkhorn',
                 cost_function='euclidean', p=2, device='cpu'):

    device = process_device_arg(device)
    if X2 is None:
        symmetric = True
        X2, Y2 = X1, Y1

    c1 = torch.unique(Y1)
    c2 = torch.unique(Y2)
    n1, n2 = len(c1), len(c2)

    ## We account for the possibility that labels are shifted (c1[0]!=0), see below

    if symmetric:
        ## If tasks are symmetric (same data on both sides) only need combinations
        pairs = list(itertools.combinations(range(n1), 2))
    else:
        ## If tasks are assymetric, need n1 x n2 comparisons
        pairs = list(itertools.product(range(n1), range(n2)))


    if cost_function == 'euclidean':
        if p == 1:
            cost_function = lambda x, y: geomloss.utils.distances(x, y)
        elif p == 2:
            cost_function = lambda x, y: geomloss.utils.squared_distances(x, y)
        else:
            raise ValueError()
            
    loss='SWGG'
    if loss == 'SWGG':
        def distance(X,Y):
            m=min(X.shape[0],Y.shape[0])
            X=X[0:m,:]
            Y=Y[0:m,:]
            n,dim=X.shape
            theta=random_slice(100,dim,'cpu').T
            X_line=torch.matmul(X,theta)
            Y_line=torch.matmul(Y,theta)
            X_line_sort,u=torch.sort(X_line,axis=0)
            Y_line_sort,v=torch.sort(Y_line,axis=0)
    
            Z_line_sort=(X_line_sort+Y_line_sort)/2
            Z = Z_line_sort[:,None]*theta[None,:]

            X_sort=X[u].transpose(1,2)
            Y_sort=Y[v].transpose(1,2)
            bary=(X_sort+Y_sort)/2
            
            bary_line=torch.einsum('ijk,jk->ik',bary,theta)
        
            W_baryZ=W2_line_W(bary,bary_line,bary_line,Z_line_sort,theta)
            W_XZ=W2_line_W(X,X_line,X_line_sort,Z_line_sort,theta)
            W_YZ=W2_line_W(Y,Y_line,Y_line_sort,Z_line_sort,theta)
            W=-4*W_baryZ+2*W_XZ+2*W_YZ
            idx=torch.argmin(W)
            u_min=u[:,idx]
            v_min=v[:,idx]
            return W[idx],u_min,v_min


    logger.info('Computing label-to-label (exact) wasserstein distances...')
    pbar = tqdm(pairs, leave=False)
    pbar.set_description('Computing label-to-label distances')
    D = torch.zeros((n1, n2), device = device, dtype=X1.dtype)
    for i, j in pbar:
        try:
            d,u,v = distance(X1[Y1==c1[i]].to(device), X2[Y2==c2[j]].to(device))
            D[i, j] = d#.item()
            #P=sort_to_plan(u,v)
            #pl.figure()
            #pl.imshow(P, interpolation='nearest')
            
        except:
            print("This is awkward. Distance computation failed. Geomloss is hard to debug" \
                  "But here's a few things that might be happening: "\
                  " 1. Too many samples with this label, causing memory issues" \
                  " 2. Datatype errors, e.g., if the two datasets have different type")
            sys.exit('Distance computation failed. Aborting.')
        if symmetric:
            D[j, i] = D[i, j]
    return D


def pwdist_smooth_SWGG(X1, Y1, X2=None, Y2=None, symmetric=False, loss='sinkhorn',
                 cost_function='euclidean', p=2, device='cpu'):

    device = process_device_arg(device)
    if X2 is None:
        symmetric = True
        X2, Y2 = X1, Y1

    c1 = torch.unique(Y1)
    c2 = torch.unique(Y2)
    n1, n2 = len(c1), len(c2)

    ## We account for the possibility that labels are shifted (c1[0]!=0), see below

    if symmetric:
        ## If tasks are symmetric (same data on both sides) only need combinations
        pairs = list(itertools.combinations(range(n1), 2))
    else:
        ## If tasks are assymetric, need n1 x n2 comparisons
        pairs = list(itertools.product(range(n1), range(n2)))


    if cost_function == 'euclidean':
        if p == 1:
            cost_function = lambda x, y: geomloss.utils.distances(x, y)
        elif p == 2:
            cost_function = lambda x, y: geomloss.utils.squared_distances(x, y)
        else:
            raise ValueError()
            
    loss='smooth_SWGG'
    if loss == 'smooth_SWGG':
        def distance(X,Y):
            m=min(X.shape[0],Y.shape[0])
            X=X[0:m,:]
            Y=Y[0:m,:]
            theta,loss_l=get_minSW_smooth_pos(X,Y,lr=5e-1,num_iter=20,s=1,std=0.5)
            #pl.figure
            #pl.plot(loss_l)
            #pl.show()
            return upperW2_smooth_pos(X,Y,theta,s=1,std=0)

    logger.info('Computing label-to-label (exact) wasserstein distances...')
    pbar = tqdm(pairs, leave=False)
    pbar.set_description('Computing label-to-label distances')
    D = torch.zeros((n1, n2), device = device, dtype=X1.dtype)
    for i, j in pbar:
        try:
            D[i, j] = distance(X1[Y1==c1[i]].to(device), X2[Y2==c2[j]].to(device))#.item()
        except:
            print("This is awkward. Distance computation failed. Geomloss is hard to debug" \
                  "But here's a few things that might be happening: "\
                  " 1. Too many samples with this label, causing memory issues" \
                  " 2. Datatype errors, e.g., if the two datasets have different type")
            sys.exit('Distance computation failed. Aborting.')
        if symmetric:
            D[j, i] = D[i, j]
    return D

#####################################################    
def random_slice(n_proj,dim,device='cpu'):
    theta=torch.randn((n_proj,dim))
    theta=torch.stack([th/torch.sqrt((th**2).sum()) for th in theta])
    return theta.to(device)
    
def W2_line_W(X,X_line,X_line_sort,Y_line_sort,theta): 
    
    n=X.shape[0]
    
    X_proj = X_line[:,None,:]*theta[None,:,:]
    if len(X.shape)==2:
        W_s = torch.norm(X[:,:,None]-X_proj,dim=(0,1))**2
    else :
        W_s = torch.norm(X-X_proj,dim=(0,1))**2
        
    W_1d = torch.sum((X_line_sort-Y_line_sort)**2,axis=0)

    return W_s/n+W_1d/n
    
def get_minSW_smooth_pos(X,Y,lr=1e-2,num_iter=100,s=1,std=0):
    theta=torch.randn((X.shape[1],), device=X.device, dtype=X.dtype,requires_grad=True)
    optimizer = torch.optim.SGD([theta], lr=lr)
    
    loss_l=torch.empty(num_iter)
    #proj_l=torch.empty((num_iter,X.shape[1]))
    
    for i in range(num_iter):
        theta.data/=torch.norm(theta.data)
        
        optimizer.zero_grad()
        loss = upperW2_smooth_pos(X,Y,theta,s=s,std=std)
        loss.backward()
        optimizer.step()
        
        loss_l[i]=loss.data
        #proj_l[i,:]=theta.data
    return theta.data, loss_l#, proj_l
    
def upperW2_smooth_pos(X,Y,theta,s=1,std=0):
    n,dim=X.shape
    
    X_line=torch.matmul(X,theta)
    Y_line=torch.matmul(Y,theta)
    
    X_line_sort,u=torch.sort(X_line,axis=0)
    Y_line_sort,v=torch.sort(Y_line,axis=0)
    
    X_sort=X[u]
    Y_sort=Y[v]
    
    Z_line=(X_line_sort+Y_line_sort)/2
    Z=Z_line[:,None]*theta[None,:]
    
    W_XZ=torch.sum((X_sort-Z)**2)/n
    W_YZ=torch.sum((Y_sort-Z)**2)/n
    
    X_line_extend = X_line_sort.repeat_interleave(s,dim=0)#.requires_grad_()
    X_line_extend_blur = X_line_extend + 0.5 * std * torch.randn(X_line_extend.shape)
    Y_line_extend = Y_line_sort.repeat_interleave(s,dim=0)#.requires_grad_()
    Y_line_extend_blur = Y_line_extend + 0.5 * std * torch.randn(Y_line_extend.shape)
    
    X_line_extend_blur_sort,u_b=torch.sort(X_line_extend_blur,axis=0)
    Y_line_extend_blur_sort,v_b=torch.sort(Y_line_extend_blur,axis=0)

    
    X_extend=X_sort.repeat_interleave(s,dim=0)
    Y_extend=Y_sort.repeat_interleave(s,dim=0)
    X_sort_extend=X_extend[u_b]
    Y_sort_extend=Y_extend[v_b]
    
    bary_extend=(X_sort_extend+Y_sort_extend)/2
    bary_blur=torch.mean(bary_extend.reshape((n,s,dim)),dim=1)
    
    W_baryZ=torch.sum((bary_blur-Z)**2)/n
    return -4*W_baryZ+2*W_XZ+2*W_YZ
    
def sort_to_plan(u,v):
    n=u.shape[0]
    temp=torch.arange(n)
    P1=torch.zeros((n,n))
    P2=torch.zeros((n,n))

    P1[u,temp]=1
    P2[v,temp]=1
    return (P1@P2.T)/n
