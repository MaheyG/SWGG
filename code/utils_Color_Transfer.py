#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 15 13:33:51 2019

@author: boris
"""

import numpy as np
import torch
import ot

def sqrtm(A):
    vecs, vals, _ = np.linalg.svd(A)
    return vecs.dot(np.sqrt(vals[:, np.newaxis]) * vecs.T)

def bures(A,B):
    sA = sqrtm(A)
    return np.trace(A + B - 2 * sqrtm(sA.dot(B).dot(sA)))


#### Monge ####

def monge(A, B):

    sA = sqrtm(A)
    sA_inv = np.linalg.inv(sA)
  
    return sA_inv.dot(sqrtm(sA.dot(B).dot(sA))).dot(sA_inv)


def Vpi(A, B):
  
    sA = sqrtm(A)
    sA_inv = np.linalg.inv(sA)
    mid = sqrtm(sA.dot(B).dot(sA))
  
    T = sA_inv.dot(mid).dot(sA_inv)
    
    return A + B - (T.dot(A) + A.dot(T))

def fidelity(A, B):
    sA = sqrtm(A)
    return  np.trace(sqrtm(sA.dot(B).dot(sA)))


#### Monge-Knothe ####

def MK(A, B, k=2):
    d = A.shape[0]
  
    Ae = A[:k, :k]
    Aeet = A[:k, k:]
    Aet = A[k:, k:]
  
    schurA = Aet - Aeet.T.dot(np.linalg.inv(Ae)).dot(Aeet)
  
    Be = B[:k, :k]
    Beet = B[:k, k:]
    Bet = B[k:, k:]
  
    schurB = Bet - Beet.T.dot(np.linalg.inv(Be)).dot(Beet)
  
    Tee = monge(Ae, Be)
    Tschur = monge(schurA, schurB)
 
    return (np.hstack([np.vstack([Tee, (Beet.T.dot(np.linalg.inv(Tee)) - Tschur.dot(Aeet.T)).dot(np.linalg.inv(Ae))]), np.vstack([np.zeros((k, d-k)), Tschur])]))

def MK_dist(A, B, k = 2):
    T = MK(A, B, k)
    return np.trace(A + B - (T.dot(A) + A.dot(T.T)))

def MK_fidelity(A, B, k = 2):
    T = MK(A, B, k)
    return np.trace(T.dot(A) + A.dot(T.T)) / 2.



#### Monge-Independent ####

def Vpi_MI(A, B, k = 2):
    d = A.shape[0]
  
    Ae = A[:k, :k]
    Aeet = A[:k, k:]
  
    Be = B[:k, :k]
    Beet = B[:k, k:]
  
    I = np.eye(d)
    Ve = I[:, :k]
    Vet = I[:, k:]
  
    sAe = sqrtm(Ae)
    sAe_inv = np.linalg.inv(sAe)
    Te = sAe_inv.dot(sqrtm(sAe.dot(Be).dot(sAe))).dot(sAe_inv)
  
    C1 = Ve.dot(Ae).dot(Te).dot(Ve.T + (np.linalg.inv(Be)).dot(Beet).dot(Vet.T))
    C2 = Vet.dot(Aeet.T).dot(Te).dot(Ve.T + (np.linalg.inv(Be)).dot(Beet).dot(Vet.T))
  
    C = C1 + C2
  
    return A + B - (C + C.T) 

def MI_dist(A, B, k = 2):
    return np.trace(Vpi_MI(A, B, k))

def MI_fidelity(A, B, k = 2):
    return np.trace((A + B) - Vpi_MI(A, B, k)) / 2.



### Knothe-Rosenblatt ###

def KR_dist(A, B):
  
    La = np.linalg.cholesky(A)
    Lb = np.linalg.cholesky(B)
  
    return ((La - Lb)**2).sum()



# ============== Pytorch Versions ================ #


def Sqrtm(A):
    vals, vecs = torch.symeig(A, True)
    return (vecs * torch.sqrt(vals)).mm(vecs.t())

def Bures(A,B):
    sA = Sqrtm(A)
    return torch.trace(A + B - 2 * Sqrtm(sA.mm(B).mm(sA)))


def Monge(A, B):
  
    sA = Sqrtm(A)
    sA_inv = torch.inverse(sA)

    return sA_inv.mm(Sqrtm(sA.mm(B).mm(sA))).mm(sA_inv)


def MK_torch(A, B, k=2):
    d = A.shape[0]
  
    Ae = A[:k, :k]
    Aeet = A[:k, k:]
    Aet = A[k:, k:]
  
    schurA = Aet - Aeet.t().mm(torch.inverse(Ae)).mm(Aeet)
  
    Be = B[:k, :k]
    Beet = B[:k, k:]
    Bet = B[k:, k:]
  
    schurB = Bet - Beet.t().mm(torch.inverse(Be)).mm(Beet)
  
    Tee = Monge(Ae, Be)
    Tschur = Monge(schurA, schurB)
 
    return torch.cat((torch.cat((Tee, Beet.t().mm(torch.inverse(Be)).mm(Tee) - Tschur.mm(Aeet.t()).mm(torch.inverse(Ae))), 0), 
                      torch.cat((torch.zeros((k, d-k)), Tschur), 0)), 1)

def MK_dist_torch(A, B, k = 2):
    T = MK_torch(A, B, k)
    return torch.trace(A + B - (T.mm(A) + A.mm(T.t())))



#### Monge-Independent ####

def Vpi_MI_torch(A, B, k = 2):
    d = A.shape[0]
  
    Ae = A[:k, :k]
    Aeet = A[:k, k:]
  
    Be = B[:k, :k]
    Beet = B[:k, k:]
  
    I = torch.eye(d)
    Ve = I[:, :k]
    Vet = I[:, k:]
  
    sAe = Sqrtm(Ae)
    sAe_inv = torch.inverse(sAe)
    Te = sAe_inv.mm(Sqrtm(sAe.mm(Be).mm(sAe))).mm(sAe_inv)
  
    C1 = Ve.mm(Ae).mm(Te).mm(Ve.t() + (torch.inverse(Be)).mm(Beet).mm(Vet.t()))
    C2 = Vet.mm(Aeet.t()).mm(Te).mm(Ve.t() + (torch.inverse(Be)).mm(Beet).mm(Vet.t()))
  
    C = C1 + C2
  
    return A + B - (C + C.t()) 

def MI_dist_torch(A, B, k = 2):
    return torch.trace(Vpi_MI_torch(A, B, k))



#### MK Subspace Selection Algorithm ####


def polar(A):
    ### The polar factor of a matrix is its projection on the set of unitary matrices
    V, _, W = np.linalg.svd(A)
    return V.dot(W.T)

def subspace_gd(A, B, k, lr = 5e-5, niter = 401, minimize=True, verbose=False):

    d = A.shape[0]

    losses = []
    Ps = []

#    L = torch.randn((d, d))
#    S = L.mm(L.t()) / d

    S = A.mm(B)

    with torch.no_grad():
        S.data = torch.from_numpy(polar(S.data))

#    S = torch.eye(d)
    S.requires_grad = True

    optimizer = torch.optim.SGD([S], lr = lr)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.98, last_epoch=-1)


    for i in range(niter):

        SAS = S.t().mm(A).mm(S)
        SBS = S.t().mm(B).mm(S)

        loss = (2 * minimize - 1) * MK_dist_torch(SAS, SBS, k)

        if loss.item() != loss.item():
            print('Nan loss')
            break

        losses.append(loss.item())
        Ps.append(S.detach())

        if i % 50 == 0 and verbose:
            print(('iteration {} : loss {}').format(i, torch.abs(loss)))


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        with torch.no_grad():
            S.data = torch.from_numpy(polar(S.data))


    P_opt = Ps[np.argmin(losses)]
    
    return P_opt, losses

def transform(Xs,xs,xt,P,batch_size=128): #xs xt training and Xs Xt all

    # perform out of sample mapping
    indices = torch.arange(Xs.shape[0])
    batch_ind = [
        indices[i:i + batch_size]
        for i in range(0, len(indices), batch_size)]

    transp_Xs_l = []
    
    for bi in batch_ind:
        # get the nearest neighbor in the source domain
        D0 = ot.dist(Xs[bi], xs)
        idx = torch.argmin(D0, axis=1)

        # transport the source samples
        transp = P/ torch.sum(P, axis=1)[:, None]
        transp[~ torch.isfinite(transp)] = 0
        
        #print(transp.shape)
        transp_Xs = torch.matmul(transp, xt) #Barycentric Projection 

        # define the transported points
        transp_Xs = transp_Xs[idx, :]#+ Xs[bi] - xs[idx, :]
        transp_Xs_l.append(transp_Xs)

    transp_Xs_l = torch.concatenate(transp_Xs_l, axis=0)

    return transp_Xs_l
    
def random_subsample(X1,X2,nb=200):
    idx1 = torch.randint(X1.shape[0], size=(nb,))
    idx2 = torch.randint(X2.shape[0], size=(nb,))
    xs = X1[idx1, :]
    xt = X2[idx2, :]
    return xs,xt

def minibatch_kmeans_subsample(X1,X2,nb=200,batch_size=5000,max_iter=300):
    kmeans = MiniBatchKMeans(n_clusters=nb,random_state=0,batch_size=batch_size,max_iter=max_iter).fit(X1)
    xs=torch.tensor(kmeans.cluster_centers_)
    kmeans = MiniBatchKMeans(n_clusters=nb,random_state=0,batch_size=batch_size,max_iter=max_iter).fit(X2)
    xt=torch.tensor(kmeans.cluster_centers_)
    return xs,xt


