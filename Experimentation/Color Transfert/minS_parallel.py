import numpy as np
import torch


def random_slice(n_proj,dim,device='cpu'):
    theta=torch.randn((n_proj,dim))
    theta=torch.stack([th/torch.sqrt((th**2).sum()) for th in theta])
    return theta.to(device)
    
def compute_bary_line(X_line_sort,Y_line_sort,theta) :

    Z_line_sort=(X_line_sort+Y_line_sort)/2
    Z = Z_line_sort[:,None]*theta[None,:]
    
    return Z_line_sort,Z

def W2_line_W(X,X_line,X_line_sort,Y_line_sort,theta): 
    
    n=X.shape[0]
    
    X_proj = X_line[:,None,:]*theta[None,:,:]
    if len(X.shape)==2:
        W_s = torch.norm(X[:,:,None]-X_proj,dim=(0,1))**2
    else :
        W_s = torch.norm(X-X_proj,dim=(0,1))**2
        
    W_1d = torch.sum((X_line_sort-Y_line_sort)**2,axis=0)

    return W_s/n+W_1d/n

def compute_bary_gene(X_sort,Y_sort):
    
    bary=(X_sort+Y_sort)/2

    return bary
    
def upperW2(X,Y,theta):
    n=X.shape[0]

    X_line=torch.matmul(X,theta)
    Y_line=torch.matmul(Y,theta)

    X_line_sort,u=torch.sort(X_line,axis=0)
    Y_line_sort,v=torch.sort(Y_line,axis=0)
   

    X_sort=X[u].transpose(1,2)
    Y_sort=Y[v].transpose(1,2)

    Z_line_sort,Z=compute_bary_line(X_line_sort,Y_line_sort,theta)
    
    bary=compute_bary_gene(X_sort,Y_sort)
    bary_line=torch.einsum('ijk,jk->ik',bary,theta)
        
    W_baryZ=W2_line_W(bary,bary_line,bary_line,Z_line_sort,theta)
    W_XZ=W2_line_W(X,X_line,X_line_sort,Z_line_sort,theta)
    W_YZ=W2_line_W(Y,Y_line,Y_line_sort,Z_line_sort,theta)
    
    return -4*W_baryZ+2*W_XZ+2*W_YZ,u,v
