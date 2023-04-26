import torch
from tqdm import trange
import numpy as np


#======== 1st version ==========
#SWGG_GG = 2W+2W-4W
#return the value and the permutations

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
    
def SWGG_GG(X,Y,theta):
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




#======== 2nd version ==========
#SWGG_CP=\|X[u]-Y[v]\|^2_2
#return the value and the permutations

def SWGG_CP(X, Y, theta):
    n = X.shape[0]

    X_line = torch.matmul(X, theta)
    Y_line = torch.matmul(Y, theta)

    X_line_sort, u = torch.sort(X_line, axis=0)
    Y_line_sort, v = torch.sort(Y_line, axis=0)

    return torch.mean(torch.sum(torch.square(X[u]-Y[v]), axis=-1), axis=0),u,v




#======== 3rd version ==========
#SWGG with optimization scheme
#return the optimal theta, the loss and all the theta

def get_SWGG_smooth(X,Y,lr=1e-2,num_iter=100,s=1,std=0,device='cpu'):
    theta=torch.randn((X.shape[1],), device=X.device, dtype=X.dtype,requires_grad=True)
    
    #optimizer = torch.optim.Adam([theta], lr=lr) #Reduce the lr if you use Adam
    optimizer = torch.optim.SGD([theta], lr=lr)
    
    loss_l=torch.empty(num_iter)
    proj_l=torch.empty((num_iter,X.shape[1]))
    pbar = trange(num_iter)
    for i in pbar:
        theta.data/=torch.norm(theta.data)
        
        optimizer.zero_grad()
        loss = SWGG_smooth(X,Y,theta,s=s,std=std,device=device)
        loss.backward()
        optimizer.step()
        
        loss_l[i]=loss.data
        proj_l[i,:]=theta.data
        pbar.set_postfix_str(f"loss = {loss.item():.3f}")
    return theta, loss_l, proj_l



def SWGG_smooth(X,Y,theta,s=1,std=0,device='cpu'):
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
    
    X_line_extend = X_line_sort.repeat_interleave(s,dim=0)
    X_line_extend_blur = X_line_extend + 0.5 * std * torch.randn(X_line_extend.shape,device=device)
    Y_line_extend = Y_line_sort.repeat_interleave(s,dim=0)
    Y_line_extend_blur = Y_line_extend + 0.5 * std * torch.randn(Y_line_extend.shape,device=device)
    
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
