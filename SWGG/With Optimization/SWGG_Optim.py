import torch
from tqdm import trange




def get_SWGG_smooth(X,Y,lr=1e-2,num_iter=100,s=1,std=0):
    theta=torch.randn((X.shape[1],), device=X.device, dtype=X.dtype,requires_grad=True)
    
    #optimizer = torch.optim.Adam([theta], lr=lr) #Reduce the lr if you use Adam
    optimizer = torch.optim.SGD([theta], lr=lr)
    
    loss_l=torch.empty(num_iter)
    proj_l=torch.empty((num_iter,X.shape[1]))
    pbar = trange(num_iter)
    for i in pbar:
        theta.data/=torch.norm(theta.data)
        
        optimizer.zero_grad()
        loss = SWGG_smooth(X,Y,theta,s=s,std=std)
        loss.backward()
        optimizer.step()
        
        loss_l[i]=loss.data
        proj_l[i,:]=theta.data
        pbar.set_postfix_str(f"loss = {loss.item():.3f}")
    return theta, loss_l, proj_l





def SWGG_smooth(X,Y,theta,s=1,std=0):
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
    X_line_extend_blur = X_line_extend + 0.5 * std * torch.randn(X_line_extend.shape)
    Y_line_extend = Y_line_sort.repeat_interleave(s,dim=0)
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
