import torch
import ot
from sklearn import neighbors
import numpy as np
import time
from utils import sinkhorn

dtype=torch.DoubleTensor
torch.set_default_tensor_type(dtype)

def ICP(X, Y,pairs='nearest', max_iter=20, tol=0.1,n_proj=200):
    start = time.time()
    '''
    The Iterative Closest Point method: finds best-fit transform that maps points X on to points Y
    Input:
        X: nxd numpy array of source n points, dim d
        Y: nxd numpy array of destination n point, dim d
        max_iterations: exit algorithm after max_iterations
        tolerance: convergence criteria
    Output:
        R_f,t_f: final homogeneous transformation that maps X on to Y
        distances: Euclidean distances (errors) of the nearest neighbor
    '''
    assert X.shape == Y.shape
    n,d=X.shape
    
    
    #Compute real transformation
    #R_real,t_real=procruste(X, Y) # Frobenius Norm
    R_f=torch.eye(d)
    t_f=torch.zeros(d)
    
    #Initialization of X_temp
    X_temp=X.clone()
    X_l=torch.empty((n,d,max_iter))
    X_l[:,:,0]=X_temp
    
    loss_l=torch.zeros(max_iter)
    loss_l[0] = sinkhorn(X_temp,Y)
    #loss_l[0]=torch.norm(R_real@torch.linalg.inv(R_f)-torch.eye(d))+(torch.linalg.norm(t_real-t_f))/torch.linalg.norm(t_real) #Frobenious norm
    #P_l=[]
    
    for i in range(1,max_iter):
        #print(i,end=' ')
       
        if pairs=='nearest':
            # Correspondance with nearest neighbor
            distances, indices = nearest_neighbor(X_temp, Y)
            Y=Y[indices,:]
            R,t = procruste(X_temp,Y)
            #P=torch.zeros((n,n))
            #P[np.arange(n),indices]=1
         
        if pairs=='ot':
            #One to one correspondance with OT
            P=exact_ot(X_temp,Y)
            R,t = procruste(X_temp,P@Y)
            
        if pairs=='swgg':
            u,v=SWGG(X_temp,Y,n_proj)
            X_temp=X_temp[u]
            Y=Y[v]
            R,t=procruste(X_temp,Y)
            #P=sort_to_plan(u,v).numpy()
            
        if pairs=='swgg_optim':
            theta,_=get_SWGG_smooth(X,Y,lr=2e0,num_iter=50,s=10,std=0.1)
            theta=theta.data.reshape((2,1))
            u,v=SWGG_parallel(X,Y,theta)
            X_temp=X_temp[u]
            Y=Y[v]
            R,t=procruste(X_temp,Y)
            
            
            
        #Stop at convergence
        mean_error=torch.mean(torch.norm(R-torch.eye(d))+torch.norm(t))
        if torch.abs(mean_error) < tol:
            #print('converge at iteration n',i)
            for h in range(i,max_iter):
                X_l[:,:,h]=X_l[:,:,i-1]
                loss_l[h]=loss_l[i-1]
            end = time.time()
            #print('Convergence at iteration = ',i)
            return R_f,t_f,X_l,loss_l,end-start,i#,P_l
               
        # update the current source
        X_temp=X_temp+t
        X_temp = X_temp@(R.T) 
        
        #Update final transformation
        R_f=R@R_f
        t_f=t_f+t
        
        
        #P_l.append(P)
        X_l[:,:,i]=X_temp
        loss_l[i]=sinkhorn(X_temp,Y)
        #loss_l[i]=torch.norm(R_real@torch.linalg.inv(R_f)-torch.eye(d))+(torch.linalg.norm(t_real-t_f))/torch.linalg.norm(t_real) #Frobenius norm
        

    end = time.time()
    return R_f,t_f,X_l,loss_l,end-start,i#,P_l

def procruste(X, Y):
    '''
    Calculates the least-squares best-fit transform that maps corresponding points X to Y in d spatial dimensions
    Input:
      X: nxd numpy array of corresponding points
      Y: nxd numpy array of corresponding points
    Returns:
      R: dxd rotation matrix
      t: dx1 translation vector
    '''
    
    assert X.shape == Y.shape
    d = X.shape[1]

    # translate points to their centroids
    m_X = torch.mean(X, axis=0)
    m_Y = torch.mean(Y, axis=0)
    
    M = (X-m_X).T@ (Y-m_Y) # size (d,d)
    U, _, V = torch.linalg.svd(M, full_matrices=False)
    
    R = V.T @ U.T
    t=m_Y-m_X
    """# special reflection case
    if np.linalg.det(R) < 0:
        Vt[m-1,:] *= -1
        R = np.dot(Vt.T, U.T)"""  
    return R,t




#####################################
##### One-to-one correspondence #####
#####################################
def nearest_neighbor(X,Y):
    '''
    Find the nearest (Euclidean) neighbor in X for each point in Y
    Input:
        X: nxd array of points
        Y: nxd array of points
    Output:
        distances: Euclidean distances of the nearest neighbor
        indices: dst indices of the nearest neighbor
    '''
    assert X.shape == Y.shape

    nearest = neighbors.NearestNeighbors(n_neighbors=1)
    nearest.fit(Y)
    distances, indices = nearest.kneighbors(X, return_distance=True)
    return np.sum(distances.ravel()), indices.ravel()
    
def exact_ot(X,Y):
    n=X.shape[0]
    C=ot.dist(X,Y)
    a,b=torch.ones((n,)),torch.ones((n,))
    return ot.emd(a,b,C,numItermax=1000000)
    
def random_slice(n_proj,dim,device='cpu'):
    theta=torch.randn((n_proj,dim))
    theta=torch.stack([th/torch.sqrt((th**2).sum()) for th in theta])
    return theta.to(device)

def SWGG_parallel(X,Y,theta):
    n=X.shape[0]

    X_line=torch.matmul(X,theta)
    Y_line=torch.matmul(Y,theta)

    X_line_sort,u=torch.sort(X_line,axis=0)
    Y_line_sort,v=torch.sort(Y_line,axis=0)
    
    idx=torch.argmin(torch.mean(torch.sum(torch.square(X[u]-Y[v]), axis=-1), axis=0))
    
    return u[:,idx],v[:,idx]

def SWGG(X,Y,n_proj=200):
    dim=X.shape[1]
    theta=random_slice(n_proj,dim).T
    u,v=SWGG_parallel(X,Y,theta)
    return u,v
    
def get_SWGG_smooth(X,Y,lr=1e-2,num_iter=100,s=1,std=0):
    theta=torch.randn((X.shape[1],), device=X.device, dtype=X.dtype,requires_grad=True)
    
    #optimizer = torch.optim.Adam([theta], lr=lr) #Reduce the lr if you use Adam
    optimizer = torch.optim.SGD([theta], lr=lr)
    
    loss_l=torch.empty(num_iter)
    #proj_l=torch.empty((num_iter,X.shape[1]))
    for i in range(num_iter):
        theta.data/=torch.norm(theta.data)
        
        optimizer.zero_grad()
        loss = SWGG_smooth(X,Y,theta,s=s,std=std)
        loss.backward()
        optimizer.step()
        
        loss_l[i]=loss.data
        #proj_l[i,:]=theta.data
    return theta.data, loss_l#, proj_l



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
    
"""def Sinkhorn(X,Y,eps=1e-1):
    n=X.shape[0]
    
    C=ot.dist(X,Y)
    a,b=np.ones((n,)),np.ones((n,))
    M = ot.sinkhorn(a, b, C,eps)
    
    idx=np.argmax(M,axis=0)#Threshold to have 1 to1 correspondance
    P=np.zeros((n,n))
    P[idx,np.arange(n)]=1
    return P"""
