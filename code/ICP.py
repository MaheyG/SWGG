import torch
import ot
from sklearn import neighbors
import numpy as np
import time
from SWGG import SWGG_CP,get_SWGG_smooth
from utils_ICP import sinkhorn
from utils import random_slice

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
       	if n_proj != 0:
       	    theta=random_slice(n_proj,d).T
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
            W,u,v=SWGG_CP(X_temp,Y,theta)
            idx=torch.argmin(W)
            u,v=u[:,idx],v[:,idx]
            X_temp=X_temp[u]
            Y=Y[v]
            R,t=procruste(X_temp,Y)
            #P=sort_to_plan(u,v).numpy()
            
        if pairs=='swgg_optim':
            theta,_=get_SWGG_smooth(X,Y,lr=2e0,num_iter=50,s=10,std=0.1)
            theta=theta.data.reshape((2,1))
            _,u,v=SWGG_CP(X,Y,theta)
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
    return R_f,t_f,X_l,loss_l,end-start,i+1#,P_l

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
    
    
"""def Sinkhorn(X,Y,eps=1e-1):
    n=X.shape[0]
    
    C=ot.dist(X,Y)
    a,b=np.ones((n,)),np.ones((n,))
    M = ot.sinkhorn(a, b, C,eps)
    
    idx=np.argmax(M,axis=0)#Threshold to have 1 to1 correspondance
    P=np.zeros((n,n))
    P[idx,np.arange(n)]=1
    return P"""
    
    
    
def ICP_fast(X, Y,pairs='nearest', max_iter=20, tol=0.1,n_proj=200):
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

    #Initialization of X_temp
    X_temp=X
    
    
    for i in range(1,max_iter):
        #print(i,end=' ')
        if n_proj != 0:
       	    theta=random_slice(n_proj,d).T
       	    
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
            W,u,v=SWGG_CP(X_temp,Y,theta)
            idx=torch.argmin(W)
            u,v=u[:,idx],v[:,idx]
            X_temp=X_temp[u]
            Y=Y[v]
            R,t=procruste(X_temp,Y)
            #P=sort_to_plan(u,v).numpy()
            
        if pairs=='swgg_optim':
            theta,_,_=get_SWGG_smooth(X,Y,lr=2e0,num_iter=50,s=10,std=0.1)
            theta=theta.data.reshape((2,1))
            u,v=SWGG_parallel(X,Y,theta)
            X_temp=X_temp[u]
            Y=Y[v]
            R,t=procruste(X_temp,Y)
            
            
        #Stop at convergence
        mean_error=torch.mean(torch.norm(R-torch.eye(d))+torch.norm(t))
        if torch.abs(mean_error) < tol:
            end = time.time()
            
            return X_temp,end-start,i
               
        # update the current source
        X_temp=X_temp+t
        X_temp = X_temp@(R.T) 
        
    end = time.time()
    return X_temp,end-start,i+1
