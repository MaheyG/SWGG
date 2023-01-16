import numpy as np
import torch
import ot
from sklearn.datasets import make_swiss_roll, make_moons, make_circles, make_blobs
from scipy.stats import random_correlation
from tqdm import trange


def w2(X,Y):
    M=ot.dist(X,Y)
    a=np.ones((X.shape[0],))/X.shape[0]
    b=np.ones((Y.shape[0],))/Y.shape[0]
    return ot.emd2(a,b,M)
    
def upperW2_smooth(X,Y,theta,s=1,std=0):
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
    
def get_minSW_smooth(X,Y,lr=1e-2,num_iter=100,s=1,std=0,init=None):
    if init is None :
         theta=torch.randn((X.shape[1],), device=X.device, dtype=X.dtype,requires_grad=True)
    else :
        theta=torch.tensor(init,device=X.device, dtype=X.dtype,requires_grad=True)
        
    #optimizer = torch.optim.Adam([theta], lr=lr) #Reduce the lr if you use Adam
    optimizer = torch.optim.SGD([theta], lr=lr)
    loss_l=torch.empty(num_iter)
    #proj_l=torch.empty((num_iter,X.shape[1]))
    pbar = trange(num_iter)
    for i in pbar:
        theta.data/=torch.norm(theta.data)
        optimizer.zero_grad()
            
        loss = upperW2_smooth(X,Y,theta,s=s,std=std)
        loss.backward()
        optimizer.step()
        
        loss_l[i]=loss.data
        #proj_l[i,:]=theta.data
        pbar.set_postfix_str(f"loss = {loss.item():.3f}")
    return theta.data, loss_l#,proj_l


def load_data(name='swiss_roll', n_samples=1000):
    N=n_samples
    if name == 'gaussian_2d':
        mu_s = np.ones(2)
        mu_s = np.array([i*float(np.random.rand(1,1))+4 for i in mu_s]) 
        cov_s = np.ones((2, 2))
        cov_s = cov_s * np.eye(2)
        cov_s = np.array([[0.5,-2], [-2, 5]])
        temp=np.random.multivariate_normal(mu_s, cov_s, N)
        #temp/=abs(temp).max()
    elif name == 'gaussian_2d_small_v':
        mu_s = np.ones(2)
        mu_s = np.array([i*float(np.random.rand(1,1)) for i in mu_s]) 
        cov_s = np.ones((2, 2))*1
        cov_s = cov_s * np.eye(2)
        temp=np.random.multivariate_normal(mu_s, cov_s, N)
        #temp/=abs(temp).max()
    elif name == 'gaussian_2d_big_v':
        mu_s = np.ones(2)
        mu_s = np.array([i*float(np.random.rand(1,1)) for i in mu_s]) 
        cov_s = random_correlation.rvs((.2, 1.8))*2
        #cov_s = cov_s * np.eye(2)
        temp=np.random.multivariate_normal(mu_s, cov_s, N)
        #temp/=abs(temp).max()
    elif name == 'gaussian_500d_small_v':
        mu_s = np.ones(500)
        mu_s = np.array([i*float(np.random.rand(1,1)) for i in mu_s]) 
        cov_s = np.ones((500, 500))*1
        cov_s = cov_s * np.eye(500)
        temp=np.random.multivariate_normal(mu_s, cov_s, N)
        #temp/=abs(temp).max()
    elif name == 'gaussian_500d_big_v':
        mu_s = np.ones(500)
        mu_s = np.array([i*float(np.random.rand(1,1)) for i in mu_s]) 
        eigs = np.random.rand(500,1)*2
        eigs = eigs / np.sum(eigs) * 500
        rr = eigs.reshape(-1)
        cov_s = random_correlation.rvs(rr)*10
        temp=np.random.multivariate_normal(mu_s, cov_s, N)
        #temp/=abs(temp).max()
    elif name == 'gaussian_500d':
        mu_s = np.ones(500)
        mu_s = np.array([i*float(np.random.rand(1,1)) for i in mu_s]) 
        cov_s = np.ones((500, 500))*50
        cov_s = cov_s * np.eye(500)
        temp=np.random.multivariate_normal(mu_s, cov_s, N)
    elif name == 'swiss_roll':
        temp=make_swiss_roll(n_samples=N)[0][:,(0,2)]
        temp/=abs(temp).max()
    elif name == 'half_moons':
        temp=make_moons(n_samples=N)[0]
        temp/=abs(temp).max()
    elif name == '8gaussians':
        # Inspired from https://github.com/caogang/wgan-gp
        scale = 2.
        centers = [
            (1, 0), (-1, 0), (0, 1), (0, -1),
            (1. / np.sqrt(2), 1. / np.sqrt(2)), (1. / np.sqrt(2), -1. / np.sqrt(2)),
            (-1. / np.sqrt(2), 1. / np.sqrt(2)), (-1. / np.sqrt(2), -1. / np.sqrt(2))
        ]
        centers = [(scale * x, scale * y) for x, y in centers]
        temp = []
        for i in range(N):
            point = np.random.randn(2) * .02
            center = centers[np.random.choice(np.arange(len(centers)))]
            point[0] += center[0]
            point[1] += center[1]
            temp.append(point)
        temp = np.array(temp, dtype='float32')
        temp /= 1.414  # stdev
    elif name == '25gaussians':
        # Inspired from https://github.com/caogang/wgan-gp
        temp = []
        for i in range(int(N / 25)):
            for x in range(-2, 3):
                for y in range(-2, 3):
                    point = np.random.randn(2) * 0.05
                    point[0] += 2 * x
                    point[1] += 2 * y
                    temp.append(point)
        temp = np.array(temp, dtype='float32')
        np.random.shuffle(temp)
        temp /= 2.828  # stdev
    elif name == 'circle':
        temp,y=make_circles(n_samples=2*N)
        temp=temp[np.argwhere(y==0).squeeze(),:]
    else:
        raise Exception("Dataset not found: name must be 'gaussian_2d', 'gaussian_500d', swiss_roll', 'half_moons', 'circle', '8gaussians' or '25gaussians'.")
    X=torch.from_numpy(temp).float()
    return X
