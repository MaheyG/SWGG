import numpy as np

import torch
from torch import optim
import ot
from scipy.stats import ortho_group



class GF():
    def __init__(self,ftype='linear',nofprojections=10,device='cpu'):
        self.ftype=ftype
        self.nofprojections=nofprojections
        self.device=device
        self.theta=None # This is for max-GSW

    def sw(self,X,Y,theta=None):
        N,dn = X.shape
        M,dm = Y.shape
        assert dn==dm and M==N
        if theta is None:
            theta=self.random_slice(dn)

        Xslices=self.get_slice(X,theta)
        Yslices=self.get_slice(Y,theta)

        Xslices_sorted=torch.sort(Xslices,dim=0)[0]
        Yslices_sorted=torch.sort(Yslices,dim=0)[0]
        return torch.sum((Xslices_sorted-Yslices_sorted)**2)
    
    def SWGG_CP(self,X,Y,theta):
        n,dn=X.shape
        if theta is None:
            theta=self.random_slice(dn).T

        X_line = torch.matmul(X, theta)
        Y_line = torch.matmul(Y, theta)

        X_line_sort, u = torch.sort(X_line, axis=0)
        Y_line_sort, v = torch.sort(Y_line, axis=0)
        
        W=torch.mean(torch.sum(torch.square(X[u]-Y[v]), axis=-1), axis=0)
        idx=torch.argmin(W)
        return W[idx],theta[:,idx]


##### GRADIENT DESCENT ######
    def SWGG_smooth(self,X,Y,theta,s=1,std=0):
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
        X_line_extend_blur = X_line_extend + 0.5 * std * torch.randn(X_line_extend.shape,device=self.device)
        Y_line_extend = Y_line_sort.repeat_interleave(s,dim=0)
        Y_line_extend_blur = Y_line_extend + 0.5 * std * torch.randn(Y_line_extend.shape,device=self.device)
    
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

    def get_minSWGG_smooth(self,X,Y,lr=1e-2,num_iter=100,s=1,std=0,init=None):
        if init is None :
             theta=torch.randn((X.shape[1],), device=X.device, dtype=X.dtype,requires_grad=True)
        else :
            theta=torch.tensor(init,device=X.device, dtype=X.dtype,requires_grad=True)
        
        #optimizer = torch.optim.Adam([theta], lr=lr) #Reduce the lr if you use Adam
        optimizer = torch.optim.SGD([theta], lr=lr)
        loss_l=torch.empty(num_iter)
        #proj_l=torch.empty((num_iter,X.shape[1]))
        for i in range(num_iter):
            theta.data/=torch.norm(theta.data)
            optimizer.zero_grad()
            
            loss = self.SWGG_smooth(X,Y,theta,s=s,std=std)
            loss.backward()
            optimizer.step()
        
            loss_l[i]=loss.data
            #proj_l[i,:]=theta.data
        res=self.SWGG_smooth(X,Y,theta.data.float(),s=1,std=0)
        return res,theta.data, loss_l#,proj_l
    

    def max_sw(self,X,Y,iterations=500,lr=1e-4):
        N,dn = X.shape
        M,dm = Y.shape
        device = self.device
        assert dn==dm and M==N
#         if self.theta is None:
        if self.ftype=='linear':
            theta=torch.randn((1,dn),device=device,requires_grad=True)
            theta.data/=torch.sqrt(torch.sum((theta.data)**2))
        self.theta=theta
        optimizer=optim.Adam([self.theta],lr=lr)
        loss_l=[]
        for i in range(iterations):
            optimizer.zero_grad()
            loss=-self.sw(X.to(self.device),Y.to(self.device),self.theta.to(self.device))
            #print('test4')
            loss_l.append(loss.data)
            loss.backward(retain_graph=True)
            optimizer.step()
            self.theta.data/=torch.norm(self.theta.data)
            #print('test5')

        res = self.sw(X.to(self.device),Y.to(self.device),self.theta.to(self.device))
        return res,self.theta.to(self.device).data,loss_l


    def get_slice(self,X,theta):
        ''' Slices samples from distribution X~P_X
            Inputs:
                X:  Nxd matrix of N data samples
                theta: parameters of g (e.g., a d vector in the linear case)
        '''
        if self.ftype=='linear':
            return self.linear(X,theta)

    def random_slice(self,dim):
        if self.ftype=='linear':
            theta=torch.randn((self.nofprojections,dim))
            theta=torch.stack([th/torch.sqrt((th**2).sum()) for th in theta])
        return theta.to(self.device)

    def linear(self,X,theta):
        if len(theta.shape)==1:
            return torch.matmul(X,theta)
        else:
            return torch.matmul(X,theta.t())
