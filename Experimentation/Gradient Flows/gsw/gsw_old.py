import numpy as np

import torch
from torch import optim
import ot
from scipy.stats import ortho_group



class GSW():
    def __init__(self,ftype='linear',nofprojections=10,degree=2,radius=2.,use_cuda=True):
        self.ftype=ftype
        self.nofprojections=nofprojections
        self.degree=degree
        self.radius=radius
        #if torch.cuda.is_available() and use_cuda:
        #    self.device=torch.device('cuda')
        #else:
        self.device=torch.device('cpu')
        self.theta=None # This is for max-GSW

    def gsw(self,X,Y,theta=None):
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

    def minsw_negative(self,X,Y,theta=None):
        N,dn = X.shape
        M,dm = Y.shape
        assert dn==dm and M==N
        if theta is None:
            theta=self.random_slice(dn)
            theta=theta.T
        value=self.upperW2(X,Y,theta)
        W_upper=min(value)
        q_min=theta[:,torch.argmin(value)]
        
        return W_upper,q_min
        
    def compute_bary_line(self,X_line_sort,Y_line_sort,theta) :

        Z_line_sort=(X_line_sort+Y_line_sort)/2
        Z = Z_line_sort[:,None]*theta[None,:]
    
        return Z_line_sort,Z    

   
    def W2_line_W(self,X,X_line,X_line_sort,Y_line_sort,theta): # X distrib and Y on the line q
    
        n=X.shape[0]
    
        X_proj = X_line[:,None,:]*theta[None,:,:]
        if len(X.shape)==2:
            W_s = torch.norm(X[:,:,None]-X_proj,dim=(0,1))**2
        else :
            W_s = torch.norm(X-X_proj,dim=(0,1))**2
        
        W_1d = torch.sum((X_line_sort-Y_line_sort)**2,axis=0)

        return W_s/n+W_1d/n
    
    
    def compute_bary_gene(self,X_sort,Y_sort):
    
        bary=(X_sort+Y_sort)/2

        return bary
    
    def upperW2(self,X,Y,theta):
        n=X.shape[0]

        X_line=torch.matmul(X,theta)
        Y_line=torch.matmul(Y,theta)

        X_line_sort,u=torch.sort(X_line,axis=0)
        Y_line_sort,v=torch.sort(Y_line,axis=0)
   
        W_1d=torch.sum((X_line_sort-Y_line_sort)**2)/n
        
        X_sort=X[u].transpose(1,2)
        Y_sort=Y[v].transpose(1,2)

        Z_line_sort,Z=self.compute_bary_line(X_line_sort,Y_line_sort,theta)
    
        bary=self.compute_bary_gene(X_sort,Y_sort)
        bary_line=torch.einsum('ijk,jk->ik',bary,theta)
        
        W_baryZ=self.W2_line_W(bary,bary_line,bary_line,Z_line_sort,theta)#W_s is zero in this situation
        W_XZ=self.W2_line_W(X,X_line,X_line_sort,Z_line_sort,theta)
        W_YZ=self.W2_line_W(Y,Y_line,Y_line_sort,Z_line_sort,theta)
    
        return -4*W_baryZ+2*W_XZ+2*W_YZ


##### GRADIENT DESCENT ######
    def upperW2_blur(X,Y,theta,s=1,std=0):
        n=X.shape[0]
    
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
    
        X_line_extend=X_line.repeat_interleave(s,dim=0)
        X_line_extend_blur=torch.normal(X_line_extend,std)
        Y_line_extend=Y_line.repeat_interleave(s,dim=0)

        X_line_extend_blur_sort,u=torch.sort(X_line_extend_blur,axis=0)
        Y_line_extend_sort,v=torch.sort(Y_line_extend,axis=0)
    
        X_extend=X.repeat_interleave(s,dim=0)
        Y_extend=Y.repeat_interleave(s,dim=0)
        X_sort_extend=X_extend[u]
        Y_sort_extend=Y_extend[v]

        Z_line_extend=Z_line.repeat_interleave(s,dim=0)
        Z_extend=Z_line_extend[:,None]*theta[None,:]
    
        bary_extend=(X_sort_extend+Y_sort_extend)/2
        W_baryZ=torch.sum((bary_extend-Z_extend)**2)/(s*n)
        
    return -4*W_baryZ+2*W_XZ+2*W_YZ

    def upperW2p(self,X,Y,theta,reg=1):
        n=X.shape[0]

        X_line=torch.matmul(X,theta)
        Y_line=torch.matmul(Y,theta)

        X_line_sort,u=torch.sort(X_line,axis=0)
        Y_line_sort,v=torch.sort(Y_line,axis=0)

        X_sort=X[u]
        Y_sort=Y[v]

        Z_line_sort,Z=self.compute_bary_linep(X_line_sort,Y_line_sort,theta)
      
        bary=self.compute_bary_genep(X_sort,Y_sort)
        bary_line=torch.matmul(bary,theta)
        
        W_baryZ=self.W2_line_Wp(bary,bary_line,bary_line,Z_line_sort,theta,reg=reg)#W_s is zero in this situation
        W_XZ=self.W2_line_Wp(X,X_line,X_line_sort,Z_line_sort,theta,reg=reg)
        W_YZ=self.W2_line_Wp(Y,Y_line,Y_line_sort,Z_line_sort,theta,reg=reg)
    
        return -4*W_baryZ+2*W_XZ+2*W_YZ
      
    def get_minSW(self,Xa,Xb,lr=1e-2,num_iter=100,reg=1):
        theta=torch.ones(Xa.shape[1],requires_grad=True, device=Xa.device, dtype=Xa.dtype)
        optimizer = torch.optim.Adam([theta], lr=lr)
        loss_l=[]
        for i in range(num_iter):
            # build projector
            proj = theta/torch.norm(theta)
            loss = self.upperW2p(Xa.float(),Xb.float(),(proj.T).float(),reg=reg)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_l.append(loss.data)
            
        res=self.upperW2p(Xa.float(),Xb.float(),((theta.data).T).float(),reg=0)
        return res,theta.data,loss_l
    
### OLD ######""          
    def minSW2(self,X,Y,proj): 
    
        proj=proj[0]
    
        X_line=torch.matmul(X.float(),proj[:,None].float())
        Y_line=torch.matmul(Y.float(),proj[:,None].float())
        
        X_proj = X_line * proj
        Y_proj = Y_line * proj
    
        X_line_sorted,_=torch.sort(X_line,axis=0)
        Y_line_sorted,_=torch.sort(Y_line,axis=0)
    

        W_s = torch.norm(X-X_proj)**2
        W_t = torch.norm(Y-Y_proj)**2
    
        W_1d = torch.sum((X_line_sorted-Y_line_sorted)**2)


        return torch.sqrt(W_1d)+torch.sqrt(2*W_s)+torch.sqrt(2*W_t)
        
        
        
    def minSW2_sq(self,X,Y,proj): #add a seed?
    
        proj=proj[0]
    	
        X_line=torch.matmul(X.float(),proj[:,None].float())
        Y_line=torch.matmul(Y.float(),proj[:,None].float())
        
        X_proj = X_line * proj
        Y_proj = Y_line * proj
    
        X_line_sorted,_=torch.sort(X_line,axis=0)
        Y_line_sorted,_=torch.sort(Y_line,axis=0)
    

        W_s = torch.norm(X-X_proj)**2
        W_t = torch.norm(Y-Y_proj)**2
    
        W_1d = torch.sum((X_line_sorted-Y_line_sorted)**2)
    
        return W_1d+2*W_s+2*W_t
        
           
    def minSW2_1d(self,X,Y,proj): #add a seed?
    
        proj=proj[0]
    	
        X_line=torch.matmul(X.float(),proj[:,None].float())
        Y_line=torch.matmul(Y.float(),proj[:,None].float())
        
        X_line_sorted,_=torch.sort(X_line,axis=0)
        Y_line_sorted,_=torch.sort(Y_line,axis=0)
    
        W_1d = torch.sum((X_line_sorted-Y_line_sorted)**2)
    
        return W_1d   

    def max_gsw(self,X,Y,iterations=500,lr=1e-4):
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
        total_loss=np.zeros((iterations,))
        for i in range(iterations):
            optimizer.zero_grad()
            loss=-self.gsw(X.to(self.device),Y.to(self.device),self.theta.to(self.device))
            total_loss[i]=loss.item()
            loss.backward(retain_graph=True)
            optimizer.step()
            self.theta.data/=torch.norm(self.theta.data)

        res = self.gsw(X.to(self.device),Y.to(self.device),self.theta.to(self.device))
        return res,self.theta.data


    def min_sw2(self,X,Y,iterations=500,lr=1e-4):
        N,dn = X.shape
        M,dm = Y.shape
        device = self.device
        theta=torch.randn((1,dn),device=device,requires_grad=True)
        theta.data/=torch.norm(theta)
        self.theta=theta

        optimizer = optim.Adam([self.theta], lr=lr)
        for i in range(iterations):
            loss = self.minSW2(X.to(self.device),Y.to(self.device),self.theta.to(self.device))
            optimizer.zero_grad()
            loss.backward()#retain_graph=True
            optimizer.step()
            self.theta.data/=torch.norm(theta)
        res=self.minSW2(X.to(self.device),Y.to(self.device),self.theta.to(self.device))
        return res,self.theta.data
        
    def min_sw2_sq(self,X,Y,iterations=500,lr=1e-4):
        N,dn = X.shape
        M,dm = Y.shape
        device = self.device
        theta=torch.randn((1,dn),device=device,requires_grad=True)
        theta.data/=torch.norm(theta)
        self.theta=theta

        optimizer = optim.Adam([self.theta], lr=lr)
        for i in range(iterations):
            loss = self.minSW2_sq(X.to(self.device),Y.to(self.device),self.theta.to(self.device))
            optimizer.zero_grad()
            loss.backward()#retain_graph=True
            optimizer.step()
            self.theta.data/=torch.norm(theta)
        res=self.minSW2_sq(X.to(self.device),Y.to(self.device),self.theta.to(self.device))
        return res,self.theta.data
        

    def min_sw2_sq_cp(self,X,Y,iterations=500,lr=1e-4):
        N,dn = X.shape
        M,dm = Y.shape
        device = self.device
        theta=torch.randn((1,dn),device=device,requires_grad=True)
        theta.data/=torch.norm(theta)
        self.theta=theta

        optimizer = optim.Adam([self.theta], lr=lr)
        for i in range(iterations):
            loss = self.minSW2_sq(X.to(self.device),Y.to(self.device),self.theta.to(self.device))
            optimizer.zero_grad()
            loss.backward()#retain_graph=True
            optimizer.step()
            self.theta.data/=torch.norm(theta)
            
        proj=self.theta.data
        
        proj=proj[0]
        C = torch.cdist(X.contiguous(),Y.contiguous(),p=2)**2 
    	
        X_line=torch.matmul(X.float(),proj[:,None].float())
        Y_line=torch.matmul(Y.float(),proj[:,None].float())
        res = torch.sum(ot.emd_1d(X_line, Y_line, p=2) * C)
        return res,self.theta.data 
        
    def min_sw2_cp(self,X,Y,iterations=500,lr=1e-4):
        N,dn = X.shape
        M,dm = Y.shape
        device = self.device
        theta=torch.randn((1,dn),device=device,requires_grad=True)
        theta.data/=torch.norm(theta)
        self.theta=theta

        optimizer = optim.Adam([self.theta], lr=lr)
        for i in range(iterations):
            loss = self.minSW2(X.to(self.device),Y.to(self.device),self.theta.to(self.device))
            optimizer.zero_grad()
            loss.backward()#retain_graph=True
            optimizer.step()
            self.theta.data/=torch.norm(theta)
            
        proj=self.theta.data
        C = torch.cdist(X.contiguous(),Y.contiguous(),p=2)**2 
        proj=proj[0]
    	
        X_line=torch.matmul(X.float(),proj[:,None].float())
        Y_line=torch.matmul(Y.float(),proj[:,None].float())
        res = torch.sum(ot.emd_1d(X_line, Y_line, p=2) * C)
        return res,self.theta.data
        
    def min_sw2_1d_cp(self,X,Y,iterations=500,lr=1e-4):
        N,dn = X.shape
        M,dm = Y.shape
        device = self.device
        theta=torch.randn((1,dn),device=device,requires_grad=True)
        theta.data/=torch.norm(theta)
        self.theta=theta

        optimizer = optim.Adam([self.theta], lr=lr)
        for i in range(iterations):
            loss = self.minSW2_1d(X.to(self.device),Y.to(self.device),self.theta.to(self.device))
            optimizer.zero_grad()
            loss.backward()#retain_graph=True
            optimizer.step()
            self.theta.data/=torch.norm(theta)
            
        proj=self.theta.data
        C = torch.cdist(X.contiguous(),Y.contiguous(),p=2)**2 
        proj=proj[0]
    	
        X_line=torch.matmul(X.float(),proj[:,None].float())
        Y_line=torch.matmul(Y.float(),proj[:,None].float())
        res = torch.sum(ot.emd_1d(X_line, Y_line, p=2) * C)
        return res,self.theta.data 

    def shift(self, xs,xt):
        m_xs=torch.mean(xs,dim=0)
        m_xt=torch.mean(xt,dim=0)
        dim=xs.shape[1]
        tau_xs=-torch.max(xs,dim=0)[0]#-1*torch.ones(dim)
        tau_xt=-torch.min(xt,dim=0)[0]#+1*torch.ones(dim)
        mult = np.zeros(dim) # pour les gaussiennes
        #mult[0] = 1
        #mult = np.zeros(dim) # pour les swiss rolls?
        mult[1] = 1
        #mult = np.ones(dim)
        mult = torch.from_numpy(mult).float()
        xs_shift=xs+tau_xs#*mult
        xt_shift=xt+tau_xt#*mult
        #print(xs, xs_shift)
        res=-2*torch.dot(-tau_xs+tau_xt,m_xs-m_xt)+torch.linalg.norm(tau_xs-tau_xt)**2
        return xs_shift,xt_shift,res
        
    def min_gsw_shift(self,X,Y):

        N,dn = X.shape
        M,dm = Y.shape
        assert dn==dm and M==N

        theta=self.random_slice(dn)
        
        C = torch.cdist(X.contiguous(),Y.contiguous(),p=2)**2
        
        X, Y, res_shift = self.shift(Y, X)
        
        Xslices=self.get_slice(X,theta)
        Yslices=self.get_slice(Y,theta)
        
        X_line_sorted,_=torch.sort(Xslices,axis=0)
        Y_line_sorted,_=torch.sort(Yslices,axis=0)
        #print(X_line_sorted)
    
        W_1d = torch.sum((X_line_sorted-Y_line_sorted)**2,axis=0)

       
        best_idx = torch.argmin(W_1d)

        res = torch.sqrt(torch.sum(ot.emd_1d(Xslices[:, best_idx], Yslices[:, best_idx], p=2) * C))
        
        return res, theta[best_idx], X, Y



    def alea_gsw(self,X,Y, p=2, theta=None):
        '''
        Estimates GSW between two empirical distributions.
        It takes the transport map that minimizes the 
        distance of the two distributions projected on a line.
        The X and Y distributions are translated to make them 
        'separable' (not compulsory here). The estimated W is computed
        as the transport cost obtained by the transportation matrix given
        by the alignment on the line.
        '''

        #X, Y, res_shift = self.shift(X, Y)


        N,dn = X.shape
        M,dm = Y.shape
        assert dn==dm and M==N

        if theta is None:
            theta=self.random_slice(dn)
        
        if p==2:
            C = torch.cdist(X.contiguous(),Y.contiguous(),p=2)**2
        elif p==1:
            C = torch.cdist(X.contiguous(),Y.contiguous(),p=1)
 
        
        if p==2:
            res_shift=0

            res = torch.sqrt(torch.sum(torch.eye(M)/M * C))
        elif p==1:
            res = torch.sum(ot.emd_1d(Xslices[:, best_idx], Yslices[:, best_idx],a, b, p=p) * C)
        return res

    
    def gsl2(self,X,Y,theta=None):
        '''
        Calculates GSW between two empirical distributions.
        Note that the number of samples is assumed to be equal
        (This is however not necessary and could be easily extended
        for empirical distributions with different number of samples)
        '''
        N,dn = X.shape
        M,dm = Y.shape
        assert dn==dm and M==N
        if theta is None:
            theta=self.random_slice(dn)

        Xslices=self.get_slice(X,theta)
        Yslices=self.get_slice(Y,theta)

        Yslices_sorted=torch.sort(Yslices,dim=0)

        return torch.sqrt(torch.sum((Xslices-Yslices)**2))

    def get_slice(self,X,theta):
        ''' Slices samples from distribution X~P_X
            Inputs:
                X:  Nxd matrix of N data samples
                theta: parameters of g (e.g., a d vector in the linear case)
        '''
        if self.ftype=='linear':
            return self.linear(X,theta)
        elif self.ftype=='poly':
            return self.poly(X,theta)
        elif self.ftype=='circular':
            return self.circular(X,theta)
        else:
            raise Exception('Defining function not implemented')

    def random_slice(self,dim):
        if self.ftype=='linear':
            theta=torch.randn((self.nofprojections,dim))
            theta=torch.stack([th/torch.sqrt((th**2).sum()) for th in theta])
        elif self.ftype=='poly':
            dpoly=self.homopoly(dim,self.degree)
            theta=torch.randn((self.nofprojections,dpoly))
            theta=torch.stack([th/torch.sqrt((th**2).sum()) for th in theta])
        elif self.ftype=='circular':
            theta=torch.randn((self.nofprojections,dim))
            theta=torch.stack([self.radius*th/torch.sqrt((th**2).sum()) for th in theta])
        return theta.to(self.device)

    def linear(self,X,theta):
        if len(theta.shape)==1:
            return torch.matmul(X,theta)
        else:
            return torch.matmul(X,theta.t())

    def poly(self,X,theta):
        ''' The polynomial defining function for generalized Radon transform
            Inputs
            X:  Nxd matrix of N data samples
            theta: Lxd vector that parameterizes for L projections
            degree: degree of the polynomial
        '''
        N,d=X.shape
        assert theta.shape[1]==self.homopoly(d,self.degree)
        powers=list(self.get_powers(d,self.degree))
        HX=torch.ones((N,len(powers))).to(self.device)
        for k,power in enumerate(powers):
            for i,p in enumerate(power):
                HX[:,k]*=X[:,i]**p
        if len(theta.shape)==1:
            return torch.matmul(HX,theta)
        else:
            return torch.matmul(HX,theta.t())

    def circular(self,X,theta):
        ''' The circular defining function for generalized Radon transform
            Inputs
            X:  Nxd matrix of N data samples
            theta: Lxd vector that parameterizes for L projections
        '''
        N,d=X.shape
        if len(theta.shape)==1:
            return torch.sqrt(torch.sum((X-theta)**2,dim=1))
        else:
            return torch.stack([torch.sqrt(torch.sum((X-th)**2,dim=1)) for th in theta],1)

    def get_powers(self,dim,degree):
        '''
        This function calculates the powers of a homogeneous polynomial
        e.g.

        list(get_powers(dim=2,degree=3))
        [(0, 3), (1, 2), (2, 1), (3, 0)]

        list(get_powers(dim=3,degree=2))
        [(0, 0, 2), (0, 1, 1), (0, 2, 0), (1, 0, 1), (1, 1, 0), (2, 0, 0)]
        '''
        if dim == 1:
            yield (degree,)
        else:
            for value in range(degree + 1):
                for permutation in self.get_powers(dim - 1,degree - value):
                    yield (value,) + permutation

    def homopoly(self,dim,degree):
        '''
        calculates the number of elements in a homogeneous polynomial
        '''
        return len(list(self.get_powers(dim,degree)))
        

