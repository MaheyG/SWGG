import numpy as np
import torch 
from torch import optim
from mlp import MLP
import ot

class GSW_NN():
    def __init__(self,din=2,nofprojections=10,model_depth=3,num_filters=32,use_cuda=True):        

        self.nofprojections=nofprojections

        if torch.cuda.is_available() and use_cuda:
            self.device=torch.device('cuda')
        else:
            self.device=torch.device('cpu')
        
        self.parameters=None # This is for max-GSW
        self.din=din
        self.dout=nofprojections
        self.model_depth=model_depth
        self.num_filters=num_filters
        self.model=MLP(din=self.din,dout=self.dout,num_filters=self.num_filters)
        if torch.cuda.is_available() and use_cuda:
            self.model.cuda()
 
    def gsw(self,X,Y,random=True):
        '''
        Calculates GSW between two empirical distributions.
        Note that the number of samples is assumed to be equal
        (This is however not necessary and could be easily extended
        for empirical distributions with different number of samples)
        '''
        N,dn = X.shape
        M,dm = Y.shape
        assert dn==dm and M==N
        
        if random:
            self.model.reset()
        
        Xslices=self.model(X.to(self.device))
        Yslices=self.model(Y.to(self.device))

        Xslices_sorted=torch.sort(Xslices,dim=0)[0]
        Yslices_sorted=torch.sort(Yslices,dim=0)[0]
        
        return torch.sqrt(torch.sum((Xslices_sorted-Yslices_sorted)**2))
    
    def min_gsw(self,X,Y,random=True):
        '''
        Calculates GSW between two empirical distributions.
        Note that the number of samples is assumed to be equal
        (This is however not necessary and could be easily extended
        for empirical distributions with different number of samples)
        '''
        N,dn = X.shape
        M,dm = Y.shape
        assert dn==dm and M==N

        #add some directions???

         
        if random:
            self.model.reset()
        
        #print("theta ", theta)
        C = ot.dist(X,Y)
        X_translated = X + 50
        Y_translated = Y - 50

        #print("X", X)
        #print("Y", Y)

        Xslices=self.model(X_translated.to(self.device))
        Yslices=self.model(Y_translated.to(self.device))
        #Xslices=self.get_slice(X_translated,theta)
        #Yslices=self.get_slice(Y_translated,theta)
        #print("X sliced " , Xslices)

        #print(Xslices.shape)

        #Xproj=self.get_proj(X_translated, theta)
        #Yproj=self.get_proj(Y_translated,theta)

        xs_line_sorted=torch.sort(Xslices,dim=0)[0]

        xt_line_sorted=torch.sort(Yslices,dim=0)[0]
        
        W_1d = torch.sqrt(torch.sum((xs_line_sorted-xt_line_sorted)**2, dim=0))
        #print(" W_1d", W_1d)
        best_idx = torch.argmin(W_1d)
        a = torch.ones((N,))/N
        b = torch.ones((M,))/M
        true_W = ot.emd2(a, b, C)
        print("true W", true_W)
        #print("W1D", W_1d[best_idx])
        #print("estimated W", torch.sqrt(torch.sum(ot.emd_1d(Xslices[:, best_idx], Yslices[:, best_idx], p=2) * C)))
        res = torch.sqrt(torch.sum(ot.emd_1d(Xslices[:, best_idx], Yslices[:, best_idx],a, b, p=2) * C))
        print("est W_min", res)#, " true_W" , true_W)
        #print("estimated W", res)
        #print("shape XSlice", Xslices.shape)
        #print("emd2 en full d", ot.emd2(a, b, C))
        #print(Xslices[:, best_idx], Yslices[:, best_idx])
        return res
        #torch.sqrt(torch.sum((Xslices_sorted-Yslices_sorted)**2))

        #Xslices_sorted=torch.sort(Xslices,dim=0)[0]
        #Yslices_sorted=torch.sort(Yslices,dim=0)[0]
        #return torch.sqrt(torch.sum((Xslices_sorted-Yslices_sorted)**2))

    def max_gsw(self,X,Y,iterations=50,lr=1e-4):
        N,dn = X.shape
        M,dm = Y.shape
        assert dn==dm and M==N

        self.model.reset()
        
        optimizer=optim.Adam(self.model.parameters(),lr=lr)
        total_loss=np.zeros((iterations,))
        for i in range(iterations):
            optimizer.zero_grad()
            loss=-self.gsw(X.to(self.device),Y.to(self.device),random=False)
            total_loss[i]=loss.item()
            loss.backward(retain_graph=True)
            optimizer.step()
        
        return self.gsw(X.to(self.device),Y.to(self.device),random=False)
