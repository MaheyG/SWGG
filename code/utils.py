import torch

def random_slice(n_proj,dim,device='cpu'):
    theta=torch.randn((n_proj,dim))
    theta=torch.stack([th/torch.sqrt((th**2).sum()) for th in theta])
    return theta.to(device)
    
def sort_to_plan(u,v):
    n=u.shape[0]
    temp=torch.arange(n)
    P1=torch.zeros((n,n))
    P2=torch.zeros((n,n))

    P1[u,temp]=1
    P2[v,temp]=1
    return (P1@P2.T)/n
