import torch
import numpy as np
import math
from scipy.stats import ortho_group
from imageio.v2 import imread
from random import choices
import ot
from geomloss import SamplesLoss
import matplotlib.pylab as pl

dtype=torch.DoubleTensor
torch.set_default_tensor_type(dtype)

def Y_to_X(Y,trans=0,eps=0):
    n,dim=Y.shape
    
    X = Y.clone()
    #Translation
    t = torch.normal(torch.zeros(dim),1)
    t = t/torch.norm(t)
    t = (trans*t)
    X += t
    # Rotate
    R = rotation_matrix(dim)
    X = X@(R.T)
    # Add noise
    X += (torch.randn(n, dim) * eps)
    return X


def rotation_matrix(dim=2):
    return torch.tensor(ortho_group.rvs(dim))
    """
    if dim==2:
        c, s = np.cos(theta), np.sin(theta)
        return np.array(((c, -s), (s, c)))
        
    if dim==3:
        axis[-1]=0
        axis = axis / math.sqrt(np.dot(axis, axis))
        a = math.cos(theta / 2.0)
        b, c, d = -axis * math.sin(theta / 2.0)
        aa, bb, cc, dd = a * a, b * b, c * c, d * d
        bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
        return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                         [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                         [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])"""
    
    
def load_image(fname):
    img = imread(fname, as_gray=True)  # Grayscale
    img = (img[::-1, :]) / 255.0
    return 1 - img

def draw_samples(fname, n):
    A = load_image(fname)
    xg, yg = np.meshgrid(
        np.linspace(0, 1, A.shape[0]),
        np.linspace(0, 1, A.shape[1]),
        indexing="xy",
    )

    grid = list(zip(xg.ravel(), yg.ravel()))
    dens = A.ravel() / A.sum()
    dots = np.array(choices(grid, dens, k=n))
    dots += (0.5 / A.shape[0]) * np.random.standard_normal(dots.shape)

    return torch.from_numpy(dots).type(dtype)
    
def sinkhorn(X,Y,eps=1e-1):
    if torch.cuda.is_available():
        device='cuda'
        dtype = torch.cuda.DoubleTensor
        #print('device = ',device)
        
        X=X.to(device)
        Y=Y.to(device)
        loss = SamplesLoss("sinkhorn", p=2)#, blur=5, scaling=0.99)
        return loss(X, Y).item()
       
    else:
        device='cpu'
        dtype=torch.DoubleTensor
        #print('device = ',device)
        
        n=X.shape[0]
        C=ot.dist(X,Y)
        #C/=C.max()
        a=torch.ones((n,))/n
        CX=ot.dist(X,X)
        #CX/=CX.max()
        CY=ot.dist(Y,Y)
        #CY/=CY.max()
        return ot.sinkhorn2(a, a, C,eps)-(ot.sinkhorn2(a, a, CX,eps)+ot.sinkhorn2(a, a, CY,eps))/2
        
def plot_cloud(xs,xt,s=1):
    n,d=xs.shape
    if d==2:
        fig = pl.figure(figsize=(4,4))
        pl.scatter(xs[:, 0], xs[:, 1], c='C0', label='Source',s=s)
        pl.scatter(xt[:, 0], xt[:, 1], c='C1', label='Target',s=s)
        #pl.xlim(-7,7)
        #pl.ylim(-7,7)
        pl.axis('off')
        pl.tight_layout()
        pl.legend()
        #fig.savefig(foldername + '/Source.pdf')
    
    if d==3:
        size = s
        fig = pl.figure(figsize=(10,10))
        ax = fig.add_subplot(121, projection='3d')
        ax.grid(False)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        # Now set color to white (or whatever is "invisible")
        ax.xaxis.pane.set_edgecolor('w')
        ax.yaxis.pane.set_edgecolor('w')
        ax.zaxis.pane.set_edgecolor('w')
        pl.axis('off')
        pl.scatter(xs[:,0],xs[:,1],zs=xs[:,2],s=size,label='Source')
        pl.scatter(xt[:,0],xt[:,1],zs=xt[:,2],s=size,label='Target')
        pl.tight_layout()
        pl.legend()
        #fig.savefig(foldername + '/Source.png')
