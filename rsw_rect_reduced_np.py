import numpy as np
import numpy.linalg as nlg
import scipy
import scipy.sparse as sp
from scipy.misc import factorial
import scipy.linalg as spalg
from scipy.sparse.linalg import eigs
import matplotlib.pyplot as plt

## numpy version of rsw_rect_reduced.m

# Function for creation of chebyshev differentiation matrices
# CHEB  compute D = differentiation matrix, x = Chebyshev grid
def cheb(N):
    if N == 0:
        D = 0; x = 1;
    else:
        x = np.cos(np.pi*np.array(range(0,N+1))/N).reshape([N+1,1])
        c = np.ravel(np.vstack([2, np.ones([N-1,1]), 2])) \
            *(-1)**np.ravel(np.array(range(0,N+1)))
        c = c.reshape(c.shape[0],1)
        X = np.tile(x,(1,N+1))
        dX = X-(X.conj().transpose())
        D  = (c*(1/c).conj().transpose())/(dX+(np.eye(N+1)))   # off-diagonal entries
        D  = D - np.diag(np.sum(D,1))   # diagonal entries
    return D,x

def rsw_rect(grid):

    H    = 5e2            # Fluid Depth
    beta = 2e-11          # beta parameter
    f0   = 1e-4           # Mean Coriolis parameter
    g    = 9.81           # gravity
    Lx   = np.sqrt(2)*1e6 # Zonal qidth
    Ly   = 1e6            # Meridional width

    Nx = grid[0]
    Ny = grid[1]

    # x derivative
    Dx,temp = cheb(Nx)
    Dx      = 2/Lx*Dx
    # y derivative
    Dy,y = cheb(Ny)
    y    = Ly/2*y
    Dy   = 2/Ly*Dy

    # Define Differentiation Matrices using kronecker product

    F = np.kron(np.diag(np.ravel(f0+beta*y)), np.eye(Nx+1))
    Z = np.zeros([(Nx+1)*(Ny+1),(Nx+1)*(Ny+1)])
    DX = np.kron(np.eye(Ny+1), Dx)
    DY = np.kron(Dy, np.eye(Nx+1))

    # Sx and Sy are used to select which rows/columns need to be
    # deleted for the boundary conditions.
    Sx = np.ones((Nx+1)*(Ny+1), dtype=bool)
    Sx[0:Nx+1] = 0
    Sx[Ny*(Nx+1):] = 0

    Sy = np.ones((Nx+1)*(Ny+1),dtype=bool)
    Sy[0:((Nx+1)*(Ny+1)):(Nx+1)] = 0
    Sy[Nx:((Nx+1)*(Ny+1)):(Nx+1)] = 0

    # Define Matrices
    Zx  =  Z[Sx,:];  Fx =  F[Sx,:]
    Zxx = Zx[:,Sx]; Fxy = Fx[:,Sy]
    
    Fy  =  F[Sy,:];  Zy =  Z[Sy,:]
    Fyx = Fy[:,Sx]; Zyy = Zy[:,Sy]

    A0 = np.hstack([Zxx,          Fxy,        -g*DX[Sx,:]])
    A1 = np.hstack([-Fyx,         Zyy,        -g*DY[Sy,:]])
    A2 = np.hstack([-H*DX[:,Sx], -H*DY[:,Sy],  Z])

    #size = (Nx-1)(Ny+1)+(Nx+1)(Ny-1)+(Nx+1)(Ny+1) = 3*Nx*Ny + Ny + Nx + 1  ^2
    A = np.vstack([A0,A1,A2])
    B = np.eye(A.shape[0])

    # Using eig
    eigVals, eigVecs = spalg.eig(1j*A,B)
    ind = (np.real(eigVals)).argsort() #get indices in ascending order
    eigVecs = eigVecs[:,ind]
    eigVals = eigVals[ind]
    omega = eigVals

    # Using eigs
    evals_all, evecs_all = eigs(1j*A,5,B,which='SR',maxiter=500)


if __name__ == '__main__':
    grid = np.array([10,10])
    rsw_rect(grid)
