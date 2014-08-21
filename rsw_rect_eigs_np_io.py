import numpy as np
import numpy.linalg as nlg
import scipy.sparse as sp
from scipy.misc import factorial
import scipy.linalg as spalg
from scipy.sparse.linalg import eigs
import matplotlib.pyplot as plt
from FinDif import FiniteDiff
import os

## numpy version of rsw_rect_reduced.m
## Running this will create the files: eigVals, eigVecs, InputData, x, y
## To see the plots, run the file: python read_rsw_rect.py

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

def rsw_rect(grid,diff_typex,diff_typey,solve,store,sig0):

    if store:
        # For writing eigenvalues, eigenvectors, and data to files
        OutpDir = "storage"
        if not os.path.exists(OutpDir):
            os.mkdir(OutpDir)
        eigValsFile = open('storage/eigVals','wb')
        eigVecsFile = open('storage/eigVecs','wb')
        data = open('storage/InputData', 'wb')
        xFile = open('storage/x','wb')
        yFile = open('storage/y','wb')

    H    = 5e2            # Fluid Depth
    beta = 2e-11          # beta parameter
    f0   = 1e-4           # Mean Coriolis parameter
    g    = 9.81           # gravity
    Lx   = np.sqrt(2)*1e5 # Zonal qidth
    Ly   = 1e5            # Meridional width

    Nx = grid[0]
    Ny = grid[1]

    #diff_typex  = 'FD' # `Cheb'yschev differentiation of `FD' (finite difference)
    diff_ordx   = 2 # Order for finite difference differentiation
    #diff_typey  = 'FD' # `Cheb'yschev differentiation of `FD' (finite difference)
    diff_ordy   = 2 # Order for finite difference differentiation

    # x derivative
    if diff_typex == 'cheb':
        Dx,x = cheb(Nx)
        x    = Lx/2*x
        Dx   = 2/Lx*Dx
    elif diff_typex == 'FD':
        x   = np.linspace(Lx/2,-Lx/2,Nx+1)
        Dx  = FiniteDiff(x, diff_ordx, True, True)

    # y derivative
    if diff_typey == 'cheb':
        Dy,y = cheb(Ny)
        y    = Ly/2*y
        Dy   = 2/Ly*Dy
    elif diff_typey == 'FD':
        y   = np.linspace(Ly/2, -Ly/2, Ny+1)
        Dy  = FiniteDiff(y, diff_ordy, True, True)

    # Define Differentiation Matrices using kronecker product
    if diff_typex == 'cheb':
        F = np.kron(np.diag(np.ravel(f0+beta*y)), np.eye(Nx+1))
        Z = np.zeros([(Nx+1)*(Ny+1),(Nx+1)*(Ny+1)])
        DX = np.kron(np.eye(Ny+1), Dx)
        DY = np.kron(Dy, np.eye(Nx+1))
    else:
        F = sp.kron(np.diag(np.ravel(f0+beta*y)), np.eye(Nx+1),format='csr')
        Z = sp.csr_matrix(((Nx+1)*(Ny+1),(Nx+1)*(Ny+1)))
        DX = sp.kron(np.eye(Ny+1), Dx,format='csr')
        DY = sp.kron(Dy, np.eye(Nx+1),format='csr')

    # Sx and Sy are used to select which rows/columns need to be
    # deleted for the boundary conditions.
    Sy = np.ones((Nx+1)*(Ny+1), dtype=bool)
    Sy[0:Nx+1] = 0
    Sy[Ny*(Nx+1):] = 0

    Sx = np.ones((Nx+1)*(Ny+1),dtype=bool)
    Sx[0:((Nx+1)*(Ny+1)):(Nx+1)] = 0
    Sx[Nx:((Nx+1)*(Ny+1)):(Nx+1)] = 0

    # Define Matrices
    Zx  =  Z[Sx,:];  Fx =  F[Sx,:]
    Zxx = Zx[:,Sx]; Fxy = Fx[:,Sy]
    
    Fy  =  F[Sy,:];  Zy =  Z[Sy,:]
    Fyx = Fy[:,Sx]; Zyy = Zy[:,Sy]

    if diff_typex == 'cheb':
        A0 = np.hstack([Zxx,          Fxy,        -g*DX[Sx,:]])
        A1 = np.hstack([-Fyx,         Zyy,        -g*DY[Sy,:]])
        A2 = np.hstack([-H*DX[:,Sx], -H*DY[:,Sy],  Z])
        #size(A) = (Nx-1)(Ny+1)+(Nx+1)(Ny-1)+(Nx+1)(Ny+1) = 3*Nx*Ny + Ny + Nx + 1  ^2
        A = np.vstack([A0,A1,A2])
    else:
        A0 = sp.hstack([Zxx,          Fxy,        -g*DX[Sx,:]])
        A1 = sp.hstack([-Fyx,         Zyy,        -g*DY[Sy,:]])
        A2 = sp.hstack([-H*DX[:,Sx], -H*DY[:,Sy],  Z])
        A = sp.vstack([A0,A1,A2])

    if solve == 'eig':
        # Using eig
        if diff_typex=='cheb':
            eigVals, eigVecs = spalg.eig(1j*A)
        else:
            eigVals, eigVecs = spalg.eig(1j*A.todense())
    else:
        # Using eigs
        eigVals, eigVecs = eigs(1j*A,150,ncv=330,sigma=sig0)

    ind = (np.real(eigVals)).argsort() #get indices in ascending order
    eigVecs = eigVecs[:,ind]
    eigVals = eigVals[ind]
    omega = eigVals
    evals = len(eigVals) # how many eigenvalues we have

    if store:
        plt.plot(np.arange(0,eigVals.shape[0]),eigVals[:].real, 'o')
        plt.title("Plot of Real Part of Eigenvalues")
        plt.show()

    dataArr = np.array([H,beta,f0,g,Lx,Ly,Nx,Ny,evals,eigVecs.shape[0],eigVecs.shape[1]])
    if (evals == 0):
        if store:
            eigValsFile.close; eigVecsFile.close(); data.close(); xFile.close(); yFile.close()
        print "No eigenvalues have converged"
        return
    if store:
        eigVals.tofile(eigValsFile)
        eigVecs.tofile(eigVecsFile)
        dataArr.tofile(data)
        x.tofile(xFile)
        y.tofile(yFile)
        eigValsFile.close; eigVecsFile.close(); data.close(); xFile.close(); yFile.close()
    om = abs(omega.real)
    om = om[om>f0]
    ii = (om).argmin(0)
    if eigVals[ii] < 0: return eigVals[ii]*-1
    else: return eigVals[ii]

if __name__ == '__main__':
    grid1 = np.array([10,10])
    grid2 = np.array([30,30])
    sig0 = rsw_rect(grid1,'cheb','cheb','eig',False,None)
    rsw_rect(grid2,'FD','FD','eigs',True,sig0)


