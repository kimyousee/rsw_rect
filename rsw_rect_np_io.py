import numpy as np
import numpy.linalg as nlg
import scipy
import scipy.sparse as sp
from scipy.misc import factorial
import scipy.linalg as spalg
from scipy.sparse.linalg import eigs
import matplotlib.pyplot as plt
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

def fd2(N):
    if N==0: D=0; x=1; return
    x = np.linspace(-1,1,N+1) #double check syntax
    h = 2./N
    e = np.ones(N+1)

    data = np.array([-1*e, 0*e, e])/(2*h)
    D = sp.spdiags(data, [-1, 0, 1], N+1,N+1).todense()
    D[0, 0:2] = np.array([-1, 1])/h
    D[N, N-1:N+1] = np.array([-1, 1])/h
    sp.dia_matrix(D)

    D2 = sp.spdiags(np.array([e, -2*e, e])/h**2, [-1, 0, 1], N+1, N+1).todense()
    D2[0, 0:3] = np.array([1, -2, 1])/h**2
    D2[N, N-2:N+1] = np.array([1,-2,1])/h**2
    sp.dia_matrix(D2)
    return D, D2, x

def rsw_rect(grid):
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

    xMethod = 'cheb'
    xOrd    = 2
    yMethod = 'cheb'
    yOrd    = 2

    if xMethod == 'cheb':
        Dx,x = cheb(Nx)
    else:
        [Dx,Dx2,x]  = fd2(Nx)
    x  = Lx/2*x
    Dx = 2/Lx*Dx

    if yMethod == 'cheb':
        Dy,y = cheb(Ny)
    else:
        [Dy,Dy2,y]  = fd2(Ny)
    y  = Ly/2*y
    Dy = 2/Ly*Dy

    # Define Differentiation Matrices using kronecker product

    F = np.kron(np.diag(np.ravel(f0+beta*y)), np.eye(Nx+1))
    Z = np.zeros([(Nx+1)*(Ny+1),(Nx+1)*(Ny+1)])
    DX = np.kron(np.eye(Ny+1), Dx)
    DY = np.kron(Dy, np.eye(Nx+1))

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

    A0 = np.hstack([Zxx,          Fxy,        -g*DX[Sx,:]])
    A1 = np.hstack([-Fyx,         Zyy,        -g*DY[Sy,:]])
    A2 = np.hstack([-H*DX[:,Sx], -H*DY[:,Sy],  Z])

    #size = (Nx-1)(Ny+1)+(Nx+1)(Ny-1)+(Nx+1)(Ny+1) = 3*Nx*Ny + Ny + Nx + 1  ^2
    A = np.vstack([A0,A1,A2])
    # B = np.eye(A.shape[0])

    # Using eig
    eigVals, eigVecs = spalg.eig(1j*A)
    ind = (np.real(eigVals)).argsort() #get indices in ascending order
    eigVecs = eigVecs[:,ind]
    eigVals = eigVals[ind]
    omega = eigVals

    # # Using eigs
    # evals_all, evecs_all = eigs(1j*A,80,which='SR',maxiter=500)
    # print evals_all[0:5]
    # omega = eigVals

    evals = len(eigVals) # how many eigenvalues we have

    # for i in range(evals):
    #     plt.plot(np.arange(0,evals_all.shape[0]),evals_all[:].real, 'o')
    #     plt.title("Plot of Real Part of Eigenvalues Using eigs")
    # plt.show()

    dataArr = np.array([H,beta,f0,g,Lx,Ly,Nx,Ny,evals,eigVecs.shape[0],eigVecs.shape[1]])
    if (evals == 0):
        eigValsFile.close; eigVecsFile.close(); data.close(); xFile.close(); yFile.close()
        print "No eigenvalues have converged"
        return
    
    eigVals.tofile(eigValsFile)
    eigVecs.tofile(eigVecsFile)
    dataArr.tofile(data)
    x.tofile(xFile)
    y.tofile(yFile)
    eigValsFile.close; eigVecsFile.close(); data.close(); xFile.close(); yFile.close()

    plt.plot(np.arange(0,eigVals.shape[0]),eigVals[:].real, 'o')
    plt.title("Plot of Real Part of Eigenvalues Using eig")
    plt.show()


if __name__ == '__main__':
    grid = np.array([10,10])
    rsw_rect(grid)
