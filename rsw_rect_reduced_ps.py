import sys, slepc4py
slepc4py.init(sys.argv)
from petsc4py import PETSc
from slepc4py import SLEPc
import numpy as np
import numpy.linalg as nlg
import matplotlib.pyplot as plt
import scipy
import scipy.sparse as sp
from scipy.misc import factorial
import scipy.linalg as spalg
from scipy.sparse.linalg import eigs

rank = PETSc.COMM_WORLD.Get_rank()
Print = PETSc.Sys.Print

## petsc4py/slepc4py version of rsw_rect_reduced.m 
## (also uses some numpy/scipy for setup matrices)

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

def rsw_rect(grid, nEV=5):

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
    gDX = -g*DX[Sx,:]

    Fy  =  F[Sy,:];  Zy =  Z[Sy,:]
    Fyx = Fy[:,Sx]; Zyy = Zy[:,Sy]
    gDY = -g*DY[Sy,:]
    
    HDX = -H*DX[:,Sx]
    HDY = -H*DY[:,Sy]

    dim = (Nx-1)*(Ny+1)+(Nx+1)*(Ny-1)+(Nx+1)*(Ny+1) # size
    A = PETSc.Mat().createAIJ([dim,dim])
    A.setFromOptions(); A.setUp()
    B = PETSc.Mat().createAIJ([dim,dim])
    B.setFromOptions(); B.setUp()
    start,end = A.getOwnershipRange()
    
    # Making A and B 
    for i in range(start,end):
        # First row of matrices
        if 0 <= i < (Nx-1)*(Ny+1):
            A[i,:] = np.hstack([Zxx[i,:],Fxy[i,:],gDX[i,:]])
        # second row of matrices:
        elif (Nx-1)*(Ny+1) <= i < (Nx-1)*(Ny+1)+(Nx+1)*(Ny-1):
            ii = i - (Nx-1)*(Ny+1)
            A[i,:] = np.hstack([-Fyx[ii,:],Zyy[ii,:],gDY[ii,:]])
        # last row of matrices
        else:
            ii = i - ((Nx-1)*(Ny+1)+(Nx+1)*(Ny-1))
            A[i,:] = np.hstack([HDX[ii,:], HDY[ii,:], Z[ii,:]])
        B[i,i] = 1

    A.assemble()
    B.assemble()

    E = SLEPc.EPS(); E.create(comm=SLEPc.COMM_WORLD)
    E.setOperators(A,B); E.setDimensions(nEV, SLEPc.DECIDE)
    E.setProblemType(SLEPc.EPS.ProblemType.GNHEP);E.setFromOptions()
    E.setWhichEigenpairs(SLEPc.EPS.Which.SMALLEST_REAL)
    E.setTolerances(1e-9,max_it=500)

    E.solve()

    nconv = E.getConverged()
    vr, wr = A.getVecs()
    vi, wi = A.getVecs()

    if nconv <= nEV: evals = nconv
    else: evals = nEV
    Print("Eigenvalues: ")
    for i in range(evals):
        eigVal = E.getEigenvalue(i)
        Print(eigVal)

if __name__ == '__main__':
    opts = PETSc.Options()
    Nx = opts.getInt('Nx', 50)
    Ny = opts.getInt('Ny', 50)
    nEV = opts.getInt('nev', 5)

    grid = np.array([10,10])
    rsw_rect(grid,nEV)
    
