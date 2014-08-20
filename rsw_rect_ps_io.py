import sys, slepc4py
slepc4py.init(sys.argv)
from petsc4py import PETSc
from slepc4py import SLEPc
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
from FinDif import FiniteDiff
import time
import os

rank = PETSc.COMM_WORLD.Get_rank()
Print = PETSc.Sys.Print
size = PETSc.COMM_WORLD.Get_size()

## petsc4py version of rsw_rect_reduced.m 
## (also uses some numpy/scipy for setup matrices)
## Builds A from nonzero values; can also use cheb or fd2
## Running this will create the files: eigVals, eigVecs, InputData, x, y
## To see the plots, run the file: python read_rsw_rect.py
## Set a target value in command line: -eps_target 1.55e-3

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

def build_A(Nx,Ny,Zx,Zxx,gDX,Fxy,Fyx,gDY,HDX,HDY):

    dim = (Nx-1)*(Ny+1) + (Nx+1)*(Ny-1) + (Nx+1)*(Ny+1)
    dim1 = (Nx-1)*(Ny+1)
    dim2 = (Nx+1)*(Ny-1) + dim1
    dim3 = (Nx+1)*(Ny+1) + dim2

    A = PETSc.Mat().createAIJ([dim,dim])
    A.setFromOptions(); A.setUp()
    #B = PETSc.Mat().createAIJ([dim,dim])
    #B.setFromOptions(); B.setUp()
    astart,aend = A.getOwnershipRange()

    if astart < dim1: # first row
        A = assignNonzeros(A,Fxy,0,dim1)
        A = assignNonzeros(A,gDX,0,dim2)

    if ((dim1 <= astart < dim2) or (astart < dim1 and aend > dim1)): #2nd row
        A = assignNonzeros(A,-Fyx,dim1,0)
        A = assignNonzeros(A,gDY,dim1,dim2)

    if (dim2 <= astart <= dim3) or \
       (astart < dim2 and aend > dim2): #last row
        A = assignNonzeros(A,HDX,dim2,0)
        A = assignNonzeros(A,HDY,dim2,dim1)


    A.assemble()

    # bst,be = B.getOwnershipRange()
    # for i in range(bst,be):
    #     B[i,i] = 1 #eye
    # B.assemble()
    return A

def assignNonzeros(A,submat,br,bc):
    # br, bc Used to know which block in A we're in (i.e A01 for Fxy)
    smr,smc = np.nonzero(submat)
    smr = np.asarray(smr).ravel(); smc = np.asarray(smc).ravel()
    astart,aend = A.getOwnershipRange()

    for i in xrange(len(smr)):
        #ar,ac is where the nonzero value belongs in A
        ar = smr[i]+br
        ac = smc[i]+bc

        #check if location of nonzero is within processor's range
        if astart <= ar < aend: 
            A[ar,ac] = submat[smr[i],smc[i]] # assign nonzero value to it's position in A
    return A

def rsw_rect(grid, nEV):
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
    Lx   = np.sqrt(2)*1e5 # Zonal width
    Ly   = 1e5            # Meridional width

    Nx = grid[0]
    Ny = grid[1]

    diff_typex  = 'FD' # `Cheb'yschev differentiation of `FD' (finite difference)
    diff_ordx   = 2 # Order for finite difference differentiation
    diff_typey  = 'FD' # `Cheb'yschev differentiation of `FD' (finite difference)
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
        Z = sp.lil_matrix(((Nx+1)*(Ny+1),(Nx+1)*(Ny+1)))
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
    gDX = -g*DX[Sx,:]

    Fy  =  F[Sy,:];  Zy =  Z[Sy,:]
    Fyx = Fy[:,Sx]; Zyy = Zy[:,Sy]
    gDY = -g*DY[Sy,:]
    
    HDX = -H*DX[:,Sx]
    HDY = -H*DY[:,Sy]

    A = build_A(Nx,Ny,Zx,Zxx,gDX,Fxy,Fyx,gDY,HDX,HDY)

    E = SLEPc.EPS(); E.create(comm=SLEPc.COMM_WORLD)
    E.setOperators(1j*A); E.setDimensions(nEV, SLEPc.DECIDE)
    # E.setType(SLEPc.EPS.Type.LAPACK)
    E.setProblemType(SLEPc.EPS.ProblemType.NHEP);E.setFromOptions()
    E.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_REAL)
    E.setTolerances(1e-6,max_it=25)

    E.solve()

    nconv = E.getConverged()
    vr, wr = A.getVecs()
    vi, wi = A.getVecs()
    freq = np.zeros([nEV])

    if nconv <= nEV: evals = nconv
    else: evals = nEV
    
    if evals == 0: 
        eigValsFile.close; eigVecsFile.close(); data.close(); xFile.close(); yFile.close()
        Print("\nNo eigenvalues have converged\n"); return

    eigVecs = np.empty([vr.getSize(),evals],dtype='complex')
    eigVals = np.empty([evals],dtype='complex')
    dataArr = np.array([H,beta,f0,g,Lx,Ly,Nx,Ny,evals,eigVecs.shape[0],eigVecs.shape[1]])
    
    for i in range(evals):
        eigVal = E.getEigenvalue(i)
        eigVals[i] = eigVal # Store eigenvalue into np array

        E.getEigenvector(i,vr,vi)
        
        # Put all values of vr into 1 processor
        scatter, vrSeq = PETSc.Scatter.toZero(vr)
        im = PETSc.InsertMode.INSERT_VALUES
        sm = PETSc.ScatterMode.FORWARD
        scatter.scatter(vr,vrSeq,im,sm)

        if rank == 0:
            # store eigenvector in numpy array
            for j in range(0,vrSeq.getSize()):
                eigVecs[j,i] = vrSeq[j].real+vrSeq[j].imag*1j
            freq[i] = eigVal.real

    if rank == 0:
        eigVals.tofile(eigValsFile)
        eigVecs.tofile(eigVecsFile)
        dataArr.tofile(data)
        x.tofile(xFile)
        y.tofile(yFile)

        plt.plot(np.arange(0,freq.shape[0]), freq[:], 'o')
        plt.title("petsc4py/slepc4py Plot of Real Part of Eigenvalues")
        plt.show()

    eigValsFile.close; eigVecsFile.close(); data.close(); xFile.close(); yFile.close()
    

if __name__ == '__main__':
    opts = PETSc.Options()
    Nx = opts.getInt('Nx', 10)
    Ny = opts.getInt('Ny', 10)
    nEV = opts.getInt('nev', 400) # increase with more grid points

    grid = np.array([Nx,Ny])
    rsw_rect(grid,nEV)
