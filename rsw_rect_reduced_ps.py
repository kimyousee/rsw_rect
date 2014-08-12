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
import time

rank = PETSc.COMM_WORLD.Get_rank()
Print = PETSc.Sys.Print
size = PETSc.COMM_WORLD.Get_size()

## petsc4py version of rsw_rect_reduced.m 
## (also uses some numpy/scipy for setup matrices)
## Builds A from nonzero values; can also use cheb or fd2

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
    x = np.linspace(-1,1,N+1)
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

    H    = 5e2            # Fluid Depth
    beta = 2e-11          # beta parameter
    f0   = 1e-4           # Mean Coriolis parameter
    g    = 9.81           # gravity
    Lx   = np.sqrt(2)*1e5 # Zonal qidth
    Ly   = 1e5            # Meridional width

    Nx = grid[0]
    Ny = grid[1]

    # # For fd2
    # [Dx,Dx2,x]  = fd2(Nx);        [Dy,Dy2,y]  = fd2(Ny)
    # x           = Lx/2*x;         y           = Ly/2*y
    # Dx          = 2/Lx*Dx;        Dy          = 2/Ly*Dy
    # Dx2         = (2/Lx)**2*Dx2;  Dy2         = (2/Ly)**2*Dy2

    # For cheb
    # x derivative
    Dx,x = cheb(Nx)
    x    = Ly/2*x
    Dx   = 2/Lx*Dx
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
    E.setWhichEigenpairs(SLEPc.EPS.Which.SMALLEST_REAL)
    E.setTolerances(1e-8,max_it=200)

    E.solve()

    nconv = E.getConverged()
    vr, wr = A.getVecs()
    vi, wi = A.getVecs()
    freq = np.zeros([nEV])

    if nconv <= nEV: evals = nconv
    else: evals = nEV
    eigVecs = np.empty([vr.getSize(),evals],dtype='complex')
    eigVals = np.empty([evals],dtype='complex')

    for i in range(evals):
        eigVal = E.getEigenvalue(i)
        eigVals[i] = eigVal

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
        print "First 5 eigenvalues:"
        posReal = eigVals[eigVals.real>1e-10]
        print posReal[0:5]

        omega = eigVals.real
        fieldNames = ["u_x", "u_y", "eta"]
        nsol = eigVecs.shape[1]
        
        fields = np.empty(3,dtype='object')
        # fields = [np.reshape(eigVecs[0:(Nx-1)*(Ny+1),:], [Nx-1,Ny+1,nsol]), \
        #           np.reshape(eigVecs[(Nx-1)*(Ny+1):2*Nx*Ny-2,:], [Nx+1, Ny-1, nsol]), \
        #           np.reshape(eigVecs[2*Nx*Ny-2:,:], [Nx+1, Ny+1, nsol])]
        fields = [np.reshape(eigVecs[0:(Nx-1)*(Ny+1),:], [Ny+1,Nx-1,nsol]), \
                  np.reshape(eigVecs[(Nx-1)*(Ny+1):2*Nx*Ny-2,:], [Ny-1, Nx+1, nsol]), \
                  np.reshape(eigVecs[2*Nx*Ny-2:,:],   [Nx+1, Ny+1, nsol])]

        om = omega.real
        om[om<=f0] = np.Inf
        ii = (abs(om.real)).argmin(0)
        for i in range(ii-2,ii+9):
            uf = fields[0]; u = np.squeeze(uf[:,:,i])
            vf = fields[1]; v = np.squeeze(vf[:,:,i])
            hf = fields[2]; h = np.squeeze(hf[:,:,i])

            # u = np.vstack([np.zeros([1,Ny+1]), u, np.zeros([1,Ny+1])])
            # v = np.hstack([np.zeros([Nx+1,1]), v, np.zeros([Nx+1,1])])
            v = np.vstack([np.zeros([1,Nx+1]), v, np.zeros([1,Nx+1])])
            u = np.hstack([np.zeros([Ny+1,1]), u, np.zeros([Ny+1,1])])

            X, Y = np.meshgrid(x,y)

            plt.subplot(3,2,1)
            plt.contourf(X/1e3,Y/1e3,(u.real).conj().transpose(), 20)
            plt.colorbar()

            plt.subplot(3,2,2)
            plt.contourf(X/1e3,Y/1e3,(u.imag).conj().transpose(), 20)
            plt.colorbar()

            plt.subplot(3,2,3)
            plt.contourf(X/1e3,Y/1e3,(v.real).conj().transpose(), 20)
            plt.colorbar()

            plt.subplot(3,2,4)
            plt.contourf(X/1e3,Y/1e3,(v.imag).conj().transpose(), 20)
            plt.colorbar()

            plt.subplot(3,2,5)
            plt.contourf(X/1e3,Y/1e3,(h.real).conj().transpose(), 20)
            plt.colorbar()

            plt.subplot(3,2,6)
            plt.contourf(X/1e3,Y/1e3,(h.imag).conj().transpose(), 20)
            plt.colorbar()
            plt.show()

    #     if rank == 0:
    #         plt.plot(np.arange(0,freq.shape[0]), freq[:], 'o')
    #         plt.title("petsc4py/slepc4py Plot of Real Part of Eigenvalues")
    
    # plt.show()

if __name__ == '__main__':
    opts = PETSc.Options()
    Nx = opts.getInt('Nx', 10)
    Ny = opts.getInt('Ny', 10)
    nEV = opts.getInt('nev', 400) # increase with more grid points

    grid = np.array([Nx,Ny])
    rsw_rect(grid,nEV)
