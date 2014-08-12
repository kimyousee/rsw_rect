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

    H    = 5e2            # Fluid Depth
    beta = 2e-11          # beta parameter
    f0   = 1e-4           # Mean Coriolis parameter
    g    = 9.81           # gravity
    Lx   = np.sqrt(2)*1e5 # Zonal qidth
    Ly   = 1e5            # Meridional width

    Nx = grid[0]
    Ny = grid[1]

    # # Using Finite Difference
    # [Dx,Dx2,x]  = fd2(Nx);        [Dy,Dy2,y]  = fd2(Ny)
    # x           = Lx/2*x;         y           = Ly/2*y
    # Dx          = 2/Lx*Dx;        Dy          = 2/Ly*Dy
    # Dx2         = (2/Lx)**2*Dx2;  Dy2         = (2/Ly)**2*Dy2

    # Using cheb
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

    # for i in range(evals):
    #     plt.plot(np.arange(0,eigVals.shape[0]),eigVals[:].real, 'o')
    #     plt.title("Plot of Real Part of Eigenvalues Using eig")
    # plt.show()

    print "First 5 eigenvalues:"
    posReal = eigVals[eigVals.real>1e-10]
    print posReal[0:5]

    omega = eigVals.real
    fieldNames = ["u_x", "u_y", "eta"]
    nsol = eigVecs.shape[1]
    
    fields = np.empty(3,dtype='object')
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
    

if __name__ == '__main__':
    grid = np.array([10,10])
    rsw_rect(grid)
