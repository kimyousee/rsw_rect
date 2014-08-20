import numpy as np
import numpy.linalg as nlg
import scipy
import scipy.sparse as sp
from scipy.misc import factorial
from FinDif import FiniteDiff
import scipy.linalg as spalg
from scipy.sparse.linalg import eigs
import matplotlib.pyplot as plt

## numpy version of rsw_rect_reduced.m. Finds eigenvalues and also plots

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

    diff_typex  = 'FD' # `Cheb'yschev differentiation of `FD' (finite difference)
    diff_ordx   = 2 # Order for finite difference differentiation
    diff_typey  = 'FD' # `Cheb'yschev differentiation of `FD' (finite difference)
    diff_ordy   = 2 # Order for finite difference differentiation

    H    = 5e2            # Fluid Depth
    beta = 2e-11          # beta parameter
    f0   = 1e-4           # Mean Coriolis parameter
    g    = 9.81           # gravity
    Lx   = np.sqrt(2)*1e5 # Zonal qidth
    Ly   = 1e5            # Meridional width

    Nx = grid[0]
    Ny = grid[1]

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

    # B = np.eye(A.shape[0])

    # Using eig
    if diff_typex=='cheb':
        eigVals, eigVecs = spalg.eig(1j*A)
    else:
        eigVals, eigVecs = spalg.eig(1j*A.todense())
    ind = (np.real(eigVals)).argsort() #get indices in ascending order
    eigVecs = eigVecs[:,ind]
    eigVals = eigVals[ind]
    omega = eigVals

    # # Using eigs
    # eigVals, eigVecs = eigs(1j*A,80,which='SR',maxiter=500)
    # print eigVals[0:5]
    # omega = eigVals

    evals = len(eigVals) # how many eigenvalues we have

    # plt.plot(np.arange(0,eigVals.shape[0]),eigVals[:].real, 'o')
    # plt.title("Plot of Real Part of Eigenvalues Using eigs")
    # plt.show()

    plt.plot(np.arange(0,eigVals.shape[0]),eigVals[:].real, 'o')
    plt.title("Plot of Real Part of Eigenvalues Using eig")
    plt.show()

    print "First 5 positive eigenvalues (real):"
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
    # for i in range(300,311):
        uf = fields[0]; u = np.squeeze(uf[:,:,i])
        vf = fields[1]; v = np.squeeze(vf[:,:,i])
        hf = fields[2]; h = np.squeeze(hf[:,:,i])

        v = np.vstack([np.zeros([1,Nx+1]), v, np.zeros([1,Nx+1])])
        u = np.hstack([np.zeros([Ny+1,1]), u, np.zeros([Ny+1,1])])

        X, Y = np.meshgrid(x,y)

        fig = plt.figure()
        plt.subplot(3,2,1)
        plt.tight_layout(w_pad=2.5)
        plt.contourf(X/1e3,Y/1e3,u.real, 20)
        rcbar = plt.colorbar(format='%.1e')
        cl = plt.getp(rcbar.ax,'ymajorticklabels')
        plt.setp(cl,fontsize=8)

        plt.subplot(3,2,2)
        plt.contourf(X/1e3,Y/1e3,u.imag, 20)
        rcbar = plt.colorbar(format='%.1e')
        cl = plt.getp(rcbar.ax,'ymajorticklabels')
        plt.setp(cl,fontsize=8)

        plt.subplot(3,2,3)
        plt.contourf(X/1e3,Y/1e3,v.real, 20)
        rcbar = plt.colorbar(format='%.1e')
        cl = plt.getp(rcbar.ax,'ymajorticklabels')
        plt.setp(cl,fontsize=8)

        plt.subplot(3,2,4)
        plt.contourf(X/1e3,Y/1e3,v.imag, 20)
        rcbar = plt.colorbar(format='%.1e')
        cl = plt.getp(rcbar.ax,'ymajorticklabels')
        plt.setp(cl,fontsize=8)

        plt.subplot(3,2,5)
        plt.contourf(X/1e3,Y/1e3,h.real, 20)
        rcbar = plt.colorbar(format='%.1e')
        cl = plt.getp(rcbar.ax,'ymajorticklabels')
        plt.setp(cl,fontsize=8)

        plt.subplot(3,2,6)
        plt.contourf(X/1e3,Y/1e3,h.imag, 20)
        rcbar = plt.colorbar(format='%.1e')
        cl = plt.getp(rcbar.ax,'ymajorticklabels')
        plt.setp(cl,fontsize=8)

        # fig = "figs/RSW_rect_m%d.eps" % i
        # plt.savefig(fig, format='eps', dpi=1000)
        plt.show()

if __name__ == '__main__':
    grid = np.array([10,10])
    rsw_rect(grid)
