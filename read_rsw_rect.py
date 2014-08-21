import numpy as np
import matplotlib.pyplot as plt
import os

### Script to read in data that was made already with rsw_rect_io ###

# Where the plots will be stored
FigDir = "figs"
if not os.path.exists(FigDir):
    os.mkdir(FigDir)

# Read from necessary files
data = np.fromfile('storage/InputData')
eigVals = np.fromfile('storage/eigVals',dtype=np.complex128)
eigVecs = np.fromfile('storage/eigVecs',dtype=np.complex128)
x = np.fromfile('storage/x')
y = np.fromfile('storage/y')

# Get parameters from data file
H    = data[0]
beta = data[1]
f0   = data[2]
g    = data[3]
Lx   = data[4]
Ly   = data[5]
Nx   = int(data[6])
Ny   = int(data[7])
evals= int(data[8])

eigVecs = eigVecs.reshape([data[9],data[10]])

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
ii = (abs(om)).argmin(0)
for i in range(ii,ii+11):
    print i,"Frequency:",eigVals[i].real
    uf = fields[0]; u = np.squeeze(uf[:,:,i])
    vf = fields[1]; v = np.squeeze(vf[:,:,i])
    hf = fields[2]; h = np.squeeze(hf[:,:,i])

    v = np.vstack([np.zeros([1,Nx+1]), v, np.zeros([1,Nx+1])])
    u = np.hstack([np.zeros([Ny+1,1]), u, np.zeros([Ny+1,1])])

    X, Y = np.meshgrid(x,y)

    fig = plt.figure()
    plt.rcParams["axes.titlesize"] = 8
    plt.subplot(3,2,1)
    plt.title("u_real")
    plt.tight_layout(w_pad=2.5)
    plt.contourf(X/1e3,Y/1e3,u.real, 20)
    cv = abs(u.real).max()
    plt.clim(-cv,cv)
    rcbar = plt.colorbar(format='%.1e')
    cl = plt.getp(rcbar.ax,'ymajorticklabels')
    plt.setp(cl,fontsize=8)

    plt.subplot(3,2,2)
    plt.title("u_imag")
    plt.contourf(X/1e3,Y/1e3,u.imag, 20)
    cv = abs(u.imag).max()
    plt.clim(-cv,cv)
    rcbar = plt.colorbar(format='%.1e')
    cl = plt.getp(rcbar.ax,'ymajorticklabels')
    plt.setp(cl,fontsize=8)

    plt.subplot(3,2,3)
    plt.title("v_real")
    plt.contourf(X/1e3,Y/1e3,v.real, 20)
    cv = abs(v.real).max()
    plt.clim(-cv,cv)
    rcbar = plt.colorbar(format='%.1e')
    cl = plt.getp(rcbar.ax,'ymajorticklabels')
    plt.setp(cl,fontsize=8)

    plt.subplot(3,2,4)
    plt.title("v_imag")
    plt.contourf(X/1e3,Y/1e3,v.imag, 20)
    cv = abs(v.imag).max()
    plt.clim(-cv,cv)
    rcbar = plt.colorbar(format='%.1e')
    cl = plt.getp(rcbar.ax,'ymajorticklabels')
    plt.setp(cl,fontsize=8)

    plt.subplot(3,2,5)
    plt.title("h_real")
    plt.contourf(X/1e3,Y/1e3,h.real, 20)
    cv = abs(h.real).max()
    plt.clim(-cv,cv)
    rcbar = plt.colorbar(format='%.1e')
    cl = plt.getp(rcbar.ax,'ymajorticklabels')
    plt.setp(cl,fontsize=8)

    plt.subplot(3,2,6)
    plt.title("h_imag")
    plt.contourf(X/1e3,Y/1e3,h.imag, 20)
    cv = abs(h.imag).max()
    plt.clim(-cv,cv)
    rcbar = plt.colorbar(format='%.1e')
    cl = plt.getp(rcbar.ax,'ymajorticklabels')
    plt.setp(cl,fontsize=8)

    fig = "figs/RSW_rect_m%d.eps" % i
    plt.savefig(fig, format='eps', dpi=1000)
    plt.show()
