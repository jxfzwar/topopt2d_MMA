from __future__ import division
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
from matplotlib import colors
import matplotlib.pyplot as plt
import nlopt
import time


#Default input parameters
class para:
    nelx = 60
    nely = 20
    volfrac = 0.5
    rmin = 1.5
    penal = 3.0
    ft = 0

class eMat:
    def __init__(self,nelx,nely):
        self.nelx=nelx
        self.nely=nely
    def Mat(self):
        nelx=self.nelx
        nely=self.nely
        Mat = np.zeros((nelx * nely, 8), dtype=int)
        for elx in range(nelx):
            for ely in range(nely):
                el = ely + elx * nely
                n1 = (nely + 1) * elx + ely
                n2 = (nely + 1) * (elx + 1) + ely
                Mat[el, :] = np.array(
                    [2 * n1 + 2, 2 * n1 + 3, 2 * n2 + 2, 2 * n2 + 3, 2 * n2, 2 * n2 + 1, 2 * n1, 2 * n1 + 1])
        return Mat

class FE:
    def __init__(self,x,nelx,nely,volfrac,rmin,penal,ft,Emin,Emax,KE):
        self.x=x
        self.nelx=nelx
        self.nely=nely
        self.volfrac=volfrac
        self.rmin=rmin
        self.penal=penal
        self.ft=ft
        self.Emin=Emin
        self.Emax=Emax
        self.KE=KE
    def Usolution(self):
        x=self.x
        nelx=self.nelx
        nely=self.nely
        volfrac=self.volfrac
        rmin=self.rmin
        penal=self.penal
        ft=self.ft
        Emin=self.Emin
        Emax=self.Emax
        KE=self.KE
        # dofs:
        ndof = 2 * (nelx + 1) * (nely + 1)
        # FE: Build the index vectors for the for coo matrix format.
        ee = eMat(nelx,nely)
        edofMat = ee.Mat()
        # Construct the index pointers for the coo format
        iK = np.kron(edofMat, np.ones((8, 1))).flatten()
        jK = np.kron(edofMat, np.ones((1, 8))).flatten()
        # BC's and support
        dofs = np.arange(2 * (nelx + 1) * (nely + 1))
        fixed = np.union1d(dofs[0:2 * (nely + 1):2], np.array([2 * (nelx + 1) * (nely + 1) - 1]))
        free = np.setdiff1d(dofs, fixed)
        # Solution and RHS vectors
        f = np.zeros((ndof, 1))
        u = np.zeros((ndof, 1))
        # Set load
        f[1, 0] = -1
        # Setup and solve FE problem
        sK = ((KE.flatten()[np.newaxis]).T * (Emin + x ** penal * (Emax - Emin))).flatten(order='F')
        K = coo_matrix((sK, (iK, jK)), shape=(ndof, ndof)).tocsc()
        # Remove constrained dofs from matrix
        K = K[free, :][:, free]
        # Solve system
        u[free, 0] = spsolve(K, f[free, 0])
        return u


class FILTERMATRIX:
    def __init__(self,nelx,nely,rmin):
        self.nelx=nelx
        self.nely=nely
        self.rmin=rmin
    def assembly(self):
        nelx=self.nelx
        nely=self.nely
        rmin=self.rmin
        nfilter = nelx * nely * ((2 * (np.ceil(rmin) - 1) + 1) ** 2)
        iH = np.zeros(nfilter)
        jH = np.zeros(nfilter)
        sH = np.zeros(nfilter)
        cc = 0
        for i in range(nelx):
            for j in range(nely):
                row = i * nely + j
                kk1 = int(np.maximum(i - (np.ceil(rmin) - 1), 0))
                kk2 = int(np.minimum(i + np.ceil(rmin), nelx))
                ll1 = int(np.maximum(j - (np.ceil(rmin) - 1), 0))
                ll2 = int(np.minimum(j + np.ceil(rmin), nely))
                for k in range(kk1, kk2):
                    for l in range(ll1, ll2):
                        col = k * nely + l
                        fac = rmin - np.sqrt(((i - k) * (i - k) + (j - l) * (j - l)))
                        iH[cc] = row
                        jH[cc] = col
                        sH[cc] = np.maximum(0.0, fac)
                        cc = cc + 1
        # Finalize assembly and convert to csc format
        H = coo_matrix((sH, (iH, jH)), shape=(nelx * nely, nelx * nely)).tocsc()
        return H


#Optimization Problem Definition
def objfunc(x, grad):
    if grad.size > 0:
        # Default input parameters
        p = para()
        nelx = p.nelx
        nely = p.nely
        volfrac = p.volfrac
        rmin = p.rmin
        penal = p.penal
        ft = p.ft  # ft==0 -> sens, ft==1 -> dens
        # Max and min stiffness
        Emin = 1e-3
        Emax = 1.0
        # dofs:
        ndof = 2 * (nelx + 1) * (nely + 1)
        # Finalize assembly and convert to csc format
        HH = FILTERMATRIX(nelx, nely, rmin)
        H = HH.assembly()
        Hs = H.sum(1)
        # KE Matrix
        KE = lk()
        # FE
        uu = FE(x, nelx, nely, volfrac, rmin, penal, ft, Emin, Emax, KE)
        u = uu.Usolution()
        # sensitivity
        dv = np.ones(nely * nelx)
        dc = np.ones(nely * nelx)
        ce = np.ones(nely * nelx)
        ee = eMat(nelx, nely)
        edofMat = ee.Mat()
        ce[:] = (np.dot(u[edofMat].reshape(nelx * nely, 8), KE) * u[edofMat].reshape(nelx * nely, 8)).sum(1)
        dc[:] = (-penal * x ** (penal - 1) * (Emax - Emin)) * ce
        dv[:] = np.ones(nely * nelx)
        # Sensitivity filtering:
        if ft == 0:
            dc[:] = np.asarray((H * (x * dc))[np.newaxis].T / Hs)[:, 0] / np.maximum(0.001, x)
        elif ft == 1:
            dc[:] = np.asarray(H * (dc[np.newaxis].T / Hs))[:, 0]
            dv[:] = np.asarray(H * (dv[np.newaxis].T / Hs))[:, 0]
        # gradient of obj
        grad[:] = dc
    f = ((Emin + x ** penal * (Emax - Emin)) * ce).sum()
    return f

def Constraint1(x, grad):
    if grad.size > 0:
        p = para()
        nelx = p.nelx
        nely = p.nely
        # gradient of con
        dg = list(1 / (0.5 * nelx * nely + 1e-2) for i in range(nelx * nely))
        grad[:] = np.array(dg)
    return x.sum()/(0.5 * nelx * nely  + 1e-2)-1

def Constraint2(x, grad):
    if grad.size > 0:
        p = para()
        nelx = p.nelx
        nely = p.nely
        # gradient of con
        dg = list(-1 / (0.5 * nelx * nely - 1e-2) for i in range(nelx * nely))
        grad[:] = np.array(dg)
    return -x.sum()/(0.5 * nelx * nely - 1e-2)+1

# MAIN DRIVER
def main():
    # Default input parameters
    p=para()
    nelx = p.nelx
    nely = p.nely
    volfrac = p.volfrac
    rmin = p.rmin
    penal = p.penal
    ft = p.ft # ft==0 -> sens, ft==1 -> dens
    # Max and min stiffness
    Emin = 1e-3
    Emax = 1.0
    # Set loop counter and gradient vectors
    loop = 0
    change = 1
    # Allocate design variables (as array), initialize and allocate sens.
    xx = volfrac * np.ones(nely * nelx, dtype=float)
    # Initialize plot and plot the initial design
    plt.ion()
    fig, ax = plt.subplots()
    im = ax.imshow(-xx.reshape((nelx, nely)).T, cmap='gray', \
                   interpolation='none', norm=colors.Normalize(vmin=-1, vmax=0))
    fig.show()

    t1 = 0

    while change > 0.01 and loop < 2000:

        t1old=t1
        t1 = time.clock()

        xxold = xx.copy()
        loop = loop + 1
        #use pyOpt to solve the problem
        opt = nlopt.opt(nlopt.LD_MMA, nelx*nely)
        #minimize the objective function
        opt.set_min_objective(objfunc)
        #bond constraints
        opt.set_lower_bounds(Emin)
        opt.set_upper_bounds(Emax)
        #Nonlinear constraints
        opt.add_inequality_constraint(Constraint1, 1e-6)
        opt.add_inequality_constraint(Constraint2, 1e-6)
        #Stopping criteria
        opt.set_ftol_rel(1e-4)
        #Performing the optimization
        x = opt.optimize(xx)

        t2=time.clock()

        # Filter design variables
        # Finalize assembly and convert to csc format
        HH = FILTERMATRIX(nelx, nely, rmin)
        H = HH.assembly()
        Hs = H.sum(1)
        if ft == 0:
            xx = x
        elif ft == 1:
            xx = np.asarray(H * x[np.newaxis].T / Hs)[:, 0]

        obj = opt.last_optimum_value()
        vol = xx.sum()/(nelx*nely)
        #Compute the change by the inf. norm
        change = np.linalg.norm(xx.reshape(nelx * nely, 1) - xxold.reshape(nelx * nely, 1), np.inf)

        t3=time.clock()

        # Plot to screen
        im.set_array(-xx.reshape((nelx, nely)).T)
        fig.canvas.draw()
        #plt.pause(0.01)

        t4=time.clock()

        # Write iteration history to screen (req. Python 2.6 or newer)
        print("it.: {0} , obj.: {1:.3f} Vol.: {2:.3f}, ch.: {3:.3f}".format( \
            loop, obj, vol, change))
        print("t1-t1old.: {0:.5f} , t2-t1.: {1:.5f} , t3-t2.: {2:.5f} , t3-t2.: {3:.5f}". \
            format(t1-t1old, t2-t1, t3-t2, t4-t3))

    # Make sure the plot stays and that the shell remains
    plt.show()

    raw_input("Press any key...")

#element stiffness matrix
def lk():
	E=1
	nu=0.3
	k=np.array([1/2-nu/6,1/8+nu/8,-1/4-nu/12,-1/8+3*nu/8,-1/4+nu/12,-1/8-nu/8,nu/6,1/8-3*nu/8])
	KE = E/(1-nu**2)*np.array([ [k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7]],
	[k[1], k[0], k[7], k[6], k[5], k[4], k[3], k[2]],
	[k[2], k[7], k[0], k[5], k[6], k[3], k[4], k[1]],
	[k[3], k[6], k[5], k[0], k[7], k[2], k[1], k[4]],
	[k[4], k[5], k[6], k[7], k[0], k[1], k[2], k[3]],
	[k[5], k[4], k[3], k[2], k[1], k[0], k[7], k[6]],
	[k[6], k[3], k[4], k[1], k[2], k[7], k[0], k[5]],
	[k[7], k[2], k[1], k[4], k[3], k[6], k[5], k[0]] ]);
	return (KE)

# The real main driver
if __name__ == "__main__":
	main()