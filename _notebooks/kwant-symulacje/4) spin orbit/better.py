import kwant
import numpy as np
import matplotlib.pyplot as plt

import tinyarray

sigma_0 = tinyarray.array([[1, 0], [0, 1]])
sigma_x = tinyarray.array([[0, 1], [1, 0]])
sigma_y = tinyarray.array([[0, -1j], [1j, 0]])
sigma_z = tinyarray.array([[1, 0], [0, -1]])


def make_system(L=50,W=51,t=1.0,e_z=0.3, alpha=1):
    
    lat  = kwant.lattice.square(norbs=2)
    syst = kwant.Builder()
    
    def onsite(site):
        (x, y) = site.pos
        return 4 * t* sigma_0 + e_z * sigma_z
    
    def hopping(sitei, sitej):
        xt, yt = sitei.pos
        xs, ys = sitej.pos
        
        t0=-t * sigma_0 
        
        if(xt-xs==1): #hopping in x direction
            t1=1j * alpha * sigma_y / 2
        if(yt-ys==1):
            t1=-1j * alpha * sigma_x / 2
        
        return t0+t1
    
    
    def onsite_L(site):
        (x, y) = site.pos
        return 4 * t* sigma_0 + e_z * sigma_z
    
    
    def hopping_L(sitei, sitej):
        xt, yt = sitei.pos
        xs, ys = sitej.pos
        
        t0=-t * sigma_0
        
        if(xt-xs==1): #hopping in x direction
            t1=1j * alpha * sigma_y / 2
        if(yt-ys==1):
            t1=-1j * alpha * sigma_x / 2
        
        return t0+t1
    
    def central_region(pos):
        x, y = pos
        return (x < L and x>=0 and y < W and y>=0)
    
    syst[lat.shape(central_region, (0, 0))] = onsite
    syst[lat.neighbors()] = hopping
    
    def lead_shape(pos):
        x, y = pos
        return y < W and y>=0


    
    lead = kwant.Builder(kwant.TranslationalSymmetry((-1, 0)))
    lead[lat.shape(lead_shape, (-1,0))] = onsite_L
    lead[lat.neighbors()] = hopping_L


    lead2=lead.finalized()
    kwant.plotter.bands(lead2, show=False)
    plt.xlabel("momentum [(lattice constant)^-1]")
    plt.ylabel("energy [t]")
    plt.ylim(-0.5,0.83)
    #plt.xlim(-1,1)
    plt.show()
    

    #### Attach the leads and return the finalized system. ####
    syst.attach_lead(lead)
    syst.attach_lead(lead.reversed())

    return syst
    
    
def plot_conductance(syst, energies):
    # Compute conductance
    data = []
    for energy in energies:
        smatrix = kwant.smatrix(syst, energy)
        data.append(smatrix.transmission(1, 0))

    plt.figure()
    plt.plot(energies, data)
    plt.xlabel("energy [t]")
    plt.ylabel("conductance [e^2/h]")
    plt.show()

def plot_density(sys,ener):
    
    wf = kwant.wave_function(sys, energy=ener)
    
    t=np.shape(wf(0))
    nwf=np.empty_like(wf(0)[0][::2])
    
    for i in range(t[0]):
        i=3
        psi=wf(0)[i]
        up, down = psi[::2], psi[1::2]
        nwf+=(down+up)
        break
    
    
    
    
    title="E = "+ str(ener)
    kwant.plotter.map(sys, abs(down), method='linear',show=False)
    plt.title(title)
    plt.show()
    
    plt.close()


def main():
    syst = make_system()

    # Check that the system looks as intended.
    kwant.plot(syst)

    # Finalize the system.
    syst = syst.finalized()

    # We should see non-monotonic conductance steps.
    plot_conductance(syst, energies=[0.004 * i - 0.3 for i in range(4)])
    
    for i in [-0.3,0,0.4]:
        plot_density(syst,i)

main()