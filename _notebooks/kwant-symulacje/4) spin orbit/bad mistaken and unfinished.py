import kwant
import numpy as np
import matplotlib.pyplot as plt

import tinyarray

sigma_0 = tinyarray.array([[1, 0], [0, 1]])
sigma_x = tinyarray.array([[0, 1], [1, 0]])
sigma_y = tinyarray.array([[0, -1j], [1j, 0]])
sigma_z = tinyarray.array([[1, 0], [0, -1]])


def make_system(L=5,W=150,t=1,Ez=0.01, alfa=0.1):
    
    lat = kwant.lattice.square()
    syst = kwant.Builder()
    
    def onsite(site):
        (x, y) = site.pos
        return 4 * t* sigma_0 + Ez * sigma_z
    
    def hopping(sitei, sitej):
        xt, yt = sitei.pos
        xs, ys = sitej.pos
        
        t0=-t * sigma_0 
        
        if(xt-xs==0): #hopping in y direction
            t1=1j * alfa * sigma_y / 2
        if(yt-ys==0):
            t1=-1j * alfa * sigma_x / 2
        
        return - t0+t1
    
    
    def onsite_L(site):
        (x, y) = site.pos
        return 4 * t* sigma_0 
    
    
    def hopping_L(sitei, sitej):
        xt, yt = sitei.pos
        xs, ys = sitej.pos
        
        t0=-t * sigma_0 
        
        if(xt-xs==0): #hopping in y direction
            t1=1j * alfa * sigma_y / 2
        if(yt-ys==0):
            t1=-1j * alfa * sigma_x / 2
        
        return -t0+t1
    
    def central_region(pos):
        x, y = pos
        return abs(x) < L and abs(y) < W
    
    syst[lat.shape(central_region, (-1, 1))] = onsite
    syst[lat.neighbors()] = hopping
    
    ### Define the left lead. ####
    lead = kwant.Builder(kwant.TranslationalSymmetry((-1, 0)))

    lead[(lat(0, j) for j in range(-W+1,W))] = 4 * t * sigma_0 + Ez * sigma_z
    # hoppings in x-direction
    lead[kwant.builder.HoppingKind((1, 0), lat, lat)] = \
        -t * sigma_0 + 1j * alfa * sigma_y / 2
    # hoppings in y-directions
    lead[kwant.builder.HoppingKind((0, 1), lat, lat)] = \
        -t * sigma_0 - 1j * alfa * sigma_x / 2
    


    lead2=lead.finalized()
    kwant.plotter.bands(lead2, show=False)
    plt.xlabel("momentum [(lattice constant)^-1]")
    plt.ylabel("energy [t]")
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


def main():
    syst = make_system()

    # Check that the system looks as intended.
    kwant.plot(syst)

    # Finalize the system.
    syst = syst.finalized()

    # We should see non-monotonic conductance steps.
    plot_conductance(syst, energies=[0.01 * i - 0.3 for i in range(100)])


main()