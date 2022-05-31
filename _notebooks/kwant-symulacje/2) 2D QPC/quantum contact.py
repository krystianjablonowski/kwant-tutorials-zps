import kwant
import numpy as np
import matplotlib.pyplot as plt







def make_system(V=1,  W=10, L=30, a=1, t=1.0, plot=0, plot2=0,n=0.25):
    sigx=L/3
    sigy=W*n
    
    # Start with an empty tight-binding system and a single square lattice.
    # `a` is the lattice constant (by default set to 1 for simplicity.
    lat = kwant.lattice.square(a)

    syst = kwant.Builder()
    
    def potential(site):
        (x,y)=site.pos
        
        return V*np.exp(-((x-L/2.)/sigx)**2)*(1-np.exp(-((y-W/2.)/sigy)**2))
        
    def onsite(site):
        return 4 * t + potential(site)
    syst[(lat(x, y) for x in range(L) for y in range(W))] = onsite
    
    syst[lat.neighbors()] = -t
    
    
    
    
    lead = kwant.Builder(kwant.TranslationalSymmetry((-a, 0)))
    lead[(lat(0, j) for j in range(W))] = 4 * t
    lead[lat.neighbors()] = -t
    
    syst.attach_lead(lead)
    syst.attach_lead(lead.reversed())
    
    if (plot==1): ##plot bands and system
        
        lead=lead.finalized()
        kwant.plotter.bands(lead, show=False)
        plt.xlabel("momentum [(lattice constant)^-1]")
        plt.ylabel("energy [meV]")
        plt.show()
        
        plt.figure()
        kwant.plot(syst)
        
        
    if (plot2==1): ##plot potential
    
        plt.figure()
        x=np.linspace(0,L,100)
        y=np.linspace(0,W,int(100*W/L))
        
        X,Y=np.meshgrid(x,y)
         ## to co wyżej
        Z=V*np.exp(-((X-L/2.)/sigx)**2)*(1-np.exp(-((Y-W/2.)/sigy)**2))
        
        im = plt.imshow(Z, cmap=plt.cm.RdBu, extent=(0,L,0,W),
                        interpolation='bilinear', vmax=5)
 
        plt.colorbar(im);
 
        plt.title(('Quantum contact   V='+ '%.2f' % V))
 
        plt.show()
    
    
    syst = syst.finalized()
    return syst
    



def plot_conductanc_energy(syst, energies):
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
    
    
def plot_conductance_pots(energy, pots):
    data= []
    
    pots=np.array(pots)
    
    for i in range(np.size(pots)):
        if ((i+1)%30==0):
            p=1
        else:
            p=0
        
        syst=make_system(pots[i],plot=i,plot2=p)
        smatrix = kwant.smatrix(syst, energy)
        data.append(smatrix.transmission(1, 0))
    
    
    plt.figure()
    plt.plot(pots, data)
    plt.xlabel("V")
    plt.ylabel("conductance [e^2/h]")
    plt.show()
    
    
#sys=make_system()

#plot_conductanc_energy(sys,np.linspace(0.0001,1,100))
    
plot_conductance_pots(3, np.linspace(0.001,5,100))
    
    