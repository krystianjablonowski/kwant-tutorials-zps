import kwant
from matplotlib import pyplot as plt
import tinyarray as ta
import numpy as np
import scipy.sparse.linalg as sla

def make_system(a=1, t_1=1.0, t_2=1.0, L=10):
    lat = kwant.lattice.Polyatomic([[3*a/2, np.sqrt(3)*a/2], [3*a/2, -np.sqrt(3)*a/2]], [[0, 0], [a,0]])
    lat.a, lat.b = lat.sublattices
    
    syst = kwant.Builder()
    onsite = 0.8
    # Onsites
    syst[(lat.a(n, m) for n in range(L) for m in range(L))] = onsite
    syst[(lat.b(n, m) for n in range(L) for m in range(L))] = onsite
    # Hopping
    hoppings = (((0, a), lat.a, lat.b), ((a, 0), lat.a, lat.b), ((-a/2, -np.sqrt(3)*a/2), lat.a, lat.b), ((-a/2, np.sqrt(3)*a/2), lat.a, lat.b))
    syst[[kwant.builder.HoppingKind(*hopping) for hopping in hoppings]] = t_1
    hoppings2_a = (((0, a), lat.a, lat.a), ((a, 0), lat.a, lat.a))
    hoppings2_b = (((0, a), lat.b, lat.b), ((a, 0), lat.b, lat.b))
    syst[[kwant.builder.HoppingKind(*hopping) for hopping in hoppings2_a]] = t_2
    syst[[kwant.builder.HoppingKind(*hopping) for hopping in hoppings2_b]] = t_2

    
    LEADS = True
    if LEADS:
        # Right lead
        sym_right_lead = kwant.TranslationalSymmetry([-3*a/2, -np.sqrt(3)*a/2])
        right_lead = kwant.Builder(sym_right_lead)
        right_lead[(lat.a(n, m) for n in range(L) for m in range(L))] = onsite
        right_lead[(lat.b(n, m) for n in range(L) for m in range(L))] = onsite
        
        # Hopping
        right_lead[[kwant.builder.HoppingKind(*hopping) for hopping in hoppings]] = t_1
        right_lead[[kwant.builder.HoppingKind(*hopping) for hopping in hoppings2_a]] = t_2
        right_lead[[kwant.builder.HoppingKind(*hopping) for hopping in hoppings2_b]] = t_2
        
        syst.attach_lead(right_lead)
        right_lead_fin = right_lead.finalized()
        
        syst.attach_lead(right_lead.reversed())
    
    return syst, right_lead_fin

def plot_bandstructure(flead, momenta, label=None, title=None):
    bands = kwant.physics.Bands(flead)
    energies = [bands(k) for k in momenta]
    
    plt.figure()
    plt.title(title)
    plt.plot(momenta, energies, label=label)
    plt.xlabel("momentum [(lattice constant)^-1]")
    plt.ylabel("energy [t]")

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

def plot_density(sys,ener, it=-1):
    
    wf = kwant.wave_function(sys, energy=ener)
    
    t=np.shape(wf(0))
    nwf=np.zeros(np.shape(wf(0)[0][::2]),dtype=float)
    
    
    for i in range(min(t[0],5)):

        psi=wf(0)[i]
        up, down = psi[::2], psi[1::2]
        up2=(np.abs(up))**2
        nwf+=up2
    
    
    f= nwf

    
    if it==-1:
        title="density"
    elif it>-1:
        title= "density"
        
    title2=title+".png"
    plt.title(title)
    kwant.plotter.map(sys, f,method='linear',vmax=max(nwf),vmin=0,show=False,file=title2)
    
    plt.show()
    
    plt.close()

sys, right_lead = make_system(t_1 = 1, t_2 = 0.5*np.exp(1j))
kwant.plot(sys, file='haldane.pdf')
sys = sys.finalized()

#plot_conductance(sys, np.linspace(0,2,1000))
#plot_density(sys, 1)

for phi in np.linspace(2*np.pi/5, 3*np.pi/2, 10):
    t_2 = 0.1 * np.exp(1j*phi)
    sys, right_lead = make_system(t_1 = 1, t_2 = t_2)
    plot_bandstructure(right_lead, np.linspace(-np.pi, np.pi, 100), t_2, title=f"t_1 = 1, t_2 = {t_2}")
    plt.savefig(f't_1 = 1 t_2 = {round(t_2,4)}.pdf')
plt.show()