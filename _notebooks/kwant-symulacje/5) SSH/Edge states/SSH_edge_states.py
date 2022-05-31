import kwant
import matplotlib.pyplot as plt
import scipy.sparse.linalg as sla
import scipy as scp
import numpy as np
from kwant.physics import dispersion

def ssh_model(t_1 = 1, t_2 = 1, L = 100, return_only_ham = 1):
    
    syst = kwant.Builder()
    a = 1
    lat = kwant.lattice.chain(a)

    # Define the scattering region
    for n in range(L):
        syst[lat(n)] = 0

    # Left hopping
    for n in range(L):
        if n%2:
            syst[lat(n-1), lat(n)] = t_1

    # Right hopping
    for n in range(1,L):      
        if not n%2:
            syst[lat(n-1), lat(n)] = t_2

    sym_left_lead = kwant.TranslationalSymmetry([-a])
    left_lead = kwant.Builder(sym_left_lead)
    left_lead[lat(0)] = 0
    left_lead[lat(1), lat(0)] = t_1
    left_lead[lat(2), lat(1)] = t_2
    syst.attach_lead(left_lead)
    left_lead_fin = left_lead.finalized()

    sym_right_lead = kwant.TranslationalSymmetry([a])
    right_lead = kwant.Builder(sym_right_lead)
    right_lead[lat(0)] = 0
    right_lead[lat(1), lat(0)] = t_1
    right_lead[lat(2), lat(1)] = t_2
    syst.attach_lead(right_lead)
    right_lead_fin = right_lead.finalized()

    #kwant.plot(syst)
    syst = syst.finalized()
    
    if(return_only_ham):
        return syst.hamiltonian_submatrix(sparse=False)
    else:
        return syst, syst.hamiltonian_submatrix(sparse=True)
    
# Plots energy spectrum
kwant.plotter.spectrum(ssh_model, x=('t_1', np.linspace(0,4,50)), file = 'energy_spectrum_ssh.png', dpi = 300)

# Plots wavefunction of the system for given energy
syst, ham = ssh_model(0.5, 1, return_only_ham = 0)

def plot_probability(syst, energy):
    wf = kwant.solvers.default.wave_function(syst, energy)
    L=100
    p_1 = []
    p_2 = []
    for i in range(len(wf(0)[0])):
        p_1.append(abs(wf(0)[0][i])**2)
        p_2.append(abs(wf(1)[0][i])**2)
    plt.plot(np.linspace(0,L,L), p_1, '.', linestyle='--')
    plt.plot(np.linspace(0,L,L), p_2,  '.', linestyle='--')

energies = np.linspace(0.3,0.7,10)
for i in range(len(energies)):
    plt.title(f'E = {energies[i]}')
    plot_probability(syst, energies[i])
    plt.savefig(f'states/E_{energies[i]}.png')
    plt.show()