def make_system(a=1, t_1=1.0, t_2=1.0, L=10):
    lat = kwant.lattice.Polyatomic([[2*a, 0]], [[0, 0], [a, 0]])
    lat.a, lat.b = lat.sublattices
    
    syst = kwant.Builder()
    # Onsites
    syst[(lat.a(n) for n in range(L))] = 0
    syst[(lat.b(n) for n in range(L))] = 0
    # Hopping t1
    syst[((lat.a(n), lat.b(n)) for n in range(L))] = t_1
    # Hopping t2
    syst[((lat.b(n-1), lat.a(n)) for n in range(1,L))] = t_1
    
    # Left lead
    sym_left_lead = kwant.TranslationalSymmetry([-2*a, 0])
    left_lead = kwant.Builder(sym_left_lead)
    left_lead[lat.a(0)] = 0
    left_lead[lat.b(0)] = 0
    left_lead[lat.a(0), lat.b(0)] = t_1
    left_lead[lat.b(0), lat.a(-1)] = t_2
    syst.attach_lead(left_lead)
    left_lead_fin = left_lead.finalized()
    
    # Right lead
    sym_right_lead = kwant.TranslationalSymmetry([2*a, 0])
    right_lead = kwant.Builder(sym_right_lead)
    right_lead[lat.a(0)] = 0
    right_lead[lat.b(0)] = 0
    right_lead[lat.a(0), lat.b(0)] = t_1
    right_lead[lat.a(0), lat.a(1)] = t_2
    syst.attach_lead(right_lead)
    right_lead_fin = right_lead.finalized()
    
    return syst, left_lead_fin, right_lead_fin

def plot_bandstructure(flead, momenta, label=None, title=None):
    bands = kwant.physics.Bands(flead)
    energies = [bands(k) for k in momenta]
    
    plt.figure()
    plt.title(title)
    plt.plot(momenta, energies, label=label)
    plt.xlabel("momentum [(lattice constant)^-1]")
    plt.ylabel("energy [t]")
    
sys, left_lead, right_lead = make_system()
kwant.plot(sys)

for t_2 in np.linspace(0.5, 1.5, 5):
    sys, left_lead, right_lead = make_system(t_1 = 1, t_2 = t_2)
    plot_bandstructure(left_lead, np.linspace(-np.pi, np.pi, 100), t_2, title=f"t_1 = 1, t_2 = {t_2}")
plt.show()
