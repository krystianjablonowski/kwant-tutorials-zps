#!/usr/bin/env python
# coding: utf-8

# # Quantum Hall edge states in a constriction

# In[1]:


#get_ipython().run_line_magic('matplotlib', 'inline')
import kwant


# In[2]:



import numpy as np
from cmath import exp
import matplotlib.pyplot as plt
from types import SimpleNamespace


#get_ipython().run_line_magic('matplotlib', 'inline')
# We make a system that has a constriction in the middle

# In[3]:


def inf(params,periodic=False):
    
    """ Creates a ribbon with magnetic field through it.

    Square lattice, one orbital per site.
    Returns finalized kwant system.

    Arguments required in onsite/hoppings: t, mu, mu_lead

    If we have periodic boundary conditions, the flux through a single 
    unit cell is quantized."""
    
    W=2*(params.W//2)
    
    def ribbon_shape(pos):
        (x, y) = pos
        return (-W / 2 <= y <= W / 2) 

    def onsite2(site, params):
        (x, y) = site.pos
        return 4 * params.t - params.mu

    def hopping(site1, site2, params):
        xt, yt = site1.pos
        xs, ys = site2.pos
        return - params.t
        #return - params.t * np.exp(-0.5j * params.B * (xt - xs) * (yt + ys))

    def hopping_periodic(site1, site2, params):
        xt, yt = site1.pos
        xs, ys = site2.pos
        return -params.t * np.exp(-0.5j * int(params.B) * 2 * np.pi / (W + 1) * (xt - xs) * (yt + ys))

    lat = kwant.lattice.square()
    sym_syst = kwant.TranslationalSymmetry((-1, 0))
    syst = kwant.Builder(sym_syst)

    syst[lat.shape(ribbon_shape, (0, 0))] = onsite2

    if periodic:
        syst[lat.neighbors()] = hopping_periodic
        syst[lat(0, - W / 2), lat(0, + W / 2)] = hopping_periodic
    else:
        syst[lat.neighbors()] = hopping

    return syst    



def make_system(params):
    
    
    def hopping(sitei, sitej,params):
        xi, yi = sitei.pos
        xj, yj = sitej.pos
    
        B=params.B
    #B=p.B*np.exp(-((xi+xj)/(p.L))**10)
        if(xi<=-params.L//2):
            B=B/params.turn_of*(xj+params.L//2+params.turn_of)
        elif(xj>params.L//2):
            B=B-B/params.turn_of*(xi-params.L//2)
    
        return - exp(-0.5j * B * (xi - xj) * (yi + yj))*params.t

    def hopping_l(sitei, sitej, p):
        xi, yi = sitei.pos
        xj, yj = sitej.pos
        return -p.t

    def onsite(site,params):
        x, y = site.pos
    
        sigx=(params.L)/3
        sigy=params.W*params.n
        
        po=params.V*np.exp(-((x)/sigx)**2)*(1-np.exp(-(y/sigy)**2))
        return 4 * params.t + po- params.mu
    
    
    def central_region(pos):
        x, y = pos
        return abs(x) < params.L//2+params.turn_of and abs(y) < params.W//2.
    


    

    lat = kwant.lattice.square()
    sys = kwant.Builder()

    sys[lat.shape(central_region, (-1, 1))] = onsite
    sys[lat.neighbors()] = hopping
        
    
    def lead_shape(pos):
        x, y = pos
        return abs(y) < params.W/(2)


    
    lead = kwant.Builder(kwant.TranslationalSymmetry((-1, 0)))
    lead[lat.shape(lead_shape, (1,1))] = 4*params.t
    lead[lat.neighbors()] = hopping_l
    
    sys.attach_lead(lead)
    sys.attach_lead(lead.reversed())
           
    syst_f=sys.finalized()    
        
    plt.figure()

    return syst_f



def plot_things(sys ,params, n=0, plot_sys=0 ,plot_p=0 , plot_d=1, plot_b=0, ener=1):
    
    if(plot_b!=0):
       
        band_sys=inf(params)
        
        fsyst = band_sys.finalized()

        kwant.plotter.bands(fsyst, args=([params]), show=False)
        plt.grid()
        plt.ylim(-1,6)
        plt.show()
    
    
    
    if plot_sys!=0:
        
        kwant.plot(sys);
        plt.show()
        
        
    if plot_p!=0:
        plot_pot(params)
    


    if plot_d!=0:
        wf = kwant.wave_function(sys, energy=ener, args=[params])
        
        t=np.shape(wf(0))
        nwf=np.empty_like(wf(0)[0])
        
        for i in range(t[0]):
            nwf+=wf(0)[i]
        
        title="B=" + f'{params.B:9.4f}' + "   V=" + str(params.V) + "    E = "+ str(ener)
        kwant.plotter.map(sys, abs(nwf)**2, method='linear',show=False)
        plt.title(title)
        plt.savefig(title+".png")
        
        plt.close()

    #smatrix = kwant.smatrix(sys, energy=ener, args=[params])

    return sys
      

    


def plot_pot(p):
    
    sigx=(p.L)/3
    sigy=p.W*p.n
        
    a=plt.figure()
   
    x=np.linspace(-int(p.L)//2-p.turn_of,int(p.L)//2+p.turn_of,100)
    y=np.linspace(-int(p.W)//2,int(p.W)//2,int(100*p.W/p.L))
        
    X,Y=np.meshgrid(x,y)
         ## to co wyżej
    Z=p.V*np.exp(-((X)/sigx)**2)*(1-np.exp(-((Y)/sigy)**2))
        
    im = plt.imshow(Z, cmap=plt.cm.twilight, extent=(-p.L//2-p.turn_of,p.L//2+p.turn_of,-p.W//2,p.W//2),
                        interpolation='bilinear', vmax=p.V)
    
    a.colorbar(im);
 
    plt.title(('V='+ '%.2f' % p.V+" B="+ '%.2f' % p.B))
 
    plt.show()

    
    
    
    
    
    
    
params=SimpleNamespace(L=100 ,B=0, V=10, t=1., W=200, n=0.4, mu=0.1, turn_of=50)
sys = make_system(params=params)


plot_things(sys,params=params, ener=1.2,n=0,plot_d=0,plot_p=0,plot_sys=1,plot_b=1)


B=np.linspace(0,0.1,1000)




def plot_conductance_b(energy, B):
    data= []
    
    
    
    for i in range(np.size(B)):
        if ((i)%30==0):
            p=1
            #print(B[i])
        else:
            p=0
        params.B=B[i]
        print(B[i])
        syst=plot_things(sys,params, plot_d=p, ener=energy)
        smatrix = kwant.smatrix(syst, energy=energy, args=[params])
        data.append(smatrix.transmission(1, 0))
    
    
    plt.figure()
    plt.plot(B, data)
    plt.xlabel("B")
    plt.ylabel("conductance [e^2/h]")
    plt.title("   V=" + str(params.V) + "    E = "+ str(energy))
    plt.savefig(" Cond  V=" + str(params.V) + "    E = "+ str(energy)+".png")
    plt.show()
    

    
    return data

results=plot_conductance_b(1.2,B)

plt.figure()
plt.plot(B, 1/np.array(results))
plt.xlabel("B")
plt.ylabel("resistance [h/e^2]")
plt.title("   V=" + str(params.V) + "    E = "+ str(1.2))
plt.savefig(" Res  V=" + str(params.V) + "    E = "+ str(1.2)+".png")
plt.show()




#posumować wavefunctions po n

#pętla po params a nie nowe układy





