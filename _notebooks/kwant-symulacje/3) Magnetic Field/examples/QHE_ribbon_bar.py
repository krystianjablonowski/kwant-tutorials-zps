#!/usr/bin/env python
# coding: utf-8

# # Quantum Hall edge states in a constriction

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import kwant


# In[2]:


from kwant.digest import gauss
import numpy as np
from cmath import exp, phase
import matplotlib.pyplot as plt


# We make a system that has a constriction in the middle

# In[3]:


def hopping(sitei, sitej, B):
    xi, yi = sitei.pos
    xj, yj = sitej.pos
    return - exp(-0.5j * B * (xi - xj) * (yi + yj))


def make_system(W=40, L=40, W_qpc=10):
    def central_region(pos):
        x, y = pos
        return abs(x) < L/2. and abs(y) < (W-W_qpc)/2. * np.sin(np.pi * x/L)**2 + W_qpc/2.

    lat = kwant.lattice.square()
    sys = kwant.Builder()

    sys[lat.shape(central_region, (-1, 1))] = 4
    sys[lat.neighbors()] = hopping
        
    
    def lead_shape(pos):
        x, y = pos
        return abs(y) < W/2.
    
    lead = kwant.Builder(kwant.TranslationalSymmetry((-1, 0)))
    lead[lat.shape(lead_shape, (1,1))] = 4
    lead[lat.neighbors()] = hopping
    
    sys.attach_lead(lead)
    sys.attach_lead(lead.reversed())
        
    return sys.finalized()


# In[23]:


sys = make_system(L=60, W=40, W_qpc=40)
kwant.plot(sys);


# In[39]:


def plot_wavefunction(n, W_qpc=40):
    sys = make_system(L=60, W=40, W_qpc=W_qpc)
    
    wf = kwant.wave_function(sys, energy=0.5, args=(0.05,))
    
    kwant.plotter.map(sys, abs(wf(0)[n])**2, method='linear')
    
#interact(plot_wavefunction, n=(0,2), W_qpc=(2,40))

plot_wavefunction(n=1,W_qpc=20);


# In[ ]:




