#!/usr/bin/env python
# coding: utf-8

# In[5]:


import sys
import os
import re
import types
import warnings

import numpy as np
from matplotlib import pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')

import kwant

class SimpleNamespace(types.SimpleNamespace):
    def update(self, **kwargs):
        self.__dict__.update(kwargs)
        return self

## Simple Namespace in Python 2.7
#class SimpleNamespace(object):
#    def __init__(self, **kwargs):
#        self.__dict__.update(kwargs)  


# In[6]:


def qhe_ribbon(W, periodic=False):
    """ Creates a ribbon with magnetic field through it.

    Square lattice, one orbital per site.
    Returns finalized kwant system.

    Arguments required in onsite/hoppings: t, mu, mu_lead

    If we have periodic boundary conditions, the flux through a single 
    unit cell is quantized.
    """
    W = 2 * (W // 2)

    def ribbon_shape(pos):
        (x, y) = pos
        return (-W / 2 <= y <= W / 2)

    def onsite(site, p):
        (x, y) = site.pos
        return 4 * p.t - p.mu

    def hopping(site1, site2, p):
        xt, yt = site1.pos
        xs, ys = site2.pos
        return - p.t * np.exp(-0.5j * p.B * (xt - xs) * (yt + ys))

    def hopping_periodic(site1, site2, p):
        xt, yt = site1.pos
        xs, ys = site2.pos
        return -p.t * np.exp(-0.5j * int(p.B) * 2 * np.pi / (W + 1) * (xt - xs) * (yt + ys))

    lat = kwant.lattice.square()
    sym_syst = kwant.TranslationalSymmetry((-1, 0))
    syst = kwant.Builder(sym_syst)

    syst[lat.shape(ribbon_shape, (0, 0))] = onsite

    if periodic:
        syst[lat.neighbors()] = hopping_periodic
        syst[lat(0, - W / 2), lat(0, + W / 2)] = hopping_periodic
    else:
        syst[lat.neighbors()] = hopping

    return syst

def qhe_hall_bar(L=50, W=10, w_lead=10, w_vert_lead=None):
    """Create a hall bar system. 

    Square lattice, one orbital per site.
    Returns finalized kwant system.

    Arguments required in onsite/hoppings: 
        t, mu, mu_lead
    """

    L = 2 * (L // 2)
    W = 2 * (W // 2)
    w_lead = 2 * (w_lead // 2)
    if w_vert_lead is None:
        w_vert_lead = w_lead
    else:
        w_vert_lead = 2 * (w_vert_lead // 2)

    # bar shape
    def bar(pos):
        (x, y) = pos
        return (x >= -L / 2 and x <= L / 2) and (y >= -W / 2 and y <= W / 2)

    # Onsite and hoppings
    def onsite(site, p):
        (x, y) = site.pos
        return 4 * p.t - p.mu

    def hopping_Ax(site1, site2, p):
        xt, yt = site1.pos
        xs, ys = site2.pos
        return -p.t * np.exp(-0.5j * p.B * (xt + xs) * (yt - ys))

    def make_lead_hop_y(x0):
        def hopping_Ay(site1, site2, p):
            xt, yt = site1.pos
            xs, ys = site2.pos
            return -p.t * np.exp(-1j * p.B * x0 * (yt - ys))
        return hopping_Ay

    def lead_hop_vert(site1, site2, p):
        return -p.t

    # Building system
    lat = kwant.lattice.square()
    syst = kwant.Builder()

    syst[lat.shape(bar, (0, 0))] = onsite
    syst[lat.neighbors()] = hopping_Ax

    # Attaching leads
    sym_lead = kwant.TranslationalSymmetry((-1, 0))
    lead = kwant.Builder(sym_lead)

    def lead_shape(pos):
        (x, y) = pos
        return (-w_lead / 2 <= y <= w_lead / 2)

    lead_onsite = lambda site, p: 4 * p.t - p.mu_lead

    sym_lead_vertical = kwant.TranslationalSymmetry((0, 1))
    lead_vertical1 = kwant.Builder(sym_lead_vertical)
    lead_vertical2 = kwant.Builder(sym_lead_vertical)

    def lead_shape_vertical1(pos):
        (x, y) = pos
        return (-L / 4 - w_vert_lead / 2 <= x <= -L / 4 + w_vert_lead / 2)

    def lead_shape_vertical2(pos):
        (x, y) = pos
        return (+L / 4 - w_vert_lead / 2 <= x <= +L / 4 + w_vert_lead / 2)

    lead_vertical1[lat.shape(lead_shape_vertical1, (-L / 4, 0))] = lead_onsite
    lead_vertical1[lat.neighbors()] = lead_hop_vert
    lead_vertical2[lat.shape(lead_shape_vertical2, (L / 4, 0))] = lead_onsite
    lead_vertical2[lat.neighbors()] = lead_hop_vert

    syst.attach_lead(lead_vertical1)
    syst.attach_lead(lead_vertical2)

    syst.attach_lead(lead_vertical1.reversed())
    syst.attach_lead(lead_vertical2.reversed())

    lead[lat.shape(lead_shape, (-1, 0))] = lead_onsite
    lead[lat.neighbors()] = make_lead_hop_y(-L / 2)

    syst.attach_lead(lead)

    lead = kwant.Builder(sym_lead)
    lead[lat.shape(lead_shape, (-1, 0))] = lead_onsite
    lead[lat.neighbors()] = make_lead_hop_y(L / 2)

    syst.attach_lead(lead.reversed())

    return syst


# In[67]:


p = SimpleNamespace(t=1, mu=0.5, B=0.05)

syst = qhe_ribbon(W=40)

#kwant.plot(syst,show=False);

# finalize the system
fsyst = syst.finalized()

kwant.plotter.bands(syst.finalized(), args=([p]), show=False)
plt.grid()
plt.ylim(-0.5,2)


# In[65]:


p = SimpleNamespace(t=1, mu=0.5, B=0.15)

syst = qhe_hall_bar(L=40,W=20)

kwant.plot(syst,show=False);


# In[66]:


p = SimpleNamespace(t=1, mu = 0.6, mu_lead=0.6, B=0.15, phi=0.0)
syst = qhe_hall_bar(L=200, W=100).finalized()
ldos = kwant.ldos(syst, energy=0.0, args=[p])

fig = plt.figure(figsize=[20,20])
ax = fig.add_subplot(1,2,1)
ax.axis('off')
kwant.plotter.map(syst, ldos, num_lead_cells=20, colorbar=False, ax=ax);


# In[ ]:




