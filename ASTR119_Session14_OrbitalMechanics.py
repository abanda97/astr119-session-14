#!/usr/bin/env python
# coding: utf-8

# # Solar System Model

# In[ ]:





# ### Create a simple solar system model

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np
from collections import namedtuple


# ### Define a planet class

# In[2]:


class planet():
    "A planet in our solar system"
    def _init_(self,semimajor,eccentricity):
        self.x = np.array(2)   #x and y position
        self.v = np.array(2)   #x and y velocity
        self.a_g = np.array(2) #x and y acceleration
        self.t = 0.0           #current time
        self.d = 0.0           #current timestep
        self.a = semimajor     #semimajor axis of the orbit
        self.e = eccentricity  #eccentricity of the orbit
        self.istep = 0         #current integer timestepl


# ### Define a dictionary with some constants

# In[3]:


solar_system = {"M_sun":1.0, "G":39.4784176043574320}


# ### Define some functions for setting circular velocity and acceleration

# In[4]:


def solar_circular_velocity(p,solar_system):
    
    G = solar_system["G"]
    M = solar_system[M_sun]
    r = ( p.x[0]**2 + p.x[1]**2 )**0.5
    
    #return the circular velocity
    return (G*M/r)**0.5


# In[6]:


def solar_gravitational_acceleration(p,solar_system):
    
    G = solar_system["G"]
    M = solar_system[M_sun]
    r = ( p.x[0]**2 + p.x[1]**2 )**0.5
    
    #acceleration in AU/yr/yr
    a_grav = -1.0*G*M/r**2
    
    #Find the angle at this position
    if(p.x[0] == 0.0):
        if(p.x[1]>0.0):
            theta = 0.5*np.pi
        else:
            theta = 1.5*np.pi
    else:
        theta = np.atan(p.x[1], p.x[0])
        
    #Set the x and y components of the velocity
    p.a_g[0] = a_grav*np.cos(theta)
    p.a_g[1] = a_grav*np.sin(theta)


# ### Compute the timestep

# In[7]:


def calc_dt(p):
    
    #integration tolerance
    ETA_TIME_STEP = 0.004
    
    #COMPUTE TIMESTEP
    eta = ETA_TIME_STEP
    v = (p.v[0]**2 + p.v[1]**2)**0.5
    a = (p.a_g[0]**2 + p.a_g[1]**2)**0.5
    dt = eta*fp.min(1./np.fabs(v),1./fabs(a))
    
    return dt


# ### Define the initial conditions

# In[ ]:


def SetPlanet(p,i):
    
    AU_in_km = 1.495979e8 #an AU in km

