
from numpy import sin, cos
import numpy as np 
import pandas as pd 
import logging

from pathlib import Path
import os 

"""
All variables that are related to the specific PTA configuration used.
"""
class Pulsars:


    def __init__(self,SystemParameters):


        #Define some universal constants
        pc = 3e16     # parsec in m
        c  = 3e8      # speed of light in m/s


        #Load the pulsar data
        #root = get_project_root()
        pulsars = pd.read_csv(SystemParameters.PTA_file)

        if SystemParameters.Npsr != 0:
            pulsars = pulsars.sample(SystemParameters.Npsr,random_state=SystemParameters.seed) #can also use  pulsars.head(N) to sample  

        
        #Extract the parameters
        self.f         = pulsars["F0"].to_numpy()                    # Hz
        self.fdot      = pulsars["F1"] .to_numpy()                   # s^-2
        self.d         = pulsars["DIST"].to_numpy()*1e3*pc/c         # this is in units of s^-1
        self.γ         = np.ones_like(self.f) * SystemParameters.γ   # for every pulsar let γ be 1e-13. Hardcoded value, i.e. not a variable in SystemParameters
        self.δ         = pulsars["DECJD"].to_numpy()                 # radians
        self.α         = pulsars["RAJD"].to_numpy()                  # radians
        self.σp        = np.ones_like(self.f)*SystemParameters.σp    #For now, let all pulsars have the same magnitude of process noise. # ToDo
    
        #Pulsar positions as unit vectors
        self.q         = _unit_vector(np.pi/2.0 -self.δ, self.α) # 3 rows, N columns


        #Discrete timesteps
        self.dt      = SystemParameters.cadence * 24*3600 #from days to step_seconds
        end_seconds  = SystemParameters.T* 365*24*3600 #from years to second
        self.t       = np.arange(0,end_seconds,self.dt)
        

        # Assign some other useful quantities to self
        # Some of these are already defined in SystemParameters, but I don't want to pass
        # the SystemParameters class to the Kalman filter - it should be completely blind
        # to the true parameters - it only knows what we tell it!
        self.Npsr    = len(self.f) 
        self.σm =  SystemParameters.σm
        self.ephemeris = self.f + np.outer(self.t,self.fdot) 



        
        
    
"""
Given a latitude theta and a longitude phi, get the xyz unit vector which points in that direction 
"""
def _unit_vector(theta,phi):
    qx = sin(theta) * cos(phi)
    qy = sin(theta) * sin(phi)
    qz = cos(theta)
    return np.array([qx, qy, qz]).T

