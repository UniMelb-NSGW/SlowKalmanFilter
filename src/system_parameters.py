


import logging
logging.basicConfig()
logging.getLogger(name="KalmanGW").setLevel(logging.INFO)
import configparser
from pathlib import Path

import numpy as np 

def get_project_root() -> Path:
    return Path(__file__).resolve().parent.parent







"""
Class of parameters which completely define the system that is being modelled
"""
class BrownianMotionParameters:


    def __init__(self,
                 T       = 10,               # how long to integrate for in years
                 cadence = 7,                # the interval between observations in days
                 σp = 1e-20,
                 σm = 1e-11,                 # measurement noise standard deviation
                 seed=1,
                 γ  = 1e-7                  # Mean reversion. The same for every pulsar ): 
                 ):



        self.T = T 
        self.cadence = cadence
        self.σp = σp
        self.σm=σm
        self.seed = seed 
        self.γ=γ



        #Use the variables to also create the following quantities

        #Discrete timesteps
        self.dt      = self.cadence * 24*3600 #from days to step_seconds
        end_seconds  = self.T* 365*24*3600 #from years to second
        self.t       = np.arange(0,end_seconds,self.dt)
        self.t_years = self.t/(365*24*3600)


