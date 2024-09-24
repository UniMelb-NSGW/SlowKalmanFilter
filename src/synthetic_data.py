

import sdeint
import logging
import numpy as np 
import matplotlib.pyplot as plt 
import scienceplots
plt.style.use('science')



class ScalarBrownianMotion:


    def __init__(self,P):

        #Define general matrices to be accepted by sdeint
        A = -P.γ
        B = P.σp
        
        #State equation
        #e.g. https://pypi.org/project/sdeint/
        def f(x,t):
            return A*x
        def g(x,t):
            return B

 
        # Initial condition.
        x0 = 0
        #Random seeding
        generator = np.random.default_rng(P.seed)


        #Discrete timesteps
        self.t = P.t

        #Integrate 
        self.state = sdeint.itoint(f,g,x0, self.t,generator=generator) 
        
        #Create some mean-zero measurement noise
        measurement_noise = generator.normal(0, P.σm,self.state.shape) # Measurement noise. Seeded.

        #...and add it on to the states
        self.measurement =  self.state + measurement_noise

       
      
    """
    A plotting function used to plot the synthetic data.
    Can plot either the states, the measurements, or both.
    """
    def plot(self,plot_state=True, plot_observations=True,plot_points=False):


        #Setup the figure
        h,w = 12,12
        rows = 1
        cols = 1
        fig, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(h,w),sharex=True)
    
        marker=None 
        if plot_points:
            marker='o'


        if plot_state:
            ax.plot(self.t,self.state,label='state',c='C0',marker=marker,scalex=1e-8)
        
        if plot_observations:
            ax.plot(self.t,self.measurement,label='measurement',c='C2',marker=marker,scalex=1e-8)


        fs=20
        ax.set_xlabel('t [s]', fontsize=fs)


        ax.xaxis.set_tick_params(labelsize=fs-4)
        ax.yaxis.set_tick_params(labelsize=fs-4)

        ax.legend(prop={'size':fs})

