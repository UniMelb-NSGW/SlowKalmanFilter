

import sdeint
# import numpy as np 

from gravitational_waves import gw_measurement_effect


import logging
import numpy as np 
from utils import block_diag_view_jit
import numpy as np 











"""
Create noisy synthetic data to be consumed by the Kalman filter.
In this example we integrate the 2D vector Ito equation dx = Ax dt + BdW
We assume the state is x = (f).
For e.g. 2 pulsars it is x=(f_1,f_2) 
"""
class SyntheticData:
    
    

    def __init__(self,pulsars,P):


      
        #Create diagonal matrices that can be accepted by vectorized sdeint
        A = np.diag(-pulsars.γ)
        B = np.diag(pulsars.σp)

        #State equation
        #e.g. https://pypi.org/project/sdeint/
        def f(x,t):
            return A.dot(x)
        def g(x,t):
            return B

 
        # Initial condition. All initial heterodyned frequencies are zero
        x0 = np.zeros((pulsars.Npsr)) 


        #Random seeding
        generator = np.random.default_rng(P.seed)


        #Discrete timesteps
        self.t = pulsars.t



        #Integrate 
        self.state = sdeint.itoint(f,g,x0, self.t,generator=generator) #This has shape (Ntimes x Npsr)
        
        
        #Now map from the state space to the measurement space via a measurement equation
        GW = gw_measurement_effect(Ω=P.Ω,
                                     Φ0=P.Φ0,
                                     ψ=P.ψ,
                                     ι=P.ι,
                                     δ=P.δ,
                                     α=P.α,
                                     h=P.h,
                                     q=pulsars.q,
                                     d=pulsars.d,
                                     t=self.t)
 
       
        self.measurement_without_noise = (1.0-GW)*self.state - GW*pulsars.ephemeris
        measurement_noise = generator.normal(0, pulsars.σm,self.measurement_without_noise.shape) # Measurement noise. Seeded
        self.f_measured = self.measurement_without_noise + measurement_noise



