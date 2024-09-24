import numpy as np

class KalmanBrownianMotion():

    """ 
    A simple, linear Kalman filter model used to track Brownian motion
    """

    def __init__(self,dt,N_states):
        self.dt = dt
        self.N_states = N_states

    def F_matrix(self,γ):
        scalar = np.exp(-γ*self.dt)
        return  scalar * np.ones((self.N_states,self.N_states))

    def Q_matrix(self,γ,σp):
        scalar = σp**2 * (1. - np.exp(-2.0*γ* self.dt)) / (2.0 * γ)
        return scalar * np.ones((self.N_states,self.N_states))

    def H_matrix(self):
        return np.ones((self.N_states,self.N_states)) #N states = N measurements for this model

    def R_matrix(self,σm):
        scalar = σm**2
        return scalar * np.ones((self.N_states,self.N_states))












