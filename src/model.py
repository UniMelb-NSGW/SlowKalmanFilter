#import numpy as np
from jax import config
config.update("jax_enable_x64", True)
import jax.numpy as jnp


class KalmanPendulum():
    """ 
    The Kalman filter equations for the noisy Pendulum oscillator with process noise.
    Please see NoisyPendulum.ipynb for a detailed explanation
    """


    def __init__(self,dt):
        self.dt = dt
        self.n_states = 2 # the system has 2 hidden states (x,y)
        self.n_y = 1      # the system has one observation state z

    """The discrete measurement noise covariance matrix"""
    def R_matrix(self,σm):
        scalar = σm**2
        return scalar * jnp.ones((self.n_y,self.n_y)) #todo fix ny nomenclature


    """The discrete process noise covariance matrix"""
    def Q_matrix(self,x,σp):
      
        Q11 = σp**2*self.dt**3 / 3
        Q12 = σp**2*self.dt**2 / 2
        Q21 = σp**2*self.dt**2 / 2
        Q22 = σp**2*self.dt
        Q = jnp.array([[Q11,Q12],[Q21,Q22]])

        return Q



    def 



    """The state evolution function dot{x} = f(x)"""
    def f(self,x,g):
        dx = jnp.zeros_like(x)
        
        x1 = x[0] + self.dt*x[1]
        x2 = x[1] - self.dt*g*jnp.sin(x[0])

        dy = -g*jnp.sin(x[0])
        y = x[1] + self.dt*dy

        return jnp.array([x1,x2]) 

    """The Jacobian matrix of the state evolution function F = partial f / partial x 
    This is a matrix of dimension (2,2)"""
    def F_jacobian(self,x,g):
        return jnp.array(([1.0,self.dt],[-g*jnp.cos(x[0])*self.dt,1]))


    """The measurement function z = h(x)"""
    def h(self,x):
        return jnp.sin(x[0])


    """The Jacobian matrix of the measurement function H = partial h / partial x 
    This is a matrix of dimension (2,1)"""
    def H_jacobian(self,x):
        return  jnp.array([jnp.cos(x[0]),0]).reshape(1,2) 

