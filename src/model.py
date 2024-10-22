#import numpy as np
from jax import config
config.update("jax_enable_x64", True)
import jax.numpy as jnp

import numpy as np 
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







class UKFKalmanPendulum():
    """ 
    The UKF equations for the noisy Pendulum oscillator with process noise.
    Please see NoisyPendulum.ipynb for a detailed explanation
    It might be nicer to have this as a joint class with KalmanPendulum, but the additional dimension due to the
    UKF sigma points make it a bit cumbersome.
    Note how we need to define far fewer objects for the UKF
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

    def derivative_function(self,x, g):
        x0 = x[0]
        x1 = x[1]
        rhs = np.asarray([x1, -g*np.sin(x0)]) #jnp seems to slow things down a lot here 
        return rhs

    """The measurement function z = h(x)"""
    def sigma_point_measurement_function(self,x):
        return np.sin(x[:,0]) # x has dimension (number_of_sigma_points, number_of_states). We want all the sigma points, but just the 0th state

    """The measurement function z = h(x)"""
    def measurement_function(self,x):
        return np.sin(x[0])  # x has dimension (number_of_states). We want all the sigma points, but just the 0th state


    #currently we have two measurement functions. One which operates on the sigma points and one which operates on the states
    #we can probably clean this up to be more concise. todo
    #Note that measurment_function is not actually required for the filter to run, it is just useful for plotting.







class UKFKalmanPendulumEstimateG():
    """ 
    As UKFKalmanPendulum but now with the state extended to include the gravitational field strength parameter g
    """


    def __init__(self,dt):
        self.dt = dt
        self.n_states = 3 # the system has 2 hidden states (x,y)
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
        Q33 = 1e-4
        Q = np.array([[Q11,Q12],[Q21,Q22]])

        return Q

    def derivative_function(self,x, g):
        x0 = x[0]
        x1 = x[1]
        rhs = np.asarray([x1, -g*np.sin(x0)]) #jnp seems to slow things down a lot here 
        return rhs

    """The measurement function z = h(x)"""
    def sigma_point_measurement_function(self,x):
        return np.sin(x[:,0]) # x has dimension (number_of_sigma_points, number_of_states). We want all the sigma points, but just the 0th state

    """The measurement function z = h(x)"""
    def measurement_function(self,x):
        return np.sin(x[0])  # x has dimension (number_of_states). We want all the sigma points, but just the 0th state


    #currently we have two measurement functions. One which operates on the sigma points and one which operates on the states
    #we can probably clean this up to be more concise. todo
    #Note that measurment_function is not actually required for the filter to run, it is just useful for plotting.