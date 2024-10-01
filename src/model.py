import numpy as np


class KalmanVDP():
    """ 
    The Kalman filter equations for the Van der Pol oscillator with process noise.
    Please see xx.ipynb for a detailed explanation
    """


    def __init__(self,dt):
        self.dt = dt
        self.n_states = 2 # the system has 2 hidden states (x,y)
        self.n_y = 1 # the system has one observation state z

    """
    The discrete measurement noise covariance matrix
    """
    def R_matrix(self,σm):
        scalar = σm**2
        return scalar * np.ones((self.n_y,self.n_y)) #todo fix ny nomenclature


    """
    The discrete process noise covariance matrix
    """
    def Q_matrix(self,x,μ,σp):
      
        F21 = -2.0*μ*x[0]*x[1]-1.0

        Q11 = 1 
        Q12 = self.dt*F21/2 
        Q21 = Q12 
        Q22 = self.dt**2*(F21**2) / 3 

        return self.dt*σp**2*np.array(([Q11,Q12],[Q21,Q22]))


    """
    The state evolution function \dot{x} = f(x)
    """
    def f(self,x,μ):
        dx = np.zeros_like(x)
        dx[0] = x[1]
        dx[1] = μ*(1-x[0]**2)*x[1] - x[0]

        return dx 

    """
    The Jacobian matrix of the state evolution function F = \partial f / \partial x 
    This is a matrix of dimension (2,2)
    """
    def F_jacobian(self,x,μ):
        return np.array(([0.0,1.0],[-2.0*μ*x[0]*x[1]-1.0,μ*(1-x[0]**2)]))


    """
    The measurement function z = h(x)
    """
    def h(self,x):
        return x[0]**2 * x[1] + x[1]


    """
    The Jacobian matrix of the measurement function H = \partial h / \partial x 
    This is a matrix of dimension (2,1)
    """
    def H_jacobian(self,x):
        return np.ndarray((1,2),buffer=np.array([2*x[0]*x[1],x[0]**2 + 1])) #y,x #todo fix this to be general




class KalmanPendulum():
    """ 
    The Kalman filter equations for the noisy Pendulum oscillator with process noise.
    Please see xx.ipynb for a detailed explanation
    """


    def __init__(self,dt):
        self.dt = dt
        self.n_states = 2 # the system has 2 hidden states (x,y)
        self.n_y = 1 # the system has one observation state z

    """
    The discrete measurement noise covariance matrix
    """
    def R_matrix(self,σm):
        scalar = σm**2
        return scalar * np.ones((self.n_y,self.n_y)) #todo fix ny nomenclature


    """
    The discrete process noise covariance matrix
    """
    def Q_matrix(self,x,σp):
      
        Q11 = σp*self.dt**3 / 3
        Q12 = σp*self.dt**2 / 2
        Q21 = σp*self.dt**2 / 2
        Q22 = σp*self.dt
        Q = np.array([[Q11,Q12],[Q21,Q22]])

        return Q


    """
    The state evolution function \dot{x} = f(x)
    """
    def f(self,x,g):
        dx = np.zeros_like(x)
        
        x1 = x[0] + self.dt*x[1]
        x2 = x[1] - self.dt*g*np.sin(x[0])

        dy = -g*np.sin(x[0])
        y = x[1] + self.dt*dy
        #print("f func:", x2,y)
        #print("f func:", x[1],self.dt,dy)
       
        return np.array([x1,x2]) 

    """
    The Jacobian matrix of the state evolution function F = \partial f / \partial x 
    This is a matrix of dimension (2,2)
    """
    def F_jacobian(self,x,g):
        return np.array(([1.0,self.dt],[-g*np.cos(x[0])*self.dt,1]))


    """
    The measurement function z = h(x)
    """
    def h(self,x):
        return np.sin(x[0])


    """
    The Jacobian matrix of the measurement function H = \partial h / \partial x 
    This is a matrix of dimension (2,1)
    """
    def H_jacobian(self,x):
        return np.ndarray((1,2),buffer=np.array([np.cos(x[0]),0])) #y,x #todo fix this to be general

