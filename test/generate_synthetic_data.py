import numpy as np 
import matplotlib.pyplot as plt 
import scienceplots
plt.style.use('science')

class NoisyPendulum:
    """ Simulates the signal from a noisy pendulum with process noise. 
    
    The observation is an arbitrary, non-linear combination of the states
    """
    
    def __init__(self,g,σp,σm, x=1.0, y=-0.1,seed=1):
    
        self.g  = g
        self.σp = σp 
        self.σm = σm
        self.x  = x
        self.y  = y
        self.generator = np.random.default_rng(seed)
        
    def take_a_timestep(self):
        """ Call once for each new measurement at dt time from last call.
            Uses a basic Euler method.
        """
        

        #Discretised Q-matrix\
        Q11 = self.σp*self.dt**3 / 3
        Q12 = self.σp*self.dt**2 / 2
        Q21 = self.σp*self.dt**2 / 2
        Q22 = self.σp*self.dt
        Q = np.array([[Q11,Q12],[Q21,Q22]])
        Qnoise = self.generator.multivariate_normal(np.zeros(2),Q)

 
        #Derivatives
        dx = self.y
        dy = -self.g*np.sin(self.x)
        
        
        #Euler timestep
        self.x = self.x  + self.dt*dx + Qnoise[0]
        self.y = self.y  + self.dt*dy + Qnoise[1]

     


        #The observation, no noise
        observation_no_noise = np.sin(self.x)

        #add noise to the observation
        observation = observation_no_noise + self.generator.normal(0, self.σm) 

        return self.x,self.y,observation_no_noise,observation

    def integrate(self,dt,n_steps):
        """ Integrate for n_steps timesteps and return an array that holds the states and observations
        """

        self.dt = dt
        self.n_steps      = n_steps
        self.t            = np.arange(0,self.n_steps*self.dt,self.dt)
        self.results      = np.zeros((self.n_steps,4)) # 4 columns: x,y,observation_no_noise,observation 

        for i in range(self.n_steps):
            self.results[i,:] = self.take_a_timestep()



