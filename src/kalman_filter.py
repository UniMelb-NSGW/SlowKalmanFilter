import numpy as np 
import scipy
import sys

class ExtendedKalmanFilter:
    """
    A class to implement the extended (non-linear) Kalman filter.

    It takes two initialisation arguments:

        `Model`: class which defines all the Kalman machinery e.g. state transition models, covariance matrices etc. 

        `Observations`: 2D array which holds the noisy observations recorded at the detector

        `x0`: A 1D array which holds the initial guess of the initial states

        `P0`: The uncertainty in the guess of P0

    ...and a general **kwargs that can be modified for the specific problem. todo


    The states are passed mix of functional and OOP, the states are passed as arguments i.e. no self.x
    just makes it easier to track where they are updated
    todo

    """

    def __init__(self,Model,Observations,x0,P0,**kwargs):

        """
        Initialize the class. 
        """

   
        self.model        = Model
        self.observations = Observations


   
        assert self.observations.ndim == 2, f'This filter requires that input observations is a 2D array. The observations here have {self.observations.ndim} dimensions '
      
     
        self.n_measurement_states  = self.observations.shape[-1] #number of measurement states
        self.n_steps               = self.observations.shape[0]  #number of observations/timesteps
        self.n_states              = self.model.n_states


 
        self.x0 = x0 #Guess of the initial state 
        self.P0 = P0 #Guess of the initial state covariance

      



    """
    Given the innovation and innovation covariance, get the likelihood.
    """
    def _log_likelihood(self,y,cov):
        N = len(y)
        sign, log_det = np.linalg.slogdet(cov)
        ll = -0.5 * (log_det + np.dot(y.T, np.linalg.solve(cov, y))+ N*np.log(2 * np.pi))
        return ll


    """
    Predict step.
    """
    def _predict(self,x,P,parameters):
        f_function = self.model.f(x,parameters['μ'])
        F_jacobian = self.model.F_jacobian(x,parameters['μ'])
        Q          = self.model.Q_matrix(x,parameters['μ'],parameters['σp'])
  
        return f_function, F_jacobian@P@F_jacobian.T + Q


    """
    Update step
    """
    def _update(self,x, P, observation):

        h_function = self.model.h(x)
        H_jacobian = self.model.H_jacobian(x)

        #Now do the standard Kalman steps
        y_predicted = self.model.h(x)                       # The predicted y
        y           = observation - y_predicted # The innovation/residual w.r.t actual data
       
     

        S           = H_jacobian@P@H_jacobian.T + self.R               # Innovation covariance
        Sinv        = scipy.linalg.inv(S)       # Innovation covariance inverse

    

        K           = P@H_jacobian.T@Sinv                # Kalman gain
        xnew        = x + K@y                   # update x
        Pnew        = P - K@H_jacobian@P                 # update P
        ll          = self._log_likelihood(y,S) # and get the likelihood
        y_updated   =self.model.h(xnew)# H@xnew         # and map xnew to measurement space


        return xnew, Pnew,ll,y_updated


    
    def run(self,parameters):


        #Initialise x and P
        x = self.x0 
        P = self.P0

        #Initialise the likelihood
        self.ll = 0.0

        #Define any matrices which are constant in time
        self.R          = self.model.R_matrix(parameters['σm'])


        # #Define all the Kalman matrices, which are constant in time
        # self.F = self.model.F_matrix(parameters['γ'])
        # self.Q = self.model.Q_matrix(parameters['γ'],parameters['σp'])
        # self.H = self.model.H_matrix()
        # self.R = self.model.R_matrix(parameters['σm'])
     
       
        #Define arrays to store results
        self.state_predictions       = np.zeros((self.n_steps,self.n_states))
        self.measurement_predictions = np.zeros((self.n_steps,self.n_measurement_states))



        #Do the first update step
        i = 0
        x,P,likelihood_value,y_predicted = self._update(x,P, self.observations[i,:])
        
        
           
        self.state_predictions[i,:] = x
        self.measurement_predictions[i,:]  = y_predicted
        self.ll +=likelihood_value

 
     
        for i in np.arange(1,self.n_steps):
            x_predict, P_predict             = self._predict(x,P,parameters)                                        # The predict step
            x,P,likelihood_value,y_predicted = self._update(x_predict,P_predict, self.observations[i,:]) # The update step
            
            #Update the running sum of the likelihood and save the state and measurement predictions
            self.ll +=likelihood_value
            self.state_predictions[i,:] = x
            self.measurement_predictions[i,:]  = y_predicted
       



