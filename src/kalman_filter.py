import numpy as np 
import scipy


class KalmanFilter:
    """
    A class to implement the linear Kalman filter.

    It takes two initialisation arguments:

        `Model`: definition of all the Kalman machinery: state transition models, covariance matrices etc. 

        `Observations`: class which holds the noisy observations recorded at the detector

    ...and a general **kwargs that can be modified for the specific problem. In this example, we use **kwargs to pass the filter information about the PTA,
        which seems separate from either the observations or the model.

    """

    def __init__(self,Model,Observations,**kwargs):

        """
        Initialize the class. 
        """

        #General Kalman filter bits
        self.model = Model
        self.observations = Observations.measurement
        self.t = Observations.t


        self.N_measurement_states  = self.observations.shape[-1] #number of measurement states
        self.N_steps               = self.observations.shape[0]  #number of observations/timesteps
        self.N_states              = self.model.N_states


        self.x0 = np.zeros(self.N_states) #Guess of the initial state
        self.P0 = 1e-10*np.eye(self.N_states) #Guess of the initial state covariance






    """
    Given the innovation and innovation covariance, get the likelihood. Pure function.
    """
    def _log_likelihood(self,y,cov):
        N = len(y)
        sign, log_det = np.linalg.slogdet(cov)
        ll = -0.5 * (log_det + np.dot(y.T, np.linalg.solve(cov, y))+ N*np.log(2 * np.pi))
        return ll


    """
    Predict step. Pure function
    """
    def _predict(self,x,P):


        #these are time-indpendetn and so really could go elsewhere. ToDo
        F = self.F
        Q = self.Q



        return F@x, F@P@F.T + Q


    """
    Update step as a pure function
    """
    def _update(self,x, P, observation):


        H = self.H
        R = self.R

        #Now do the standard Kalman steps
        y_predicted = H@x           # The predicted y
        y           = observation - y_predicted # The innovation/residual w.r.t actual data
        S           = H@P@H.T + R      # Innovation covariance
        Sinv        = scipy.linalg.inv(S)       # Innovation covariance inverse
        K           = P@H.T@Sinv                # Kalman gain
        xnew        = x + K@y                   # update x
        Pnew        = P - K@H@P                 # update P
        ll          = self._log_likelihood(y,S) # and get the likelihood
        y_updated   = H@xnew         # and map xnew to measurement space
        return xnew, Pnew,ll,y_updated


    
    def run(self,parameters):


        #Initialise x and P
        x = self.x0 
        P = self.P0

        #Initialise the likelihood
        ll = 0.0

        #Define all the Kalman matrices, which are constant in time
        self.F = self.model.F_matrix(parameters['γ'])
        self.Q = self.model.Q_matrix(parameters['γ'],parameters['σp'])
        self.H = self.model.H_matrix()
        self.R = self.model.R_matrix(parameters['σm'])
     
       
        #Do the first update step
        i = 0
        x,P,likelihood_value,y_predicted = self._update(x,P, self.observations[i,:])
        ll +=likelihood_value

        #Place to store results
        x_results = np.zeros((self.N_steps,self.N_states))
        y_results = np.zeros((self.N_steps,self.N_measurement_states))

        
        #y_results[0,:] = y_predicted 
        for i in np.arange(1,self.N_steps):
            x_predict, P_predict             = self._predict(x,P)                                           #The predict step
            
            x,P,likelihood_value,y_predicted = self._update(x_predict,P_predict, self.observations[i,:])  
            
            ll +=likelihood_value

         
    
            x_results[i,:] = x
            y_results[i,:] = y_predicted
       





        return x_results,y_results,ll






