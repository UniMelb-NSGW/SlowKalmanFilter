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
        self.observations = Observations


        self.N_measurement_states  = self.observations.shape[-1] #number of measurement states
        self.N_steps               = self.observations.shape[0]  #number of observations/timesteps
        self.N_states              = self.model.N_states




        #Bits specific to this example problem. You may want to change this depending on your particular problem


        self.x0 = np.zeros(self.N_states) #Guess of the initial state
        self.P0 = 0.0*np.eye(self.N_states) #Guess of the initial state covariance




        # PTA related quantities
        PTA = kwargs['PTA']
        self.dt = PTA.dt
        self.q = PTA.q
        self.t = PTA.t
        self.q = PTA.q
    




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
    def _predict(self,x,P,parameters):


        #these are time-indpendetn and so really could go elsewhere. ToDo
        F = self.model.F_matrix(parameters)
        B = self.model.B_control_vector(parameters)
        Q = self.model.Q_matrix(parameters)



        return F@x + B, F@P@F.T + Q


    """
    Update step as a pure function
    """
    def _update(self,x, P, observation, parameters,t):

        #Define measurement matrix and control vector for this timestep, for these parameters
        H           = self.model.H_matrix(parameters,t)
        H_control   = self.model.H_control_vector(parameters,t)
        R           = self.model.R_matrix()

        
        #Now do the standard Kalman steps
        y_predicted = H@x + H_control           # The predicted y
        y           = observation - y_predicted # The innovation/residual w.r.t actual data
        S           = np.diag(H@P@H.T) + R      # Innovation covariance
        Sinv        = scipy.linalg.inv(S)       # Innovation covariance inverse
        K           = P@H.T@Sinv                # Kalman gain
        xnew        = x + K@y                   # update x
        Pnew        = P - K@H@P                 # update P
        ll          = self._log_likelihood(y,S) # and get the likelihood
        y_updated   = H@xnew + H_control        # and map xnew to measurement space
        return xnew, Pnew,ll,y_updated




    
    def run(self,parameters):


        #Initialise x and P
        x = self.x0 
        P = self.P0

        
        #Map from "Bilby parameters dictionary" to "Kalman parameters dictionary" 
        kalman_parameters_dictionary = self.model.create_parameters_dictionary(parameters)


        #Initialise the likelihood
        ll = 0.0
              
       
        #Do the first update step
        i = 0
        x,P,likelihood_value,y_predicted = self._update(x,P, self.observations[i,:],kalman_parameters_dictionary,self.t[i])
        ll +=likelihood_value

        #Place to store results
        x_results = np.zeros((self.N_steps,self.N_states))
        y_results = np.zeros((self.N_steps,self.N_measurement_states))

        
        #y_results[0,:] = y_predicted 
        for i in np.arange(1,self.N_steps):
            x_predict, P_predict             = self._predict(x,P,kalman_parameters_dictionary)                                           #The predict step
            
            x,P,likelihood_value,y_predicted = self._update(x_predict,P_predict, self.observations[i,:],kalman_parameters_dictionary,self.t[i])  
            
            ll +=likelihood_value

         
    
            x_results[i,:] = x
            y_results[i,:] = y_predicted
       





        return x_results,y_results,ll






