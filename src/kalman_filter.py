import numpy as np 
import scipy

"""
Kalman likelihood
"""

def log_likelihood(y,cov):
    N = len(y)
    x = y/cov
    #The innovation covariance is diagonal
    #Therefore we can calculate the determinant of its logarithm as below
    #A normal np.linalg.det(cov)fails due to overflow, since we just have
    #det ~ 10^{-13} * 10^{-13} * 10^{-13}*...
    log_det_cov = np.sum(np.log(cov)) # Uses log rules and diagonality of covariance matrix
    ll = -0.5 * (log_det_cov + y@x+ N*np.log(2*np.pi))
    return ll




class KalmanFilter:
    """
    A class to implement the linear Kalman filter.

    It takes two initialisation arguments:

        `Model`: definition of all the Kalman machinery: state transition models, covariance matrices etc. 

        `Observations`: class which holds the noisy observations recorded at the detector

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
    





    def log_likelihood(y,cov):
        N = len(y)
        sign, log_det = np.linalg.slogdet(cov)
        ll = -0.5 * (log_det + np.dot(y.T, np.linalg.solve(cov, y))+ N*np.log(2 * np.pi))
        return ll


    """
    Predict step. Pure function
    """
    def _predict(self,x,P,F,B,Q,t):
        return F@x + B, F@P@F.T + Q


    """
    Update step as a pure function
    """
    def update(self,x, P, observation, parameters,t):

        #Define measurement matrix and control vector for this timestep, for these parameters
        H           = self.model.H_matrix(parameters,t)
        H_control   =  self.model.H_control_vector(parameters,t)
        R           = self.model.R_matrix()
        
        
        #Now do the standard Kalman steps
        y_predicted = H@x + H_control           # The predicted y
        y           = observation - y_predicted # The innovation/residual w.r.t actual data
        S           = np.diag(H@P@H.T) + R      # Innovation covariance
        Sinv        = scipy.linalg.inv(S)       # Innovation covariance inverse
        K           = P@H.T@Sinv                # Kalman gain


        xnew        = x + K@y                   # update x
        Pnew        = P - K@H@P                 # update P
        ll          = log_likelihood(y,S)       #a nd get the likelihood
        return xnew, Pnew,ll




    
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
        x,P,likelihood_value = update(x,P, self.observations[i,:],kalman_parameters_dictionary)
        ll +=likelihood_value

        #Place to store results
        x_results = np.zeros((self.Nsteps,2*self.Npsr))
        y_results = np.zeros((self.Nsteps,self.Npsr))

        
        #y_results[0,:] = y_predicted 
        for i in np.arange(1,self.Nsteps):
            x_predict, P_predict             = predict(x,P,F,F_transpose,Q)                                           #The predict step
            x,P,likelihood_value = update(x_predict,P_predict, self.observations[i,:],self.R,GW[i,:],ephemeris[i,:]) #The update step    
            ll +=likelihood_value

         


            H              = construct_H_matrix(GW[i,:])    #Determine the H matrix for this step
            y_predicted =  H@x - GW[i,:]*ephemeris[i,:]
    
            x_results[i,:] = x
            y_results[i,:] = y_predicted
       





        return ll






