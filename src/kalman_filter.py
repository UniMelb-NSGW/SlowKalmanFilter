import numpy as np 
import scipy
import sys


"""
Given the innovation and innovation covariance, get the likelihood.
Shared by multiple classes
"""
def log_likelihood(y,cov):
    N = len(y)
    sign, log_det = np.linalg.slogdet(cov)
    ll = -0.5 * (log_det + np.dot(y.T, np.linalg.solve(cov, y))+ N*np.log(2 * np.pi))
    return ll


class ExtendedKalmanFilter:
    """
    A class to implement the extended (non-linear) Kalman filter.

    It takes four initialisation arguments:

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
        f_function = self.model.f(x,parameters['g'])




        F_jacobian = self.model.F_jacobian(x,parameters['g'])
        Q          = self.model.Q_matrix(x,parameters['σp'])
    
        x_predict = f_function
        P_predict = F_jacobian@P@F_jacobian.T + Q

      
        return x_predict, P_predict




    """
    Update step
    """
    def _update(self,x, P, observation):

        #Evaluate the Jacobian of the measurement matrix
        H_jacobian = self.model.H_jacobian(x)

        #Now do the standard Kalman steps
        y_predicted = self.model.h(x)                       # The predicted y
        y           = observation - y_predicted             # The innovation/residual w.r.t actual data         
        S           = H_jacobian@P@H_jacobian.T + self.R    # Innovation covariance
        Sinv        = scipy.linalg.inv(S)                   # Innovation covariance inverse
        K           = P@H_jacobian.T@Sinv                   # Kalman gain
        xnew        = x + K@y                               # update x
        Pnew        = P -K@S@K.T                            # Update P 
        
        ##-------------------
        # The Pnew update equation is equation 5.27 in Sarkka  
        # Other equivalent options for computing Pnew are:
            # P = (I-KH)P
            # P = (I-KH)P(I-KH)' + KRK' (this one may have advantages for numerical stability)
        ##-------------------


        ll          = self._log_likelihood(y,S)          # and get the likelihood
        y_updated   = self.model.h(xnew)                 # and map xnew to measurement space


        return xnew, Pnew,ll,y_updated


    
    def run(self,parameters):


        #Initialise x and P
        x = self.x0 
        P = self.P0

        #Initialise the likelihood
        self.ll = 0.0

        #Define any matrices which are constant in time
        self.R          = self.model.R_matrix(parameters['σm'])

     
        #Define arrays to store results
        self.state_predictions       = np.zeros((self.n_steps,self.n_states))
        self.measurement_predictions = np.zeros((self.n_steps,self.n_measurement_states))
        self.state_covariance        = np.zeros((self.n_steps,self.n_states,self.n_states))


        # #Do the first update step
        i = 0
        x,P,likelihood_value,y_predicted = self._update(x,P, self.observations[i,:])
        self.state_predictions[i,:] = x
        self.state_covariance[i,:,:] = P
        self.measurement_predictions[i,:]  = y_predicted
        self.ll +=likelihood_value

 
     
        for i in np.arange(1,self.n_steps):

            #Predict step
            x_predict, P_predict             = self._predict(x,P,parameters)                                        # The predict step
            
            #Update step
            x,P,likelihood_value,y_predicted = self._update(x_predict,P_predict, self.observations[i,:]) # The update step
            
            #Update the running sum of the likelihood and save the state and measurement predictions
            self.ll +=likelihood_value
            self.state_predictions[i,:] = x
            self.state_covariance[i,:,:] = P
            self.measurement_predictions[i,:]  = y_predicted
       


class UnscentedKalmanFilter:
    """
    A class to implement the Unscented Kalman filter (UKF).

    Please see Wan & Van Der Merwe, "The unscented Kalman filter for nonlinear estimation" https://ieeexplore.ieee.org/document/882463. W&V2000 hereafter

    It takes seven initialisation arguments:

        `Model`: class which defines all the Kalman machinery e.g. state transition models, covariance matrices etc. 

        `Observations`: 2D array which holds the noisy observations recorded at the detector

        `x0`: A 1D array which holds the initial guess of the initial states

        `P0`: The uncertainty in the guess of x0

        `α`: a float used to determine the spread of the sigma points, usually a small number
        
        `β`: a float used to incrporate prior knowledge on distribution of the states. For Gaussians, use β=2

        `κ`: a secondary scaling parameters, usually zero

    """

    def __init__(self,Model,Observations,x0,P0,α=1e-3,β=2.0,κ=0):

        """
        Initialize the class. 
        """

   
        #Load the mathematical model and the empirical observations
        self.model        = Model
        self.observations = Observations
        assert self.observations.ndim == 2, f'This filter requires that input observations is a 2D array. The observations here have {self.observations.ndim} dimensions ' # even if the observations are a scalar timeseries, we require this to be an array (n_steps,1)
      
     
        # Define some useful quantities related to the dimension of the problem
        self.dt = self.model.dt
        self.n_measurement_states  = self.observations.shape[-1] #number of measurement states
        self.n_steps               = self.observations.shape[0]  #number of observations/timesteps
        self.n_states              = self.model.n_states


        #Guess of the initial state and the uncertainty in that guess
        self.x0 = x0 
        self.P0 = P0 

        # parameters of the UKF
        self.α = α
        self.β = β
        self.κ = κ

    """
    Calculate the sigma points weight vectors, see Equation (15) from W&V2000
    """
    def CalculateWeights(self):

        # First define the scaling parameter
        λ = self.α**2 * (self.n_states + self.κ) - self.n_states


        self.Wm = np.concatenate(([λ/(self.n_states+λ)], (0.5/(self.n_states+λ))*np.ones(2*self.n_states)), axis=None) # todo, sources for these equations
        self.Wc = np.concatenate(([λ/(self.n_states+λ)+(1-self.α**2+self.β)], (0.5/(self.n_states+λ))*np.ones(2*self.n_states)), axis=None)
        self.γ = np.sqrt(self.n_states + λ)

        # Developer note: most other functions in this class are of the form a = f(x), i.e. we don't update the state but return a value
        # This is done as it makes it a bit easier to read what quantities are being updated where
        # In contrast for this function we just update the state directly. It is only called once. 


    """
    Calculate the sigma points vectors, see Equation (15) from W&V2000
    """
    def CalculateSigmaPoints(self, x, P):


        # Here we enforce the posdef property of the covariance matrix, which can be lost sometimes due to numerical errors
        # The value of epsilon has been found to work well in practice. Are there circumstances where we will need to modify this?
        epsilon = 1e-24 
        positive_definite_check= 0.5*(P + P.T) + epsilon*np.eye(len(x))
        U               = scipy.linalg.cholesky(positive_definite_check).T # sqrt
        
        
        # Now calculate the sigma points by Equation (15)
        sigma_points    = np.zeros((2*self.n_states + 1, self.n_states))
        sigma_points[0] = x
        for i in range(self.n_states):
            sigma_points[i+1]               = x + self.γ*U[:, i]
            sigma_points[self.n_states+i+1] = x - self.γ*U[:, i]
        return sigma_points



    """
    A fourth order Runge-Kutta integration step
    """
    def rk4_step(self,x):
        k1 = self.dt * self.model.derivative_function(x, self.g)
        k2 = self.dt * self.model.derivative_function(x + k1 / 2, self.g)
        k3 = self.dt * self.model.derivative_function(x + k2 / 2, self.g)
        k4 = self.dt * self.model.derivative_function(x + k3, self.g)

        return x + (k1 + 2 * k2 + 2 * k3 + k4) / 6


    """
    Use the RK4 integrator to propagate the sigma points forward by one timestep
    todo: I think we can vectorise this and combine with the rk4_step function
    """
    def propagate_sigma_points(self,sigma_points):
        propagated_sigma_points = np.empty_like(sigma_points)
        for i, x in enumerate(sigma_points):
            propagated_sigma_points[i] = self.rk4_step(x)
        return propagated_sigma_points


    """ 
    The predict stage of the UKF. See Algorithm 3.1 in  W&V2000
    """
    def predict(self,propagated_sigma_points):
        # Predict the state
        xp = np.zeros((self.n_states, 1))
        for i in range(len(self.Wm)):
            xp += self.Wm[i] * propagated_sigma_points[i].reshape(self.n_states, 1)

            
        #Predict the covariance
        sigma_point_diff = np.zeros((len(propagated_sigma_points), self.n_states))
        for i, point in enumerate(propagated_sigma_points):
            sigma_point_diff[i] = point - xp.ravel()

        Pp = np.zeros((self.n_states, self.n_states))
        for i, diff in enumerate(sigma_point_diff):
            Pp += self.Wc[i] * np.outer(diff, diff)
            
        Pp += self.Q
        return xp, Pp



    """ 
    The update stage of the UKF. See Algorithm 3.1 in  W&V2000
    """
    def update(self,xp, Pp, propagated_sigma_points, observation):
        


        # Use your current sigma points to make a prediction and compare that with the actual observation to get the innovation, y
        sigma_points_in_measurement_space = np.column_stack((self.model.sigma_point_measurement_function(propagated_sigma_points)))
        y_predicted                       = np.einsum('i,ij->j', self.Wm, sigma_points_in_measurement_space.T) # Equation (17) from W&V2000
        y                                 = observation - y_predicted


        #Calculate the innovation covariance, Equation (18) from from W&V2000
        measurement_difference = sigma_points_in_measurement_space - y_predicted
        P_yy = np.zeros((self.n_measurement_states, self.n_measurement_states))
        for i in range(len(self.Wc)):
            P_yy += self.Wc[i] * np.outer((measurement_difference.T)[i],(measurement_difference.T)[i]) # Equation (18) from W&V2000
        P_yy += self.R




        #Calculate the cross-covariance 
        states_difference = propagated_sigma_points - xp.T
        P_xy = np.zeros((self.n_states, self.n_measurement_states))
        for i in range(len(self.Wc)):
            P_xy += self.Wc[i] * np.outer(states_difference[i], (measurement_difference.T)[i])


        # Calculate the Kalman gain and update the state and covariance estimates
        kalman_gain = np.linalg.solve(P_yy.T, P_xy.T).T #todo, check all these transposes. 
        x           = xp + kalman_gain*y
        P           = Pp - np.dot(kalman_gain, np.dot(P_yy, kalman_gain.T))


        #and finally calculate the likelihood for this step
        ll = log_likelihood(y,P_yy)
        return x.flatten(), P, ll, self.model.measurement_function(x) 










    """ 
    Run the UKF given a set of free parameters
    Updates 
        state_predictions
        measurement_predictions
        state_covariance
        ll (likelihood)
    """
    def run(self,parameters):


        #Initialise x and P
        x = self.x0 
        P = self.P0

        #Initialise the likelihood
        self.ll = 0.0

        #Define any matrices which are constant in time
        self.R          = self.model.R_matrix(parameters['σm'])
        self.Q          = self.model.Q_matrix(x,parameters['σp']) # todo - not a funciton of x. Is it ever? 

        self.g = parameters['g'] # this is not the right place to unpack this parameter. Leaving here for now while strucutre comes in 

     
        #Define arrays to store results
        self.state_predictions       = np.zeros((self.n_steps,self.n_states))
        self.measurement_predictions = np.zeros((self.n_steps,self.n_measurement_states))
        self.state_covariance        = np.zeros((self.n_steps,self.n_states,self.n_states))


        # Calculate the weights of the UKF
        self.CalculateWeights()


        for i in range(self.n_steps):
            
            sigma_points            = self.CalculateSigmaPoints(x,P)
            propagated_sigma_points = self.propagate_sigma_points(sigma_points)


            x_predict, P_predict    = self.predict(propagated_sigma_points)
            propagated_sigma_points = self.CalculateSigmaPoints(x_predict.squeeze(),P_predict) #todo, handle the squeeze. Also, check this is the same as what Joe does

            #WeightedMeas,MeasurementDiff = self.sigma_point_measurement(propagated_sigma_points)
            x,P,likelihood_value,y_predicted = self.update(x_predict, P_predict, propagated_sigma_points, self.observations[i,:])

      
            #IO
            self.state_predictions[i,:] = x
            self.state_covariance[i,:,:] = P
            self.measurement_predictions[i,:]  = y_predicted
            self.ll +=likelihood_value

 
     
 