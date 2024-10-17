import numpy as np 
import scipy
import sys


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

    def __init__(self,Model,Observations,x0,P0,α,β,κ,**kwargs):

        """
        Initialize the class. 
        """

   
        self.model        = Model
        self.observations = Observations


   
        assert self.observations.ndim == 2, f'This filter requires that input observations is a 2D array. The observations here have {self.observations.ndim} dimensions '
      
     
        self.dt = self.model.dt
        self.n_measurement_states  = self.observations.shape[-1] #number of measurement states
        self.n_steps               = self.observations.shape[0]  #number of observations/timesteps
        self.n_states              = self.model.n_states


 
        self.x0 = x0 #Guess of the initial state 
        self.P0 = P0 #Guess of the initial state covariance

        # parameters of the UKF
        self.α = α
        self.β = β
        self.κ = κ

    """
    What does this funciton do exactly..... 
    """
    def CalculateWeights(self):

        # Compute sigma point weights
        λ = self.α**2 * (self.n_states + self.κ) - self.n_states


        self.Wm = np.concatenate(([λ/(self.n_states+λ)], (0.5/(self.n_states+λ))*np.ones(2*self.n_states)), axis=None) # todo, sources for these equations
        self.Wc = np.concatenate(([λ/(self.n_states+λ)+(1-self.α**2+self.β)], (0.5/(self.n_states+λ))*np.ones(2*self.n_states)), axis=None)
        self.γ = np.sqrt(self.n_states + λ)


    """_summary_
    """
    def CalculateSigmaPoints(self, x, P):


  
        epsilon = 1e-24 #todo, does this value matter?
        Pos_definite_Check= 0.5*(P + P.T) + epsilon*np.eye(len(x))
        
        U               = scipy.linalg.cholesky(Pos_definite_Check).T # sqrt
        sigma_points    = np.zeros((2*self.n_states + 1, self.n_states))
        
        sigma_points[0] = x

        for i in range(self.n_states):
            sigma_points[i+1]               = x + self.γ*U[:, i]
            sigma_points[self.n_states+i+1] = x - self.γ*U[:, i]
        return sigma_points




    def rk4_step(self,x):
        k1 = self.dt * self.model.derivative_function(x, self.g)
        k2 = self.dt * self.model.derivative_function(x + k1 / 2, self.g)
        k3 = self.dt * self.model.derivative_function(x + k2 / 2, self.g)
        k4 = self.dt * self.model.derivative_function(x + k3, self.g)

        return x + (k1 + 2 * k2 + 2 * k3 + k4) / 6


    def propagate_sigma_points(self,sigma_points):

        propagated_sigma_points = np.empty_like(sigma_points)
        for i, x in enumerate(sigma_points):
            propagated_sigma_points[i] = self.rk4_step(x)
        return propagated_sigma_points


    def predict(self,propagated_sigma_points):

        xp = np.zeros((self.n_states, 1))
        for i in range(len(self.Wm)):
            xp += self.Wm[i] * propagated_sigma_points[i].reshape(self.n_states, 1)

            
        SigmaPointDiff = np.zeros((len(propagated_sigma_points), self.n_states))
        for i, point in enumerate(propagated_sigma_points):
            SigmaPointDiff[i] = point - xp.ravel()

        Pp = np.zeros((self.n_states, self.n_states))
        for i, diff in enumerate(SigmaPointDiff):
            Pp += self.Wc[i] * np.outer(diff, diff)
            
        Pp += self.Q

        return xp, Pp




    def update(self,xp, Pp, propagated_sigma_points, observation):
        
        #todo, update this notatoin
        SP_Meas = np.column_stack((np.sin(propagated_sigma_points[:, 0])))  # this should come in from th emodel
        WeightedMeas = np.einsum('i,ij->j', self.Wm, SP_Meas.T)
        MeasurementDiff = SP_Meas - WeightedMeas

        
        Inn = observation - WeightedMeas

        SigmaPointDiff = propagated_sigma_points - xp.T

        PredMeasurementCov = np.zeros((self.n_measurement_states, self.n_measurement_states))
        for i in range(len(self.Wc)):
            PredMeasurementCov += self.Wc[i] * np.outer((MeasurementDiff.T)[i],( MeasurementDiff.T)[i])
        PredMeasurementCov += self.R

        CrossCovariance = np.zeros((self.n_states, self.n_measurement_states))
        for i in range(len(self.Wc)):
            CrossCovariance += self.Wc[i] * np.outer(SigmaPointDiff[i], (MeasurementDiff.T)[i])

        kalman_gain = np.linalg.solve(PredMeasurementCov.T, CrossCovariance.T).T

        x = xp + kalman_gain*Inn

        P = Pp - np.dot(kalman_gain, np.dot(PredMeasurementCov, kalman_gain.T))

        sign, log_det = np.linalg.slogdet(PredMeasurementCov)
        ll = -0.5 * (log_det + np.dot(Inn.T, np.linalg.solve(PredMeasurementCov, Inn))
                    + np.log(2 * np.pi))
        
        return x.flatten(), P, ll, np.sin(x[0]) #need the measurement model here










    
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

 
     
 