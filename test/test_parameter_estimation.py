from test import generate_synthetic_data
from src import model as KalmanModel
from src import kalman_filter, priors
import numpy as np 






"""Check that the final estimate of g is reasonably accurate"""
def test_reasonable_estimate():

    #Generate some data 

    g    = 10.0
    σp   = 0.1
    σm   = 0.1
    x0   = 1.0
    y0   = -0.1
    seed = 1
    dt   = 0.001

    #Initialise the class
    Pendulum = generate_synthetic_data.NoisyPendulum(g,σp,σm, x0, y0,seed)

    #Integrate
    Pendulum.integrate(dt=dt,n_steps=5000)


    #Define the mathematical model
    model = KalmanModel.UKFKalmanPendulumEstimateG(dt=dt)


    #Setup the filter
    x_guess = np.array(([0,0,0]))      # guess of the initial states
    P0      = 10*np.eye(3)             # uncertainty in that guess
    data    = Pendulum.results[:,3].reshape(len(Pendulum.results),1)  # data to be ingested by the filter. Here reshaped to be a 2d array which the KF expects



    #Run the filter 
    α  =1e-4 
    β = 2 
    κ = 0
    UKF      = kalman_filter.UnscentedKalmanFilter(model,data,x_guess,P0,α,β,κ)
    init_parameters,parameters_dict = priors.constant_priors(g,σp,σm) #g is not actually used here
    UKF.run(parameters_dict.sample())


    # check that the estimate is ok
    final_estimate_of_g       =  UKF.state_predictions[-1,2]
    final_uncertainty_of_g    = UKF.state_covariance[-1,-1,-1] 

    lower_limit = final_estimate_of_g - np.sqrt(final_uncertainty_of_g)
    upper_limit = final_estimate_of_g + np.sqrt(final_uncertainty_of_g)

    assert lower_limit <= g <= upper_limit
