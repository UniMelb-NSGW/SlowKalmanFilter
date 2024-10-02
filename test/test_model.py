# from src import system_parameters,pulsars,synthetic_data,model, kalman_filter, priors
# import numpy as np 


#     P   = system_parameters.SystemParameters()    # User-specifed system parameters
#     PTA = pulsars.Pulsars(P)
#     data = synthetic_data.SyntheticData(PTA,P)            # Given the system parameters and the PTA configuration, create some synthetic data

#     state_model = model.LinearModel(P,PTA)
    
#     #Initialise the Kalman filter
#     KF = kalman_filter.KalmanFilter(state_model,data.f_measured,PTA=PTA)

#     #Run the KF with the correct parameters.
#     #We get the correct parameters via Bilby dictionary, looking towards when we will run this with nested sampling
#     init_parameters,optimal_parameters_dict = priors.bilby_priors_dict(PTA,P,set_state_parameters_as_known=True,set_measurement_parameters_as_known=True)
#     optimal_parameters                      = optimal_parameters_dict.sample(1)    
#     xx,yy,optimal_model_likelihood                = KF.run(optimal_parameters)
    

#     for i in range(10):
    
#         init_parameters,sub_optimal_parameters_dict = priors.bilby_priors_dict(PTA,P,set_state_parameters_as_known=True,set_measurement_parameters_as_known=False)
#         sub_optimal_parameters                      = sub_optimal_parameters_dict.sample(1)    
#         xx,yy,suboptimal_model_likelihood           = KF.run(sub_optimal_parameters)
        
        
#         assert optimal_model_likelihood > suboptimal_model_likelihood


from test import generate_synthetic_data
from src import model as KalmanModel
from src import kalman_filter, priors
import numpy as np 


"""Check that the likelihood using true parameters is greater than using some randomly sampled parameters """
def test_likelihood_hierarchy():

    #Parameters of the system
    g    = 10.0
    σp   = 0.1
    σm   = 1.0
    x0   = 1.0
    y0   = -0.1
    seed = 1
    dt   = 0.001

    #Initialise the class
    Pendulum = generate_synthetic_data.NoisyPendulum(g,σp,σm, x0, y0,seed)

    #Integrate
    Pendulum.integrate(dt=dt,n_steps=5000)


    #Define the model
    model = KalmanModel.KalmanPendulum(dt=dt)

    #Setup the filter
    x_guess = np.array(([0,0]))      # guess of the initial states
    P0      = 10*np.eye(2)           # uncertainty in that guess
    data    = Pendulum.results[:,3].reshape(len(Pendulum.results),1)  # data to be ingested by the filter. Here reshaped to be a 2d array which the KF expects
    
    
    
    #Filter using correct parameters
    KF      = kalman_filter.ExtendedKalmanFilter(model,data,x_guess,P0)
    init_parameters,parameters_dict = priors.constant_priors(g,σp,σm)
    KF.run(parameters_dict.sample())
    likelihood_true = KF.ll 

    #Filter using incorrect parameters
    KF      = kalman_filter.ExtendedKalmanFilter(model,data,x_guess,P0)
    init_parameters,parameters_dict = priors.constant_priors(g/10,σp*1.2,σm*4.5)
    KF.run(parameters_dict.sample())
    likelihood_false = KF.ll 


    assert likelihood_true > likelihood_false



"""Check that the P matrix is always positive definite """
def test_posdef():

    #https://stackoverflow.com/questions/16266720/find-out-if-a-matrix-is-positive-definite-with-numpy
    def is_pos_def(x):
        return np.all(np.linalg.eigvals(x) > 0)


    #Parameters of the system
    g    = 10.0
    σp   = 0.1
    σm   = 1.0
    x0   = 1.0
    y0   = -0.1
    seed = 1
    dt   = 0.001

    #Initialise the class
    Pendulum = generate_synthetic_data.NoisyPendulum(g,σp,σm, x0, y0,seed)

    #Integrate
    Pendulum.integrate(dt=dt,n_steps=5000)


    #Define the model
    model = KalmanModel.KalmanPendulum(dt=dt)

    #Setup the filter
    x_guess = np.array(([0,0]))      # guess of the initial states
    P0      = 10*np.eye(2)           # uncertainty in that guess
    data    = Pendulum.results[:,3].reshape(len(Pendulum.results),1)  # data to be ingested by the filter. Here reshaped to be a 2d array which the KF expects
    
    
    
    #Filter using correct parameters
    KF      = kalman_filter.ExtendedKalmanFilter(model,data,x_guess,P0)
    init_parameters,parameters_dict = priors.constant_priors(g,σp,σm)
    KF.run(parameters_dict.sample())
  

    for i in range(KF.n_steps):
        P = KF.state_covariance[i,:,:]
        assert is_pos_def(P)
