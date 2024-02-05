from cProfile import Profile
from pstats import Stats
from src import system_parameters,pulsars,synthetic_data,model, kalman_filter, priors
import numpy as np 



"""Check the runtimes for a single likelihood evaluation. No assertions. Run with pytest -s for outputs """
def test_likelihood_runtimes():

   
    P   = system_parameters.SystemParameters()    # User-specifed system parameters
    PTA = pulsars.Pulsars(P)
    data = synthetic_data.SyntheticData(PTA,P)            # Given the system parameters and the PTA configuration, create some synthetic data

    state_model = model.LinearModel(P,PTA)
    
     # Initialise the Kalman filter
    KF = kalman_filter.KalmanFilter(state_model,data.f_measured,PTA=PTA)

    # Run the KF with the correct parameters.
    # We get the correct parameters via Bilby dictionary, looking towards when we will run this with nested sampling
    init_parameters,optimal_parameters_dict = priors.bilby_priors_dict(PTA,P,set_state_parameters_as_known=True,set_measurement_parameters_as_known=True)
    optimal_parameters                      = optimal_parameters_dict.sample(1)    
    

    #Run it again to profile
    with Profile() as profile:
        x_results,y_results,model_likelihood    = KF.run(optimal_parameters)
        stats = Stats(profile)
        stats.sort_stats('tottime').print_stats(10)

    
