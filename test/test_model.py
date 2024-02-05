from src import system_parameters,pulsars,synthetic_data,model,priors
import numpy as np 


#These tests will need to be modified depending on the particular model used



"""Check the shapes of the matrices are all as expected """
def test_matrix_shapes():

    Npsr = 5
    P   = system_parameters.SystemParameters(Npsr=Npsr)    # User-specifed system parameters
    PTA = pulsars.Pulsars(P)
    state_model = model.LinearModel(P,PTA)

    #Set some parameters
    init_parameters,optimal_parameters_dict = priors.bilby_priors_dict(PTA,P,set_state_parameters_as_known=True,set_measurement_parameters_as_known=True)
    optimal_parameters                      = optimal_parameters_dict.sample(1)
    parameters_dictionary                   = state_model.create_parameters_dictionary(optimal_parameters)

    #The Kalman matrices
    F = state_model.F_matrix(parameters_dictionary)
    B = state_model.B_control_vector(parameters_dictionary)
    Q = state_model.Q_matrix(parameters_dictionary)
    H = state_model.H_matrix(parameters_dictionary,1)
    Hcontrol = state_model.H_control_vector(parameters_dictionary,1)


    assert F.shape == (Npsr,Npsr)
    assert B.shape == (Npsr,)
    assert F.shape == Q.shape
    assert H.shape == F.shape
    assert Hcontrol.shape == B.shape





"""Check the matrices reduced to what we expect in the zero case"""
def test_zero_values():


    #Check F becomes an identity matrix
    P   = system_parameters.SystemParameters(γ=0.0)    # User-specifed system parameters
    PTA = pulsars.Pulsars(P)
    state_model = model.LinearModel(P,PTA)

    #Set some parameters
    init_parameters,optimal_parameters_dict = priors.bilby_priors_dict(PTA,P,set_state_parameters_as_known=True,set_measurement_parameters_as_known=True)
    optimal_parameters                      = optimal_parameters_dict.sample(1)
    parameters_dictionary                   = state_model.create_parameters_dictionary(optimal_parameters)


    F = state_model.F_matrix(parameters_dictionary)
    assert np.all(F == np.eye(PTA.Npsr))     



    #Check Q becomes all zeros
    P   = system_parameters.SystemParameters(σp=0.0)    # User-specifed system parameters
    PTA = pulsars.Pulsars(P)
    state_model = model.LinearModel(P,PTA)

    #Set some parameters
    init_parameters,optimal_parameters_dict = priors.bilby_priors_dict(PTA,P,set_state_parameters_as_known=True,set_measurement_parameters_as_known=True)
    optimal_parameters                      = optimal_parameters_dict.sample(1)
    parameters_dictionary                   = state_model.create_parameters_dictionary(optimal_parameters)


    Q = state_model.Q_matrix(parameters_dictionary)
    assert np.all(Q == 0.0)   





    #Check H becomes identity
    P   = system_parameters.SystemParameters(h=0.0)    # User-specifed system parameters
    PTA = pulsars.Pulsars(P)
    state_model = model.LinearModel(P,PTA)

    #Set some parameters
    init_parameters,optimal_parameters_dict = priors.bilby_priors_dict(PTA,P,set_state_parameters_as_known=True,set_measurement_parameters_as_known=True)
    optimal_parameters                      = optimal_parameters_dict.sample(1)
    parameters_dictionary                   = state_model.create_parameters_dictionary(optimal_parameters)


    H = state_model.H_matrix(parameters_dictionary,1)

    assert np.all(H == np.eye(PTA.Npsr))  




















