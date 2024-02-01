import numpy as np
from numba import njit
import logging 
import sys
from gravitational_waves import gw_measurement_effect
from utils import dict_to_array


class LinearModel:

    """
    A linear model of the state evolution x = (f)

    To create a new model, the following are required:

    """
    def __init__(self,P,PTA):

        """
        Initialize the class. 
        Define any "global" variables here
        """
       

        self.γ        = PTA.γ
        self.dt       = PTA.dt
        self.N_states = PTA.Npsr
        self.σm       = PTA.σm 
        self.q        = PTA.q




        #Some problem specific stuff 


        # Define a list_of_keys arrays. 
        # This is useful for parsing the Bibly dictionary into arrays efficiently
        # There may be a less verbose way of doing this, but works well in practice
        self.list_of_f_keys        = [f'f0{i}' for i in range(PTA.Npsr)]
        self.list_of_fdot_keys     = [f'fdot{i}' for i in range(PTA.Npsr)]
        self.list_of_distance_keys = [f'distance{i}' for i in range(PTA.Npsr)]
        self.list_of_sigma_p_keys  = [f'sigma_p{i}' for i in range(PTA.Npsr)]




    """
    State transition matrix.
    This one is time-independent, only depends on dt
    """
    def F_matrix(self,parameters):
        return np.diag(np.exp(-self.γ*self.dt))


    """
    Control vector.
    This one is time-independent, only depends on dt
    """
    def B_control_vector(self,parameters):
        return np.zeros(self.N_states)


    """
    Process noise covariance matrix
    This one is time-independent, only depends on dt
    """
    def Q_matrix(self,parameters):
        σp = parameters['σp']
        diagonal = σp**2 * (1. - np.exp(-2.0*self.γ* self.dt)) / (2.0 * self.γ)
        return np.diag(diagonal)


    """
    Measurement matrix
    """
    def H_matrix(self,parameters,t):
        X = gw_measurement_effect(Ω=parameters['Ω'],
                                  Φ0=parameters['Φ0'],
                                  ψ=parameters['ψ'],
                                  ι=parameters['ι'],
                                  δ=parameters['δ'],
                                  α=parameters['α'],
                                  h=parameters['h'],
                                  q=self.q,
                                  d=parameters['d'],
                                  t=t).flatten() 



        return np.diag(1.0 - X)


    """
    A control vector added to the measurement matrix. Must be independent of the states!
    """
    def H_control_vector(self,parameters,t):

        X = gw_measurement_effect(Ω=parameters['Ω'],
                                  Φ0=parameters['Φ0'],
                                  ψ=parameters['ψ'],
                                  ι=parameters['ι'],
                                  δ=parameters['δ'],
                                  α=parameters['α'],
                                  h=parameters['h'],
                                  q=self.q,
                                  d=parameters['d'],
                                  t=t).flatten()


        ephemeris = parameters['f'] + t*parameters['fdot']

        return -X * ephemeris



    def R_matrix(self):
        return self.σm**2 * np.eye(self.N_states)
    


    """
    From the Bilby dictionary, create a "nice" dictionary for use by the Kalman filter 
    
    The likelihood function accepts as an argument a `parameters` Bilby dictionary.

    This is used in conjunction with nested sampling. 
    
    But we need to tell our model / Kalman filter how to read this parameters dictionary 
    
    """
    def create_parameters_dictionary(self, bilby_parameters_dict):
        
        
        #All the GW parameters can just be directly accessed as variables
        Ω   = bilby_parameters_dict["omega_gw"].item()
        Φ0  = bilby_parameters_dict["phi0_gw"].item()
        ψ   = bilby_parameters_dict["psi_gw"].item()
        ι   = bilby_parameters_dict["iota_gw"].item()
        δ   = bilby_parameters_dict["delta_gw"].item()
        α   = bilby_parameters_dict["alpha_gw"].item()
        h   = bilby_parameters_dict["h"].item()

        #Now read in the pulsar parameters as vectors
        f       = dict_to_array(bilby_parameters_dict,self.list_of_f_keys)
        fdot    = dict_to_array(bilby_parameters_dict,self.list_of_fdot_keys)
        d       = dict_to_array(bilby_parameters_dict,self.list_of_distance_keys)
        σp      = dict_to_array(bilby_parameters_dict,self.list_of_sigma_p_keys)


        #Create a new dictionary
        output_dictionary = { 'Ω': Ω,
                              'Φ0':Φ0,
                              'ψ': ψ,
                              'ι':ι,
                              'δ':δ,
                              'α':α,
                              'h':h,
                              'f':f,
                              'fdot':fdot,
                              'd':d,
                              'σp':σp}

        return output_dictionary





























