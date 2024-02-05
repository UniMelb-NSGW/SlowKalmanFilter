


import bilby
import numpy as np
import logging 
logging.getLogger().setLevel(logging.INFO)



"""
Main external function for defining priors
"""
def bilby_priors_dict(PTA,P,set_state_parameters_as_known=False,set_measurement_parameters_as_known=False):


    #logging.info('Setting the bilby priors dict')


    init_parameters = {}
    priors = bilby.core.prior.PriorDict()
    
    #Measurement priors
    init_parameters,priors = _set_prior_on_measurement_parameters(init_parameters,priors,P,set_measurement_parameters_as_known) 

    #State priors
    init_parameters,priors = _set_prior_on_state_parameters(init_parameters,priors,PTA, set_state_parameters_as_known)
 
    return init_parameters,priors
    



"""
Create a delta function prior about the true value
"""
def _add_to_bibly_priors_dict_constant(x,label,init_parameters,priors):


    
    i = 0
    for f in x:
        key = label+str(i)
        init_parameters[key] = None
        priors[key] = f
        i+= 1

    return init_parameters,priors


"""
Add  logarithmic prior vector
"""
def _add_to_bibly_priors_dict_log(x,label,init_parameters,priors,lower,upper): #same lower/upper for every one
    
    i = 0
    for f in x:
        key = label+str(i)
        init_parameters[key] = None
      
        priors[key] = bilby.core.prior.LogUniform(lower,upper, key)
        
        i+= 1

    return init_parameters,priors


"""
Add uniform prior vector
"""
def _add_to_bibly_priors_dict_uniform(x,label,init_parameters,priors,tol):
    
    i = 0
    for f in x:
        key = label+str(i)
        init_parameters[key] = None
      
        priors[key] = bilby.core.prior.Uniform(f-np.abs(f*tol),f+ np.abs(f*tol), key)
        
        i+= 1

    return init_parameters,priors



"""
Add uniform prior vector in the range 0 - 2pi
"""
def _add_to_bibly_priors_dict_radians(x,label,init_parameters,priors):
    
    i = 0
    for f in x:
        key = label+str(i)
        init_parameters[key] = None
      
        priors[key] = bilby.core.prior.Uniform(0.0,2*np.pi, key)
        
        i+= 1

    return init_parameters,priors





import sys
"""
Set a prior on the state parameters
"""
def _set_prior_on_state_parameters(init_parameters,priors,PTA,set_parameters_as_known):

    #In the below we assume that γ is known exactly a priori
    #We also take there to just be one σp parameter which applies to all pulsars
    #This condition will need to be loosened once MWE is running. TODO.

    if set_parameters_as_known:
        #logging.info('Setting fully informative priors on PSR parameters')

        init_parameters,priors = _add_to_bibly_priors_dict_constant(PTA.f,"f0",init_parameters,priors)     
        init_parameters,priors = _add_to_bibly_priors_dict_constant(PTA.fdot,"fdot",init_parameters,priors)           
        init_parameters,priors = _add_to_bibly_priors_dict_constant(PTA.σp,"sigma_p",init_parameters,priors) 
        init_parameters,priors = _add_to_bibly_priors_dict_constant(PTA.d,"distance",init_parameters,priors) 

    
    else:
        
        sys.exit('This has not been done yet')

    return init_parameters,priors 




"""
Set a prior on the measurement parameters
"""
def _set_prior_on_measurement_parameters(init_parameters,priors,P,set_parameters_as_known):

    if set_parameters_as_known: #don't set a prior, just assume these are known exactly a priori

        #logging.info('Setting fully informative priors on GW parameters')
        
        #Add all the GW quantities
        init_parameters[f"omega_gw"] = None
        priors[f"omega_gw"] = P.Ω

        init_parameters[f"phi0_gw"] = None
        priors[f"phi0_gw"] = P.Φ0

        init_parameters[f"psi_gw"] = None
        priors[f"psi_gw"] = P.ψ

        init_parameters[f"iota_gw"] = None
        priors[f"iota_gw"] = P.ι

        init_parameters[f"delta_gw"] = None
        priors[f"delta_gw"] = P.δ

        init_parameters[f"alpha_gw"] = None
        priors[f"alpha_gw"] = P.α

        init_parameters[f"h"] = None
        priors[f"h"] = P.h

    else:


            
        #Add all the GW quantities
        init_parameters["omega_gw"] = None
        priors["omega_gw"] = bilby.core.prior.LogUniform(1e-8, 1e-6, 'omega_gw')


        init_parameters["phi0_gw"] = None
        priors["phi0_gw"] = bilby.core.prior.Uniform(0.0, 2.0*np.pi, 'phi0_gw')


        init_parameters["psi_gw"] = None
        priors["psi_gw"] = bilby.core.prior.Uniform(0.0, np.pi, 'psi_gw')

        init_parameters["iota_gw"] = None
        priors["iota_gw"] = bilby.core.prior.Sine(0.0, np.pi, 'iota_gw')


        init_parameters["delta_gw"] = None
        priors["delta_gw"] = bilby.core.prior.Cosine(-np.pi/2, np.pi/2, 'delta_gw')


        init_parameters["alpha_gw"] = None
        priors["alpha_gw"] = bilby.core.prior.Uniform(0.0, np.pi, 'alpha_gw')


        init_parameters["h"] = None
        priors["h"] = bilby.core.prior.LogUniform(P.h/100.0, P.h*10.0, 'h')


    return init_parameters,priors 



