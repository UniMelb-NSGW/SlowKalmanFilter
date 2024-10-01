


import bilby
import numpy as np
import logging 
logging.getLogger().setLevel(logging.INFO)



"""
Main external function for defining priors
"""
def bilby_priors_dict(g,σp,σm):


    init_parameters = {}
    priors = bilby.core.prior.PriorDict()

    init_parameters[f""] = None
    priors[f"g"] = g

    init_parameters[f"σp"] = None
    priors[f"σp"] = σp

    init_parameters[f"σm"] = None
    priors[f"σm"] = σm
 
    return init_parameters,priors
    







# """
# Main external function for defining priors
# """
# def bilby_priors_dict(P):


#     init_parameters = {}
#     priors = bilby.core.prior.PriorDict()

#     init_parameters[f"γ"] = None
#     priors[f"γ"] = P.γ

#     init_parameters[f"σp"] = None
#     priors[f"σp"] = P.σp


#     init_parameters[f"σm"] = None
#     priors[f"σm"] = P.σm
 
#     return init_parameters,priors
    


