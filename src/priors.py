import bilby

"""
Use bilby to define constant (delta-function) priors on the parameters
"""
def constant_priors(g,σp,σm):


    init_parameters = {}
    priors = bilby.core.prior.PriorDict()

    init_parameters[f""] = None
    priors[f"g"] = g

    init_parameters[f"σp"] = None
    priors[f"σp"] = σp

    init_parameters[f"σm"] = None
    priors[f"σm"] = σm
 
    return init_parameters,priors
    