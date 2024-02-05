from numpy import sin,cos 
import numpy as np 

import sys





"""
Reparameterise h0 and ι into hplus and hcross amplitudes
"""
def _h_amplitudes(h,ι): 

    hp =  h*(1.0 + cos(ι)**2)
    hx =  -2.0*h*cos(ι)

    return hp, hx



"""
Given the principal axes vectors `m` and `n`, return the two polarisation tensors e_+, e_x
"""
def _polarisation_tensors(m, n):

    # For e_+,e_x, Tensordot might be a bit faster, but list comprehension has JIT support and is more explicit
    e_plus              = np.array([m[i]*m[j]-n[i]*n[j] for i in range(3) for j in range(3)]).reshape(3,3)
    e_cross             = np.array([m[i]*n[j]+n[i]*m[j] for i in range(3) for j in range(3)]).reshape(3,3)

    return e_plus,e_cross



def _principal_axes(theta,phi,psi):
    
    m1 = sin(phi)*cos(psi) - sin(psi)*cos(phi)*cos(theta)
    m2 = -(cos(phi)*cos(psi) + sin(psi)*sin(phi)*cos(theta))
    m3 = sin(psi)*sin(theta)
    m = np.array([m1,m2,m3])

    n1 = -sin(phi)*sin(psi) - cos(psi)*cos(phi)*cos(theta)
    n2 = cos(phi)*sin(psi) - cos(psi)*sin(phi)*cos(theta)
    n3 = cos(psi)*sin(theta)
    n = np.array([n1,n2,n3])

    return m,n



"""
The GW measurement equation. 
Note that it is convenient to just define X here where g(t) = 1 - X.
This is because we operate on heterodyned states.
"""
def gw_measurement_effect(Ω,Φ0,ψ,ι,δ,α,h,q,d,t): #7 GW parameters, 2 "pulsar parameters", and the discrete timesteps


    hp,hx               = _h_amplitudes(h,ι)                # plus and cross GW amplitudes. Scalars
    m,n                 = _principal_axes(np.pi/2.0 - δ,α,ψ) # The principal axes of the GW. np.pi/2.0 - δ converts declination to an altitude. length-3 vectors
    e_plus,e_cross      = _polarisation_tensors(m.T,n.T)    # The polarization tensors. 3x3 matrices
    Hij                 = hp * e_plus + hx * e_cross        # Amplitude tensor. Shape (3,3)
    

    #Verbose, explicit, slow
    H                   = np.zeros(len(q))
    for k in range(len(H)):
        H_k = 0
        q_k = q[k,:]
        for i in range(3):
            for j in range(3):
                H_k += Hij[i,j]*q_k[i]*q_k[j]
        H[k] = H_k

                       

    gw_direction        = np.cross(m,n)                     # The GW source direction. 
    dot_product         = 1.0 + q @ gw_direction            # The dot product of PSR direction and GW direction




    #The GW phases
    earth_term_phase =  -Ω*t + Φ0       #This has length = len (t). 1D vector
    pulsar_term_phase = Ω*dot_product*d #This has length = Npsr. 1D vector


    #Reshapes to allow broadcasting and correct summation 
    earth_term_phase  = np.expand_dims(earth_term_phase, axis=-1) #Make it 2D
    pulsar_term_phase = np.expand_dims(pulsar_term_phase, axis=-1).T #Make it 2D and transpose


    #Put it all together
    GW_factor = H/(2.0*dot_product) *(cos(earth_term_phase) - cos(earth_term_phase +pulsar_term_phase))




    return GW_factor



