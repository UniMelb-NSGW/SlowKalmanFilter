from src import system_parameters
import numpy as np 
from pathlib import Path

def get_project_root() -> Path:
    return Path(__file__).resolve().parent.parent


"""Check that we can call SystemParameters and the variables are generally set properly"""
def test_basic_call():

    N = 5
    
    #Just check the GW parameters
    Ω = np.random.uniform(low=0.0,high=2*np.pi,size=N)
    Φ0= np.random.uniform(low=0.0,high=2*np.pi,size=N)
    ψ= np.random.uniform(low=0.0,high=2*np.pi,size=N)
    ι= np.random.uniform(low=0.0,high=2*np.pi,size=N)
    δ= np.random.uniform(low=0.0,high=2*np.pi,size=N)
    α= np.random.uniform(low=0.0,high=2*np.pi,size=N)
    h= np.random.uniform(low=0.0,high=2*np.pi,size=N)

    #And observation time
    T = np.random.uniform(1,10,size=N)

    for i in range(N):
        P   = system_parameters.SystemParameters(Ω=Ω[i],Φ0=Φ0[i],ψ=ψ[i],ι=ι[i],δ=δ[i],α=α[i],h=h[i],T=T[i]) 
        assert P.Ω==Ω[i]
        assert P.Φ0==Φ0[i]
        assert P.ψ==ψ[i]
        assert P.ι==ι[i]
        assert P.δ==δ[i]
        assert P.α==α[i]
        assert P.h==h[i]
        assert P.T==T[i]



"""Check that basic config file can be read without errors"""
def test_config_file():

    root = get_project_root()
    P   = system_parameters.SystemParameters(config_file=root / 'src/configs/testing.ini') 
