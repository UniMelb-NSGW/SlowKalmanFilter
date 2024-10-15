from src import model as KalmanModel
import numpy as np 
from jax import jacfwd
from jax import config
config.update("jax_enable_x64", True)



"""Check that the F-jacobian specified in the model is correct given the f(x) function """
def test_F_jacobian():

    #First do a basic test with standard parameter values
    dt    = 0.001
    model = KalmanModel.KalmanPendulum(dt=dt)
    x     = np.array([1.0,1.0])
    g     = 10 
    jacobian_AD = jacfwd(model.f)(x,g) #argnums = 0 by default which differentiates w.r.t. x only
    jacobian_manual = model.F_jacobian(x,g)

    np.testing.assert_array_almost_equal(jacobian_AD,jacobian_manual)



    #Now test on some random numbers
    for i in range(10):
        x1, x2, g = np.random.uniform(size=3)
        x     = np.array([x1,x2])
    
        jacobian_AD = jacfwd(model.f)(x,g) #argnums = 0 by default which differentiates w.r.t. x only
        jacobian_manual = model.F_jacobian(x,g)
        np.testing.assert_array_almost_equal(jacobian_AD,jacobian_manual)



"""Check that the H-jacobian specified in the model is correct given the h(x) function """
def test_H_jacobian():

    #First do a basic test with standard parameter values
    dt    = 0.001
    model = KalmanModel.KalmanPendulum(dt=dt)
    x     = np.array([1.0,1.0])
    jacobian_AD = jacfwd(model.h)(x).reshape(1,2)
    jacobian_manual = model.H_jacobian(x)


    np.testing.assert_array_almost_equal(jacobian_AD,jacobian_manual)


    #Now test on some random numbers
    for i in range(10):
        x1, x2 = np.random.uniform(size=2)
        x     = np.array([x1,x2])
    
        jacobian_AD = jacfwd(model.h)(x).reshape(1,2)
        jacobian_manual = model.H_jacobian(x)
        np.testing.assert_array_almost_equal(jacobian_AD,jacobian_manual)
