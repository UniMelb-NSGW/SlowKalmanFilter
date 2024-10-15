import scipy.linalg as la
import numpy as np
from numba import jit
from numpy.linalg import solve, slogdet



################################


################################

@jit(nopython=True)
def f(t, x, g):
    x0 = x[0]
    x1 = x[1]

    rhs = np.asarray([x1, -g*np.sin(x0)])
    return rhs

@jit(nopython=True)
def rk4_step(func, x, t, dt, *args):
    k1 = dt * func(t, x, *args)
    k2 = dt * func(t + dt / 2, x + k1 / 2, *args)
    k3 = dt * func(t + dt / 2, x + k2 / 2, *args)
    k4 = dt * func(t + dt, x + k3, *args)

    return x + (k1 + 2 * k2 + 2 * k3 + k4) / 6

@jit(nopython=True)
def propagate_sigma_points(SigmaPoints, params, tcur, dt):
    g = params 

    PropagatedSigmaPoints = np.empty_like(SigmaPoints)
    for i, x in enumerate(SigmaPoints):
        PropagatedSigmaPoints[i] = rk4_step(f, x, tcur, dt, g)

    return PropagatedSigmaPoints

################################

@jit(nopython=True)
def Predict(Wc, Wm, PropagatedSigmaPoints, nstates, Q):
    Xp = np.zeros((nstates, 1))
    for i in range(len(Wm)):
        Xp += Wm[i] * PropagatedSigmaPoints[i].reshape(nstates, 1)

        
    SigmaPointDiff = np.zeros((len(PropagatedSigmaPoints), nstates))
    for i, point in enumerate(PropagatedSigmaPoints):
        SigmaPointDiff[i] = point - Xp.ravel()

    Pp = np.zeros((nstates, nstates))
    for i, diff in enumerate(SigmaPointDiff):
        Pp += Wc[i] * np.outer(diff, diff)
        
    Pp += Q

    return Xp, Pp

@jit(nopython=True)
def Update(Xp, Pp, Wc, PropagatedSigmaPoints, Observation, WeightedMeas, MeasurementDiff, R, nstates, nmeas):
    Inn = Observation - WeightedMeas

    SigmaPointDiff = PropagatedSigmaPoints - Xp.T

    PredMeasurementCov = np.zeros((nmeas, nmeas))
    for i in range(len(Wc)):
        PredMeasurementCov += Wc[i] * np.outer((MeasurementDiff.T)[i],( MeasurementDiff.T)[i])
    PredMeasurementCov += R

    CrossCovariance = np.zeros((nstates, nmeas))
    for i in range(len(Wc)):
        CrossCovariance += Wc[i] * np.outer(SigmaPointDiff[i], (MeasurementDiff.T)[i])

    kalman_gain = np.linalg.solve(PredMeasurementCov.T, CrossCovariance.T).T

    X = Xp + kalman_gain*Inn

    P = Pp - np.dot(kalman_gain, np.dot(PredMeasurementCov, kalman_gain.T))

    sign, log_det = np.linalg.slogdet(PredMeasurementCov)
    ll = -0.5 * (log_det + np.dot(Inn.T, np.linalg.solve(PredMeasurementCov, Inn))
                 + np.log(2 * np.pi))
    
    return X, P, ll

class KalmanFilterUpdateUKF(object):
    def __init__(self, Obs, R, Q, t0, nstates, nmeas):
        self.Obs = Obs
        self.R = R
        self.Q = Q
        self.t0 = t0
        self.nstates = nstates
        self.nmeas = nmeas
        self.ll = 0

    def Predict(self, Q):
        self.Xp, self.Pp = Predict(self.Wc, self.Wm, self.PropagatedSigmaPoints, self.nstates, Q)

    def Update(self, Observation):
        self.X, self.P, ll = Update(self.Xp, self.Pp, self.Wc, self.PropagatedSigmaPoints,
                             Observation, self.WeightedMeas, self.MeasurementDiff, self.R, self.nstates, self.nmeas)

        self.ll += ll

    
    def Propagate(self, params, tcur, dt):
        params_tuple = params['g']
        self.PropagatedSigmaPoints = propagate_sigma_points(self.SigmaPoints, params_tuple, tcur, dt)

    def SigmaPointMeasurement(self):

        SP_Meas = np.column_stack((np.sin(self.PropagatedSigmaPoints[:, 0])))

        self.WeightedMeas = np.einsum('i,ij->j', self.Wm, SP_Meas.T)

        self.MeasurementDiff = SP_Meas - self.WeightedMeas

    def CalculateWeights(self):
        L = self.nstates
        alpha = 1e-4
        beta =  2
        kappa = 3 - self.nstates
        kappa = 0

        # Compute sigma point weights
        lambda_ = alpha**2 * (self.nstates + kappa) - self.nstates

        self.Wm = np.concatenate(([lambda_/(self.nstates+lambda_)], 
                             (0.5/(self.nstates+lambda_))*np.ones(2*self.nstates)), axis=None)
        
        self.Wc = np.concatenate(([lambda_/(self.nstates+lambda_)+(1-alpha**2+beta)], 
                             (0.5/(self.nstates+lambda_))*np.ones(2*self.nstates)), axis=None)

        self.gamma = np.sqrt(self.nstates + lambda_)

    def CalculateSigmaPoints(self, X, P):

        epsilon = 1e-24
        Pos_definite_Check= 0.5*(P + P.T) + epsilon*np.eye(len(X))
        
        U = la.cholesky(Pos_definite_Check).T # sqrt
        sigma_points = np.zeros((2*self.nstates + 1, self.nstates))
        sigma_points[0] = X
        for i in range(self.nstates):
            sigma_points[i+1] = X + self.gamma*U[:, i]
            sigma_points[self.nstates+i+1] = X - self.gamma*U[:, i]

        self.SigmaPoints = sigma_points


    def ll_on_data(self, params, returnstates=False):
        self.ll = 0

        self.X = np.zeros((self.nstates,1))
        # self.X[0] = np.pi/5
        # self.X[1] = 1

        self.P = np.eye(self.nstates)*10

        NObs = len(self.Obs)
        if returnstates:
            xx = np.zeros((NObs,self.nstates))
            px = np.zeros((NObs,self.nstates))

        self.CalculateWeights()
        
        i = 0
        tcur = self.t0
        for step, Obs in enumerate(self.Obs):
            tcur+=Obs[0]

            self.CalculateSigmaPoints(self.X.squeeze(), self.P)

            self.Propagate(params, tcur, Obs[0])

            self.Predict(self.Q)

            self.CalculateSigmaPoints(self.Xp.squeeze(), self.Pp)
            self.PropagatedSigmaPoints = self.SigmaPoints

            self.SigmaPointMeasurement()

            self.Update(Obs[1])

            if returnstates:
                xx[i,:] = self.X.squeeze()
                px[i,:] = np.diag(self.P)
                i+=1

        if returnstates:
            return xx, px, self.ll
        else:
            return self.ll



