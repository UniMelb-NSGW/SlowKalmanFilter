import numpy as np
from Obs import pendulum
from UKF import KalmanFilterUpdateUKF
from plot import plot_parameter, plot_results, plot_tracks

dt = 0.001
nstates = 3
nmeas = 1
t = np.arange(0,10,dt)
t0 = t[0]
g = 10.0
sigma_x1 = 0.1
sigma_y1 = 0.1
seed = 1
x = np.asarray([1.0, -0.1])

states, Obs, no_noise = pendulum(t, x, g, sigma_x1, sigma_y1, seed)

R = sigma_y1



Q = np.zeros((nstates,nstates))
S_x1 = sigma_x1**2
S_g = 1e-4
Q[0,0] = S_x1*dt**3 / 3
Q[0,1] = S_x1*dt**2 / 2
Q[1,0] = S_x1*dt**2 / 2
Q[1,1] = S_x1*dt
Q[2,2] = S_g


params = {}

model = KalmanFilterUpdateUKF(Obs, R, Q, t0, nstates, nmeas)
x, px, ll = model.ll_on_data(params=params, returnstates=True)

# plot_results(t, Obs, no_noise, x, px)
plot_parameter(t, x, px, g)
plot_results(t, Obs, no_noise, x, px)
plot_tracks(t, x, px, states)
