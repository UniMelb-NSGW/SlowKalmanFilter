import sdeint
import numpy as np


def pendulum(t, x, g, sigma_x1, sigma_y1, seed=None):

    if seed is not None:
        np.random.seed(seed)

    def f(x, t):
        x0 = x[0]
        x1 = x[1]

        rhs = np.asarray([x1, -g*np.sin(x0)])

        return rhs

    def h(x, t):
        return np.diag([0, sigma_x1])

    states = sdeint.itoint(f, h, x, t)
    x0 = states[:, 0]
    x1 = states[:, 1]

    data = np.zeros((t.size, 2))
    no_noise = np.zeros(t.size)
    data[:,0] = t[1]-t[0]
    data[:,1] = np.sin(x0) + np.random.randn(t.size) * np.sqrt(sigma_y1)
    no_noise[:] = np.sin(x0)
    return states, data, no_noise



