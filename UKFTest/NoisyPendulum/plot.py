import numpy as np
import matplotlib.pyplot as plt

def plot_data(t, states, Obs, no_noise):

    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    axs[0,0].plot(t, states[:,0])
    axs[0,1].plot(t, states[:,1])
    axs[1,0].plot(states[:,0], states[:,1])
    axs[1,1].plot(t, Obs[:,1])
    axs[1,1].plot(t, no_noise)
    plt.savefig('Data.png')


def plot_results(t, Obs, no_noise, x, px):

    plt.figure(0)
    plt.plot(t, Obs[:,1], label='Measurement')
    plt.plot(t, no_noise, label='True')
    # plt.plot(t, np.sin(x[:,1]))
    plt.plot(t, np.sin(x[:,0]), label='KF Track')
    plt.legend()
    plt.savefig("KalmanTrack.png")
    