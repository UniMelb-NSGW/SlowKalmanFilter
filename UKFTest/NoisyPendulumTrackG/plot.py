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

def plot_parameter(t, x, px, g_true):

    plt.figure(0, figsize=(10, 10))
    plt.plot(t, x[:,2], 'g')
    plt.fill_between(t, x[:,2]+np.sqrt(px[:,2]),x[:,2]-np.sqrt(px[:,2]), alpha=0.4, color='g')
    plt.axhline(y=g_true)
    plt.savefig('G_Track.png')

def plot_tracks(t, x, px, states):

    fig, axs = plt.subplots(2, 1, figsize=(10, 10))
    axs[0].plot(t, x[:,0], 'g')
    axs[0].fill_between(t, x[:,0]+np.sqrt(px[:,0]),x[:,0]-np.sqrt(px[:,0]), alpha=0.4, color='g')
    axs[0].plot(t, states[:,0])
    axs[1].plot(t, x[:,1], 'g')
    axs[1].fill_between(t, x[:,1]+np.sqrt(px[:,1]),x[:,1]-np.sqrt(px[:,1]), alpha=0.4, color='g')
    axs[1].plot(t, states[:,1])
    plt.savefig('KF_State.png')


def plot_results(t, Obs, no_noise, x, px):

    plt.figure(1)
    plt.plot(t, Obs[:,1], label='Measurement')
    plt.plot(t, no_noise, label='True')
    # plt.plot(t, np.sin(x[:,1]))
    plt.plot(t, np.sin(x[:,0]), label='KF Track')
    plt.legend()
    plt.savefig("KalmanTrack.png")
    