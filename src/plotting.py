
import matplotlib.pyplot as plt 
import numpy as np 
import scienceplots # noqa: F401

import warnings
import random
import glob
#from parse import * 
warnings.filterwarnings("error")
plt.style.use('science')


"""
Plotting routine.

Plots the actual state and the actual observations, along with the corresponding predictions by the Kalman filter

state_index and observation_index specify which of the states and `observation states` to plot



"""
def plot_synthetic_data(t,states,observations,predicted_states,predicted_observations,state_index=0,observation_index=0):

    #Setup the figure
    h,w = 12,8
    rows = 2
    cols = 1
    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(h,w),sharex=True)
    
    
    
    
    
    # #Variables to plot
    tplot = t / (365*24*3600) #asssume t is in seconds, convert to years

    state = states[:,state_index]
    obs   = observations[:,state_index]


    predicted_state = predicted_states[:,state_index]
    predicted_obs   = predicted_observations[:,observation_index]


    #Plot 'em 
    axes[0].plot(tplot,state,label='Truth')
    axes[1].plot(tplot,obs,label='Truth')


    axes[0].plot(tplot,predicted_state,label='Kalman')
    axes[1].plot(tplot,predicted_obs,label='Kalman')



    # print("first state phase value = ", state_phi_i[0])

    # axes[1].plot(tplot,state_f_i)
    # print("first state frequency value = ", state_f_i[0])
    # axes[2].plot(tplot,phi_measured_i)

    # #Plot the predictions too, if you have them
    # if state_phi_pred is not None:
    #     axes[0].plot(tplot,state_phi_pred[:,psr_index])
    #     #print("first predicted state phase value = ", state_phi_pred[0,psr_index])

    # if state_f_pred is not None:
    #     axes[1].plot(tplot,state_f_pred[:,psr_index])
    #     #print("first predicted state frequency value = ", state_f_pred[0,psr_index])

    # if phi_measured_pred is not None:
    #     axes[2].plot(tplot,phi_measured_pred[:,psr_index])
    #     #print("first predicted measured phase value = ", phi_measured_pred[0,psr_index])


    # # #Make it pretty

    # fs=20
    # axes[2].set_xlabel('t [years]', fontsize=fs)
    # axes[0].set_ylabel(r'$\phi^*$ [rad]', fontsize=fs)
    # axes[1].set_ylabel(r'$f^*$ [Hz]', fontsize=fs)
    # axes[2].set_ylabel(r'$\phi^*_{\rm m}$ [rad]', fontsize=fs)

    # plt.subplots_adjust(hspace=0.0,wspace=0.0)
    # plt.suptitle(f'Synthetic data for PSR index: {psr_index}',fontsize=fs)

    # for ax in axes:    
    #     ax.xaxis.set_tick_params(labelsize=fs-4)
    #     ax.yaxis.set_tick_params(labelsize=fs-4)



    axes[1].legend()
