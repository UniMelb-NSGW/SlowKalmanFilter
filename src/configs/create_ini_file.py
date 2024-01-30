import configparser
import numpy as np
from pathlib import Path

def get_project_root() -> Path:
    return Path(__file__).resolve().parent.parent
root = get_project_root()


# A script for constructing a config file to be read by the files in `src`
# This should enforce reproducibility - no free parameters in `src`

config = configparser.ConfigParser()
config.optionxform = lambda option: option #enforce case sensitivity

config['GW_PARAMETERS'] = {'Ω':  5e-7, # GW angular frequency
                           'Φ0': 0.20,# GW phase offset at t=0
                           'ψ':  2.50, # GW polarisation angle
                           'ι':  1.0,  # GW source inclination
                           'δ':  1.0,  # GW source declination
                           'α':  1.0,  # GW source right ascension
                           'h':  1e-15 # GW strain
                           }

config['PSR_PARAMETERS'] = {'process_noise': 'Fixed', # the process noise on the pulsars. Any of "True", "Fixed", "Random". See pulsars.py for example
                            'Npsr': 0,                # Number of pulsars to use in PTA. 0 = all
                            'σp': 1e-20,              # only used if process_noise != True. Assign the process noise s.d. = σp for all pulsars if "Fixed". Assign randomly within U(σp/10,σp*10) if random. 
                            'γ': 1e-13,               # mean reversion. the same for every pulsar
                            'PTA_data_file': "../data/NANOGrav_pulsars.csv"
                            } 


config['OBS_PARAMETERS'] = {'T': 10,       # how long to integrate for in years
                            'cadence': 7,  # the interval between observations in days
                            'σm':1e-11,    # measurement noise standard deviation
                            'seed':1230,   # this is the noise seed. It is used for realisations of process noise and measurement noise and also if random pulsars or random process noise covariances are requested 
                             }


config['KF_PARAMETERS'] = {'measurement_model': 'pulsar', #which model to use for the KF. This will need updating ToDo
                             }



with open(root / 'configs/sandbox.ini', 'w') as configfile:
  config.write(configfile)


