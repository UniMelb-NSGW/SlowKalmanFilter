# Slow Kalman Filter


<!-- [![codecov](https://codecov.io/gh/tomkimpson/KalmanPhasePTA/graph/badge.svg?token=Y2TSEX32BI)](https://codecov.io/gh/tomkimpson/KalmanPhasePTA) -->


[![Build Status](https://github.com/tomkimpson/SlowKalmanFilter/actions/workflows/run_test.yml/badge.svg?branch=main)](https://github.com/tomkimpson/SlowKalmanFilter/actions/workflows/run_test.yml?query=branch%3Amain)


This repository is a generalized Kalman filter that focuses on correctness, explicitness and good unit tests over computational performance.

It is used as a "source of truth" to check the results of other more performant filters for application to astrophysical data analysis and modeling.

It is purely a filter - there is no Bayesian inference, nested sampling, etc. However, for compatibility with other projects it does use the `Bilby` package to set input parameters to the filter.

The demonstration is is based around Kalman filtering of a state-space model of pulsar timing array frequency data. However the filter itself (`src/kalman_filter.py`) is general and the repository as a whole should be easily extensible to other systems or projects



#### Getting started

1. Use `configs/create_ini_file.py` to create a `.ini` file. All settings for synthetic data are contained in this `.ini` file. 

2. Pass the `.ini` file as a command line argument to `src/main.py`. FOr example, `python main.py configs/sandbox.ini`


If no config file is passed, then the filter will run with default parameters. 

