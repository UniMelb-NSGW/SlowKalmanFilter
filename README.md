# Slow Kalman Filter


[![codecov](https://codecov.io/gh/tomkimpson/SlowKalmanFilter/graph/badge.svg?token=MXT5Y10BX7)](https://codecov.io/gh/tomkimpson/SlowKalmanFilter)

[![Build Status](https://github.com/tomkimpson/SlowKalmanFilter/actions/workflows/run_test.yml/badge.svg?branch=main)](https://github.com/tomkimpson/SlowKalmanFilter/actions/workflows/run_test.yml?query=branch%3Amain)


This repository is a generalized non-linear (extended) Kalman filter that focuses on correctness, explicitness and good tests over computational performance.

It is used as a "source of truth" to check the results of other more performant filters for application to astrophysical data analysis and modeling. It is also a useful starting point for other projects which employ state-space models. 

It is purely a filter - there is no parameter estimation, Bayesian inference etc. However, with an eye on this use case it does use the the [Bilby](https://lscsoft.docs.ligo.org/bilby/) package to specify the model parameters which are passed to the filter. 

We demonstrate the capabilities of the filter using a noisy pendulum model. See `notebooks/NoisyPendulum.py` for a worked example. However the filter itself (`src/kalman_filter.py`) is general and the repository as a whole should be easily extensible to other systems or projects.


---

### Where to start

* Install the Python environment from the env.yml file via `conda env create -f environment.yml`

* Run the notebook `notebooks/NoisyPendulum.ipynb`. This notebook gives an overview of applying the Kalman filter to the problem of a noisy pendulum.


---

### Contributing

Please feel free to [open an issue](https://github.com/Melbourne-AstroCode/SlowKalmanFilter/issues) or create a [pull request](https://github.com/Melbourne-AstroCode/SlowKalmanFilter/pulls) for any questions or edits.


