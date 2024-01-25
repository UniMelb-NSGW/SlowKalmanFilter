# Slow Kalman Filter


<!-- [![codecov](https://codecov.io/gh/tomkimpson/KalmanPhasePTA/graph/badge.svg?token=Y2TSEX32BI)](https://codecov.io/gh/tomkimpson/KalmanPhasePTA) -->


[![Build Status](https://github.com/tomkimpson/StateSpacePTA/actions/workflows/run_test.yml/badge.svg?branch=main)](https://github.com/tomkimpson/StateSpacePTA/actions/workflows/run_test.yml?query=branch%3Amain)


This repo is a generalized Kalman filter that focuses on correctness, explicitness and good unit tests over computational performance.

It is used as a "source of truth" to check the results of other more performance filters.

It is purely a filter - no Bayesian inference, nested sampling, etc. However, for compatibility with other projects it does use the `Bilby` package to set input parameters to the filter.


