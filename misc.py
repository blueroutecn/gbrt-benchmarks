from __future__ import print_function

import os

import numpy as np

from sklearn.model_selection import train_test_split


def generate_samples(n_samples, n_features, random_state):
    """Generate random samples

    Parameters
    ----------
    n_samples: int,
        The number of samples to generate.

    n_features: int,
        The number of features to generate.

    Returns
    -------
    X: ndarray, shape (n_train_samples, n_features)

    y: ndarray, shape (n_train_samples, )

    T: ndarray, shape (n_test_samples, n_features)

    valid: ndarray, shape (n_test_samples, )
    """

    data = np.random.randn(n_samples, n_features)
    label = np.random.randint(2, size=n_samples)

    X, T, y, valid = train_test_split(data, label, test_size=.1,
                                      random_state=random_state)

    return X, y, T, valid


def dtime_to_seconds(dtime):
    return dtime.seconds + (dtime.microseconds * 1e-6)


def bench(func, data, n=10, **params):
    """
    Benchmark a given function. The function is executed n times and
    its output is expected to be of type datetime.datetime.

    All values are converted to seconds and returned in an array.

    Parameters
    ----------
    func: function,
        The function to use for benchmarking.

    data: tuple, shape (4, )
        (X, y, T, valid) containing training (X, y) and validation
        (T, valid) data.

    params:
        the parameters used in the function `func`.

    Returns
    -------
    D: ndarray, shape (2, )
        return the score and elapsed time of the function.
    """

    # for the given number of try
    #assert n > 2
    score = []
    time_data = []
    time_fit = []
    for i in range(n):
        sc, t_data, t_fit = func(*data, **params)

        # Append the values
        score.append(sc)
        time_data.append(dtime_to_seconds(t_data))
        time_fit.append(dtime_to_seconds(t_fit))

    return np.array(score), np.array(time_data), np.array(time_fit)
