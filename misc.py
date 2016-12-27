from __future__ import print_function

import os

import numpy as np
import pandas as pd
import joblib

from sklearn.datasets import fetch_covtype, get_data_home
from sklearn.model_selection import train_test_split
from sklearn.externals.joblib import Memory
from sklearn.utils import check_array

from higgs import fetch_higgs

# Memoize the data extraction and memory map the resulting
# train / test splits in readonly mode
memory_covertype = Memory(
    os.path.join(get_data_home(), 'covertype_benchmark_data'), mmap_mode='r')

memory_higgs = Memory(
    os.path.join(get_data_home(), 'higgs_benchmark_data'), mmap_mode='r')

memory_random = Memory(
    os.path.join(get_data_home(), 'random_benchmark_data'), mmap_mode='r')


@memory_random
def generate_samples(n_samples, n_features, random_state):
    """Generate random samples

    Parameters
    ----------
    n_samples : int,
        The number of samples to generate.

    n_features : int,
        The number of features to generate.

    Returns
    -------
    X : ndarray, shape (n_train_samples, n_features)

    y : ndarray, shape (n_train_samples, )

    T : ndarray, shape (n_test_samples, n_features)

    valid: ndarray, shape (n_test_samples, )
    """

    data = np.random.randn(n_samples, n_features)
    label = np.random.randint(2, size=n_samples)

    X, T, y, valid = train_test_split(
        data, label, test_size=.1, random_state=random_state)

    return X, y, T, valid


@memory_covertype.cache
def load_cover_type(random_state=None, dtype=np.float32, order='C'):
    """Load cover type data

    Parameters
    ----------
    random_state : int, np.random.RandomState or None, optional (default=None)
        The random state used to shuffle the data if needed.

    dtype : np.dtype, optional (default=np.float32)
        The type for the data to be returned.

    order : 'C', 'F' or None, optional (default='C')
        Whether an array will be forced to be fortran or c-style.
        When order is None (default), then if copy=False, nothing is ensured
        about the memory layout of the output array; otherwise (copy=True)
        the memory layout of the returned array is kept as close as possible
        to the original array.

    Returns
    -------
    X : ndarray, shape (n_train_samples, n_features)

    y : ndarray, shape (n_train_samples, )

    T : ndarray, shape (n_test_samples, n_features)

    valid: ndarray, shape (n_test_samples, )
    """
    ######################################################################
    # Load dataset
    print("Loading dataset...")
    data = fetch_covtype(
        download_if_missing=True, shuffle=True, random_state=random_state)
    X = check_array(data['data'], dtype=dtype, order=order)
    y = (data['target'] != 1).astype(np.int)

    # Create train-test split (as [Joachims, 2006])
    print("Creating train-test split...")
    n_train = 522911
    X_train = X[:n_train]
    y_train = y[:n_train]
    X_test = X[n_train:]
    y_test = y[n_train:]

    # Standardize first 10 features (the numerical ones)
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    mean[10:] = 0.0
    std[10:] = 1.0
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    return X_train, y_train, X_test, y_test


@memory_higgs.cache
def load_higgs(random_state=None, dtype=np.float32, order='C'):
    """Load Higgs data

    Parameters
    ----------
    random_state : int, np.random.RandomState or None, optional (default=None)
        The random state used to shuffle the data if needed.

    dtype : np.dtype, optional (default=np.float32)
        The type for the data to be returned.

    order : 'C', 'F' or None, optional (default='C')
        Whether an array will be forced to be fortran or c-style.
        When order is None (default), then if copy=False, nothing is ensured
        about the memory layout of the output array; otherwise (copy=True)
        the memory layout of the returned array is kept as close as possible
        to the original array.

    Returns
    -------
    X : ndarray, shape (n_train_samples, n_features)

    y : ndarray, shape (n_train_samples, )

    T : ndarray, shape (n_test_samples, n_features)

    valid : ndarray, shape (n_test_samples, )
    """
    ######################################################################
    # Load dataset
    print("Loading dataset...")
    data = fetch_higgs(
        download_if_missing=True, shuffle=False, random_state=random_state)
    X = check_array(data['data'], dtype=dtype, order=order)
    y = (data['target'] != 1).astype(np.int)

    # Create train-test split (as [Baldi, 2014])
    print("Creating train-test split...")
    n_train = 10500000
    X_train = X[:n_train]
    y_train = y[:n_train]
    X_test = X[n_train:]
    y_test = y[n_train:]

    return X_train, y_train, X_test, y_test


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
    # assert n > 2
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


def load_benchmark_data(filename, lgbm=False):
    """Function to load the data collected during the benchmark inside a
    DataFrame

    Parameters
    ----------
    filename: str,
        The pickle to load containing all the benchmark info.

    lgbm: bool, optional (default=False)
        Indicate if the data correspond to LightGBM. The `max_depth` needs to
        be changed. This is a temporary trick for now.

    Returns
    -------
    data: DataFrame,
        The pandas DataFrame with all information for benchmarking.
    """
    # Load the data
    pickle_data = joblib.load(filename)

    # Collect the data iteratively
    num_samples = []
    num_features = []
    max_depth = []
    n_estimators = []
    avg_fit_time = []
    std_fit_time = []
    avg_data_time = []
    std_data_time = []
    avg_score = []
    std_score = []
    for conf in pickle_data:
        num_samples.append(conf[0][0])
        num_features.append(conf[0][1])
        if lgbm:
            max_depth.append(conf[1]['max_depth'] - 1)
        else:
            max_depth.append(conf[1]['max_depth'])
        n_estimators.append(conf[1]['n_estimators'])
        avg_score.append(np.mean(conf[2][0]))
        std_score.append(np.std(conf[2][0]))
        avg_data_time.append(np.mean(conf[2][1]))
        std_data_time.append(np.std(conf[2][1]))
        avg_fit_time.append(np.mean(conf[2][2]))
        std_fit_time.append(np.std(conf[2][2]))

    # Create the DataFrame
    # Start with the data dictionary
    d = {
        'num_samples': num_samples,
        'num_features': num_features,
        'max_depth': max_depth,
        'n_estimators': n_estimators,
        'avg_fit_time': avg_fit_time,
        'std_fit_time': std_fit_time,
        'avg_data_time': avg_data_time,
        'std_data_time': std_data_time,
        'avg_score': avg_score,
        'std_score': std_score
    }

    return pd.DataFrame(data=d)
