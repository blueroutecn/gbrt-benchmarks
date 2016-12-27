from __future__ import print_function
import sys
import os
from datetime import datetime

import numpy as np
import joblib

from sklearn.model_selection import ParameterGrid

import lightgbm as lgb

import misc


def bench_lgbm(X, y, T, valid, **params):
    """Execute the gradient boosting pipeline"""

    # Extract the parameter required for the dataset
    max_bin = params.pop('max_bin')

    # Measure the time to prepare the data
    start_data_t = datetime.now()
    # Prepare the data
    lgbm_training = lgb.Dataset(X, label=y, max_bin=max_bin)
    end_data_t = datetime.now() - start_data_t
    lgbm_testing = lgb.Dataset(T, label=valid, max_bin=max_bin)

    # Pop the number of trees
    n_est = params.pop('n_estimators')
    # Create the number of leafs depending of the max depth
    params['num_leaves'] = np.power(2, params['max_depth'] - 1)

    # Set the fitting time
    start_fit_t = datetime.now()
    gbm = lgb.train(params, lgbm_training, num_boost_round=n_est)
    end_fit_t = datetime.now() - start_fit_t

    pred = gbm.predict(T)
    pred[np.nonzero(pred >= 0.5)] = 1
    pred[np.nonzero(pred < 0.5)] = 0

    score = np.mean(pred == valid)

    return score, end_data_t, end_fit_t


if __name__ == '__main__':
    USAGE = """usage: python %s dataset path_results n_try

    where:
        - dataset is one of {'random', 'cover_type'}
        - path_results is the location to store the results
        - n_try is the number of run
    """

    N_ESTIMATORS = np.array([1, 1e1], dtype=int)
    LEARNING_RATE = 0.1
    MIN_IMPURITY_SPLIT = 1e-7
    # The max depth needs to be increased of 1
    MAX_DEPTH = np.array([2, 4, 6, 9], dtype=int)
    MIN_SAMPLES_LEAF = 1
    SUBSAMPLES = 1.
    N_THREADS = 1
    RND_SEED = 42

    N_SAMPLES = np.array([10e2, 10e3, 10e4], dtype=int)
    N_FEATURES = np.array([1, 5, 10], dtype=int)

    # Setup the parameters
    params = {}
    params['n_estimators'] = N_ESTIMATORS
    params['application'] = ['binary']
    params['boosting'] = ['gbdt']
    params['learning_rate'] = [LEARNING_RATE]
    # params['num_leaves'] = [31]  # We set this depending of the depth
    params['tree_learner'] = ['serial']
    params['num_threads'] = [N_THREADS]
    params['max_depth'] = MAX_DEPTH
    params['min_data_in_leaf'] = [MIN_SAMPLES_LEAF]
    params['feature_fraction'] = [1.0]
    params['feature_fraction_seed'] = [RND_SEED]
    params['bagging_fraction'] = [SUBSAMPLES]
    params['bagging_freq'] = [0]
    params['bagging_seed'] = [RND_SEED]
    params['max_bin'] = [255]
    params['data_random_seed'] = [RND_SEED]
    params['is_sparse'] = [False]
    params['metric'] = ['binary_logloss']
    params['min_gain_to_split'] = [MIN_IMPURITY_SPLIT]
    params['verbosity'] = [1]
    params_list = list(ParameterGrid(params))

    print(__doc__ + '\n')
    if not len(sys.argv) == 4:
        print(USAGE % __file__)
        sys.exit(-1)
    else:
        dataset = sys.argv[1]
        store_dir = sys.argv[2]
        n_try = int(sys.argv[3])

    # Create several array for the data
    if dataset == 'random':
        array_data = [
            misc.generate_samples(ns, nf, RND_SEED)
            for ns in N_SAMPLES for nf in N_FEATURES
        ]
    elif dataset == 'cover_type':
        array_data = [misc.load_cover_type(RND_SEED)]
    else:
        raise ValueError('The dataset is not known. The possible choices are:'
                         ' random')

    # Save only the time for the moment
    res_lgbm = [[data[0].shape, p, misc.bench(
        bench_lgbm, data, n=n_try, **p)]
                for p in params_list for data in array_data]

    # Check that the path is existing
    if not os.path.exists(store_dir):
        os.makedirs(store_dir)

    filename = 'lightgbm_' + dataset + '.pk'
    store_filename = os.path.join(store_dir, filename)

    joblib.dump(res_lgbm, store_filename)
