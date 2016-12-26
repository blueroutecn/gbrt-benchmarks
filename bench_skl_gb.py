from __future__ import print_function
import sys
import os
from datetime import datetime

import numpy as np
import joblib

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import ParameterGrid

import misc


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


def bench_skl(X, y, T, valid, **params):
    """Execute the gradient boosting pipeline"""

    # Create a list of Gradient Boosting
    clf = GradientBoostingClassifier()
    clf.set_params(**params)

    start_fit_t = datetime.now()
    clf.fit(X, y)
    end_fit_t = datetime.now() - start_fit_t

    score = np.mean(clf.predict(T) == valid)

    return score, datetime.now() - datetime.now(), end_fit_t


if __name__ == '__main__':
    USAGE = """usage: python %s dataset presort growth_style path_results n_try

    where:
        - dataset is one of {'random'}
        - presort is one of {True, False}. To presort the data or not.
        - growth_style is one of {'leaf', 'depth'}.
        - path_results is the location to store the results
        - n_try is the number of run
    """

    N_ESTIMATORS = np.array([1, 1e1], dtype=int)
    LEARNING_RATE = 0.1
    MIN_IMPURITY_SPLIT = 1e-7
    MAX_DEPTH = np.array([1, 3, 5, 8], dtype=int)
    MIN_SAMPLES_LEAF = 1
    SUBSAMPLES = 1.
    N_THREADS = 1
    RND_SEED = 42

    print(__doc__ + '\n')
    if not len(sys.argv) == 6:
        print(USAGE % __file__)
        sys.exit(-1)
    else:
        dataset = sys.argv[1]
        presort = str2bool(sys.argv[2])
        growth = sys.argv[3]
        store_dir = sys.argv[4]
        n_try = int(sys.argv[5])

    # Setup the parameters
    params = {}
    params['max_depth'] = MAX_DEPTH
    params['learning_rate'] = [LEARNING_RATE]
    params['n_estimators'] = N_ESTIMATORS
    params['loss'] = ['deviance']
    params['min_weight_fraction_leaf'] = [0.]
    params['subsample'] = [SUBSAMPLES]
    params['max_features'] = [None]
    params['min_samples_split'] = [2]
    params['min_samples_leaf'] = [MIN_SAMPLES_LEAF]
    params['min_impurity_split'] = [MIN_IMPURITY_SPLIT]
    params['presort'] = [presort]
    params['init'] = [None]
    params['warm_start'] = [False]
    params['verbose'] = [0]
    params['random_state'] = [RND_SEED]
    params['criterion'] = ['friedman_mse']

    params_list = list(ParameterGrid(params))
    for p_idx in len(params_list):
        if growth == 'leaf':
            params_list[p_idx]['max_leaf_nodes'] = np.power(
                2,
                params_list[p_idx]['max_depth'])
        elif growth == 'depth':
            params_list[p_idx]['max_leaf_nodes'] = None

    N_SAMPLES = np.array([10e2, 10e3, 10e4], dtype=int)
    N_FEATURES = np.array([1, 5, 10], dtype=int)

    # Create several array for the data
    if dataset == 'random':
        array_data = [misc.generate_samples(ns, nf, RND_SEED)
                      for ns in N_SAMPLES for nf in N_FEATURES]
    else:
        raise ValueError('The dataset is not known. The possible choices are:'
                         ' random')

    # Save only the time for the moment
    res_skl = [[data[0].shape, p, misc.bench(bench_skl, data, **p)]
               for p in params_list for data in array_data]

    # Check that the path is existing
    if not os.path.exists(store_dir):
        os.makedirs(store_dir)

    if presort:
        name_presort = '_with_presort_'
    else:
        name_presort = '_without_presort_'
    filename = 'skl_' + growth + name_presort + dataset + '.pk'
    store_filename = os.path.join(store_dir, filename)

    joblib.dump(res_skl, store_filename)
