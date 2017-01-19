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

    # Presort the data
    start_data_t = datetime.now()
    X_idx_sorted = np.asfortranarray(np.argsort(X, axis=0),
                                     dtype=np.int32)
    end_data_t = datetime.now() - start_data_t

    # Create a list of Gradient Boosting
    clf = GradientBoostingClassifier()
    clf.set_params(**params)

    start_fit_t = datetime.now()
    clf.fit(X, y, X_idx_sorted=X_idx_sorted)
    end_fit_t = datetime.now() - start_fit_t

    score = np.mean(clf.predict(T) == valid)

    return score, end_data_t, end_fit_t


if __name__ == '__main__':
    USAGE = """usage: python %s dataset presort growth_style path_results n_try

    where:
        - dataset is one of {'random', 'cover_type', 'higgs'}
        - presort is one of {True, False}. To presort the data or not.
        - growth_style is one of {'leaf', 'depth'}.
        - path_results is the location to store the results
        - n_try is either an integer or None. Integer is the number of try
        the benchmark will be repeated.
    """

    DATASET_CHOICE = ('random', 'cover_type', 'higgs')

    N_ESTIMATORS = [10]  # np.array([1, 1e1], dtype=int)
    LEARNING_RATE = 0.1
    MIN_IMPURITY_SPLIT = 1e-7
    MAX_DEPTH = [8]  # np.array([1, 3, 5, 8], dtype=int)
    MIN_SAMPLES_LEAF = 1
    SUBSAMPLES = 1.
    N_THREADS = 1
    RND_SEED = 42
    N_SAMPLES_SPLIT = np.array([1e3, 1e4, 1e5, 1e6], dtype=int)

    print(__doc__ + '\n')
    if not len(sys.argv) == 6:
        print(USAGE % __file__)
        sys.exit(-1)
    else:
        dataset = sys.argv[1]
        presort = str2bool(sys.argv[2])
        growth = sys.argv[3]
        store_dir = sys.argv[4]
        # Try to perform a conversion to int
        try:
            n_try = int(sys.argv[5])
        except:
            if sys.argv[5] == 'None':
                n_try = None
            else:
                raise ValueError('Choose None or an integer for n_try')
        if dataset not in DATASET_CHOICE:
            raise ValueError('Unknown dataset')

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
    params['n_samples_split'] = N_SAMPLES_SPLIT

    params_list = list(ParameterGrid(params))
    for p_idx in range(len(params_list)):
        if growth == 'leaf':
            params_list[p_idx]['max_leaf_nodes'] = np.power(
                2, params_list[p_idx]['max_depth'])
        elif growth == 'depth':
            params_list[p_idx]['max_leaf_nodes'] = None

    N_SAMPLES = np.array([1e5, 1e6, 1e7], dtype=int)
    N_FEATURES = np.array([1, 5, 10], dtype=int)

    # Create several array for the data
    if dataset == 'random':
        array_data = [
            misc.generate_samples(ns, nf, RND_SEED)
            for ns in N_SAMPLES for nf in N_FEATURES
        ]
    elif dataset == 'cover_type':
        array_data = [misc.load_cover_type(RND_SEED)]
    elif dataset == 'higgs':
        # We will select some samples from the higgs dataset
        array_data = [misc.load_higgs(random_state=RND_SEED, n_samples=ns)
                      for ns in N_SAMPLES]
    else:
        raise ValueError('The dataset is not known. The possible choices are:'
                         ' random')

    # Save only the time for the moment
    res_skl = [[data[0].shape, p, misc.bench(bench_skl, data, n=n_try, **p)]
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
