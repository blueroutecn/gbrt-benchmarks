from __future__ import print_function
import sys
import os
from datetime import datetime

import numpy as np
import xgboost as xgb
import joblib

from sklearn.model_selection import ParameterGrid

import misc


def bench_xgb(X, y, T, valid, **params):
    """Execute the gradient boosting pipeline"""

    # Create the data matrix
    start_data_t = datetime.now()
    xgb_training = xgb.DMatrix(X, label=y, missing=None, weight=None,
                               silent=False, feature_names=None,
                               feature_types=None)
    end_data_t = datetime.now() - start_data_t

    xgb_testing = xgb.DMatrix(T, label=valid, missing=None, weight=None,
                              silent=False, feature_names=None,
                              feature_types=None)

    n_est = params.pop('n_estimators')
    start_fit_t = datetime.now()
    bst = xgb.train(params, xgb_training, n_est)
    end_fit_t = datetime.now() - start_fit_t

    pred = bst.predict(xgb_testing)
    pred[np.nonzero(pred >= 0.5)] = 1
    pred[np.nonzero(pred < 0.5)] = 0

    score = np.mean(pred == valid)

    return score, end_data_t, end_fit_t


if __name__ == '__main__':
    USAGE = """usage: python %s dataset path_results n_try

    where:
        - dataset is one of {random}
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

    N_SAMPLES = np.array([10e2, 10e3, 10e4], dtype=int)
    N_FEATURES = np.array([1, 5, 10], dtype=int)

    # Setup the parameters
    params = {}
    params['n_estimators'] = N_ESTIMATORS
    params['booster'] = ['gbtree']
    params['nthread'] = [N_THREADS]
    params['eta'] = [LEARNING_RATE]
    params['gamma'] = [MIN_IMPURITY_SPLIT]
    params['max_depth'] = MAX_DEPTH
    params['min_child_weight'] = [MIN_SAMPLES_LEAF]
    params['max_delta_step'] = [0]
    params['subsample'] = [SUBSAMPLES]
    params['colsample_bytree'] = [1.]
    params['colsample_bylevel'] = [1.]
    params['alpha'] = [0.]
    params['delta'] = [0.]
    params['tree_method'] = ['exact']
    params['scale_pos_weight'] = [1.]
    params['objective'] = ['binary:logistic']
    params['seed'] = [RND_SEED]
    params['verbose_eval'] = [False]
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
        array_data = [misc.generate_samples(ns, nf, RND_SEED)
                      for ns in N_SAMPLES for nf in N_FEATURES]
    else:
        raise ValueError('The dataset is not known. The possible choices are:'
                         ' random')

    # Save only the time for the moment
    res_xgb = [[data[0].shape, p, misc.bench(bench_xgb, data, **p)]
               for p in params_list for data in array_data]

    # Check that the path is existing
    if not os.path.exists(store_dir):
        os.makedirs(store_dir)

    filename = 'xgboost_' + dataset + '.pk'
    store_filename = os.path.join(store_dir, filename)

    joblib.dump(res_xgb, store_filename)
