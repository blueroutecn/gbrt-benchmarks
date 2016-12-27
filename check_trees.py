from __future__ import print_function
import sys
import os

import json
import numpy as np
import xgboost as xgb
import lightgbm as lgb

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from sklearn.ensemble import GradientBoostingClassifier
from sklearn import tree

import misc

if __name__ == '__main__':
    USAGE = """usage: python %s dataset path_results

    where:
        - dataset is one of {random}
        - path_results is the location to store the output of the tree
    """

    N_ESTIMATORS = 1
    LEARNING_RATE = 0.1
    MIN_IMPURITY_SPLIT = 1e-7
    MAX_DEPTH = 5
    MIN_SAMPLES_LEAF = 1
    SUBSAMPLES = 1.
    N_THREADS = 1
    RND_SEED = 42

    # Setup the parameters for the gradient boosting in sklearn
    params_sklearn = {}
    params_sklearn['max_depth'] = MAX_DEPTH
    params_sklearn['learning_rate'] = LEARNING_RATE
    params_sklearn['n_estimators'] = N_ESTIMATORS
    params_sklearn['loss'] = 'deviance'
    params_sklearn['min_weight_fraction_leaf'] = 0.
    params_sklearn['subsample'] = SUBSAMPLES
    params_sklearn['max_features'] = None
    params_sklearn['min_samples_split'] = 2
    params_sklearn['min_samples_leaf'] = MIN_SAMPLES_LEAF
    params_sklearn['min_impurity_split'] = MIN_IMPURITY_SPLIT
    params_sklearn['max_leaf_nodes'] = None
    params_sklearn['presort'] = 'auto'
    params_sklearn['init'] = None
    params_sklearn['warm_start'] = False
    params_sklearn['verbose'] = 0
    params_sklearn['random_state'] = RND_SEED
    params_sklearn['criterion'] = 'friedman_mse'

    # Setup the parameters for the gradient boosting in xgboost
    params_xgboost = {}
    params_xgboost['n_estimators'] = N_ESTIMATORS
    params_xgboost['booster'] = 'gbtree'
    params_xgboost['nthread'] = N_THREADS
    params_xgboost['eta'] = LEARNING_RATE
    params_xgboost['gamma'] = MIN_IMPURITY_SPLIT
    params_xgboost['max_depth'] = MAX_DEPTH
    params_xgboost['min_child_weight'] = MIN_SAMPLES_LEAF
    params_xgboost['max_delta_step'] = 0
    params_xgboost['subsample'] = SUBSAMPLES
    params_xgboost['colsample_bytree'] = 1.
    params_xgboost['colsample_bylevel'] = 1.
    params_xgboost['alpha'] = 0.
    params_xgboost['delta'] = 0.
    params_xgboost['tree_method'] = 'exact'
    params_xgboost['scale_pos_weight'] = 1.
    params_xgboost['objective'] = 'binary:logistic'
    params_xgboost['seed'] = RND_SEED
    params_xgboost['verbose_eval'] = False

    # Setup the parameters for the gradient boosting in lightgbm
    params_lgbm = {}
    params_lgbm['n_estimators'] = N_ESTIMATORS
    params_lgbm['application'] = 'binary'
    params_lgbm['boosting'] = 'gbdt'
    params_lgbm['learning_rate'] = LEARNING_RATE
    # params['num_leaves'] = [31]  # We set this depending of the depth
    params_lgbm['tree_learner'] = 'serial'
    params_lgbm['num_threads'] = N_THREADS
    params_lgbm['max_depth'] = MAX_DEPTH + 1
    params_lgbm['min_data_in_leaf'] = MIN_SAMPLES_LEAF
    params_lgbm['feature_fraction'] = 1.0
    params_lgbm['feature_fraction_seed'] = RND_SEED
    params_lgbm['bagging_fraction'] = SUBSAMPLES
    params_lgbm['bagging_freq'] = 0
    params_lgbm['bagging_seed'] = RND_SEED
    params_lgbm['max_bin'] = 255
    params_lgbm['data_random_seed'] = RND_SEED
    params_lgbm['is_sparse'] = False
    params_lgbm['metric'] = 'binary_logloss'
    params_lgbm['min_gain_to_split'] = MIN_IMPURITY_SPLIT
    params_lgbm['verbosity'] = 1
    params_lgbm['num_leaves'] = np.power(2, params_lgbm['max_depth'] - 1)

    N_SAMPLES = 10e4
    N_FEATURES = 1

    print(__doc__ + '\n')
    if not len(sys.argv) == 3:
        print(USAGE % __file__)
        sys.exit(-1)
    else:
        dataset = sys.argv[1]
        store_dir = sys.argv[2]

    # Create several array for the data
    if dataset == 'random':
        X, y, T, valid = misc.generate_samples(N_SAMPLES, N_FEATURES, RND_SEED)
    else:
        raise ValueError('The dataset is not known. The possible choices are:'
                         ' random')

    # Fit the sklearn gradient boosting
    clf = GradientBoostingClassifier()
    clf.set_params(**params_sklearn)
    clf.fit(X, y)

    # Fit the xgboost gradient boosting
    xgb_training = xgb.DMatrix(
        X,
        label=y,
        missing=None,
        weight=None,
        silent=False,
        feature_names=None,
        feature_types=None)
    n_est = params_xgboost.pop('n_estimators')
    bst = xgb.train(params_xgboost, xgb_training, n_est)

    # Fit the LightGBM gradient boosting
    max_bin = params_lgbm.pop('max_bin')
    lgbm_training = lgb.Dataset(X, label=y, max_bin=max_bin)
    n_est = params_lgbm.pop('n_estimators')
    # params_lgbm['num_leaves'] = np.power(2, params_lgbm['max_depth'] - 1)
    params_lgbm['num_leaves'] = 8
    gbm = lgb.train(params_lgbm, lgbm_training, num_boost_round=n_est)

    # Export the trees
    if not os.path.exists(store_dir):
        os.makedirs(store_dir)

    # sklearn
    filename = 'sklearn_tree.dot'
    store_filename = os.path.join(store_dir, filename)
    tree.export_graphviz(clf[0][0], out_file=store_filename)

    # xgboost
    filename = 'xgboost_tree.pdf'
    store_filename = os.path.join(store_dir, filename)
    ax = xgb.plot_tree(bst)
    plt.savefig(store_filename, bbox_inches='tight', dpi=2000)

    # lightgbm
    filename = 'lightgbm_tree.txt'
    store_filename = os.path.join(store_dir, filename)
    gbm.save_model(store_filename)
    model_json = gbm.dump_model()
    filename = 'lightgbm_tree.json'
    store_filename = os.path.join(store_dir, filename)
    with open(store_filename, 'w+') as f:
        json.dump(model_json, f, indent=4)
