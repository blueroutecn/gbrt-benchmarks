# first line: 19
@memory.cache
def bench_lgbm(X, y, T, valid, **params):
    """Execute the gradient boosting pipeline"""

    # Extract the parameter required for the dataset
    max_bin = params.pop('max_bin')

    # Measure the time to prepare the data
    start_data_t = datetime.now()
    # Prepare the data
    lgbm_training = lgb.Dataset(X, label=y, max_bin=max_bin)
    end_data_t = datetime.now() - start_data_t
    # lgbm_testing = lgb.Dataset(T, label=valid, max_bin=max_bin)

    # Pop the number of trees
    n_est = params.pop('n_estimators')
    # Create the number of leafs depending of the max depth
    params['num_leaves'] = np.power(2, params['max_depth'] - 1)
    # Do not limit the depth of the trees
    params['max_depth'] = -1

    # Set the fitting time
    start_fit_t = datetime.now()
    gbm = lgb.train(params, lgbm_training, num_boost_round=n_est)
    end_fit_t = datetime.now() - start_fit_t

    # Predict on the training
    pred = gbm.predict(X)
    pred[np.nonzero(pred >= 0.5)] = 1
    pred[np.nonzero(pred < 0.5)] = 0
    score_training = np.mean(pred == y)

    pred = gbm.predict(T)
    pred[np.nonzero(pred >= 0.5)] = 1
    pred[np.nonzero(pred < 0.5)] = 0
    score_testing = np.mean(pred == valid)

    return score_training, score_testing, end_data_t, end_fit_t
