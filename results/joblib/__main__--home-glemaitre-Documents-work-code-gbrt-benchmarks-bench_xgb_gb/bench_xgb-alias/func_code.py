# first line: 18
@memory.cache
def bench_xgb(X, y, T, valid, **params):
    """Execute the gradient boosting pipeline"""

    # Create the data matrix
    start_data_t = datetime.now()
    xgb_training = xgb.DMatrix(
        X,
        label=y,
        missing=None,
        weight=None,
        silent=False,
        feature_names=None,
        feature_types=None)
    end_data_t = datetime.now() - start_data_t

    xgb_testing = xgb.DMatrix(
        T,
        label=valid,
        missing=None,
        weight=None,
        silent=False,
        feature_names=None,
        feature_types=None)

    n_est = params.pop('n_estimators')
    start_fit_t = datetime.now()
    bst = xgb.train(params, xgb_training, n_est)
    end_fit_t = datetime.now() - start_fit_t

    pred = bst.predict(xgb_training)
    pred[np.nonzero(pred >= 0.5)] = 1
    pred[np.nonzero(pred < 0.5)] = 0
    score_training = np.mean(pred == y)

    pred = bst.predict(xgb_testing)
    pred[np.nonzero(pred >= 0.5)] = 1
    pred[np.nonzero(pred < 0.5)] = 0
    score_testing = np.mean(pred == valid)

    return score_training, score_testing, end_data_t, end_fit_t
