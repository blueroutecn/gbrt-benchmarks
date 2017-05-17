# first line: 22
@memory.cache
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

    score_training = np.mean(clf.predict(X) == y)

    score_testing = np.mean(clf.predict(T) == valid)

    return score_training, score_testing, end_data_t, end_fit_t
