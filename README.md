# Gradient boosting benchmark

The benchmark tests:

* `scikit-learn` implementation
* `xgboost` implementation

### Dataset

The following parameters were used to build the dataset using a grid:

* `n_samples`: 1k, 10k, 100k
* `n_features`: 1, 5, 10

### Parameters GBRT

The following parameters were used to build create the classifier using a grid:

* `max_depth`: 1, 5, 8
* `n_estimators`: 1, 10, 100
