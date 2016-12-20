# Gradient boosting benchmark

The benchmark tests:

* `scikit-learn` implementation
* `xgboost` implementation
* `LightGBM` implementation

## Install

We used conda environment. The toolbox were installed as followed:

### `scikit-learn`

```
git clone https://github.com/scikit-learn/scikit-learn.git
cd scikit-learn
python setup.py install
```

### `xgboost`

```
git clone https://github.com/dmlc/xgboost.git
cd xgboost
git submodule init
git submodule update
make
cd python-package
python setup.py install
```

### `LightGBM`

```
git clone https://github.com/Microsoft/LightGBM.git
cd LightGBM
mkdir build
cd build
cmake ../
make
cd ../python-package
python setup.py install
```

## Parameters

### `scikit-learn`

We used the following list of parameters:

| Parameters                   | Value            |
|------------------------------|------------------|
| `'learning_rate'`            | `0.1`            |
| `'loss'`                     | `'deviance'`     |
| `'min_weight_fraction_leaf'` | `0.`             |
| `'subsample'`                | `1.`             |
| `'max_features'`             | `None`           |
| `'min_samples_split'`        | `2`              |
| `'min_samples_leaf'`         | `1`              |
| `'min_impurity_split`'       | `1`              |
| `'max_leaf_nodes'`           | `None`           |
| `'presort'`                  | `'auto'`         |
| `'init'`                     | `None`           |
| `'warm_start'`               | `False`          |
| `'verbose'`                  | `0`              |
| `'random_state'`             | `42`             |
| `'criterion'`                | `'friedman_mse'` |

### `xgboost`

We fixed the following parameters to be similar of `scikit-learn`.

| Parameters                   | Value               |
|------------------------------|---------------------|
| `'booster'`                  | `'gbtree'`          |
| `'eta'`                      | `0.1`               |
| `'objective'`                | `'binary:logistic'` |
| `'subsample'`                | `1.`                |
| `'colsample_bytree'`         | `1.`                |
| `'colsample_bylevel'`        | `1.`                |
| `'min_child_weight'`         | `1`                 |
| `'gamma`'                    | `1`                 |
| `'max_delta_step'`           | `0`                 |
| `'alpha'`                    | `0.`                |
| `'delta'`                    | `0.`                |
| `'tree_method'`              | `'exact'`           |
| `'scale_pos_weight'`         | `1.`                |
| `'presort'`                  | `'auto'`            |
| `'init'`                     | `None`              |
| `'verbose_eval'`             | `False`             |
| `'random_state'`             | `42`                |

### `LightGBM`

We fixed the following parameters to be similar of `scikit-learn`.

| Parameters                   | Value              |
|------------------------------|--------------------|
| `'boosting'`                 | `'gbdt'`           |
| `'learning_rate'`            | `0.1`              |
| `'application'`              | `'binary'`         |
| `'metric'`                   | `'binary_logloss'` |
| `'tree_learner'`             | `'serial`'         |
| `'feature_fraction'`         | `1.`               |
| `'bagging_fraction'`         | `1.`               |
| `'bagging_freq'`             | `0`                |
| `'max_bin'`                  | `255`              |
| `'is_sparse'`                | `False`            |
| `'min_gain_to_split`'        | `1`                |
| `'verbose'`                  | `1`                |
| `'feature_fraction_seed'`    | `42`               |
| `'bagging_seed'`             | `42`               |
| `'data_random_seed'`         | `42`               |

A useful list of alias between parameters is available in [`config.h`](https://github.com/Microsoft/LightGBM/blob/master/include/LightGBM/config.h#L316).

## Dataset

We used the following datasets to benchmark the libraries:

* Randomly generated dataset
* Real dataset

### Randomly generated dataset

The following parameters were used to build the dataset using a grid:

* `n_samples`: 1k, 10k, 100k
* `n_features`: 1, 5, 10

#### Parameters GBRT

The following parameters were used to build create the classifier using a grid:

* `max_depth`: 1, 3, 5, 8
* `n_estimators`: 1, 10

#### Results

In the `results` folder, the results of the benchmark have been dumped using `joblib`.

### Real dataset


## Check of tree structure

The file [`check_trees.py`](https://github.com/glemaitre/gbrt-benchmarks/blob/master/check_trees.py)
is intended to check the structure of a tree created within the gradient
boosting algorithm.

The parameters are fixed in the python file. The resulting structures are:

* [`sklearn` tree structure](https://github.com/glemaitre/gbrt-benchmarks/blob/master/results/sklearn_tree.png)
* [`xgboost` tree structure](https://github.com/glemaitre/gbrt-benchmarks/blob/master/results/xgboost_tree.pdf)
* `LighGBM` tree structure
