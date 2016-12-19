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

### `xgboost`

We fixed the following parameters to be similar of `scikit-learn`.

### `LightGBM`

We fixed the following parameters to be similar of `scikit-learn`.

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
