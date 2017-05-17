=====
Notes
=====

Install
-------

A python environment have been created with

- python 3.6
- scipy 0.19
- numpy 1.12.1
- ipython 5.3.0
- scikit-learn 0.18.1
- pandas 0.19.2
- matplotlib 2.0
- seaborn 0.7.1

Activate the environment with `source activate python36`.

The following libraries were installed: XGBoost, LightGBM, FastBDT

XGBoost
~~~~~~~

```
$ git fetch origin
$ git rebase origin master
$ git submodule update
$ make clean
$ make -j
$ cd python-package
$ pip install -e .
```

LightGBM
~~~~~~~~

```
$ git fetch origin
$ git rebase origin master
$ git submodule update
$ cd build
$ make clean
$ rm -rf *
$ ccmake .. # in release mode
$ make -j
$ cd ../python-package
$ pip install -e .
```

FastBDT
~~~~~~~

```
$ mkdir build
$ cmake ..
$ build
$ cmake ..
$ make -j
$ cd ..
$ pip install -U .
```

Benchmark
---------

We need to define a common set of parameters to be tried between the different
packages. In the following, we described which parameters have been used.

XGBoost
~~~~~~~


LightGBM
~~~~~~~~

FastBDT
~~~~~~~
