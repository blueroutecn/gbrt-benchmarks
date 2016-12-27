"""Higgs bosons dataset
This is a classification problem to distinguish between a signal process which
produces Higgs bosons and a background process which does not.
The dataset page is available from UCI Machine Learning Repository
    https://archive.ics.uci.edu/ml/datasets/HIGGS
Courtesy of Daniel Whiteson Assistant Professor, Physics & Astronomy,
Univ. of California Irvine
"""

# Author: Guillaume Lemaitre
# License: BSD 3 clause

from gzip import GzipFile
from io import BytesIO
# import logging
from os.path import exists, join
try:
    from urllib2 import urlopen
except ImportError:
    from urllib.request import urlopen

import numpy as np

from sklearn.datasets import get_data_home
from sklearn.datasets.base import Bunch, _pkl_filepath
from sklearn.utils.fixes import makedirs
from sklearn.externals import joblib
from sklearn.utils import check_random_state

URL = ('https://archive.ics.uci.edu/ml/'
       'machine-learning-databases/00280/HIGGS.csv.gz')

# logger = logging.getLogger()


def fetch_higgs(data_home=None,
                download_if_missing=True,
                random_state=None,
                shuffle=False):
    """Load the Higgs dataset, downloading it if necessary.

    Read more in the :ref:`User Guide <datasets>`.

    Parameters
    ----------
    data_home : string, optional
        Specify another download and cache folder for the datasets. By default
        all scikit learn data is stored in '~/scikit_learn_data' subfolders.

    download_if_missing : boolean, default=True
        If False, raise a IOError if the data is not locally available
        instead of trying to download the data from the source site.

    random_state : int, RandomState instance or None, optional (default=None)
        Random state for shuffling the dataset.
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    shuffle : bool, default=False
        Whether to shuffle dataset.

    Returns
    -------
    dataset : dict-like object with the following attributes:

    dataset.data : numpy array of shape (11000000, 28)
        Each row corresponds to the 28 features in the dataset.

    dataset.target : numpy array of shape (11000000,)
        The targets are binary: 0 (background) - 1 (signal).

    dataset.DESCR : string
        Description of the Higgs dataset.
    """

    data_home = get_data_home(data_home=data_home)
    higgs_dir = join(data_home, "higgs")
    samples_path = _pkl_filepath(higgs_dir, "samples")
    targets_path = _pkl_filepath(higgs_dir, "targets")
    available = exists(samples_path)

    if download_if_missing and not available:
        makedirs(higgs_dir, exist_ok=True)
        # logger.warning("Downloading %s" % URL)
        print("Downloading {}".format(URL))
        f = BytesIO(urlopen(URL).read())
        Xy = np.genfromtxt(GzipFile(fileobj=f), delimiter=',')

        X = Xy[:, 1:-1]
        y = Xy[:, 0].astype(np.int32)

        joblib.dump(X, samples_path, compress=9)
        joblib.dump(y, targets_path, compress=9)
        print("Dumped the data")

    try:
        X, y
    except NameError:
        X = joblib.load(samples_path)
        y = joblib.load(targets_path)
        print("Data loaded from pickle")

    if shuffle:
        ind = np.arange(X.shape[0])
        rng = check_random_state(random_state)
        rng.shuffle(ind)
        X = X[ind]
        y = y[ind]

    return Bunch(data=X, target=y, DESCR=__doc__)
