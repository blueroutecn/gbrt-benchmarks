from __future__ import print_function
import sys
import yaml
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import ParameterGrid
sys.path.insert(0, '../datasets')
from misc import load_higgs


configuration_path = "../params_benchmark/parameters_higgs.conf"
config_name = 'lightgbm'
with open(configuration_path, 'r') as stream:
    params = yaml.load(stream)[config_name]

params = {key: (value if isinstance(value, list) else [value])
          for key, value in params.items()}
params_grid = list(ParameterGrid(params))
params_selected = [d for d in params_grid if d['max_depth'] == 8][0]

N_SAMPLES = 1e7
data = load_higgs(random_state=42, n_samples=int(N_SAMPLES))

# Extract the parameter required for the dataset
max_bin = params_selected.pop('max_bin')

lgbm_training = lgb.Dataset(data[0], label=data[1], max_bin=max_bin)
n_est = params_selected.pop('n_estimators')
# Create the number of leafs depending of the max depth
params_selected['num_leaves'] = np.power(2, params_selected['max_depth'] - 1)
# Do not limit the depth of the trees
params_selected['max_depth'] = -1

gbm = lgb.train(params_selected, lgbm_training, num_boost_round=n_est)
