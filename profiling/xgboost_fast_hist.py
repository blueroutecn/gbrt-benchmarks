from __future__ import print_function
import sys
import yaml
import xgboost as xgb
from sklearn.model_selection import ParameterGrid
sys.path.insert(0, '../datasets')
from misc import load_higgs


configuration_path = "../params_benchmark/parameters_higgs.conf"
config_name = 'xgboost-fast-hist'
with open(configuration_path, 'r') as stream:
    params = yaml.load(stream)[config_name]

params = {key: (value if isinstance(value, list) else [value])
          for key, value in params.items()}
params_grid = list(ParameterGrid(params))
params_selected = [d for d in params_grid if d['max_depth'] == 8][0]

N_SAMPLES = 1e7
data = load_higgs(random_state=42, n_samples=int(N_SAMPLES))

# Create the data matrix
xgb_training = xgb.DMatrix(
    data[0],
    label=data[1],
    missing=None,
    weight=None,
    silent=False,
    feature_names=None,
    feature_types=None)

n_est = params_selected.pop('n_estimators')

bst = xgb.train(params_selected, xgb_training, n_est)
