import importlib
import tensorflow as tf
from common.yaml_ops import load_yaml
assert tf.__version__[0] == '2'
# algorithms based on TF 2.0
algos = {
    'td3':      {'class': 'TD3',    'policy': 'off-policy', 'update': 'perStep'},
    'sac':      {'class': 'SAC',    'policy': 'off-policy', 'update': 'perStep'},
}


def get_model_info(name: str):
    '''
    Args:
        name: name of algorithms
    Return:
        class of the algorithm model named `name`.
        defaulf config of specified algorithm.
        mode of policy, `on-policy` or `off-policy`
    '''
    if name not in algos.keys():
        raise NotImplementedError
    else:
        class_name = algos[name]['class']
        policy_mode = algos[name]['policy']
        model_file = importlib.import_module('Algorithms.tf2algos.' + name)
        model = getattr(model_file, class_name)
        algo_config = load_yaml(f'Algorithms/config.yaml')[name]
        return model, algo_config, policy_mode
