# -*- coding: utf-8 -*-
import sys
import torch

from pprint import pprint
from utils.util import get_common_params, dict_merge , seed_everything
def run(params):
    seed_everything(params['seed'])
    pprint(params)
    model = params['model']
    if model == 'ranketpa_route':
        from algorithm.rankepta.train_route import main
        main(params)
    if model == 'ranketpa_time':
        from algorithm.rankepta.train import main
        main(params)

def get_params():
    parser = get_common_params()
    args, _ = parser.parse_known_args()
    return args

if __name__ == "__main__":
    params = vars(get_params())
    datasets = [params["dataset"]]  # the name of datasets
    target_models = [params["model"]]

    args_lst = []
    params['cuda_id'] = 0
    params['is_test'] = False
    params['inference'] = False
    for (
        model
    ) in target_models:  # ['ranketpa_route', 'ranketpa_time']
        if model in ['ranketpa_route', 'ranketpa_time']:
            for hs in [64]: 
                for dataset in datasets:
                    deeproute_params = {'model': model, 'hidden_size': hs, 'dataset': dataset}
                    deeproute_params = dict_merge([params, deeproute_params])
                    args_lst.append(deeproute_params)
        print(model)

    for p in args_lst:
        run(p)
