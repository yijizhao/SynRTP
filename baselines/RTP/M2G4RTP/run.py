# -*- coding: utf-8 -*-
import os
from pprint import pprint
from utils.util import get_common_params, dict_merge, seed_everything

def run(params):
    seed_everything(params['seed'])
    pprint(params)
    model = params['model']
    if model == 'm2g4rtp_pickup':
        from algorithm.m2g4rtp_pickup.train import main
        main(params)
def get_params():
    parser = get_common_params()
    args, _ = parser.parse_known_args()
    return args


if __name__ == "__main__":
    params = vars(get_params())
    params['cuda_id'] = 1
    params['is_test'] = False
    datasets = [params["dataset"]]  # the name of datasets
    target_models = [params['model']]

    args_lst = []
    for model in target_models:
        if model in ['m2g4rtp_pickup']:
            for hs in [64]: # 32, 64
                for dataset in datasets:
                    deeproute_params = {'model': model, 'hidden_size': hs, 'dataset': dataset, 'num_epoch': 100, 'batch_size': 8}
                    deeproute_params = dict_merge([params, deeproute_params])
                    args_lst.append(deeproute_params)

    for p in args_lst:
        run(p)
