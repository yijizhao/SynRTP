# -*- coding: utf-8 -*-
import os
from pprint import pprint
from utils.util import get_common_params, dict_merge

def run(params):
    pprint(params)
    model = params['model']
    # for route prediction task
    if model in ['Distance-Greedy', 'Time-Greedy']:
        from algorithm.basic.basic_model import main
        main(params)
    if model ==  'deeproute':
        from algorithm.deeproute.train import main
        main(params)
    if model == 'osqure':
        from algorithm.osqure.train import main
        main(params)
    if model == 'graph2route':
        from algorithm.graph2route.train import main
        main(params)
    if model == 'm2g4rtp_pickup':
        from algorithm.m2g4rtp_pickup.train import main
        main(params)
    if model == 'drl4route':
        from algorithm.drl4route.train import main
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
    # for model in ['Distance-Greedy', 'Time-Greedy',  'osqure', 'deeproute', 'graph2route' , 'drl4route', 'm2g4rtp_pickup']:
    for model in target_models:
        if model in ['Distance-Greedy', 'Time-Greedy']:
            for dataset in datasets:
                basic_params = dict_merge([params, {'model': model,'dataset': dataset}])
                args_lst.append(basic_params)

        if model in ['osqure']:
            for dataset in datasets:
                osqure_params = {'model': model, 'dataset': dataset}
                osqure_params = dict_merge([params, osqure_params])
                args_lst.append(osqure_params)

        if model in ['drl4route']:
            for hs in [32]: # [64, 32]
                for rl_r in [0.4]: # [ 0.4, 0.6]
                    for dataset in datasets:
                        dl_params = {'model': model, 'hidden_size': hs, 'dataset': dataset,  'rl_ratio':rl_r}
                        dl_params = dict_merge([params, dl_params])
                        args_lst.append(dl_params)

        if model in ['deeproute', 'm2g4rtp_pickup']:
            for hs in [64]: # 32, 64
                for dataset in datasets:
                    if dataset == 'pickup_sh_dataset':          
                        deeproute_params = {'model': model, 'hidden_size': hs, 'dataset': dataset, 'num_epoch': 100}
                    else:
                        deeproute_params = {'model': model, 'hidden_size': hs, 'dataset': dataset, 'num_epoch': 100, 'batch_size': 64}
                    deeproute_params = dict_merge([params, deeproute_params])
                    args_lst.append(deeproute_params)

        if model in ['graph2route']:
            for hs in [64]: # [32, 64]
                for gcn_num_layers in [2]: # [2, 3]
                    for dataset in datasets:
                        for knn in ['n']: # 'n-1', 'n'
                            graph2route_params = {'model': model, 'hidden_size': hs, 'gcn_num_layers': gcn_num_layers,
                                                  'worker_emb_dim': 10, 'dataset': dataset, 'k_nearest_neighbors': knn}
                            graph2route_params = dict_merge([params, graph2route_params])
                            args_lst.append(graph2route_params)

    for p in args_lst:
        run(p)
