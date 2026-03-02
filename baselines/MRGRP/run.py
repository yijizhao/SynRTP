# -*- coding: utf-8 -*-
import os
import time
import torch
import signal
import argparse
import pickle
from multiprocessing import Process
import multiprocessing
from utils.functions import seed_everything, set_cpu_num, AutoGPU, display_results
from dotmap import DotMap
import warnings

warnings.filterwarnings("ignore")

def subworker(args, flags, T, is_print=False, random_seed=729, mode='train'):

    time.sleep(0.1)
    seed_everything(random_seed)
    set_cpu_num(1)
    gpu_id = 1  
    torch.cuda.set_device(gpu_id)
    current_device = torch.cuda.current_device()
    print(f"Current GPU ID: {current_device}")


    if args.model == 'mrgrp':
        from methods import mrgrp_train as model_train
        from methods import mrgrp_test as model_test 
        
    else:
        raise ValueError(f"Wrong model name {args.model}")
    
    file_name = "xl_"+ str(args.batch_size) + "_" + args.dataset_name  + ".txt"
    model_path = './logs/'+ args.model +'/'+args.dataset_name+ f'/seed{random_seed}/checkpoints/'+ T +'_ckpt.pt'

    if mode == 'train':
        model_train(args, flags, T, model_path, file_name, is_print, random_seed)
    elif mode == 'test':
        model_test(args, flags, T, model_path, file_name, is_print, random_seed)
    else:
        raise ValueError(f"Wrong mode {mode}")

def experiment(args, T, mode='train'):

    print(f'Start {mode}ing...')

    flags_path = './MRGRP/flags.pkl'
    flags = pickle.load(open(flags_path, 'rb'))
    flags = DotMap(flags)

    workers = []
    gpu_controller = AutoGPU(args.memory_size, args)

    for random_seed in range(1, args.seed_num+1):

        if args.parallel:
            is_print = True if (len(workers)==0 and mode=='train') else False
        else:
            subworker(args, flags, T, True, random_seed, mode)

    while args.parallel and any([sub.exitcode==None for sub in workers]):
        pass

def dict_merge(dict_list = []):
    dict_ =  {}
    for dic in dict_list:
        assert isinstance(dic, dict), "object is not a dict!"
        dict_ = {**dict_, **dic}
    return dict_


if __name__ == '__main__':

    multiprocessing.set_start_method('spawn')

    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_name', type=str, default='yt_dataset', help='dataset name')  
    parser.add_argument('--model', type=str, default='mrgrp', help='model name')
    parser.add_argument('--device', type=str, default='cuda', help='device')
    parser.add_argument('--memory_size', type=int, default=9000, help='memory size')
    parser.add_argument('--seed_num', type=int, default=1, help='seed num')
    parser.add_argument('--log_dir', type=str, default='./logs', help='log dir')
    parser.add_argument('--parallel', action='store_true', help='parallel')
    parser.add_argument('--force', action='store_true', help='force')
    parser.add_argument('--cpu_num', type=int, default=1, help='cpu num')
    parser.add_argument('--batch_size', type=int, default=8, help='256 batch size')
    parser.add_argument('--epochs', type=int, default=100, help='epochs')
    parser.add_argument('--train_use_ratio', type=float, default=1.0, help='use train ratio')
    parser.add_argument('--val_use_ratio', type=float, default=1.0, help='use val ratio')
    parser.add_argument('--test_use_ratio', type=float, default=1.0, help='use test ratio')
    parser.add_argument('--dataset_type', type=str, default='')
    parser.add_argument('--test_only', action='store_true', help='test only')

    parser.add_argument('--ab_gcn', action='store_true', help='ab_gcn')
    parser.add_argument('--no_de_edge', action='store_true', help='no_de_edge')
    parser.add_argument('--no_pi_edge', action='store_true', help='no_pi_edge')
    parser.add_argument('--no_dist_edge', action='store_true', help='no_dist_edge')
    parser.add_argument('--no_direct_edge', action='store_true', help='no_direct_edge')
    parser.add_argument('--ab_distance', action='store_true', help='ab_distance')
    parser.add_argument('--ab_etr', action='store_true', help='ab_etr')

    args, _ = parser.parse_known_args()
    params = vars(args)

    datasets = [args.dataset_name]
    parser.add_argument("--train_path", type=str, default=None)
    parser.add_argument("--val_path", type=str, default=None)
    parser.add_argument("--test_path", type=str, default=None)
    # the name of datasets
    args_lst = []
    for batch_size in [1024]:
        for dataset in datasets:
            basic_params = dict_merge([params, {'batch_size':batch_size, 'dataset_name': dataset}])
            args_lst.append(basic_params)    

    for p in args_lst:
        args = argparse.Namespace(**p)

        T = time.strftime("%mM%dD%Hh%Mm", time.localtime())

        if not args.test_only:
            experiment(args, T, 'train')

        experiment(args, T, 'test')
