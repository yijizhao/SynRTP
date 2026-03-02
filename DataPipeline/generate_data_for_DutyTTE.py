import numpy as np
import os
import math
import torch
import argparse
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import time
import pprint
import hashlib

import hashlib 

def generate_segment_idx_in_range(node_id1, node_id2, max_idx=200001):
   
    num_available_indices = max_idx - 2
    if num_available_indices <= 0:
        raise ValueError("max_idx must be greater than 2.")
    sorted_node_ids = sorted([int(node_id1), int(node_id2)])
    
    edge_str = f"{sorted_node_ids[0]}_{sorted_node_ids[1]}"
    hash_object = hashlib.sha1(edge_str.encode('utf-8'))
    full_hash_int = int(hash_object.hexdigest(), 16)
    generated_idx = full_hash_int % num_available_indices
    
    return generated_idx + 2

os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MKL_THREADING_LAYER']  = 'GNU'
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'
np.random.seed(2021)
torch.manual_seed(2021)
torch.cuda.manual_seed(2021)


def get_workspace():
    cur_path = os.path.abspath(__file__)
    file = os.path.dirname(cur_path)
    file = os.path.dirname(file)
    return file
ws =  get_workspace()

def dir_check(path):
    """
    check weather the dir of the given path exists, if not, then create it
    """
    import os

    dir = path if os.path.isdir(path) else os.path.split(path)[0]
    if not os.path.exists(dir):
        os.makedirs(dir)
    return path

def get_common_params():
    parser = argparse.ArgumentParser(description='Entry Point of the code')
    parser.add_argument('--dataset', default='yt_dataset', type=str, help='logistics service of picking up')
    parser.add_argument("--train_path", type=str, default=None)
    parser.add_argument("--val_path", type=str, default=None)
    parser.add_argument("--test_path", type=str, default=None)
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="input batch size for training (default: 512)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="number of data loading workers (default: 4)",
    )
    parser.add_argument('--max_task_num',  type=int, default=25, help = 'maxmal number of task') 
    parser.add_argument("--max_shortest_path_len", type=int, default=25, help='maximum shortest path length') 
    return parser


def get_params():
    parser = get_common_params()
    args, _ = parser.parse_known_args()
    if args.train_path is None:
        args.train_path = os.path.join(
            ws, f"datasets/lade/{args.dataset}/train_mini.npy"
        )
    if args.val_path is None:
        args.val_path = os.path.join(ws, f"datasets/lade/{args.dataset}/val_mini.npy")
    if args.test_path is None:
        args.test_path = os.path.join(ws, f"datasets/lade/{args.dataset}/test_mini.npy")

    return args


def collate_fn(batch):
    return batch


def process_and_save_dataset(data_obj, dataset_name, N_nodes, max_shortest_path_len, output_dir):
    print(f"\n--- Starting processing for {dataset_name} dataset ---")
    

    start_time_total = time.time()

    V_np = data_obj['V']
    V_len_np = data_obj['V_len']
    V_reach_mask_np = data_obj['V_reach_mask']
    E_np = data_obj['E_static_fea'] 
    A_np = data_obj['A']
    start_fea_np = data_obj['start_fea']
    start_idx_np = data_obj['start_idx']
    cou_fea_np = data_obj['cou_fea']
    route_label_np = data_obj['route_label']
    label_len_np = data_obj['label_len']
    time_label_np = data_obj['time_label']

    Total_B_original, T, N, _ = V_np.shape
    total_samples_to_process = Total_B_original * T

    processed_train_xs = []
    processed_segment_travel_time_mean = []
    processed_total_ts = []
    processed_segment_travel_time = []
    processed_segment_num = []
    processed_ts_10min = []
    processed_od = []


    print(f"\n--- Starting processing for {dataset_name} dataset ({total_samples_to_process} samples) ---")

    for b_idx in tqdm(range(Total_B_original), desc=f"Processing {dataset_name} samples"):
        for t_idx in range(T):
            current_label_len = label_len_np[b_idx, t_idx]

            if current_label_len == 0:
                continue

            current_V = V_np[b_idx, t_idx]
            current_start_fea = start_fea_np[b_idx, t_idx]
            current_start_idx = start_idx_np[b_idx, t_idx]
            current_route_label = route_label_np[b_idx, t_idx]
            current_time_label = time_label_np[b_idx, t_idx]

            valid_node_indices = current_route_label[:int(current_label_len)]
            valid_node_ids = [abs(int(current_V[idx, 0])) for idx in valid_node_indices]
            
            start_node_id = abs(int(current_start_fea[0]))
            current_train_xs = []
            if valid_node_ids:
                current_train_xs.append(generate_segment_idx_in_range(start_node_id, valid_node_ids[0]))
            
            for i in range(len(valid_node_ids) - 1):
                current_train_xs.append(generate_segment_idx_in_range(valid_node_ids[i], valid_node_ids[i+1]))
            
            train_xs_padded = current_train_xs + [0] * (51 - len(current_train_xs))
            processed_train_xs.append(train_xs_padded)

            if valid_node_ids:
                processed_start_node_id = abs(start_node_id)
                processed_end_node_id = abs(valid_node_ids[-1])
                current_od = [processed_start_node_id, processed_end_node_id]
            else:
                current_od = [0, 0] 
            processed_od.append(current_od)

            start_timestamp_min = abs(current_start_fea[3])
            ts_10min = float(round(start_timestamp_min / 10.0)) 
            processed_ts_10min.append([ts_10min])
            segment_num = abs(int(current_label_len))
            processed_segment_num.append([segment_num])

            total_ts = abs(int(current_time_label[int(current_label_len) - 1]))
            processed_total_ts.append([total_ts])

            current_segment_travel_time_mean = []
            if current_label_len > 0:
                current_segment_travel_time_mean.append(abs(current_time_label[0]))
                for i in range(1, int(current_label_len)):
                    segment_time = current_time_label[i] - current_time_label[i-1]
                    current_segment_travel_time_mean.append(abs(segment_time))
            segment_travel_time_mean_padded = current_segment_travel_time_mean + [-1] * (51 - len(current_segment_travel_time_mean))
            processed_segment_travel_time_mean.append(segment_travel_time_mean_padded)

            current_segment_travel_time = []
            for mean_time in segment_travel_time_mean_padded:
                if mean_time != -1:
                    current_segment_travel_time.append([mean_time, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, mean_time])
                else:
                    current_segment_travel_time.append([-1.0] * 11)
            processed_segment_travel_time.append(current_segment_travel_time)

    print(f"\n--- Merging all processed samples for {dataset_name} dataset ---")
    final_dataset = {}
    final_dataset[f'{dataset_name}_xs'] = processed_train_xs
    final_dataset[f'segment_travel_time_mean_{dataset_name}'] = processed_segment_travel_time_mean
    final_dataset[f'total_ts_{dataset_name}'] = processed_total_ts
    final_dataset[f'segment_travel_time_{dataset_name}'] = processed_segment_travel_time
    final_dataset[f'segment_num_{dataset_name}'] = processed_segment_num
    final_dataset[f'ts_10min_{dataset_name}'] = processed_ts_10min
    final_dataset[f'od_{dataset_name}'] = processed_od



    output_filename = os.path.join(output_dir, f"{dataset_name}.pkl")
    output_file_dir = os.path.dirname(output_filename) 
    dir_check(output_file_dir) 





    print(f"Saving processed {dataset_name} dataset to: {output_filename}")
    import pickle
    with open(output_filename, 'wb') as f:
        pickle.dump(final_dataset, f)

    end_time_total = time.time()
    print(f"Processing and saving {dataset_name} completed in {end_time_total - start_time_total:.2f} seconds.") 


if __name__ == "__main__":
    params = vars(get_params())
    pprint.pprint(params) 
    device = torch.device('cpu') 
    params['device'] = device 

    N_NODES = params['max_task_num'] 
    MAX_SPD_LEN = params['max_shortest_path_len'] 

    output_base_dir = os.path.join(ws, f'datasets/dutytte/{params["dataset"]}')
    dir_check(output_base_dir) 

    print("\n--- Loading datasets ---")
    train_data_path = params['train_path']
    val_data_path = params['val_path']
    test_data_path = params['test_path']

    train_obj = np.load(train_data_path, allow_pickle=True).item()
    val_obj = np.load(val_data_path, allow_pickle=True).item()
    test_obj = np.load(test_data_path, allow_pickle=True).item()

    print(f'Train data keys: {train_obj.keys()}')
    print(f'Val data keys: {val_obj.keys()}')
    print(f'Test data keys: {test_obj.keys()}')

    process_and_save_dataset(train_obj, 'train', N_NODES, MAX_SPD_LEN, output_base_dir)
    process_and_save_dataset(val_obj, 'val', N_NODES, MAX_SPD_LEN, output_base_dir)
    process_and_save_dataset(test_obj, 'test', N_NODES, MAX_SPD_LEN, output_base_dir)

    print("\nAll datasets processed and saved. You can now update your DATASET class to load these new files.")
