# -*- coding: utf-8 -*-
import numpy as np
import os
import math
import torch
import argparse
from tqdm import tqdm
import pickle
import time
import pprint
import itertools 

os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MKL_THREADING_LAYER']  = 'GNU'


MAX_PATH_LENGTH = 25
GRAPH_SIZE_WITH_SPECIAL_NODES = MAX_PATH_LENGTH + 2

def get_workspace():
    cur_path = os.path.abspath(__file__)
    file = os.path.dirname(cur_path)
    file = os.path.dirname(file)
    return file

def dir_check(path):
    import os
    os.makedirs(path, exist_ok=True)

def get_params():
    parser = argparse.ArgumentParser(description='Data Preprocessing for New Model Structure')
    parser.add_argument('--dataset', default='yt_dataset', type=str, help='Dataset name for constructing folder paths')
    parser.add_argument("--train_path", type=str, default=None)
    parser.add_argument("--val_path", type=str, default=None)
    parser.add_argument("--test_path", type=str, default=None)
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help=" 64,This parameter is not used in preprocessing but kept for compatibility",
    )
    parser.add_argument('--workers', type=int, default=4, help='This parameter is not used in preprocessing but kept for compatibility')
    parser.add_argument('--max_task_num', type=int, default=25, help='Maximal number of tasks in raw data (N_nodes)') 

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

def calculate_bearing(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])
    dLon = lon2 - lon1
    y = math.sin(dLon) * math.cos(lat2)
    x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dLon)
    bearing = math.degrees(math.atan2(y, x))
    return int(round(bearing))


def process_and_save_dataset(data_obj, dataset_name, output_dir):
    print(f"\n--- Starting processing for {dataset_name} dataset ---")
    start_time_total = time.time()

    V_np = data_obj['V']
    V_pt_np = data_obj['V_pt']
    E_static_fea_np = data_obj['E_static_fea']
    start_fea_np = data_obj['start_fea']
    route_label_np = data_obj['route_label']
    time_label_np = data_obj['time_label']
    label_len_np = data_obj['label_len']

    Total_B, T, _, _ = V_np.shape
    print(f"Original data shape: (B={Total_B}, T={T})")

    all_processed_samples = []
    for b_idx in tqdm(range(Total_B), desc=f"Processing {dataset_name} samples"):
        current_E_static_fea = E_static_fea_np[b_idx]

        for t_idx in range(T):
            current_label_len = int(label_len_np[b_idx, t_idx])

            if current_label_len == 0:
                continue
            
            current_V = V_np[b_idx, t_idx]  
            current_V_pt = V_pt_np[b_idx, t_idx] 
            current_start_fea = start_fea_np[b_idx, t_idx] 
            original_route_indices = route_label_np[b_idx, t_idx][:current_label_len].astype(int) 
            original_time_labels = time_label_np[b_idx, t_idx][:current_label_len] #

            ID_list = []  
            max_node_idx_in_V = current_V.shape[0] 
            
            for node_idx in range(max_node_idx_in_V):
                if not np.all(current_V[node_idx] == 0):
                    ID_list.append(node_idx)


            if not ID_list:
                continue

            wb_c_features = np.zeros((MAX_PATH_LENGTH, 5), dtype=np.float32)
            pickup_c_features = np.zeros((MAX_PATH_LENGTH, 5), dtype=np.float32)
            pickup_n_features = np.zeros((MAX_PATH_LENGTH, 19), dtype=np.float32)
            deliver_n_features = np.zeros((MAX_PATH_LENGTH, 12), dtype=np.float32)
            da_n_features = np.zeros((MAX_PATH_LENGTH, 5), dtype=np.float32)
            tempt1 = np.zeros(MAX_PATH_LENGTH, dtype=np.float32)
            wb_time_features = np.zeros((MAX_PATH_LENGTH, 3), dtype=np.float32)
            wb_info = np.zeros((MAX_PATH_LENGTH, 2), dtype=np.float32)
            line_dist_mat = np.zeros((GRAPH_SIZE_WITH_SPECIAL_NODES, GRAPH_SIZE_WITH_SPECIAL_NODES), dtype=np.float32)
            angle_mat = np.zeros((GRAPH_SIZE_WITH_SPECIAL_NODES, GRAPH_SIZE_WITH_SPECIAL_NODES), dtype=np.float32)
            context_c_features = np.zeros(3, dtype=np.float32)
            context_n_features = np.zeros(5, dtype=np.float32)
            tempt2 = np.zeros(3, dtype=np.float32)
            tempt3 = np.zeros(1, dtype=np.float32)

            labels = np.zeros((MAX_PATH_LENGTH, 5), dtype=np.float32)
            route_labels = labels[:, 0]  
            etr_labels = labels[:, 1]   
            for step, node_index in enumerate(original_route_indices):
                if node_index < MAX_PATH_LENGTH:
                    route_labels[node_index] = step + 2
                    
                    etr_labels[node_index] = original_time_labels[step]


            labels[:current_label_len, 2] = original_route_indices 
            labels[:current_label_len, 3] = original_time_labels 
            

            for i, node_id in enumerate(ID_list):
                wb_c_features[node_id] = [1.0, 1.0, 4.0, 1.0, 1.0]
                pickup_c_features[node_id] = [1.0, 1.0, 1.0, 1.0, 1.0]
                wb_info[node_id] = [float(node_id), 1.0] 
                node_lon, node_lat = current_V[node_id, 1], current_V[node_id, 2]
                deliver_n_features[node_id, :2] = [node_lon, node_lat]
                promise_time = current_V_pt[node_id]
                start_time = current_start_fea[4]
                wb_time_features[node_id] = [0.0, promise_time, start_time]


            coord_map = {}
            courier_lon, courier_lat = current_start_fea[1], current_start_fea[2]
            coord_map[-1] = (courier_lon, courier_lat) 
            for node_id in ID_list:
                coord_map[node_id] = (current_V[node_id, 1], current_V[node_id, 2]) 

            for node_id in ID_list:
                mat_idx = node_id + 2 
                
                dist = current_V[node_id, 4] 
                line_dist_mat[0, mat_idx] = line_dist_mat[mat_idx, 0] = dist 
                line_dist_mat[1, mat_idx] = line_dist_mat[mat_idx, 1] = dist 

                (lon1, lat1) = coord_map[-1] 
                (lon2, lat2) = coord_map[node_id] 
                angle_mat[0, mat_idx] = calculate_bearing(lon1, lat1, lon2, lat2)
                angle_mat[mat_idx, 0] = calculate_bearing(lon2, lat2, lon1, lat1)
                angle_mat[1, mat_idx] = angle_mat[0, mat_idx] 
                angle_mat[mat_idx, 1] = angle_mat[mat_idx, 0]

            for node_id_i, node_id_j in itertools.combinations(ID_list, 2):
                mat_idx_i = node_id_i + 2
                mat_idx_j = node_id_j + 2

                dist = current_E_static_fea[node_id_i, node_id_j, 1] 
                line_dist_mat[mat_idx_i, mat_idx_j] = line_dist_mat[mat_idx_j, mat_idx_i] = dist 


                (lon1, lat1) = coord_map[node_id_i] 
                (lon2, lat2) = coord_map[node_id_j] 
                angle_mat[mat_idx_i, mat_idx_j] = calculate_bearing(lon1, lat1, lon2, lat2)
                angle_mat[mat_idx_j, mat_idx_i] = calculate_bearing(lon2, lat2, lon1, lat1)

            line_dist_mat[0, 1] = line_dist_mat[1, 0] = 0
            angle_mat[0, 1] = angle_mat[1, 0] = 0

            geohash_dist_mat = np.copy(line_dist_mat)
            context_c_features[0] = 1.0
            context_c_features[1] = math.floor(current_start_fea[4] / 15.0)
            context_c_features[2] = 1.0
            tempt3[0] = 5.0
            
            feature_vector = np.concatenate([
                wb_c_features.flatten(),
                pickup_c_features.flatten(),
                pickup_n_features.flatten(),
                deliver_n_features.flatten(),
                da_n_features.flatten(),
                tempt1.flatten(),
                wb_time_features.flatten(),
                wb_info.flatten(),
                labels.flatten(),
                tempt2.flatten(),
                line_dist_mat.flatten(),
                geohash_dist_mat.flatten(),
                angle_mat.flatten(),
                context_c_features.flatten(),
                context_n_features.flatten(),
                tempt3.flatten()
            ]).astype(np.float32)

            all_processed_samples.append(feature_vector)

    if not all_processed_samples:
        print(f"Warning: No valid samples found for {dataset_name}. An empty file will be created.")
        final_dataset = np.array([], dtype=np.float32)
    else:
        final_dataset = np.stack(all_processed_samples)

    output_filename = os.path.join(output_dir, f"{dataset_name}.pkl")
    dir_check(os.path.dirname(output_filename))
    
    print(f"Saving {final_dataset.shape[0]} processed samples for {dataset_name} to: {output_filename}")
    print(f"Final data shape: {final_dataset.shape}")

    with open(output_filename, 'wb') as f:
        pickle.dump(final_dataset, f)

    end_time_total = time.time()
    print(f"Processing and saving {dataset_name} completed in {end_time_total - start_time_total:.2f} seconds.")


if __name__ == "__main__":
    params = vars(get_params())
    pprint.pprint(params)
    
    ws = get_workspace()
    output_base_dir = os.path.join(ws, f'datasets/mrgrp/{params["dataset"]}')
    dir_check(output_base_dir)

    print("\n--- Loading raw datasets ---")
    train_obj = np.load(params['train_path'], allow_pickle=True).item()
    val_obj = np.load(params['val_path'], allow_pickle=True).item()
    test_obj = np.load(params['test_path'], allow_pickle=True).item()

    print(f"Train data keys: {list(train_obj.keys())}")

    process_and_save_dataset(train_obj, 'train', output_base_dir)
    process_and_save_dataset(val_obj, 'val', output_base_dir)
    process_and_save_dataset(test_obj, 'test', output_base_dir)

    print("\nAll datasets have been successfully processed and saved in the new format.")
