import numpy as np
import os
import torch
import argparse
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from algorithm.strategy_model_1 import SynRTPDataset as DATASET
from utils.utils import dir_check
import networkx as nx
import time

os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"
os.environ["MKL_THREADING_LAYER"] = "GNU"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
np.random.seed(2021)
torch.manual_seed(2021)
torch.cuda.manual_seed(2021)


def get_workspace():
    cur_path = os.path.abspath(__file__)
    file = os.path.dirname(cur_path)
    file = os.path.dirname(file)
    return file
ws = get_workspace()


def get_common_params():
    parser = argparse.ArgumentParser(description="Entry Point of the code")
    parser.add_argument(
        "--dataset", default="yt_dataset", type=str, help="the name of dataset"
    )
    parser.add_argument("--train_path", type=str, default=None)
    parser.add_argument("--val_path", type=str, default=None)
    parser.add_argument("--test_path", type=str, default=None)    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--workers", type=int, default=2, help="number of data loading workers"
    )
    parser.add_argument(
        "--max_task_num", type=int, default=25, help="maxmal number of task"
    )
    parser.add_argument(
        "--max_shortest_path_len",
        type=int,
        default=25,
        help="maximum shortest path length",
    )
    return parser


def get_params():
    parser = get_common_params()
    args, _ = parser.parse_known_args()
    
    if args.train_path is None:
        args.train_path = os.path.join(ws, f'datasets/lade/{args.dataset}/train_mini.npy')
    if args.val_path is None:
        args.val_path = os.path.join(ws, f'datasets/lade/{args.dataset}/val_mini.npy')
    if args.test_path is None:
        args.test_path = os.path.join(
            ws, f'datasets/lade/{args.dataset}/test_mini.npy'
        )
    return args


def collate_fn(batch):
    return batch


def batch2input(batch, device):
    (
        V,
        V_len,
        V_reach_mask,
        E_static_fea,
        E_mask,
        A,
        start_fea,
        start_idx,
        cou_fea,
        route_label,
        label_len,
        time_label,
    ) = zip(*batch)
    V = torch.FloatTensor(np.array(V)).to(device)
    B, T, N, _ = V.shape
    E_mask = np.array(E_mask)
    A = np.array(A)
    E_static_fea = np.array(E_static_fea)
    E = torch.zeros([B, T, N, N, 4]).to(device)  # Edge feature, E: (B, T, N, N, d_e)
    for t in range(T):
        E_t = torch.Tensor(E_static_fea).to(device)  # E_t: (B, N, N, 4)
        E_mask_t = torch.Tensor(E_mask[:, t, :, :]).to(device)  # (B, N, N)
        E_t = E_t * E_mask_t.unsqueeze(-1).expand(E_t.shape)
        E[:, t, :, :, :] = E_t

    V_reach_mask = torch.BoolTensor(np.array(V_reach_mask)).to(device)
    route_label = torch.LongTensor(np.array(route_label)).to(device)
    time_label = torch.LongTensor(np.array(time_label)).to(device)
    label_len = torch.LongTensor(np.array(label_len)).to(device)
    start_fea = torch.FloatTensor(np.array(start_fea)).to(device)
    start_idx = torch.LongTensor(np.array(start_idx)).to(device)
    cou_fea = torch.LongTensor(np.array(cou_fea)).to(device)
    A = torch.FloatTensor(np.array(A)).to(device)
    V_len = torch.LongTensor(np.array(V_len)).to(device)

    return (
        V,
        V_reach_mask,
        E,
        start_fea,
        start_idx,
        cou_fea,
        route_label,
        label_len,
        A,
        V_len,
        time_label,
    )


def calculate_graphormer_features(A_matrix, E_features, N_nodes, max_shortest_path_len):
    node_in_degree = np.zeros(N_nodes, dtype=np.int32)
    node_out_degree = np.zeros(N_nodes, dtype=np.int32)
    shortest_path_distances = np.full(
        (N_nodes, N_nodes), max_shortest_path_len + 1, dtype=np.int32
    )
    shortest_path_edge_features_aggregated = np.zeros(
        (N_nodes, N_nodes, E_features.shape[2]), dtype=np.float32
    )

    G = nx.DiGraph()
    G.add_nodes_from(range(N_nodes))

    adj_mask = A_matrix == 1.0
    node_out_degree = np.sum(adj_mask, axis=1).astype(np.int32)
    node_in_degree = np.sum(adj_mask, axis=0).astype(np.int32)

    rows, cols = np.where(A_matrix == 1.0)
    for r, c in zip(rows, cols):
        G.add_edge(r, c, feature=E_features[r, c, :])

    for i in range(N_nodes):
        shortest_path_distances[i, i] = 0
        queue = [(i, 0, [i])]
        visited = {i}
        target_shortest_paths = {i: (0, [])}

        head = 0
        while head < len(queue):
            curr_node, dist, path_nodes = queue[head]
            head += 1

            for neighbor in G.neighbors(curr_node):
                if neighbor not in visited:
                    visited.add(neighbor)
                    new_dist = dist + 1
                    new_path_nodes = path_nodes + [neighbor]

                    if (
                        neighbor not in target_shortest_paths
                        or new_dist < target_shortest_paths[neighbor][0]
                    ):
                        current_path_edge_features_list = []
                        if len(new_path_nodes) > 1:
                            for k_path in range(len(new_path_nodes) - 1):
                                src_node = new_path_nodes[k_path]
                                dest_node = new_path_nodes[k_path + 1]
                                if "feature" in G[src_node][dest_node]:
                                    current_path_edge_features_list.append(
                                        G[src_node][dest_node]["feature"]
                                    )
                                else:
                                    print(
                                        f"Warning: Edge ({src_node}, {dest_node}) missing 'feature' in NetworkX graph."
                                    )
                                    current_path_edge_features_list.append(
                                        np.zeros(E_features.shape[2], dtype=np.float32)
                                    )

                        target_shortest_paths[neighbor] = (
                            new_dist,
                            current_path_edge_features_list,
                        )
                        queue.append((neighbor, new_dist, new_path_nodes))

        for target_node, (dist, edge_feat_list) in target_shortest_paths.items():
            if dist <= max_shortest_path_len:
                shortest_path_distances[i, target_node] = dist
                if edge_feat_list:
                    shortest_path_edge_features_aggregated[i, target_node, :] = np.mean(
                        edge_feat_list, axis=0
                    )
                else:
                    shortest_path_edge_features_aggregated[i, target_node, :] = (
                        np.zeros(E_features.shape[2], dtype=np.float32)
                    )

    return (
        node_in_degree,
        node_out_degree,
        shortest_path_distances,
        shortest_path_edge_features_aggregated,
    )


def prepare_graphormer_input_features(
    A_batch_np, E_features_batch_np, N_nodes, max_shortest_path_len
):
    B, T, _, _, d_e = E_features_batch_np.shape
    B_flat = B * T

    A_flat_np = A_batch_np.reshape(B_flat, N_nodes, N_nodes)
    E_features_flat_np = E_features_batch_np.reshape(B_flat, N_nodes, N_nodes, d_e)

    all_node_in_degree_flat = np.empty((B_flat, N_nodes), dtype=np.int32)
    all_node_out_degree_flat = np.empty((B_flat, N_nodes), dtype=np.int32)
    all_shortest_path_distances_flat = np.empty(
        (B_flat, N_nodes, N_nodes), dtype=np.int32
    )
    all_shortest_path_edge_features_aggregated_flat = np.empty(
        (B_flat, N_nodes, N_nodes, d_e), dtype=np.float32
    )

    for idx in range(B_flat):
        in_d, out_d, spd, sp_edge_agg = calculate_graphormer_features(
            A_flat_np[idx], E_features_flat_np[idx], N_nodes, max_shortest_path_len
        )
        all_node_in_degree_flat[idx] = in_d
        all_node_out_degree_flat[idx] = out_d
        all_shortest_path_distances_flat[idx] = spd
        all_shortest_path_edge_features_aggregated_flat[idx] = sp_edge_agg

    node_in_degree = all_node_in_degree_flat.reshape(B, T, N_nodes)
    node_out_degree = all_node_out_degree_flat.reshape(B, T, N_nodes)
    shortest_path_distances = all_shortest_path_distances_flat.reshape(
        B, T, N_nodes, N_nodes
    )
    shortest_path_edge_features_aggregated = (
        all_shortest_path_edge_features_aggregated_flat.reshape(
            B, T, N_nodes, N_nodes, d_e
        )
    )

    return (
        node_in_degree,
        node_out_degree,
        shortest_path_distances,
        shortest_path_edge_features_aggregated,
    )


def process_and_save_dataset(
    dataloader, dataset_name, N_nodes, max_shortest_path_len, output_dir
):
    print(f"\n--- Starting processing for {dataset_name} dataset ---")

    collected_V = []
    collected_V_len = []
    collected_V_reach_mask = []
    collected_E_static_fea = []
    collected_A = []
    collected_start_fea = []
    collected_start_idx = []
    collected_cou_fea = []
    collected_route_label = []
    collected_label_len = []
    collected_time_label = []

    collected_node_in_degree = []
    collected_node_out_degree = []
    collected_shortest_path_distances = []
    collected_shortest_path_edge_features_aggregated = []

    total_batches = len(dataloader)
    start_time_total = time.time()

    for i, batch in enumerate(
        tqdm(dataloader, total=total_batches, desc=f"Processing {dataset_name}")
    ):
        (
            V,
            V_reach_mask,
            E,
            start_fea,
            start_idx,
            cou_fea,
            route_label,
            label_len,
            A,
            V_len,
            time_label,
        ) = batch2input(batch, "cpu")

        V_np = V.cpu().numpy()
        V_len_np = V_len.cpu().numpy()
        V_reach_mask_np = V_reach_mask.cpu().numpy()
        E_np = E.cpu().numpy()
        A_np = A.cpu().numpy()
        start_fea_np = start_fea.cpu().numpy()
        start_idx_np = start_idx.cpu().numpy()
        cou_fea_np = cou_fea.cpu().numpy()
        route_label_np = route_label.cpu().numpy()
        label_len_np = label_len.cpu().numpy()
        time_label_np = time_label.cpu().numpy()

        start_time_batch_prep = time.time()
        (
            batch_node_in_degree,
            batch_node_out_degree,
            batch_shortest_path_distances,
            batch_shortest_path_edge_features_aggregated,
        ) = prepare_graphormer_input_features(
            A_np, E_np, N_nodes, max_shortest_path_len
        )
        end_time_batch_prep = time.time()

        if (i + 1) % 10 == 0 or (i + 1) == total_batches:
            print(
                f"  Batch {i+1}/{total_batches} Graphormer feature prep time: {end_time_batch_prep - start_time_batch_prep:.4f} seconds"
            )

        collected_V.append(V_np)
        collected_V_len.append(V_len_np)
        collected_V_reach_mask.append(V_reach_mask_np)
        collected_E_static_fea.append(E_np)
        collected_A.append(A_np)
        collected_start_fea.append(start_fea_np)
        collected_start_idx.append(start_idx_np)
        collected_cou_fea.append(cou_fea_np)
        collected_route_label.append(route_label_np)
        collected_label_len.append(label_len_np)
        collected_time_label.append(time_label_np)
        collected_node_in_degree.append(batch_node_in_degree)
        collected_node_out_degree.append(batch_node_out_degree)
        collected_shortest_path_distances.append(batch_shortest_path_distances)
        collected_shortest_path_edge_features_aggregated.append(
            batch_shortest_path_edge_features_aggregated
        )

    print(f"\n--- Merging all batches for {dataset_name} dataset ---")
    final_dataset = {}

    if collected_V:
        final_dataset["V"] = np.concatenate(
            collected_V, axis=0
        )  # (Total_B_original, T, N, d_v)
        final_dataset["V_len"] = np.concatenate(
            collected_V_len, axis=0
        )  # (Total_B_original, T)
        final_dataset["V_reach_mask"] = np.concatenate(
            collected_V_reach_mask, axis=0
        )  # (Total_B_original, T, N)
        final_dataset["E_static_fea"] = np.concatenate(
            collected_E_static_fea, axis=0
        )  # (Total_B_original, T, N, N, d_e)
        final_dataset["A"] = np.concatenate(
            collected_A, axis=0
        )  # (Total_B_original, T, N, N)
        final_dataset["start_fea"] = np.concatenate(
            collected_start_fea, axis=0
        )  # (Total_B_original, T, d_s)
        final_dataset["start_idx"] = np.concatenate(
            collected_start_idx, axis=0
        )  # (Total_B_original, T)
        final_dataset["cou_fea"] = np.concatenate(
            collected_cou_fea, axis=0
        )  # (Total_B_original, T, d_w_orig)
        final_dataset["route_label"] = np.concatenate(
            collected_route_label, axis=0
        )  # (Total_B_original, T, N)
        final_dataset["label_len"] = np.concatenate(
            collected_label_len, axis=0
        )  # (Total_B_original, T)
        final_dataset["time_label"] = np.concatenate(
            collected_time_label, axis=0
        )  # (Total_B_original, T)

        final_dataset["node_in_degree"] = np.concatenate(
            collected_node_in_degree, axis=0
        )  # (Total_B_original, T, N)
        final_dataset["node_out_degree"] = np.concatenate(
            collected_node_out_degree, axis=0
        )  # (Total_B_original, T, N)
        final_dataset["shortest_path_distances"] = np.concatenate(
            collected_shortest_path_distances, axis=0
        )  # (Total_B_original, T, N, N)
        final_dataset["shortest_path_edge_features_aggregated"] = np.concatenate(
            collected_shortest_path_edge_features_aggregated, axis=0
        )  # (Total_B_original, T, N, N, d_e)
    else:
        print(f"Warning: No data collected for {dataset_name}. Skipping save.")
        return

    output_filename = os.path.join(output_dir, f"{dataset_name}.npy")
    output_file_dir = os.path.dirname(output_filename)
    dir_check(output_file_dir)

    print(f"Saving processed {dataset_name} dataset to: {output_filename}")
    np.save(output_filename, final_dataset, allow_pickle=True)
    end_time_total = time.time()
    print(
        f"Processing and saving {dataset_name} completed in {end_time_total - start_time_total:.2f} seconds."
    )


if __name__ == "__main__":
    params = vars(get_params())
    device = torch.device("cpu")
    params["device"] = device

    train_dataset = DATASET(mode="train", params=params)
    train_loader = DataLoader(
        train_dataset,
        batch_size=params["batch_size"],
        shuffle=False,
        collate_fn=collate_fn,
    )

    val_dataset = DATASET(mode="val", params=params)
    val_loader = DataLoader(
        val_dataset,
        batch_size=params["batch_size"],
        shuffle=False,
        collate_fn=collate_fn,
    )

    test_dataset = DATASET(mode="test", params=params)
    test_loader = DataLoader(
        test_dataset,
        batch_size=params["batch_size"],
        shuffle=False,
        collate_fn=collate_fn,
    )

    print(
        f"Number of Train Samples:{len(train_dataset)} | Number of val Samples:{len(val_dataset)} | Number of test Samples:{len(test_dataset)}"
    )
    N_NODES = params["max_task_num"]
    MAX_SPD_LEN = params["max_shortest_path_len"]

    output_base_dir = os.path.join(ws, f'datasets/synrtp/{params["dataset"]}')
    dir_check(output_base_dir)

    process_and_save_dataset(test_loader, "test", N_NODES, MAX_SPD_LEN, output_base_dir)
    process_and_save_dataset(val_loader, "val", N_NODES, MAX_SPD_LEN, output_base_dir)
    process_and_save_dataset(
        train_loader, "train", N_NODES, MAX_SPD_LEN, output_base_dir
    )

    print(
        "\nAll datasets processed and saved. You can now update your DATASET class to load these new files."
    )
