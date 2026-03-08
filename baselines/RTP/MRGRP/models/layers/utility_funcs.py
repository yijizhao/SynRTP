import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Union
from typing import Optional,List,Tuple


def normalize_adjacency_matrix(adj):
    B, N, _ = adj.shape
    degree_matrix = adj.sum(dim=-1)  
    degree_matrix_inv_sqrt = degree_matrix.pow(-0.5)
    degree_matrix_inv_sqrt[degree_matrix_inv_sqrt == float('inf')] = 0 
    D_inv_sqrt = torch.diag_embed(degree_matrix_inv_sqrt)
    
    adj_normalized = D_inv_sqrt.matmul(adj).matmul(D_inv_sqrt)
    
    return adj_normalized

def get_adj_data(point_dist_mat, time_features, wb_info_array, args):
    batch_size = point_dist_mat.shape[0]
    graph_size = 25
    device = point_dist_mat.device

    all_E_de = torch.empty(batch_size, graph_size, graph_size, device=device)
    all_E_pi = torch.empty(batch_size, graph_size, graph_size, device=device)
    all_A_direct = torch.empty(batch_size, graph_size, graph_size, device=device)
    batch_size_int = int(batch_size)
    for b in range(batch_size_int):
        pickup_remaining_n = time_features[b, :, 2]
        arrive_remaining_n = time_features[b, :, 1]
        point_ids = wb_info_array[b, :, 0]
        point_types = wb_info_array[b, :, 1]

        E_de = torch.zeros(graph_size, graph_size, device=device)
        E_pi = torch.zeros(graph_size, graph_size, device=device)
        A_direct = torch.zeros(graph_size, graph_size, device=device)

        pick_task_idxs = (point_types == -1).nonzero().t()[0]
        deliver_task_idxs = (point_types == 1).nonzero().t()[0]

        pick_time_diff = torch.abs(pickup_remaining_n[pick_task_idxs].unsqueeze(1) - pickup_remaining_n[pick_task_idxs]) / 60
        i, j = torch.meshgrid(pick_task_idxs, pick_task_idxs, indexing='ij')
        E_pi.index_put_((i.flatten(), j.flatten()), pick_time_diff.flatten())


        deliver_time_diff = torch.abs(arrive_remaining_n[deliver_task_idxs].unsqueeze(1) - arrive_remaining_n[deliver_task_idxs]) / 60
        i, j = torch.meshgrid(deliver_task_idxs, deliver_task_idxs, indexing='ij')
        E_de.index_put_((i.flatten(), j.flatten()), deliver_time_diff.flatten())

        for unique_id in torch.unique(point_ids):
            if unique_id == 0:
                continue
            idxs = (point_ids == unique_id).nonzero().t()[0]
            if len(idxs) == 2:
                A_direct[idxs[0], idxs[1]] = 1 if point_types[idxs[0]] == -1 else 0
                A_direct[idxs[1], idxs[0]] = 1 if point_types[idxs[1]] == -1 else 0
            elif len(idxs) > 2:
                raise RuntimeError('More than two tasks with the same id.')

        all_E_de[b] = E_de
        all_E_pi[b] = E_pi
        all_A_direct[b] = A_direct

    if args.no_de_edge:
        all_E_de = torch.zeros_like(all_E_de, device=device)
    if args.no_pi_edge:
        all_E_pi = torch.zeros_like(all_E_pi, device=device)
    if args.no_direct_edge:
        all_A_direct = torch.zeros_like(all_A_direct, device=device)

    return point_dist_mat, all_E_de, all_E_pi, all_A_direct

def get_adjacency(E_dist, E_de, E_pi, point_masks, args):

    batch_size, graph_size = E_dist.shape[0], E_dist.shape[1]
    graph_size_2 = E_de.shape[1]
    A_dist = torch.zeros([batch_size, graph_size, graph_size]).to(E_dist.device)
    node_masks = ~point_masks
    edge_masks = node_masks.unsqueeze(1) * node_masks.unsqueeze(-1)
    B_indices = torch.arange(batch_size, device=E_dist.device)[:, None]
    N_indices = torch.arange(graph_size, device=E_dist.device)
    min_dist_idx = torch.argsort(E_dist + 1e9 * (~edge_masks), dim=-1)[:, :, 0]
    min_dist_idx_2 = torch.argsort(E_dist + 1e9 * (~edge_masks), dim=-1)[:, :, 1]
    min_dist_idx_3 = torch.argsort(E_dist + 1e9 * (~edge_masks), dim=-1)[:, :, 2]
    A_dist[B_indices, N_indices, min_dist_idx] = -1
    A_dist[B_indices, N_indices, min_dist_idx_2] = 1
    A_dist[B_indices, min_dist_idx, N_indices] = -1
    A_dist[B_indices, min_dist_idx_2, N_indices] = 1
    A_dist[B_indices, N_indices, min_dist_idx_3] = 1
    A_dist[B_indices, min_dist_idx_3, N_indices] = 1
    A_dist = A_dist * edge_masks
    
    A_de = (E_de != 0).float()
    A_de = A_de - torch.eye(graph_size_2, device=E_de.device)
    A_pi = (E_pi != 0).float()
    A_pi = A_pi - torch.eye(graph_size_2, device=E_pi.device)
    if graph_size_2 == graph_size:
        A_de = A_de * edge_masks
        A_pi = A_pi * edge_masks
    else:
        A_de = A_de * edge_masks[:, 2:, 2:]
        A_pi = A_pi * edge_masks[:, 2:, 2:]

    if args.no_de_edge:
        A_de = torch.zeros_like(A_de, device=A_de.device)
    if args.no_pi_edge:
        A_pi = torch.zeros_like(A_pi, device=A_pi.device)
    if args.no_dist_edge:
        A_dist = torch.zeros_like(A_dist, device=A_dist.device)
    
    return A_dist, A_de, A_pi

def get_edge_features(point_dist_mat, time_features, wb_info_array, point_masks, args):
    E_dist, E_de, E_pi, A_direct = get_adj_data(point_dist_mat, time_features, wb_info_array, args)
    A_dist, A_de, A_pi = get_adjacency(E_dist, E_de, E_pi, point_masks, args)
    E_de = F.pad(E_de, (2, 0, 2, 0), mode='constant', value=0)
    E_pi = F.pad(E_pi, (2, 0, 2, 0), mode='constant', value=0)
    A_direct = F.pad(A_direct, (2, 0, 2, 0), mode='constant', value=0)
    A_de = F.pad(A_de, (2, 0, 2, 0), mode='constant', value=0)
    A_pi = F.pad(A_pi, (2, 0, 2, 0), mode='constant', value=0)
    rel_edge = torch.stack([E_dist, A_dist,E_de, A_de,  E_pi, A_pi, A_direct], dim=-1)
    return rel_edge

def get_eigh(A,pos_enc_dim):
    N = A.size(1)
    
    A = normalize_adjacency_matrix(A)
    L = torch.eye(N, device=A.device) - A

    _, EigVec = torch.linalg.eigh(L)
    pos_enc = EigVec[..., 1:pos_enc_dim + 1]
    return pos_enc 

def pad_roll(inputs, input_dim: Optional[int] = None, shift: int = 1):
    shift = torch.tensor([shift]).item()
    shift = int(shift)
    if input_dim is None:
        input_dim = len(inputs.shape)
    else:
        input_dim = input_dim
    pad_shape = [shift, 0, 0, 0] if shift > 0 else [0, -shift, 0, 0]

    pad_inputs = F.pad(inputs, pad_shape)
    
    if shift > 0:
        return pad_inputs[:, :-shift] if input_dim == 2 else pad_inputs[:, :, :-shift]
    else:
        return pad_inputs[:, -shift:] if input_dim == 2 else pad_inputs[:, :, -shift:]

def pad_roll_3(inputs, input_dim: Optional[int] = None, shift: int = 1):
    if input_dim is None:
        input_dim = len(inputs.shape)
    else:
        input_dim = input_dim
    shift = torch.tensor([shift]).item()
    shift = int(shift)
    # pad_shape = (shift, 0, 0, 0, 0, 0) if shift > 0 else (0, -shift, 0, 0, 0, 0)
    pad_shape = [shift, 0, 0, 0, 0, 0] if shift > 0 else [0, -shift, 0, 0, 0, 0]

    pad_inputs = F.pad(inputs, pad_shape)

    
    if shift > 0:
        return pad_inputs[:, :-shift] if input_dim == 2 else pad_inputs[:, :, :-shift]
    else:
        return pad_inputs[:, -shift:] if input_dim == 2 else pad_inputs[:, :, -shift:]

def get_time_adj(time_features, wb_info_array):
    batch_size = time_features.shape[0]
    graph_size = 25
    device = time_features.device

    time_matrix = torch.zeros(batch_size, graph_size, device=device)
    point_types = wb_info_array[:, :, 1]
    time_matrix[point_types==-1] = time_features[point_types == -1][:, 2]
    time_matrix[point_types==1] = time_features[point_types == 1][:, 1]

    return time_matrix