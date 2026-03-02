import torch.nn.functional as F
import torch
import math
import os

class Utils:
    def __init__(self, params):
        # global config holder (shared access via Utils.params / Utils.device)
        Utils.params = params
        Utils.device = params['device']


def whether_stop(metric_lst = [], n=5, mode='maximize'):
    # return True if best value occurred more than `n` steps ago
    if len(metric_lst) < 1: return False

    if mode == 'minimize':
        metric_lst = [-x for x in metric_lst]

    max_v = max(metric_lst)
    first_max_idx = 0
    for idx, v in enumerate(metric_lst):
        if v == max_v:
            first_max_idx = idx
            break

    return first_max_idx < len(metric_lst) - n


class EarlyStop:

    """Simple early-stopping tracker.

    Tracks metrics, best epoch and whether improvement occurred on append.
    """
    def __init__(self, mode="maximize", patience=5):
        self.mode = mode
        self.patience = patience
        self.metric_lst = []
        self.stop_flag = False
        self.best_epoch = -1
        self.is_best_change = False

    def append(self, x):
        self.metric_lst.append(x)
        self.stop_flag = whether_stop(self.metric_lst, self.patience, self.mode)
        current_epoch = len(self.metric_lst) - 1
        if self.best_epoch == -1: 
            self.best_epoch = current_epoch
            self.is_best_change = True
        else:
            current_best_metric = self.metric_lst[self.best_epoch]
            new_metric = self.metric_lst[current_epoch]
            
            is_new_better = False
            if self.mode == "maximize":
                is_new_better = new_metric > current_best_metric
            else: # minimize
                is_new_better = new_metric < current_best_metric
                
            if is_new_better:
                self.best_epoch = current_epoch
                self.is_best_change = True
            else:
                self.is_best_change = False
        return self.is_best_change

    def best_metric(self):
        if len(self.metric_lst) == 0:
            return -1
        else:
            return self.metric_lst[self.best_epoch]


def get_reinforce_samples(selections_rl, selections_greedy,  label_steps, label_len, pad_value, rl_log_probs, pred_len_steps, eta_sample, time_label):
    # filter out batches where label is all pad_value
    valid_mask = label_steps.min(dim=1)[0] != pad_value

    pred = selections_rl[valid_mask]
    pred_greedy = selections_greedy[valid_mask]

    label = label_steps[valid_mask]
    label_len_list = label_len[valid_mask]
    rl_log_probs_list = rl_log_probs[valid_mask]
    pred_lens = pred_len_steps[valid_mask]
    eta_sample = eta_sample[valid_mask]
    time_label = time_label[valid_mask]

    valid_indices = valid_mask.nonzero().squeeze()
    return pred, pred_greedy, label, label_len_list, rl_log_probs_list, pred_lens, valid_indices, eta_sample, time_label



def get_samples(
    selections,
    selections_label,
    selections_label_len,
    outputs,
    eta_sample, 
    time_label,
    pad_value):  
    mask = selections_label.min(dim=1).values != pad_value

    pred = selections[mask]
    label = selections_label[mask]
    label_len = selections_label_len[mask]
    pointers_outputs = outputs[mask]
    eta = torch.stack([eta_sample[i] for i in range(len(mask)) if mask[i]], dim=0)
    time_label_list = time_label[mask]

    return (pred, label, label_len, pointers_outputs, eta, time_label_list)


def batch2input(batch, device):
    # stack batch of tuples into tensors and move to device
    (V, V_reach_mask, E, start_fea, start_idx, cou_fea, route_label, label_len, A, V_len, time_label, node_in_degree_flat, node_out_degree_flat, shortest_path_distances_flat, shortest_path_edge_features_aggregated_flat) = zip(*batch)

    V = torch.stack(V).float().to(device)
    V_len = torch.stack(V_len).long().to(device)
    V_reach_mask = torch.stack(V_reach_mask).bool().to(device)
    E = torch.stack(E).float().to(device)
    A = torch.stack(A).float().to(device)
    start_fea = torch.stack(start_fea).float().to(device)
    start_idx = torch.stack(start_idx).long().to(device)
    cou_fea = torch.stack(cou_fea).long().to(device)
    route_label = torch.stack(route_label).long().to(device)
    label_len = torch.stack(label_len).long().to(device)
    time_label = torch.stack(time_label).long().to(device)

    node_in_degree_flat = torch.stack(node_in_degree_flat).long().to(device)
    node_out_degree_flat = torch.stack(node_out_degree_flat).long().to(device)
    shortest_path_distances_flat = torch.stack(shortest_path_distances_flat).long().to(device)
    shortest_path_edge_features_aggregated_flat = torch.stack(shortest_path_edge_features_aggregated_flat).float().to(device)

    return (V, V_reach_mask, E, start_fea, start_idx, cou_fea, route_label, label_len, A, V_len,  node_in_degree_flat, \
                node_out_degree_flat, shortest_path_distances_flat, shortest_path_edge_features_aggregated_flat, time_label )


def get_nonzeros_nrl(
    pred_steps, label_steps, label_len, pred_len, eta_prediction, time_label, pad_value
):
    # filter out fully padded rows
    mask = label_steps.min(dim=1).values != pad_value

    valid_pred_steps = pred_steps[mask]
    valid_label_steps = label_steps[mask]
    valid_label_len = label_len[mask]
    valid_pred_len = pred_len[mask]
    valid_eta_prediction = eta_prediction[mask]
    valid_time_label = time_label[mask]

    return (
        valid_pred_steps.long(),
        valid_label_steps.long(),
        valid_label_len.long(),
        valid_pred_len.long(),
        valid_eta_prediction.long(),
        valid_time_label.long()
    )


def get_log_prob_mask(pred_len, params):
    batch_size = pred_len.size(0) 
    max_task_num = params['max_task_num'] 
    
    log_prob_mask = torch.arange(max_task_num, device=pred_len.device).expand(batch_size, -1)
    
    log_prob_mask = (log_prob_mask < pred_len.unsqueeze(1)).float() 
    
    return log_prob_mask



def compute_lsd_gdrpo(true_routes, pred_routes, pred_greedy, route_label_len):
    B, G, T = pred_routes.shape
    device = pred_routes.device

    true_expanded = true_routes.unsqueeze(1).expand(-1, G, -1)
    len_expanded = route_label_len.unsqueeze(1).expand(-1, G)

    valid_mask = (true_expanded != Utils.params['pad_value'])
    true_routes_exp = true_routes.unsqueeze(1)
    weights = torch.ones(T, device=device).float().view(1, 1, -1)

    pos_matrix_base = torch.arange(T, device=device).view(1, 1, -1).expand(B, G, -1) 
    true_pos = pos_matrix_base * valid_mask 

    comparison_sample = (true_routes_exp.unsqueeze(3) == pred_routes.unsqueeze(2))
    pred_indices_sample = torch.arange(T, device=device).view(1, 1, 1, -1).expand(B, G, T, -1)
    large_value = T
    masked_indices_sample = torch.where(comparison_sample, pred_indices_sample, large_value)
    pred_pos_lookup_sample = torch.min(masked_indices_sample, dim=3)[0].float() 
    pred_pos_sample = pred_pos_lookup_sample
    found_mask_sample = (pred_pos_sample < T) & valid_mask 
    squared_diff_sample = torch.zeros_like(true_pos, dtype=torch.float) 

    valid_indices_sample = found_mask_sample.nonzero(as_tuple=True)
    if len(valid_indices_sample[0]) > 0: 
        squared_diff_sample[valid_indices_sample] = (true_pos[valid_indices_sample] - pred_pos_sample[valid_indices_sample]).pow(2)
    weighted_sum_sample = (squared_diff_sample * weights * found_mask_sample).sum(dim=2)  
    valid_counts_sample = found_mask_sample.sum(dim=2).clamp(min=1) 
    lsd_errors_sample = (weighted_sum_sample / valid_counts_sample)

    pred_greedy = pred_greedy.unsqueeze(1).expand(-1, G, -1)
    comparison_greedy = (true_routes_exp.unsqueeze(3) == pred_greedy.unsqueeze(2))
    pred_indices_greedy = torch.arange(T, device=device).view(1, 1, 1, -1).expand(B, G, T, -1)
    masked_indices_greedy = torch.where(comparison_greedy, pred_indices_greedy, large_value)
    pred_pos_lookup_greedy = torch.min(masked_indices_greedy, dim=3)[0].float() # Shape [B, G, T]
    pred_pos_greedy = pred_pos_lookup_greedy # Shape [B, G, T]
    found_mask_greedy = (pred_pos_greedy < T) & valid_mask 
    squared_diff_greedy = torch.zeros_like(true_pos, dtype=torch.float) 
    valid_indices_greedy = found_mask_greedy.nonzero(as_tuple=True)
    if len(valid_indices_greedy[0]) > 0: 
        squared_diff_greedy[valid_indices_greedy] = (true_pos[valid_indices_greedy] - pred_pos_greedy[valid_indices_greedy]).pow(2)
    
    weighted_sum_greedy = (squared_diff_greedy * weights * found_mask_greedy).sum(dim=2) 
    valid_counts_greedy = found_mask_greedy.sum(dim=2).clamp(min=1) 
    lsd_errors_greedy = (weighted_sum_greedy / valid_counts_greedy)
    
    # return (sampled-LSD, greedy-LSD) per batch
    return lsd_errors_sample, lsd_errors_greedy


def compute_lsd(true_routes, pred_routes, route_label_len):
    # compute location square deviation per route
    prediction, label, label_len = [tensor2lst(x) for x in [pred_routes, true_routes, route_label_len]]
    prediction, label, label_len = filter_len(prediction, label, label_len)

    pred = []
    for p in prediction:
        input = set([x for x in p if x < 24])
        tmp = list(filter(lambda pi: pi in input, p))
        pred.append(tmp)

    batch_size = len(pred)
    lsd_route = torch.zeros(batch_size, device=Utils.device)

    for i in range(batch_size):
        pred_list = pred[i]
        label_list = label[i][:label_len[i]]

        n = len(label_list)
        if n == 0:
            continue

        idx_1 = [idx for idx, x in enumerate(label_list)]
        idx_2 = [pred_list.index(x) for x in label_list]

        idx_diff = [math.fabs(i - j) for i, j in zip(idx_1, idx_2)]
        idx_diff = torch.square(torch.tensor(idx_diff, device=Utils.device))
        weights = torch.ones(n, device=Utils.device)

        lsd_route[i] = torch.sum(idx_diff * weights) / n

    return lsd_route


def dict_merge(dict_list = []):
    dict_ =  {}
    for dic in dict_list:
        assert isinstance(dic, dict), "object is not a dict!"
        dict_ = {**dict_, **dic}
    return dict_

def tensor2lst(x):
    try:
        return x.cpu().numpy().tolist()
    except:
        return x


def filter_len(prediction, label, label_len):
    # only keep sequences within allowed length range
    len_range = [1, Utils.params['max_task_num']]
    pred_f = []
    label_f = []
    label_len_f = []
    for i in range(len(label_len)):
        if len_range[0] <= label_len[i] <= len_range[1]:
            pred_f.append(prediction[i])
            label_f.append(label[i])
            label_len_f.append(label_len[i])
    return pred_f, label_f, label_len_f



def dir_check(path):
    import os
    dir = path if os.path.isdir(path) else os.path.split(path)[0]
    if not os.path.exists(dir):
        os.makedirs(dir)
    return path



def save2file_meta(params, file_name, head):
    def timestamp2str(stamp):
        utc_t = int(stamp)
        utc_h = utc_t // 3600
        utc_m = (utc_t // 60) - utc_h * 60
        utc_s = utc_t % 60
        hour = (utc_h + 8) % 24
        t = f"{hour}:{utc_m}:{utc_s}"
        return t

    import csv, time, os

    dir_check(file_name)
    write_header = not os.path.exists(file_name)
    with open(file_name, "a", newline="") as file:
        csv_file = csv.writer(file)
        if write_header:
            csv_file.writerow(head)

        params["log_time"] = timestamp2str(time.time())
        data = [str(params.get(k, "")) for k in head]
        csv_file.writerow(data)


def write_list_list(fp, list_, model="a", sep=","):
    os.makedirs(os.path.dirname(fp), exist_ok=True)
    with open(fp, mode=model, encoding="utf-8") as f:
        for line in list_:
            a_line = sep.join(str(l) for l in line)
            f.write(f"{a_line}\n")


from multiprocessing import Pool
def multi_thread_work(parameter_queue, function_name, thread_number=5):
    pool = Pool(thread_number)
    result = pool.map(function_name, parameter_queue)
    pool.close()
    pool.join()
    return result


def kl_divergence_gdrpo(old_log_probs, new_log_probs):
    # symmetric KL per sample between given log-prob tensors
    new_probs = torch.exp(new_log_probs)
    new_probs = new_probs / (new_probs.sum(dim=1, keepdim=True) + 1e-10)
    old_probs = torch.exp(old_log_probs)
    old_probs = old_probs / (old_probs.sum(dim=1, keepdim=True) + 1e-10)

    kl_elements = old_probs * (torch.log(old_probs + 1e-10) - torch.log(new_probs + 1e-10))

    return kl_elements.mean(dim=-1).mean()

import torch.nn as nn
class GDRPOLoss(nn.Module):
    """PPO-style clipped policy loss (no value term).

    Returns scalar policy loss.
    """
    def __init__(self, clip_eps=0.2, kl_coef=0.05):
        super().__init__()
        self.clip_eps = clip_eps
        self.kl_coef = kl_coef

    def forward(self, new_log_probs, old_log_probs, advantages):
        ratio = torch.exp(new_log_probs - old_log_probs)

        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1-self.clip_eps, 1+self.clip_eps) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        total_loss = policy_loss

        return total_loss



def eta_mae_loss_calc(time_label, label_len, eta):
    # compute MAE between predicted and true per-leg durations (from cumulative times)
    N = eta.shape[1]
    B = time_label.shape[0]
    time_label = time_label.reshape(B, N)
    eta = eta.reshape(B, N)
    label_len = label_len.reshape(B, 1)
    pred_durations_flat = torch.empty(0).to(time_label.device)
    label_durations_flat = torch.empty(0).to(time_label.device)

    for i in range(len(label_len)):
        lab_len = label_len[i]
        lab_cumulative = time_label[i][:lab_len.long()]
        prev_cumulative_times = torch.cat([torch.tensor([0.0]).to(time_label.device), lab_cumulative[:-1]])
        lab_durations = lab_cumulative - prev_cumulative_times
        pred_cumulative = eta[i][:lab_len.long()]
        prev_cumulative_pred = torch.cat([torch.tensor([0.0]).to(eta.device), pred_cumulative[:-1]])
        pred_durations = pred_cumulative - prev_cumulative_pred
        pred_durations_flat = torch.cat([pred_durations_flat, pred_durations])
        label_durations_flat = torch.cat([label_durations_flat, lab_durations])

    return F.l1_loss(pred_durations_flat, label_durations_flat)


class Uncertainty_loss(nn.Module):
    def __init__(self, num_task):
        super(Uncertainty_loss, self).__init__()
        # learnable per-task uncertainty (sigma) used to weight losses
        sigma = torch.tensor([0.0, 1.5174, 2.1460])
        self.sigma = nn.Parameter(sigma)
        self.num_task = num_task
        self.weights = [0.0, 0.0, 0.0]

    def forward(self, *inputs):
        total_loss = 0
        sigma_clamped = self.sigma.clamp(min=1e-6, max=2.65)
        self.weights = []
        for i, loss in enumerate(inputs):
            sigma_sq = sigma_clamped[i].pow(2)
            weight = torch.exp(-sigma_sq)
            total_loss += weight * loss + sigma_sq
            self.weights.append(weight.item())
        return total_loss
