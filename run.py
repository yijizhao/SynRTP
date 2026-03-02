# -*- coding: utf-8 -*-
import numpy as np
import os
import math
import torch
import argparse
import time
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from utils.eval import Metric
from algorithm.strategy_model_1 import SynRTPDataset as DATASET
from utils.utils import compute_lsd,EarlyStop,dict_merge, get_reinforce_samples,get_samples,batch2input,get_nonzeros_nrl,dir_check,compute_lsd_gdrpo,GDRPOLoss
from utils.utils import eta_mae_loss_calc, Uncertainty_loss
from torch.optim.lr_scheduler import CosineAnnealingLR
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MKL_THREADING_LAYER']  = 'GNU'
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'
import random
random.seed(2021)
np.random.seed(2021)
torch.manual_seed(2021)
torch.cuda.manual_seed(2021)


def get_workspace():
    """
    get the workspace path
    :return:
    """
    cur_path = os.path.abspath(__file__)
    file = os.path.dirname(cur_path)
    file = os.path.dirname(file)
    return file
ws =  get_workspace()


def get_common_params():
    """Build argparse parser with default training and model settings."""
    parser = argparse.ArgumentParser(description='Entry Point of the code')
    parser.add_argument('--is_test', type=bool, default=False, help='test the code')
    parser.add_argument('--is_eval', type=str, default=False, help='True means load existing model')
    # dataset
    parser.add_argument('--min_task_num', type=int, default=0, help = 'minimal number of task')
    parser.add_argument('--max_task_num',  type=int, default=25, help = 'maxmal number of task')
    parser.add_argument('--pad_value', type=int, default=24, help=' the pad value of route label')
    parser.add_argument('--num_worker_logistics', type=int, default=5000, help='number of workers in logistics dataset')
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
    parser.add_argument('--num_epoch', type=int, default=300, help='number of epochs to train (default: 300)')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR', help='learning rate (default: 0.0001)')
    parser.add_argument('--seed', type=int, default=2021, metavar='S', help='random seed (default: 2021)')
    parser.add_argument('--wd', type=float, default=1e-5, help='weight decay')
    parser.add_argument('--early_stop', type=int, default=11, help='early stop at the epoch')
    parser.add_argument('--workers', type=int, default=0, help='number of data loading workers ')
    parser.add_argument('--task', type=str, default='logistics')
    parser.add_argument('--strategy_model_path', type=str, default=None, help='best strategy model path in logistics')
    parser.add_argument('--sort_x_size', type=int, default=8)
    parser.add_argument('--cuda_id', type=int, default=0, help='which cuda to use')
    parser.add_argument('--model', type=str, default='_SynRTP_', help='the model to use')
    parser.add_argument('--long_loss_weight', type=float, default=1.5)
    parser.add_argument('--rl_ratio', type=float, default=0.3) 
    parser.add_argument('--pretrain_strategy', type=int, default=4 , help= ' number of pretrain epochs for strategy model ')

    parser.add_argument('--node_fea_dim', type=int, default=8, help = 'dimension of node input feature')
    parser.add_argument('--edge_fea_dim', type=int, default=4, help = 'dimension of edge input feature')
    parser.add_argument('--hidden_size', type=int, default=32, help = 'the hidden size of route model')
    parser.add_argument('--gcn_num_layers', type=int, default=2)
    parser.add_argument('--mlp_layers', type=int, default=2)
    parser.add_argument('--k_nearest_neighbors', type=str, default='n-1')
    parser.add_argument('--k_min_nodes', type=int, default=3)
    parser.add_argument('--b', type=int, default=2)
    parser.add_argument('--aggregation', type=str, default='mean')
    parser.add_argument('--worker_emb_dim', type=int, default=10)
    parser.add_argument("--temperature", type=float, default=1.0, help='temperature for sampling')
    parser.add_argument("--top_p", type=float, default=0.95, help='top_p for sampling')
    parser.add_argument("--temperature_t", type=float, default=1.0, help='temperature for ETA')
    parser.add_argument("--top_p_t", type=float, default=0.95, help='top_p for ETA')

    # Graphormer
    parser.add_argument("--max_shortest_path_len", type=int, default=25, help='maximum shortest path length')
    parser.add_argument("--graphormer_num_layers", type=int, default=3, help='number of Graphormer layers')
    parser.add_argument("--graphormer_num_heads", type=int, default=4, help='number of attention heads in Graphormer')
    parser.add_argument("--graphormer_dropout_rate", type=float, default=0.0, help='dropout rate in Graphormer') 

    parser.add_argument('--xr_type', type=str, default=None, help='the type of xr')

    return parser


def get_params():
    parser = get_common_params()
    args, _ = parser.parse_known_args()
    if args.train_path is None:
        args.train_path = os.path.join(
            ws, f"datasets/synrtp/{args.dataset}/train.npy"
        )
    if args.val_path is None:
        args.val_path = os.path.join(ws, f"datasets/synrtp/{args.dataset}/val.npy")
    if args.test_path is None:
        args.test_path = os.path.join(ws, f"datasets/synrtp/{args.dataset}/test.npy")

    return args


def collate_fn(batch):
    return  batch


class Attention(nn.Module):
    def __init__(self, dim, use_tanh=False, C=10):
        super(Attention, self).__init__()
        self.use_tanh = use_tanh
        self.project_query = nn.Linear(dim, dim)
        self.project_ref = nn.Conv1d(dim, dim, 1, 1)
        self.C = C
        self.tanh = nn.Tanh()
        self.v = nn.Parameter(torch.FloatTensor(dim))
        self.v.data.uniform_(-(1. / math.sqrt(dim)), 1. / math.sqrt(dim))

    def forward(self, query, ref):
        """Compute attention logits.

        query: [B, dim]
        ref: [N, B, dim] (will be permuted to [B, dim, N])
        returns: (projected_ref, logits)
        """
        ref = ref.permute(1, 2, 0)
        q = self.project_query(query).unsqueeze(2)
        e = self.project_ref(ref)

        expanded_q = q.repeat(1, 1, e.size(2))

        v_view = self.v.unsqueeze(0).expand(expanded_q.size(0), len(self.v)).unsqueeze(1)

        u = torch.bmm(v_view, self.tanh(expanded_q + e)).squeeze(1)
        logits = self.C * self.tanh(u) if self.use_tanh else u
        return e, logits


def test_model(strategy_model,test_dataloader, device, pad_value, params, save2file, mode, G):
    """Evaluate model on a dataloader and return Metric.

    Performs forward passes and updates Metric; returns last evaluator.
    """
    strategy_model.eval()
    evaluators = [Metric([1, 25])]

    with torch.no_grad():
        for batch in tqdm(test_dataloader):
            # unpack batch -> move tensors to device inside batch2input
            V, V_reach_mask, E, start_fea, start_idx, cou_fea, route_label, label_len, A, V_len,  node_in_degree_flat, \
            node_out_degree_flat, shortest_path_distances_flat, shortest_path_edge_features_aggregated_flat, time_label = batch2input(batch, device)

            # V_reach_mask: [B, T, N] (batch, tasks per worker, num nodes)
            B, T, N = V_reach_mask.shape
            # flatten batch-time for per-route labels: [B*T, route_len]
            route_label = route_label.reshape(B*T, -1)

            B_flat = B * T

            # convert per-sample node features to nodes-first layout expected by model
            node_in_degree_flat = node_in_degree_flat.reshape(B_flat, N).permute(1, 0).contiguous()
            node_out_degree_flat = node_out_degree_flat.reshape(B_flat, N).permute(1, 0).contiguous()

            # shortest-path tensors -> [B_flat, N, N] then permute to [N, N, B_flat] nodes-first convention
            shortest_path_distances_flat = shortest_path_distances_flat.reshape(B_flat, N, N).permute(1, 2, 0).contiguous()
            shortest_path_edge_features_aggregated_flat = shortest_path_edge_features_aggregated_flat.reshape(B_flat, N, N, -1).permute(1, 2, 0, 3).contiguous()

            # model forward: returns (sampling probs, predicted pointer sequences, eta predictions)
            probs_sample, pred_pointers, eta_prediction = strategy_model(
                params, V, V_reach_mask, E, start_fea, start_idx, cou_fea, route_label, V_len,
                node_in_degree_flat, node_out_degree_flat, shortest_path_distances_flat, shortest_path_edge_features_aggregated_flat,
                False, G=1)

            # pred_pointers shape -> [B*T, seq_len]; N is sequence length here
            N = pred_pointers.size(-1)
            pred_len = torch.sum((pred_pointers.reshape(-1, N) < N - 1) + 0, dim=1)

            # filter out padded examples and collect evaluation tensors
            route_pred, route_label, label_len, preds_len, eta_pred, eta_label = get_nonzeros_nrl(
                pred_pointers.reshape(-1, N), route_label.reshape(-1, N), label_len.reshape(-1), pred_len,
                eta_prediction, time_label.reshape(-1, N), pad_value)

            # update all evaluators with filtered predictions and labels
            for e in evaluators:
                e.update_route_eta(route_pred, route_label, label_len, eta_pred, eta_label)
    if mode == 'val' :
        return evaluators[-1]

    for e in evaluators:
        params_save = dict_merge([e.route_eta_to_dict(), params])
        params_save['eval_min'], params_save['eval_max'] = e.len_range
        params_save['xr_type'] = params.get('xr_type', None)
        save2file(params_save)
    return evaluators[-1]


if __name__ == "__main__":
    params = vars(get_params())
    device = torch.device(f'cuda:{params["cuda_id"]}' if torch.cuda.is_available() else 'cpu')
    params['device'] = device
    from utils.utils import Utils
    utils = Utils(params)

    train_dataset = DATASET(mode='train', params=params) 
    train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=False, collate_fn=collate_fn, num_workers=params['workers']) 

    val_dataset = DATASET(mode='val', params=params)
    val_loader = DataLoader(val_dataset, batch_size=params['batch_size'], shuffle=False, collate_fn=collate_fn, num_workers=params['workers'])

    test_dataset = DATASET(mode='test', params=params)
    test_loader = DataLoader(test_dataset, batch_size=params['batch_size'], shuffle=False, collate_fn=collate_fn, num_workers=params['workers'])

    print(f'Number of Train:{len(train_dataset)} | Number of val:{len(val_dataset)} | Number of test:{len(test_dataset)}')

    from algorithm.strategy_model_1 import SynRTP, save2file
    strategy_model, save2file = SynRTP, save2file
    strategy_model = strategy_model(params)

    strategy_model_path = params.get('strategy_model_path', None)
    if strategy_model_path != None:
        if os.path.exists(strategy_model_path):
            try:
                print('loaded strategy_model path:', strategy_model_path)
                checkpoint = torch.load(strategy_model_path, map_location=device, weights_only=True)
                strategy_model.load_state_dict(checkpoint['model_state_dict'])
                strategy_model.to(device)
                print('best model loaded !!!')
            except:
                print('load best model failed')
                exit(0)
        else:
            print(f'Model file not found at: {strategy_model_path}')

        test_result = test_model(strategy_model,test_loader, device, params['pad_value'], params, save2file, 'test', G=1)
        print('\n-------------------------------------------------------------')
        print(f'{params["model"]} Evaluation in test:', test_result.to_str())
        with open('tempt.txt', 'a') as file:
            file.write(f'Test Result: {test_result.to_str()} |\n')
        exit(0)

    strategy_model.to(device)
    multi_task_loss_fn = Uncertainty_loss(num_task=3).to(device)
    strategy_optimizer = Adam(list(strategy_model.parameters()) + list(multi_task_loss_fn.parameters()), lr=params['lr'], weight_decay=params['wd'])
    strategy_scheduler = CosineAnnealingLR(strategy_optimizer, T_max=300, eta_min=1e-6)
    strategy_early_stop = EarlyStop(mode='maximize', patience=params['early_stop'])
    strategy_model_name = strategy_model.model_file_name() + f'{time.strftime("%m-%d-%H-%M")}' + params['model'] + str(params['xr_type'])
    strategy_model_path = os.path.join(ws, 
                                   'SynRTP', 
                                   'data', 
                                   'dataset', 
                                   params["dataset"], 
                                   'sort_models', 
                                   strategy_model_name)
    params['strategy_model_path'] = strategy_model_path
    dir_check(strategy_model_path)
    results_file_name = "tempt.txt"
    with open(results_file_name, 'a', encoding='utf-8') as file:
        file.write("\n\n" + strategy_model_name + "_" + params['dataset'] + "\n")

    for epoch in range(0, params['num_epoch']):
        # training loop: stop early if EarlyStop triggered
        if strategy_early_stop.stop_flag:
            break
        postfix = {"epoch": epoch,  "current_loss": 0.0, "loss-R": 0.0, "loss-T": 0.0}
        with tqdm(train_loader, total=len(train_loader), postfix=postfix) as t:
            ave_loss = None
            loss_T = None
            strategy_model.train()
            multi_task_loss_fn.train()
            for i, batch in enumerate(t):
                # unpack batch into tensors on device
                V, V_reach_mask, E, start_fea, start_idx, cou_fea, route_label, label_len, A, V_len,  node_in_degree_flat, \
                node_out_degree_flat, shortest_path_distances_flat, shortest_path_edge_features_aggregated_flat, time_label = batch2input(batch, device)

                # V_reach_mask: [B, T, N] where B=batch size, T=tasks per worker, N=node count
                B, T, N = V_reach_mask.shape
                # flatten batch-time dimensions to B*T for per-route operations
                label_len = label_len.reshape(B*T)
                route_label = route_label.reshape(B*T, -1)
                time_label = time_label.reshape(B*T, -1)
                B_flat = B * T

                # reshape node-level features to nodes-first layout expected by model
                node_in_degree_flat = node_in_degree_flat.reshape(B_flat, N).permute(1, 0).contiguous()
                node_out_degree_flat = node_out_degree_flat.reshape(B_flat, N).permute(1, 0).contiguous()

                shortest_path_distances_flat = shortest_path_distances_flat.reshape(B_flat, N, N).permute(1, 2, 0).contiguous()
                shortest_path_edge_features_aggregated_flat = shortest_path_edge_features_aggregated_flat.reshape(B_flat, N, N, -1).permute(1, 2, 0, 3).contiguous()

                # Pretrain branch: optimize route prediction via supervised objective (LSD loss)
                if epoch < params['pretrain_strategy']:

                    probs_sample, selections, eta_sample = strategy_model(params, V, V_reach_mask, E, start_fea, start_idx, cou_fea, route_label, V_len,
                                                            node_in_degree_flat, node_out_degree_flat, shortest_path_distances_flat, shortest_path_edge_features_aggregated_flat,  False, G=1)
                    # filter out padded examples and prepare selections/labels
                    selections_sample, selections_label, sample_len, probs_sample, eta_sample, time_label = get_samples(selections, route_label,label_len,probs_sample, eta_sample, time_label, pad_value=24)
                    # LSD: location square deviation per-sample
                    lsd_route = compute_lsd(selections_label, selections_sample, sample_len)
                    loss_route = torch.mean(lsd_route.unsqueeze(1).unsqueeze(2) * probs_sample)
                    loss = loss_route

                    if ave_loss is None:
                        ave_loss = loss_route.item()
                    else:
                        ave_loss = ave_loss * i / (i + 1) + loss_route.item() / (i + 1)

                    postfix["loss-R"] = ave_loss
                    postfix["current_loss"] = loss.item()
                    t.set_postfix(**postfix)

                    strategy_optimizer.zero_grad()
                    loss.backward()
                    strategy_optimizer.step()
                else:
                    # RL branch: mix cross-entropy and GDRPO policy optimization
                    probs_sample, selections, eta_sample = strategy_model(params, V, V_reach_mask, E, start_fea, start_idx, cou_fea, route_label, V_len,
                                                              node_in_degree_flat, node_out_degree_flat, shortest_path_distances_flat, shortest_path_edge_features_aggregated_flat,  False, G=1)
                    unrolled = probs_sample.view(-1, probs_sample.size(-1))
                    N = selections.size(-1)
                    cross_entropy_loss = F.cross_entropy(unrolled, route_label.view(-1), ignore_index=params['pad_value'])

                    # sample multiple trajectories for policy gradient
                    paths, old_log_probs, new_log_probs = strategy_model(params, V, V_reach_mask, E, start_fea, start_idx, cou_fea, route_label, V_len,
                                                            node_in_degree_flat, node_out_degree_flat, shortest_path_distances_flat, shortest_path_edge_features_aggregated_flat, True, G=16)

                    selections_greedy = selections.detach()
                    seq_pred_len = torch.sum((selections.reshape(-1, N) < N - 1) + 0, dim=1)
                    # filter valid samples and collect RL-specific tensors
                    pred, pred_greedy, label, label_len_list, rl_log_probs_list, pred_lens, valid_indices, eta_sample, time_label = \
                        get_reinforce_samples(paths, selections_greedy, route_label.reshape(-1, N), label_len.reshape(-1), params['pad_value'], 
                        old_log_probs, seq_pred_len, eta_sample, time_label)

                    # compute rewards (LSD difference) and advantages
                    lsd_errors_sample, lsd_errors_greedy = compute_lsd_gdrpo(label, pred, pred_greedy, label_len_list)
                    advantages = lsd_errors_greedy - lsd_errors_sample

                    # ETA prediction loss
                    eta_mae_loss = eta_mae_loss_calc(time_label, label_len_list, eta_sample)

                    # policy loss (GDRPO) and combined multi-task loss
                    loss_fn = GDRPOLoss()
                    loss_gdrpo = loss_fn(new_log_probs[valid_indices], old_log_probs[valid_indices].detach(), advantages)
                    current_total_loss = multi_task_loss_fn(cross_entropy_loss, loss_gdrpo, eta_mae_loss)

                    strategy_optimizer.zero_grad()
                    current_total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(strategy_model.parameters(), 1.0)
                    strategy_optimizer.step()

                    if ave_loss is None:
                        ave_loss = loss_gdrpo.item()
                    else:
                        ave_loss = ave_loss * i / (i + 1) + loss_gdrpo.item()/ (i + 1)

                    if loss_T is None:
                        loss_T = eta_mae_loss.item()
                    else:
                        loss_T = loss_T * i / (i + 1) + eta_mae_loss.item()/ (i + 1)

                    postfix["loss-R"] = ave_loss * 0.05
                    postfix["current_loss"] = current_total_loss.item()
                    postfix["loss-T"] = loss_T * 0.001
                    t.set_postfix(**postfix)

        val_result = test_model(strategy_model,val_loader, device, params['pad_value'], params, save2file, 'val', G=1)
        is_best_change = strategy_early_stop.append(round(val_result.to_dict()['krc'], 4))  

        print('\nval result:', val_result.to_str(), 'Best krc:', round(strategy_early_stop.best_metric(),4), '| Best epoch:', strategy_early_stop.best_epoch)
        with open(results_file_name, 'a', encoding='utf-8') as file:
            file.write(f'Epoch {epoch}: {val_result.to_str()} | Best krc: {round(strategy_early_stop.best_metric(), 4)} | Best epoch: {strategy_early_stop.best_epoch}\n')

        if is_best_change:
            print('value:',val_result.to_dict()['krc'], strategy_early_stop.best_metric())
            torch.save({
                        'epoch': epoch,
                        'model_state_dict': strategy_model.state_dict(),
                        'optimizer_state_dict': strategy_optimizer.state_dict(),
                        'scheduler_state_dict': strategy_scheduler.state_dict(),
                    }, strategy_model_path)
            print('model path:', strategy_model_path)

        strategy_scheduler.step()  

    try:
        print('loaded model path:', strategy_model_path)
        checkpoint = torch.load(strategy_model_path)
        strategy_model.load_state_dict(checkpoint['model_state_dict'])
        print('best model loaded !!!')
    except:
        print('load best model failed')

    test_result = test_model(strategy_model,test_loader, device, params['pad_value'], params, save2file, 'test', G=1)
    print('Best epoch: ', strategy_early_stop.best_epoch)
    print(f'{params["model"]} Evaluation in test:', test_result.to_str())
    with open(results_file_name, 'a', encoding='utf-8') as file:
        file.write(f'Test Result: {test_result.to_str()} | Best krc: {round(strategy_early_stop.best_metric(), 4)} | Best epoch: {strategy_early_stop.best_epoch}\n')
