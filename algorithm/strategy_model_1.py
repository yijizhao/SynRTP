# -*- coding: utf-8 -*-
import time
import numpy as np
import math
import torch
import torch.nn as nn
from algorithm.graphormer_encoder import GraphormerEncoder
from torch.autograd import Variable
from torch.distributions import Categorical
import torch.nn.functional as F

class Attention(nn.Module):
    """Simple additive-style attention used for glimpse and pointer.

    Works with nodes-first context tensors: context shape [N, B, dim].
    Returns (projected_ref, logits).
    """
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
        # ref expected shape: [N, B, dim]; project to [B, dim, N] for conv1d
        ref = ref.permute(1, 2, 0).contiguous()

        q = self.project_query(query).unsqueeze(2)  
        e = self.project_ref(ref)  

        expanded_q = q.repeat(1, 1, e.size(2))  
        
        v_view = self.v.unsqueeze(0).expand(
            expanded_q.size(0), len(self.v)).unsqueeze(1)  

        u = torch.bmm(v_view, self.tanh(expanded_q + e)).squeeze(1)

        logits = self.C * self.tanh(u) if self.use_tanh else u

        return e, logits


class SynRTP(nn.Module):
    def __init__(self, config):
        super(SynRTP, self).__init__()
        self.config = config
        self.N = config['max_task_num']  
        self.device = config['device']
        self.hidden_size = config['hidden_size'] 
        self.max_seq_len = config['max_task_num']

        self.temperature = config['temperature']
        self.top_p =  config['top_p']
        self.T_t = config['temperature_t']
        self.P_t = config['top_p_t']

        # input feature dimension
        self.d_v = config.get('node_fea_dim', 8)  
        self.d_e = config.get('edge_fea_dim', 4)  
        self.d_s = config.get('start_fea_dim', 5)  
        self.d_h = config['hidden_size']  
        self.d_w = config.get('worker_emb_dim', 10)  

        # feature embedding module
        self.worker_emb = nn.Embedding(config['num_worker_logistics'], self.d_w).to(self.device)
        self.node_emb = nn.Linear(self.d_v, self.d_h, bias=False).to(self.device)  
        self.edge_emb = nn.Linear(self.d_e, self.d_h, bias=False).to(self.device) 
        self.start_node_emb = nn.Linear(self.d_s, self.d_h + self.d_v).to(self.device)
        self.graphormer_num_layers = config['graphormer_num_layers'] 
        self.graphormer_num_heads = config['graphormer_num_heads'] 
        self.graphormer_dropout_rate = config.get('graphormer_dropout_rate', 0.1)

        self.max_shortest_path_len = self.N 
        self.max_degree =  self.N 
        self.edge_path_feature_dim = self.d_e
        # Graphormer encoder: returns node-level representations with nodes-first layout (N, B*T, hidden)
        self.graphormer_encoder = GraphormerEncoder(
            hidden_dim=self.d_h,
            num_heads=self.graphormer_num_heads,
            num_layers=self.graphormer_num_layers,
            dropout_rate=self.graphormer_dropout_rate,
            max_shortest_path_len=self.max_shortest_path_len,
            max_degree=self.max_degree,
            edge_path_feature_dim=self.edge_path_feature_dim,
            device=self.device
        ).to(self.device)

        # aggregate graph-level sequence via GRU then project back to per-node space
        self.graph_gru = nn.GRU(self.N * self.d_h, self.d_h, batch_first=True).to(self.device)
        self.graph_linear = nn.Linear(self.d_h, self.N * self.d_h).to(self.device)

        # decoding module
        start_fea = 5
        self.hidden_dim = self.d_h + self.d_v
        self.embedding_dim = self.d_h + self.d_v
        self.worker_fea_dim = self.d_w + 5
        self.mask_glimpses = True
        self.mask_logits = True
        self.n_glimpses = 1
        self.merge_linear = nn.Linear(self.hidden_dim + 10 + 1, self.hidden_dim).to(self.device) 

        self.first_node_embed = nn.Linear(in_features=start_fea, out_features=self.hidden_dim, bias=False).to(self.device)
        self.gru_path = nn.GRUCell(self.embedding_dim, self.hidden_dim).to(self.device)
        self.gru_eta = nn.GRUCell(self.embedding_dim, self.hidden_dim).to(self.device)
        self.glimpse = Attention(self.hidden_dim, use_tanh=False).to(self.device)
        self.sm = nn.Softmax(dim=1).to(self.device)
        self.pointer = Attention(self.hidden_dim, use_tanh=True, C=10).to(self.device)
        self.eta_linear = nn.Linear(in_features=self.hidden_dim * 2, out_features=1).to(self.device)
        # small MLP for ETA prediction from concatenated route state
        self.eta_linear_plus = nn.Sequential(
            nn.Linear(self.hidden_dim * 4, self.hidden_dim),
            nn.Softplus(),
            nn.Linear(self.hidden_dim, 1),
            nn.Softplus()
        ).to(self.device)


    
    def check_mask(self, mask_):

        def mask_modify(mask):
            all_true = mask.all(1)  
            mask_mask = torch.zeros_like(mask)  
            mask_mask[:, -1] = all_true  
            return mask.masked_fill(mask_mask, False) 
        return mask_modify(mask_)

    def update_mask(self, mask, selected):
        def mask_modify(mask): 
            all_true = mask.all(1) 
            mask_mask = torch.zeros_like(mask)
            mask_mask[:, -1] = all_true 
            return mask.masked_fill(mask_mask, False) 
        result_mask = mask.clone().scatter_(1, selected.unsqueeze(-1), True)
        return mask_modify(result_mask)
    
    def recurrence(self, x, h_in, prev_mask, prev_idxs, context):

        logit_mask = self.update_mask(prev_mask, prev_idxs) if prev_idxs is not None else prev_mask 
        if prev_idxs == None:
            logit_mask = self.check_mask(logit_mask)
        logits, h_out = self.calc_logits(x, h_in, logit_mask, context, self.mask_glimpses, self.mask_logits)

        # log probabilities and probabilities over nodes
        log_p = F.log_softmax(logits, dim=1)
        probs = torch.exp(log_p)
        if not self.mask_logits:

            probs[logit_mask] = 0.
    
        return h_out, logits, log_p, probs, logit_mask
    
    def calc_logits(self, x, h_in, logit_mask, context, mask_glimpses=None, mask_logits=None):

        if mask_glimpses is None:
            mask_glimpses = self.mask_glimpses

        if mask_logits is None:
            mask_logits = self.mask_logits
        # GRU step producing hidden state used for glimpse/pointer
        h_out = self.gru_path(x, h_in)
        g_l = h_out 

        for i in range(self.n_glimpses):
            ref, logits = self.glimpse(g_l, context)
            if mask_glimpses:
                logits[logit_mask] = float('-inf')
            g_l = torch.bmm(ref, self.sm(logits).unsqueeze(2)).squeeze(2)

        # final pointer logits
        _, logits = self.pointer(g_l, context)

        if mask_logits:
            logits[logit_mask] = float('-inf')
        return logits, h_out
    

    def recurrence_eta(self, h_in, next_node_idxs, context, last_node, logits):
        scaled_probs = F.softmax(logits / self.T_t, dim=-1)
        sorted_probs, sorted_indices = torch.sort(scaled_probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        sorted_indices_to_remove = cumulative_probs > self.P_t
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices_to_remove.scatter(-1, sorted_indices, sorted_indices_to_remove)
        scaled_probs = scaled_probs.masked_fill(indices_to_remove, 0.0)
        scaled_probs = scaled_probs / (scaled_probs.sum(dim=-1, keepdim=True) + 1e-9)

        # next_node_idxs: [B] selected node indices
        B = next_node_idxs.shape[0]
        N_nodes, BT_batch_time, hidden_dim = context.shape

        current_node_emb = torch.gather(context, 0, next_node_idxs.view(1, B, 1).expand(1, B, context.shape[2])).squeeze(0)
        current_node_input = current_node_emb

        # GRU step for ETA predictor
        h_out = self.gru_eta(current_node_input.float(), h_in)      
        g_l = h_out

        route_state_hard_info = torch.cat([last_node, current_node_input.float(), g_l], dim=1)

        context_transposed_for_agg = context.permute(1, 0, 2)

        soft_next_node_embedding = torch.sum(scaled_probs.unsqueeze(-1) * context_transposed_for_agg, dim=1)

        final_eta_predictor_input = torch.cat([route_state_hard_info, soft_next_node_embedding], dim=1)
        expected_eta = self.eta_linear_plus(final_eta_predictor_input)

        return h_out, expected_eta, current_node_input



    def decode(self, probs, mask, sample, logits):
        old_log_prob = torch.tensor([0])
        # greedy selection if sample==False, otherwise sample with nucleus/top-p
        if sample == False:
            _, idxs = probs.max(1) 
            assert not mask.gather(1, idxs.unsqueeze(-1)).data.any(), \
                "Decode greedy: infeasible action has maximum probability" 
        elif sample == True: 
            scaled_probs = F.softmax(logits / self.temperature, dim=-1)
                    
            sorted_probs, sorted_indices = torch.sort(scaled_probs, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                 
            sorted_indices_to_remove = cumulative_probs > self.top_p  
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()  
            sorted_indices_to_remove[..., 0] = 0  
            indices_to_remove = sorted_indices_to_remove.scatter(-1, sorted_indices, sorted_indices_to_remove)
            scaled_probs.masked_fill_(indices_to_remove, 0.0)
            scaled_probs = scaled_probs / scaled_probs.sum(dim=-1, keepdim=True)  
            idxs = Categorical(scaled_probs).sample()
            old_log_prob = torch.log(scaled_probs.gather(1, idxs.unsqueeze(1)) + 1e-5).squeeze(1)
        else:
            assert False, "Unknown decode type"

        return idxs, old_log_prob


    def dynamic_gnn_encode(self,
                           V, 
                           node_in_degree_flat, 
                           node_out_degree_flat, 
                           shortest_path_distances_flat, 
                           shortest_path_edge_features_aggregated_flat, 
                           key_padding_mask=None 
                          ):
        B, T, N, d_v = V.shape
        d_h = self.d_h
        B_flat = B * T

        b_V_flat = V.reshape(B_flat, N, d_v)
        node_features_proj = self.node_emb(b_V_flat).permute(1, 0, 2).contiguous() 
 
        graph_encoded_nodes = self.graphormer_encoder(
            x=node_features_proj, 
            node_in_degree=node_in_degree_flat,
            node_out_degree=node_out_degree_flat,
            shortest_path_distances=shortest_path_distances_flat,
            shortest_path_edge_features_aggregated=shortest_path_edge_features_aggregated_flat,
            key_padding_mask=key_padding_mask 
        )

        b_node_h_gru_input = graph_encoded_nodes.permute(1, 0, 2).contiguous() 
        b_node_h_gru_input = b_node_h_gru_input.reshape(B, T, N * d_h) 

        b_node_h, _ = self.graph_gru(b_node_h_gru_input) 
        b_node_h = self.graph_linear(b_node_h) 

        b_node_h_reshaped = b_node_h.reshape(B_flat, N, d_h)
        V_original_flat = V.reshape(B_flat, N, d_v)
        node_H = torch.cat([b_node_h_reshaped, V_original_flat], dim=2)
        
        node_H = node_H.permute(1, 0, 2).contiguous()

        return node_H # (N,B*T,d_h+d_v) 

    def forward(self, params, V, V_reach_mask, E, start_fea, start_idx, cou_fea, route_label, V_len,
                node_in_degree_flat, node_out_degree_flat, shortest_path_distances_flat, shortest_path_edge_features_aggregated_flat, sample, G):

        B, T, N = V_reach_mask.shape
        B_flat = B * T
        actual_N_flat = V_len.reshape(B_flat) 
        # node_padding_mask: [B*T, N] True for padded nodes
        node_padding_mask = torch.arange(N, device=self.device).unsqueeze(0).expand(B_flat, N) >= actual_N_flat.unsqueeze(1)

        # Encode nodes with dynamic GNN/Graphormer -> H shape: (N, B*T, d_h + d_v)
        H = self.dynamic_gnn_encode(
            V, 
            node_in_degree_flat,
            node_out_degree_flat, 
            shortest_path_distances_flat, 
            shortest_path_edge_features_aggregated_flat,
            key_padding_mask=node_padding_mask 
        ) 

        # prepare decoder inputs: per time-step start node embedding
        d_h, d_v = self.d_h, self.d_v
        b_decoder_input = torch.zeros([B, T, d_h + d_v]).to(self.device)
        for t in range(T):
            decoder_input = self.start_node_emb(start_fea[:, t, :])
            b_decoder_input[:, t, :] = decoder_input

        b_init_hx = torch.randn(B * T, d_h + d_v).to(self.device) 
        # flatten batch*time for decoding: masks and context
        b_V_reach_mask = V_reach_mask.reshape(B * T, N)
        b_inputs = H.clone()
        b_enc_h = H.clone()

        decoder_input = b_decoder_input.reshape(B * T, d_h + d_v)
        decoder_input_clone = decoder_input.clone()
        # embedded_inputs/context use nodes-first layout expected by Attention modules
        embedded_inputs = b_inputs.reshape(N, T * B, d_h + d_v)
        hidden = b_init_hx
        context = b_enc_h.reshape(N, T * B, d_h + d_v)
        V_reach_mask_t = b_V_reach_mask
        start_idx = start_idx.reshape(B*T)

        B = V_reach_mask_t.size()[0]
        N = V_reach_mask_t.size()[1]
        log_p_list = []
        selections = []
        eta_prediction = []
        steps = range(embedded_inputs.size(0))
        idxs = start_idx
        # mask: [B*T, N] boolean reachable mask for each decoding step
        mask = Variable(V_reach_mask_t, requires_grad=False)
        first_node_embed = self.first_node_embed(start_fea.float())
        first_node_input = first_node_embed.reshape(-1, first_node_embed.size()[2])
        # init ETA predictor hidden state and route accumulator
        hidden_eta = hidden.clone()
        hidden_eta = self.gru_eta(first_node_input.float(), hidden_eta)
        current_eta = 0
        last_node = first_node_input.float()

        # Single-sample greedy/sampling decode (G==1)
        if G==1:
            for i in steps: 
                hidden, logits, log_p, probs, mask = self.recurrence(decoder_input, hidden, mask, idxs, context)
                idxs, _ = self.decode(
                    probs,
                    mask,
                    sample,
                    logits
                )

                # predict ETA incrementally using soft distribution over next nodes
                hidden_eta, eta_duration_pred, last_node = self.recurrence_eta(hidden_eta, idxs, context, last_node, logits)
                current_eta = current_eta + eta_duration_pred


                decoder_input = torch.gather(
                    embedded_inputs,  
                    0,
                    idxs.contiguous().view(1, B, 1).expand(1, B, embedded_inputs.size()[2])
                ).squeeze(0)  

                log_p_list.append(log_p)
                selections.append(idxs)
                # record running ETA prediction per decoding step
                eta_prediction.append(current_eta)


            # return log-probabilities exp -> probabilities, selected indices and ETA sequence
            probs_list = (torch.stack(log_p_list, 1)).exp()
            selections = torch.stack(selections, 1)
            eta = torch.stack(eta_prediction, 1).squeeze(-1)

            return probs_list, selections, eta
    
        else:

            all_selections = []  
            all_old_log_probs = []   
            all_new_log_probs = [] 

            # Multi-sample branch: sample G routes per instance and return their log-probs 
            for g in range(G):
                hidden = b_init_hx.clone()  
                mask = V_reach_mask_t.clone().detach()  
                idxs = start_idx.clone() 
                decoder_input = decoder_input_clone.clone() 
                selections = []    
                old_log_probs  = [] 
                new_log_probs = []  
                
                for i in steps:
                    hidden, logits, log_p, probs, mask = self.recurrence(decoder_input, hidden, mask, idxs, context )
                
                    idxs, old_log_prob = self.decode(
                        probs,
                        mask,
                        sample, 
                        logits
                    )
                    # old_log_prob: log prob from sampling distribution; new_log_prob: model log_p at chosen idx
                    old_log_probs .append(old_log_prob)
                    new_log_prob = log_p.gather(1, idxs.unsqueeze(1)).squeeze(1)
                    new_log_probs.append(new_log_prob)

                    decoder_input = torch.gather(
                        embedded_inputs,  
                        0,
                        idxs.contiguous().view(1, B, 1).expand(1, B, embedded_inputs.size()[2])
                    ).squeeze(0)
                    selections.append(idxs)

                # accumulate per-step log-probs to path-level sums: shape [B*T]
                # path_old_log_probs: sum of sampled (old) log-probs; used for importance correction
                path_old_log_probs = torch.stack(old_log_probs, 1).sum(dim=1)  # [B*T]
                path_new_log_probs = torch.stack(new_log_probs, 1).sum(dim=1)  # [B*T]
                
                all_selections.append(torch.stack(selections, 1))  # [B*T, N_nodes]
                all_old_log_probs.append(path_old_log_probs)         # [B*T]
                all_new_log_probs.append(path_new_log_probs)         # [B*T]
            
            # outputs
            selections = torch.stack(all_selections, 1)
            old_log_probs = torch.stack(all_old_log_probs, 1)
            new_log_probs = torch.stack(all_new_log_probs, 1)

            return selections, old_log_probs, new_log_probs


    
    def model_file_name(self):
        t = time.time()
        file_name = '+'.join([f'{k}-{self.config[k]}' for k in ['hidden_size']])
        file_name = f'{file_name}.'
        return file_name

from torch.utils.data import Dataset
class SynRTPDataset(Dataset):
    def __init__(
            self,
            mode: str,
            params: dict,
    ) -> None:
        super().__init__()
        if mode not in ["train", "val", "test"]:
            raise ValueError
        path_key = {'train': 'train_path', 'val': 'val_path', 'test': 'test_path'}[mode]
        path = params[path_key]
        # Load preprocessed numpy dataset (dict-like) from path
        # Expected keys include 'V','V_len','V_reach_mask', etc.
        self.data = np.load(path, allow_pickle=True).item()

    def __len__(self):
        return self.data['V'].shape[0]   

    def __getitem__(self, idx):
        V = torch.from_numpy(self.data['V'][idx]).float()
        V_len = torch.from_numpy(self.data['V_len'][idx]).long()
        V_reach_mask = torch.from_numpy(self.data['V_reach_mask'][idx]).bool()
        E_static_fea = torch.from_numpy(self.data['E_static_fea'][idx]).float()
        A = torch.from_numpy(self.data['A'][idx]).float() 
        start_fea = torch.from_numpy(self.data['start_fea'][idx]).float()
        start_idx = torch.from_numpy(self.data['start_idx'][idx]).long()
        cou_fea = torch.from_numpy(self.data['cou_fea'][idx]).long() 
        route_label = torch.from_numpy(self.data['route_label'][idx]).long()
        label_len = torch.from_numpy(self.data['label_len'][idx]).long()
        time_label = torch.from_numpy(self.data['time_label'][idx]).long()
            
        node_in_degree = torch.from_numpy(self.data['node_in_degree'][idx]).long() 
        node_out_degree = torch.from_numpy(self.data['node_out_degree'][idx]).long()
        shortest_path_distances = torch.from_numpy(self.data['shortest_path_distances'][idx]).long()
        shortest_path_edge_features_aggregated = torch.from_numpy(self.data['shortest_path_edge_features_aggregated'][idx]).float()

        return (V, V_reach_mask, E_static_fea, start_fea, start_idx, cou_fea, 
                    route_label, label_len, A, V_len, time_label,
                    node_in_degree, node_out_degree, shortest_path_distances, shortest_path_edge_features_aggregated)    
                   
from run import ws
import os
from utils.utils import save2file_meta
def save2file(params):
    # Write experiment meta/results header to CSV under workspace output
    file_name = os.path.join(ws,'SynRTP','output', f'{params["model"]}.csv')                             
    head = [
        'dataset', 'min_task_num', 'max_task_num', 'eval_min', 'eval_max',
        'model', 'hidden_size','k_nearest_neighbors', 'long_loss_weight',
        'num_epoch', 'batch_size', 'lr', 'wd', 'early_stop', 'is_test', 'log_time',
        'lsd', 'lmd', 'krc', 'hr@1', 'hr@2', 'hr@3', 'hr@4', 'hr@5', 'hr@6', 'hr@7', 'hr@8', 'hr@9', 'hr@10',
        'ed', 'acc@1', 'acc@2', 'acc@3', 'acc@4', 'acc@5', 'acc@6', 'acc@7', 'acc@8', 'acc@9', 'acc@10',
        'mae', 'mape', 'rmse', 'acc_eta@10', 'acc_eta@20', 'acc_eta@30', 'xr_type',
    ]
    save2file_meta(params,file_name,head)
